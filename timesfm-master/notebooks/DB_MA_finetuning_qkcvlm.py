from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
# import yfinance as yf
from finetuning.finetuning_torch_qkcv import FinetuningConfig, TimesFMFinetuner_qkcv
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder_qkcv import PatchedTimeSeriesDecoder_qkcv
import os
from os import path

import pandas as pd
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# v_qkcv=2

print(f"torch.__version__ {torch.__version__}")


class Config_qkcv:

  def set_config(self,
                      version: int = 1,
                      random_seed: int = 42,
          ):
    self.version = version
    self.train_FM = False
    self.max_steps = 1000
    self.use_patch_interpolate = 3
    self.use_quantile_loss = True
    self.embedding_type = 0 # 0: TFT, 1: MLP
    self.random_seed = random_seed
    self.batch_size = 256
    self.input_embedding_type = 0 # 0: None, 1: MLP, 2: TFT
    self.train_only = False
    self.use_stride = True
    self.num_layers = 50
    
    
  def __init__(self, 
         version: int = 1,
         ):
    """
    freq_type: 0: T, MIN, H, D, B, U 1: W, M 2: Q, Y
    """
    
    self.v_qkcv = 2
    self.split_predictions = False
    self.freq_type=0

    self.set_config(version)
    print(f"Using config version {version}")
    
    if self.version == 1:
      pass
    elif self.version == 2:
      self.train_FM = True
    elif self.version == 3:
      self.embedding_type = 1
    elif self.version == 4:
      self.input_embedding_type = 1 # MLP
    elif self.version == 5:
      self.input_embedding_type = 2 # TFT

    else:
      raise ValueError(f"Invalid version number. NO Supported versions {version}.")
    
    torch.manual_seed(self.random_seed)

  def get_col_forecast(self):
    return f'{self.v_qkcv}_{1 if self.train_FM else 0}_{self.max_steps}_{self.use_patch_interpolate}_{1 if self.use_quantile_loss else 0}_{self.embedding_type}_{self.input_embedding_type}_{self.num_layers}'


# _col_forecast=f'{v_qkcv}_{1 if _train_FM else 0}_{max_steps}_{use_patch_interpolate}_{1 if use_quantile_loss else 0}_{embedding_type}'
# print(_col_forecast)

# random_seed = 42
# # horizon = 28
# batch_size = 256

###

def calculate_group_sample_counts(groups, context_length, horizon_length, stride=1):
  """
  Precompute the number of samples for each group.
  """
  sample_counts = []
  for _, group in groups:
    series_length = len(group)
    if series_length < context_length + horizon_length:
      sample_counts.append(0)
    else:
      # Adjusted to account for stride
      sample_counts.append(max(0, (series_length - context_length - horizon_length)//stride + 1))
  return sample_counts


class TimeSeriesCompletedDataset(Dataset):
  """
  Optimized Dataset for time series data with additional features.
  """

  def __init__(self, 
          dataframe: pd.DataFrame, 
          static_features: Optional[pd.DataFrame],
          context_length: int, 
          horizon_length: int, 
          freq_type: int,
          stride: int = 1,
          ):
    """
    Initialize the dataset.

    Args:
      dataframe: Input dataframe with columns ['unique_id', 'ds', 'y'] and other feature columns.
      context_length: Number of past timesteps to use as input.
      horizon_length: Number of future timesteps to predict.
      freq_type: Frequency type (0, 1, or 2)
    """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")
    self.dataframe = dataframe
    self.static_features = static_features
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self.stride = stride

    # Group data by 'unique_id' for individual time series
    self.groups = list(dataframe.groupby('unique_id'))
    self.sample_counts = calculate_group_sample_counts(self.groups, context_length, horizon_length, self.stride)
    self.cumulative_counts = np.cumsum(self.sample_counts)

  def __len__(self):
    """
    Return the total number of samples across all time series.
    """
    return self.cumulative_counts[-1] if self.cumulative_counts.size > 0 else 0

  def __getitem__(self, index: int):
    """
    Get a single sample from the dataset.

    Args:
      index: Index of the sample.

    Returns:
      Tuple of (x_context, x_future, y_context, y_future).
    """
    group_idx = np.searchsorted(self.cumulative_counts, index, side='right')
    if group_idx >= len(self.groups):
      raise IndexError("Index out of range")

    group = self.groups[group_idx][1].sort_values('ds')  # Ensure the data is sorted by date
    group_start_idx = index - (self.cumulative_counts[group_idx - 1] if group_idx > 0 else 0)

    start_idx = group_start_idx * self.stride
    end_idx = start_idx + self.context_length
    x_context = group.iloc[start_idx:end_idx]
    x_future = group.iloc[end_idx:end_idx + self.horizon_length]

    if self.static_features is not None:
      static_features = self.static_features[self.static_features['unique_id'] == self.groups[group_idx][0]].drop(columns=['unique_id']).values
    else:
      static_features = None

    # Extract features and target variable
    y_context = torch.tensor(x_context['y'].values, dtype=torch.float32)
    y_future = torch.tensor(x_future['y'].values, dtype=torch.float32)

    x_context_features = torch.tensor(x_context.drop(columns=['unique_id', 'ds', 'y']).values, dtype=torch.float32)
    x_future_features = torch.tensor(x_future.drop(columns=['unique_id', 'ds', 'y']).values, dtype=torch.float32)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    # print(f"[__getitem__] got item:index {index}")

    return x_context_features, y_context, x_future_features, y_future, static_features, torch.zeros_like(y_context), freq


def prepare_datasets(dataframe: pd.DataFrame, 
                     static_features: Optional[pd.DataFrame],
                     context_length: int,
                     horizon_length: int,
                     freq_type: int,
                     train_split: float = 0.8,
                     use_stride = False) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(dataframe) * train_split)
  # train_data = dataframe[:train_size]
  # val_data = dataframe[train_size:]
  # train_data= dataframe.loc[dataframe.ds<(pd.to_datetime(dataframe.ds.max()) - pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')]
  train_data = dataframe

  val_days_delta = 1 if freq_type == 0 else 7
  val_days_delta = val_days_delta *(context_length+horizon_length)
  val_data = dataframe.loc[dataframe.ds>(pd.to_datetime(dataframe.ds.max()) - pd.Timedelta(days=val_days_delta)).strftime('%Y-%m-%d')]

  stride = 1 if not use_stride else val_days_delta//2

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesCompletedDataset(pd.DataFrame(train_data),static_features,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type,
                                    stride=stride)

  val_dataset = TimeSeriesCompletedDataset(pd.DataFrame(val_data),static_features,
                                  context_length=context_length,
                                  horizon_length=horizon_length,
                                  freq_type=freq_type,
                                  stride=stride)
  
  print(f"[prepare_datasets] TimeSeriesCompletedDataset dataframe len {dataframe.ds.nunique()} context_length {context_length} horizon_length {horizon_length}, train_dataset len {len(train_dataset)}, val_dataset len {len(val_dataset)}")
  return train_dataset, val_dataset


def get_model(_features_static,
              config_qkcv,
              horizon,
              load_weights: bool = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  repo_id = "google/timesfm-2.0-500m-pytorch"
  hparams = TimesFmHparams(
      backend=device,
      per_core_batch_size=config_qkcv.batch_size, #32,
      horizon_len=horizon, #128,
      num_layers=config_qkcv.num_layers, #50,
      use_positional_embedding=False,
      context_len=192,  # Context length can be anything up to 2048 in multiples of 32
  )
  tfm = TimesFm(hparams=hparams,
                checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))
  
  # tfm._model_config.horizon_len = horizon
  # tfm._model_config.patch_len =horizon #int(np.ceil(horizon/4))
  # tfm._model_config

  print(f"tfm._model_config {tfm._model_config}")

  model = PatchedTimeSeriesDecoder_qkcv(tfm._model_config,
                                        list_static_feature=_features_static,
                                        v_qkcv=config_qkcv.v_qkcv,
                                        input_patch_len=horizon, 
                                        input_horizon_len=horizon,
                                        use_patch_interpolate=config_qkcv.use_patch_interpolate,
                                        embedding_type=config_qkcv.embedding_type, 
                                        input_embedding_type=config_qkcv.input_embedding_type,
                                        )
  if load_weights:
    checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.patchedTimeSeriesDecoder.load_state_dict(loaded_checkpoint,strict=True)

  if not config_qkcv.train_FM:
    print(f"[get_model] _train_FM: freeze all parameters except ")
    # If not update FM, freeze all parameters except 
    for param in model.parameters():
      param.requires_grad = False
    # Except:
    for param in model.c_Generator.parameters():
      param.requires_grad = True
    if model.use_patch_interpolate==2 or model.use_patch_interpolate==3:
      for param in model.linear_transform_patch_preprocess.parameters():
        param.requires_grad = True
      for param in model.linear_transform_patch_postprocess.parameters():
        param.requires_grad = True

  return model, hparams, tfm._model_config


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions, ca, attention_score = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  pred_vals = last_patch_pred[0].cpu().numpy()

  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()



def get_data(time_series: pd.DataFrame,
             static_features: pd.DataFrame,
             context_len: int = 128,
             horizon_len: int = 28,
             freq_type: int = 0,
             use_stride: bool = False,
             ) -> Tuple[Dataset, Dataset]:
  """Prepare datasets for training and validation."""
  
  train_dataset, val_dataset = prepare_datasets(
      dataframe=time_series,
      static_features=static_features,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.8,
      use_stride=use_stride,
  )

  print(f"[get_data] Created datasets - Training samples: {len(train_dataset)}, - Validation samples: {len(val_dataset)}, - Using frequency type: {freq_type}")

  return train_dataset, val_dataset




###
def prediction_pipeline(tfm_config, horizon, config_qkcv, Y_train_df, Y_test_df, _df_static_numeric, model):
   
  ### Config and dataset

  tfm_config.horizon_len = horizon
  tfm_config.patch_len =horizon #int(np.ceil(tfm_config.horizon_len/4))
  
  _context_len = horizon*2 #192 #

  print(f"[prediction_pipeline] model.input_horizon_len {model.input_horizon_len}, tfm_config: {tfm_config}")
  # print(f"Prep time: {(time.time() - start_time)/60:.2f} mins, tfm_config.horizon_len {tfm_config.horizon_len}, tfm_config.patch_len {tfm_config.patch_len}")

  train_dataset, val_dataset = get_data(Y_train_df, _df_static_numeric,
                                          _context_len,
                                          tfm_config.horizon_len,
                                          freq_type=config_qkcv.freq_type,
                                          use_stride=config_qkcv.use_stride,
                                          )

  ### Train model
  if config_qkcv.v_qkcv<=0 and config_qkcv.train_FM==False and config_qkcv.use_patch_interpolate==4:
      print(f'No grad. Skipped training FM')
      model_tunned=model
      model_tunned.to(device='cuda')
      finetuner=-1
  else:
    config = FinetuningConfig(batch_size=config_qkcv.batch_size,
                        num_epochs=9999,
                        learning_rate=1e-4,
                        use_wandb=False,
                        freq_type=config_qkcv.freq_type,
                        log_every_n_steps=10,
                        val_check_interval=0.5,
                        use_quantile_loss=config_qkcv.use_quantile_loss,
                        max_steps=config_qkcv.max_steps,
                        )

    finetuner = TimesFMFinetuner_qkcv(model, config)
    finetuner._prep_test_data(
                  Y_train_df,
                  Y_test_df,
                  _df_static_numeric,
                  _context_len,
                  config_qkcv.split_predictions,
                  )

    print("\nStarting finetuning...")
    results = finetuner.finetune_qkcv(train_dataset=train_dataset,
                            val_dataset=val_dataset)

    print("\nFinetuning completed!")
    print(f"Training history: {len(results['history']['train_loss'])} epochs")

  
    model_tunned = model

  ### >>>>>>>>> SKIP TESTING <<<<<<<<<<
  if config_qkcv.train_only:
      print("\nSkipping testing...")
      return [], [], model_tunned, finetuner, [], []

  model_tunned.eval()

  ### Prepare the test data
  y_context=pd.DataFrame()
  y_future=pd.DataFrame()
  x_context_features=pd.DataFrame()

  x_context_features = Y_train_df[['unique_id','ds','y']].drop_duplicates().pivot(index='unique_id', columns='ds', values='y').fillna(0)
  x_context_features.reset_index(inplace=True) 
  x_context_features.sort_values(by='unique_id', inplace=True)

  x_context =torch.tensor(x_context_features.drop(columns=['unique_id']).values, dtype=torch.float32)

  y_context=x_context_features.drop(columns=['unique_id']).iloc[:, -_context_len:].values
  # y_context = torch.tensor(x_context_features['y'].values, dtype=torch.float32)

  y_future= Y_test_df[['unique_id','ds','y']].pivot(index='unique_id', columns='ds', values='y').fillna(0)
  y_future.reset_index(inplace=True)  # Optional: Reset index for a cleaner look
  y_future.sort_values(by='unique_id', inplace=True)
  y_future=y_future.drop(columns=['unique_id']).values
  # y_future =torch.tensor(y_future.drop(columns=['unique_id']).values, dtype=torch.float32)

  _df_static_numeric.sort_values(by='unique_id', inplace=True)
  # x_context = [x_context_features, y_context, x_future_features, y_future, static_features]

  print(f"[prediction_pipeline] predict y_context {y_context.shape} use split: {config_qkcv.split_predictions}")

  if config_qkcv.split_predictions:
     
    y_context_split = np.array_split(y_context, 10)
    y_future_split = np.array_split(y_future, 10)
    _df_static_numeric_split = np.array_split(_df_static_numeric, 10)
    
    predictions_split = []
    ca_split = []
    attention_score_split = []

    for i in range(10):
      _torch_type = torch.full((y_context_split[i].shape[0], 1), fill_value=config_qkcv.freq_type, device='cuda', dtype=torch.long)

      x_context_= [torch.tensor(pd.DataFrame(y_context_split[i]).values, device='cuda', dtype=torch.float32), 
                  torch.tensor(pd.DataFrame(y_context_split[i]).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(y_future_split[i]).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(y_future_split[i]).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(_df_static_numeric_split[i].drop(columns=['unique_id',])).values, device='cuda', dtype=torch.float32).unsqueeze(1),
                  ]

      with torch.no_grad():
        _predictions, _ca, _attention_score = model_tunned(x_context_, 
                                torch.zeros_like(
                                    torch.tensor(pd.DataFrame(y_context_split[i]).values, dtype=torch.float32),device='cuda'
                                    ),
                                    _torch_type,
                                # torch.zeros((y_context_split[i].shape[0]), device='cuda', dtype=torch.long).unsqueeze(1),
                                )
        predictions_split.append(_predictions.cpu())
        ca_split.append(_ca.squeeze(1).cpu())
        attention_score_split.append(_attention_score.view(_attention_score.shape[0], -1).cpu())
    predictions = np.concatenate(predictions_split, axis=0)
    ca = np.concatenate(ca_split, axis=0)
    attention_score = np.concatenate(attention_score_split, axis=0)

  else:
    _torch_type = torch.full((y_context.shape[0], 1), fill_value=config_qkcv.freq_type, device='cuda', dtype=torch.long)

    x_context_= [torch.tensor(pd.DataFrame(y_context).values, device='cuda', dtype=torch.float32), 
                torch.tensor(pd.DataFrame(y_context).values, device='cuda', dtype=torch.float32),
                torch.tensor(pd.DataFrame(y_future).values, device='cuda', dtype=torch.float32),
                torch.tensor(pd.DataFrame(y_future).values, device='cuda', dtype=torch.float32),
                torch.tensor(pd.DataFrame(_df_static_numeric.drop(columns=['unique_id'])).values, device='cuda', dtype=torch.float32).unsqueeze(1),
                ]

    with torch.no_grad():
        predictions, ca, attention_score = model_tunned(x_context_, 
                                torch.zeros_like(
                                    torch.tensor(pd.DataFrame(y_context).values, dtype=torch.float32),device='cuda'
                                    ),
                                    _torch_type,
                                # torch.zeros((y_context.shape[0]), device='cuda', dtype=torch.long).unsqueeze(1),
                                )
        
        predictions=predictions.cpu().numpy()
        ca=ca.squeeze(1).cpu().numpy()
        attention_score=attention_score.view(attention_score.shape[0], -1).cpu().numpy()

  
  return predictions, y_future, model_tunned, finetuner, pd.DataFrame(ca), pd.DataFrame(attention_score)




def post_transform(y_future, pred_vals):
    # Assuming df is the given dataframe
    df_forecast = pd.DataFrame(pred_vals)
    # Transform the dataframe
    df_forecast = df_forecast.stack().reset_index()
    df_forecast.columns = ['unique_id', 'ds', 'forecast']
    df_forecast['unique_id'] += 1
    df_forecast['ds'] += 1

    df_real = pd.DataFrame(y_future)
    # Transform the dataframe
    df_real = df_real.stack().reset_index()
    df_real.columns = ['unique_id', 'ds', 'y']
    df_real['unique_id'] += 1
    df_real['ds'] += 1
    
    df_merged = df_forecast.merge(df_real, on=['ds', 'unique_id'], how='inner').fillna(0)
    return df_merged



def post_predictions(predictions, y_future):

  predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
  # last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  # print(predictions_mean.shape)
  # predictions_mean[:, -1, :model_tunned.input_horizon_len].shape

  pred_vals = predictions_mean[:, -1, :] #last_patch_pred #.cpu().numpy()
  # print(pred_vals.shape)
  # print(y_future.T.shape)

  _h_op_tunc = y_future.T.shape[0]

  pred_vals_tunc = pred_vals[:, :_h_op_tunc]
  print(f'using _h_op_tunc {_h_op_tunc}, pred_vals_tunc {pred_vals_tunc.shape}')

  df_merged = post_transform(y_future, pred_vals_tunc)
  return df_merged, pred_vals_tunc


###
def wpe_func(forecast_base_ori, eval_horizon=[28], real='y', forecast='TFT-median'):
    objective_metric = 0
    forecast_base = forecast_base_ori.dropna()
    for horizon in eval_horizon:

        wpe = (
            forecast_base
            # [(forecast_base.forecast_step <= horizon)]
            .groupby(by=["unique_id"], as_index=False)
            .agg({real: "sum", forecast: "sum"})
        )

        wpe["gap_qty"] = abs(wpe[real] - wpe[forecast])
        wpe["wpe_{}W_qty".format(horizon)] = wpe["gap_qty"] / wpe[real]

        wpe_all = wpe.agg(
            {"gap_qty": "sum", real: "sum",}
        )
        wpe_all["wpe_{}W_qty".format(horizon)] = (
            wpe_all["gap_qty"] / wpe_all[real]
        )

        print(wpe_all["gap_qty"] / wpe_all[real])

    return wpe, wpe_all



def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    # print(np.maximum(quantile * error, 0))
    return 2* (
        quantile * np.maximum(error, 0)+ (1 - quantile) * np.maximum(-error, 0)).mean() / np.mean(np.abs(y_true))
    
def calculate_quantile_losses(df_merged, _col_forecast, real_col='y', _pf=''):
    # df_merged = df_forecast.merge(df_real, on=['ds', 'unique_id'], how='left').fillna(0)
    p50_loss = quantile_loss(df_merged[real_col], df_merged[f'{_col_forecast}{_pf}-median'], 0.5)
    p90_loss_lo = quantile_loss(df_merged[real_col], df_merged[f'{_col_forecast}{_pf}-lo-90'], 0.9)
    p90_loss_hi = quantile_loss(df_merged[real_col], df_merged[f'{_col_forecast}{_pf}-hi-90'], 0.9)
    return p50_loss, p90_loss_lo, p90_loss_hi

# Calculate MAE
def calculate_mae(df):
    mae = (df['y'] - df['forecast']).abs().mean()
    return mae

def calculate_matrix(df_with_forecast):
    overall_mse = mean_squared_error(df_with_forecast['y'], df_with_forecast['forecast'])
    print(f"Overall MSE: {overall_mse}")

    # Calculate overall MAE
    overall_mae = mean_absolute_error(df_with_forecast['y'], df_with_forecast['forecast'])
    print(f"Overall MAE: {overall_mae}")


# p50_loss, p90_loss_lo, p90_loss_hi = calculate_quantile_losses(df_merged)
# print(f"P50 Loss: {p50_loss}")
# print(f"P90 Loss lo: {p90_loss_lo}")
# print(f"P90 Loss hi: {p90_loss_hi}")

