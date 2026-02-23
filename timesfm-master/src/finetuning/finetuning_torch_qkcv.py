"""
TimesFM Finetuner: A flexible framework for finetuning TimesFM models on custom datasets.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from timesfm.pytorch_patched_decoder import create_quantiles
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb

import numpy as np
import pandas as pd


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

def wpe_func(forecast_base_ori, real='y', forecast='forecast'):
    objective_metric = 0
    forecast_base = forecast_base_ori.dropna()
    

    wpe = (
        forecast_base
        # [(forecast_base.forecast_step <= horizon)]
        .groupby(by=["unique_id"], as_index=False)
        .agg({real: "sum", forecast: "sum"})
    )

    wpe["gap_qty"] = abs(wpe[real] - wpe[forecast])
    wpe["wpe_qty"] = wpe["gap_qty"] / wpe[real]

    wpe_all = wpe.agg(
        {"gap_qty": "sum", real: "sum",}
    )
    wpe_all["wpe_qty"] = (
        wpe_all["gap_qty"] / wpe_all[real]
    )

    op = wpe_all["gap_qty"] / wpe_all[real]

    return op, wpe, wpe_all


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



def post_predictions(predictions, y_future):

  predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
  pred_vals = predictions_mean[:, -1, :] 
  _h_op_tunc = y_future.T.shape[0]
  pred_vals_tunc = pred_vals[:, :_h_op_tunc]
  print(f'using _h_op_tunc {_h_op_tunc}, pred_vals_tunc {pred_vals_tunc.shape}')

  df_merged = post_transform(y_future, pred_vals_tunc)
  return df_merged, pred_vals_tunc


class MetricsLogger(ABC):
  """Abstract base class for logging metrics during training.

    This class defines the interface for logging metrics during model training.
    Concrete implementations can log to different backends (e.g., WandB, TensorBoard).
    """

  @abstractmethod
  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:
    """Log metrics to the specified backend.

        Args:
          metrics: Dictionary containing metric names and values.
          step: Optional step number or epoch for the metrics.
        """
    pass

  @abstractmethod
  def close(self) -> None:
    """Clean up any resources used by the logger."""
    pass


class WandBLogger(MetricsLogger):
  """Weights & Biases implementation of metrics logging.

    Args:
      project: Name of the W&B project.
      config: Configuration dictionary to log.
      rank: Process rank in distributed training.
    """

  def __init__(self, project: str, config: Dict[str, Any], rank: int = 0):
    self.rank = rank
    if rank == 0:
      wandb.init(project=project, config=config)

  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:
    """Log metrics to W&B if on the main process.

        Args:
          metrics: Dictionary of metrics to log.
          step: Current training step or epoch.
        """
    if self.rank == 0:
      wandb.log(metrics, step=step)

  def close(self) -> None:
    """Finish the W&B run if on the main process."""
    if self.rank == 0:
      wandb.finish()


class DistributedManager:
  """Manages distributed training setup and cleanup.

    Args:
      world_size: Total number of processes.
      rank: Process rank.
      master_addr: Address of the master process.
      master_port: Port for distributed communication.
      backend: PyTorch distributed backend to use.
    """

  def __init__(
      self,
      world_size: int,
      rank: int,
      master_addr: str = "localhost",
      master_port: str = "12358",
      backend: str = "nccl",
  ):
    self.world_size = world_size
    self.rank = rank
    self.master_addr = master_addr
    self.master_port = master_port
    self.backend = backend

  def setup(self) -> None:
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = self.master_addr
    os.environ["MASTER_PORT"] = self.master_port

    if not dist.is_initialized():
      dist.init_process_group(backend=self.backend,
                              world_size=self.world_size,
                              rank=self.rank)

  def cleanup(self) -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
      dist.destroy_process_group()


@dataclass
class FinetuningConfig:
  """Configuration for model training.

    Args:
      batch_size: Number of samples per batch.
      num_epochs: Number of training epochs.
      learning_rate: Initial learning rate.
      weight_decay: L2 regularization factor.
      freq_type: Frequency, can be [0, 1, 2].
      use_quantile_loss: bool = False  # Flag to enable/disable quantile loss
      quantiles: Optional[List[float]] = None
      device: Device to train on ('cuda' or 'cpu').
      distributed: Whether to use distributed training.
      gpu_ids: List of GPU IDs to use.
      master_port: Port for distributed training.
      master_addr: Address for distributed training.
      use_wandb: Whether to use Weights & Biases logging.
      wandb_project: W&B project name.
      log_every_n_steps: Log metrics every N steps (batches), this is inspired from Pytorch Lightning
      val_check_interval: How often within one training epoch to check val metrics. (also from Pytorch Lightning)
        Can be: float (0.0-1.0): fraction of epoch (e.g., 0.5 = validate twice per epoch)
                int: validate every N batches
    """

  batch_size: int = 32
  num_epochs: int = 20
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  freq_type: int = 0
  use_quantile_loss: bool = False
  quantiles: Optional[List[float]] = None
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  distributed: bool = False
  gpu_ids: List[int] = field(default_factory=lambda: [0])
  master_port: str = "12358"
  master_addr: str = "localhost"
  use_wandb: bool = False
  wandb_project: str = "timesfm-finetuning"
  log_every_n_steps: int = 500
  val_check_interval: float = 0.5

  max_steps: int = -1


class TimesFMFinetuner_qkcv:
  """Handles model training and validation.

    Args:
      model: PyTorch model to train.
      config: Training configuration.
      rank: Process rank for distributed training.
      loss_fn: Loss function (defaults to MSE).
      logger: Optional logging.Logger instance.
    """

  def __init__(
      self,
      model: nn.Module,
      config: FinetuningConfig,
      rank: int = 0,
      loss_fn: Optional[Callable] = None,
      logger: Optional[logging.Logger] = None,
  ):
    self.model = model
    self.config = config
    self._remaining_steps = self.config.max_steps
    self._remaining_steps_val = max(1, config.log_every_n_steps // 10)
    self.rank = rank
    self.logger = logger or logging.getLogger(__name__)
    self.device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    self.loss_fn = loss_fn or (lambda x, y: torch.mean((x - y.squeeze(-1))**2))

    self.training_matrix = {
        "steps": [],
        "_remaining_steps": [],
        "train_loss": [],
        "val_loss": [],
        "wpe": [],
        "mae": [],
        "total_loss": [],
    }

    self.x_context_ = []
    self.split_predictions= False

    if config.use_wandb:
      self.metrics_logger = WandBLogger(config.wandb_project, config.__dict__,
                                        rank)

    if config.distributed:
      self.dist_manager = DistributedManager(
          world_size=len(config.gpu_ids),
          rank=rank,
          master_addr=config.master_addr,
          master_port=config.master_port,
      )
      self.dist_manager.setup()
      self.model = self._setup_distributed_model()
    

  def _setup_distributed_model(self) -> nn.Module:
    """Configure model for distributed training."""
    self.model = self.model.to(self.device)
    return DDP(self.model,
               device_ids=[self.config.gpu_ids[self.rank]],
               output_device=self.config.gpu_ids[self.rank])

  def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
    """Create appropriate DataLoader based on training configuration.

        Args:
          dataset: Dataset to create loader for.
          is_train: Whether this is for training (affects shuffling).

        Returns:
          DataLoader instance.
        """
    if self.config.distributed:
      sampler = torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas=len(self.config.gpu_ids),
          rank=dist.get_rank(),
          shuffle=is_train)
    else:
      sampler = None

    return DataLoader(
        dataset,
        batch_size=self.config.batch_size,
        shuffle=(is_train and not self.config.distributed),
        sampler=sampler,
    )

  def _quantile_loss(self, pred: torch.Tensor, actual: torch.Tensor,
                     quantile: float) -> torch.Tensor:
    """Calculates quantile loss.
        Args:
            pred: Predicted values
            actual: Actual values
            quantile: Quantile at which loss is computed
        Returns:
            Quantile loss
        """
    dev = actual - pred
    loss_first = dev * quantile
    loss_second = -dev * (1.0 - quantile)
    return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

  def _process_batch_qkcv(self, batch: List[torch.Tensor]) -> tuple:
    """Process a single batch of data.

        Args:
          batch: List of input tensors.

        Returns:
          Tuple of (loss, predictions).
        """
    # x_context, x_padding, freq, x_future = [
    #     t.to(self.device, non_blocking=True) for t in batch
    # ]

    # print('\r', f'[_process_batch_qkcv] Init done. batch len {len(batch)}', end = '\n', flush=False)

    x_context_features, y_context, x_future_features, y_future, static_features, input_padding, freq = [
        t.to(self.device, non_blocking=True) for t in batch
    ]

    # print('\r', f'[_process_batch_qkcv] Feature done. y_context {len(y_context)}', end = '\n', flush=False)

    x_context = [x_context_features, y_context, x_future_features, y_future, static_features]

    predictions, _, _ = self.model(x_context, input_padding.float(), freq)

    # print('\r', f'[_process_batch_qkcv] Model init done. freq {freq}', end = '\n', flush=False)

    predictions_mean = predictions[..., 0]
    last_patch_pred = predictions_mean[:, -1, :self.model.input_horizon_len]

    # print(f"\r[_process_batch_qkcv] predictions {predictions.shape}, predictions_mean {predictions_mean.shape}, last_patch_pred {last_patch_pred.shape}, y_future {y_future.shape}", end = '\n', flush=True)

    loss = self.loss_fn(last_patch_pred, y_future.squeeze(-1)[..., :self.model.input_horizon_len])
    if self.config.use_quantile_loss:
      # print("\r", f'[_process_batch_qkcv] use_quantile_loss. predictions {predictions.shape}', end = '\n', flush=True)
      quantiles = self.config.quantiles or create_quantiles()
      for i, quantile in enumerate(quantiles):
        last_patch_quantile = predictions[:, -1, :, i + 1]
        loss += torch.mean(
            self._quantile_loss(last_patch_quantile[..., :self.model.input_horizon_len], y_future.squeeze(-1)[..., :self.model.input_horizon_len],
                                quantile))

    return loss, predictions

  def _train_epoch(self, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   val_loader: DataLoader,) -> float:
    """Train for one epoch in a distributed setting.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer instance.

        Returns:
            Average training loss for the epoch.
        """
    
    total_loss = 0.0
    num_batches = len(train_loader)

    # print('\r', f'[_train_epoch] init train() done. num_batches {num_batches}', end = '\n', flush=False)

    for batch in train_loader:
      if self._remaining_steps == 0:
        print('\r', f'[_train_epoch] _remaining_steps == 0, train_loss {loss.item()}, total_loss {total_loss} break', end = '\n', flush=False)
        break

      self.model.train()
      loss, _ = self._process_batch_qkcv(batch)
      # print('\r', f'[_train_epoch] batch done. _remaining_steps {self._remaining_steps}, loss.item() {loss.item()}', end = '\n', flush=False)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      self.training_matrix["steps"].append(self.config.max_steps - self._remaining_steps+1)
      self.training_matrix["_remaining_steps"].append(self._remaining_steps-1)
      self.training_matrix["train_loss"].append(loss.item())
      self.training_matrix["total_loss"].append(total_loss)

      self._remaining_steps -= 1

      if self._remaining_steps % 500 == 0:
        wpe, mae, ca = self._test_matrix()
        print('\r', f'[_train_epoch] log metrics. _remaining_steps {self._remaining_steps}, ca {ca.shape}, wpe \t{wpe}\t{mae}', end = '\n', flush=False)

        self.training_matrix["wpe"].append(wpe)
        self.training_matrix["mae"].append(mae)

      else:
        self.training_matrix["wpe"].append(0.00)
        self.training_matrix["mae"].append(0.00)

      if self._remaining_steps % self.config.log_every_n_steps == 0:
        val_loss = self._validate(val_loader)
        print('\r', f'[_train_epoch] log metrics. _remaining_steps {self._remaining_steps}, train_loss {loss.item()}, val_loss {val_loss}', end = '', flush=True)
        self.training_matrix["val_loss"].append(val_loss)
        
      else:
        self.training_matrix["val_loss"].append(0.00)

    print('\r', f'[_train_epoch] all batch done. total_loss {total_loss}', end = '\n', flush=False)

    avg_loss = total_loss / num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss

  def _validate(self, val_loader: DataLoader) -> float:
    """Perform validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
    self.model.eval()
    total_loss = 0.0
    num_batches = self._remaining_steps_val #len(val_loader)

    with torch.no_grad():
      for batch in val_loader:
        if self._remaining_steps_val == 0:
          # print('\r', f'[_validate] _remaining_steps_val == 0, total_loss {total_loss}, num_batches {num_batches}, break', end = '\n', flush=False)
          break
        loss, _ = self._process_batch_qkcv(batch)
        total_loss += loss.item()
        self._remaining_steps_val -= 1
        # print('\r', f'[_validate] _remaining_steps_val={self._remaining_steps_val}, loss {loss.item()}, num_batches {num_batches}, break', end = '\n', flush=True)

    avg_loss = total_loss / num_batches
    self._remaining_steps_val = num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss
  

  def _prep_test_data(self,
                      Y_train_df,
                      Y_test_df,
                      _df_static_numeric,
                      _context_len,
                      split_predictions,
                      ):

    ### Prepare the test data
    self.x_context = []

    y_context=pd.DataFrame()
    y_future=pd.DataFrame()
    # static_features=[]
    x_context_features=pd.DataFrame()
    # x_future_features=pd.DataFrame()

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

    self.y_future = y_future
    _df_static_numeric.sort_values(by='unique_id', inplace=True)

    self.split_predictions=split_predictions
    
    if self.split_predictions:
      
      y_context_split = np.array_split(y_context, 10)
      y_future_split = np.array_split(y_future, 10)
      _df_static_numeric_split = np.array_split(_df_static_numeric, 10)
      
      for i in range(10):
        x_context_= [torch.tensor(pd.DataFrame(y_context_split[i]).values, device='cuda', dtype=torch.float32), 
                    torch.tensor(pd.DataFrame(y_context_split[i]).values, device='cuda', dtype=torch.float32),
                    torch.tensor(pd.DataFrame(y_future_split[i]).values, device='cuda', dtype=torch.float32),
                    torch.tensor(pd.DataFrame(y_future_split[i]).values, device='cuda', dtype=torch.float32),
                    torch.tensor(pd.DataFrame(_df_static_numeric_split[i].drop(columns=['unique_id',])).values, device='cuda', dtype=torch.float32).unsqueeze(1),
                    ]

        self.x_context.append(x_context_)

    else:
      x_context_= [torch.tensor(pd.DataFrame(y_context).values, device='cuda', dtype=torch.float32), 
                  torch.tensor(pd.DataFrame(y_context).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(y_future).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(y_future).values, device='cuda', dtype=torch.float32),
                  torch.tensor(pd.DataFrame(_df_static_numeric.drop(columns=['unique_id'])).values, device='cuda', dtype=torch.float32).unsqueeze(1),
                  ]
      self.x_context.append(x_context_)



  def _test_matrix(self):
    """Test the model on a test dataset."""
    _context_len = self.model.input_horizon_len * 2 #192 #
    self.model.eval()

    # print(f"[prediction_pipeline] predict y_context {y_context.shape} use split: {self.split_predictions}")
      
    if self.split_predictions:
      predictions_split = []
      ca_split = []
      for i in range(10):
        x_context_= self.x_context[i]

        _torch_type = torch.full((x_context_[0].shape[0], 1), fill_value=self.config.freq_type, device='cuda', dtype=torch.long)

        with torch.no_grad():
          _predictions, ca, attention_score = self.model(x_context_, 
                                  torch.zeros_like(
                                      x_context_[0],device='cuda'
                                      ),
                                      _torch_type,
                                  )
          predictions_split.append(_predictions.cpu())   
          ca_split.append(ca.cpu())
      predictions = np.concatenate(predictions_split, axis=0)
      ca= np.concatenate(ca_split, axis=0)

    else:
      x_context_= self.x_context[0]
      _torch_type = torch.full((x_context_[0].shape[0], 1), fill_value=self.config.freq_type, device='cuda', dtype=torch.long)

      with torch.no_grad():
          predictions, ca, attention_score = self.model(x_context_, 
                                  torch.zeros_like(
                                      x_context_[0],device='cuda'
                                      ),
                                      _torch_type,
                                  )
          
      predictions=predictions.cpu().numpy()
      ca=ca.cpu().numpy()

    df_merged, _ = post_predictions(predictions, self.y_future)
    wpe, _, _ = wpe_func(df_merged,forecast='forecast')
    mae = calculate_mae(df_merged)

    return wpe, mae, ca



  def finetune_qkcv(self, train_dataset: Dataset,
               val_dataset: Dataset,
               ) -> Dict[str, Any]:
    """Train the model.

        Args:
          train_dataset: Training dataset.
          val_dataset: Validation dataset.

        Returns:
          Dictionary containing training history.
        """
    
    self.model = self.model.to(self.device)
    train_loader = self._create_dataloader(train_dataset, is_train=True)
    val_loader = self._create_dataloader(val_dataset, is_train=False)

    print('\r', f'[finetune_qkcv] before optimizer, train_loader {len(train_loader)} batches, val_loader {len(val_loader)} batches', end = '\n', flush=False)

    optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=self.config.learning_rate,
                                 weight_decay=self.config.weight_decay)

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    self.logger.info(
        f"Starting training for {self.config.num_epochs} epochs...")
    self.logger.info(f"Training samples: {len(train_dataset)}")
    self.logger.info(f"Validation samples: {len(val_dataset)}")

    try:
      for epoch in range(self.config.num_epochs):
        print('\r', f'[finetune_qkcv] Starting {epoch}/{self.config.num_epochs} epochs...', end = '\n', flush=False)

        train_loss = self._train_epoch(train_loader, optimizer, val_loader)
        print('\r', f'[finetune_qkcv] train_loss {train_loss}', end = '\n', flush=False)
        # val_loss = self._validate(val_loader)
        # print('\r', f'[finetune_qkcv] val_loss {val_loss}', end = '\n', flush=False)
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "train_loss": train_loss,
            # "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
        }

        print('\r', f'[finetune_qkcv] The {epoch}/{self.config.num_epochs} epochs, metrics {metrics}', end = '\n', flush=False)

        if self.config.use_wandb:
          self.metrics_logger.log_metrics(metrics)

        history["train_loss"].append(train_loss)
        # history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # if self.rank == 0:
        #   self.logger.info(
        #       f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        #   )
        if self._remaining_steps == 0:
          print('\r', f'[finetune_qkcv] _remaining_steps == 0, Train Loss: {train_loss:.4f} | Val Loss: , break', end = '\n', flush=False)
          break

    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")

    if self.config.distributed:
      self.dist_manager.cleanup()

    if self.config.use_wandb:
      self.metrics_logger.close()

    return {"history": history}
