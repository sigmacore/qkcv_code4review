# TimesFM

**TimesFM (Time Series Foundation Model)** is a pre-trained time-series foundation model developed by **Google Research** for time-series forecasting tasks.

This repository contains code to:
- Load public TimesFM model checkpoints.
- Run the model with modifications for **QKCV attention**.

---

## Scripts Included

The repository includes three Jupyter Notebook scripts for testing:
1. `timesfm-master/notebooks/Meal_test.ipynb`
2. `timesfm-master/notebooks/Fav_test.ipynb`
3. `timesfm-master/notebooks/M5_test.ipynb`

---

## Configuration Parameters

Each script requires setting **two configuration parameters**: `_v` and `v_qkcv`.

### 1. Parameter `_v` (Tuning Mode)
This parameter controls which parts of the model are tuned during training. Possible values:
- **`1`**: Tune only the patching layer.
- **`2`**: Tune all parameters, including Stacked Transformers.
- **`3`**: Tune the patching layer (same as `1`) while using an **MLP (Multi-Layer Perceptron)** as the static encoder.
- **`4`**: Tune the patching layer (same as `1`) while using an **MLP** as the input encoder, accepting static features alongside the target variable `Y`.
- **`5`**: Tune the patching layer (same as `1`) while using a **TFT (Temporal Fusion Transformer)** as the input encoder, accepting static features alongside the target variable `Y`.

### 2. Parameter `v_qkcv` (QKCV Version)
This parameter specifies the QKCV attention variant to use. Possible values:
- **`-1`**: Disable QKCV (no QKCV attention).
- **`1`**: Use **QKCV v1**.
- **`2`**: Use **QKCV v2**.
- **`3`**: Use **QKCV v3**.