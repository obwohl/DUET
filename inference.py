# -*- coding: utf-8 -*-
"""
A simple, standalone script to perform inference using a pre-trained DUET model checkpoint.

This script is designed to:
1. Load a multivariate time series from a long-format CSV file.
2. Load a DUET model from a saved checkpoint file (.pt).
3. Use the specific hyperparameters the model was trained with.
4. Perform a forecast for a given horizon.
5. Save the prediction to a new CSV file.
"""
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- Important: Adjust these imports based on your project structure ---
# This assumes you run the script from the root of the DUET project directory.
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ts_benchmark.baselines.duet.duet import TransformerConfig

# ========================== CONFIGURATION ==========================
# --- Path Settings ---
# 1. Path to your trained model checkpoint file.
CHECKPOINT_PATH = "inference_checkpoints/0104592.pt" 

# 2. Path to your input data CSV file (in long format).
INPUT_CSV_PATH = "inference_live.csv" # TODO: UPDATE THIS PATH

# 3. Path where the output forecast CSV will be saved.
OUTPUT_CSV_PATH = "forecast_output.csv"

# --- Data Format Settings ---
# 4. Column names in your input CSV.
DATE_COL = 'date'
VALUE_COL = 'data'
SERIES_NAME_COL = 'cols'

# 5. The names of the time series columns in the desired order for the model.
#    This MUST match the order used during training.
SERIES_ORDER = ['wassertemp', 'airtemp']

# --- Model Hyperparameter Settings ---
# 6. Fill this dictionary with the EXACT hyperparameters from the Optuna trial
#    that produced your best model checkpoint.
#    Using the wrong parameters will lead to errors or poor results.
MODEL_HYPERPARAMS = {
    "seq_len": 192,
    "horizon": 96,
    "patch_len": 48,
    "d_model": 128,
    "d_ff": 128,
    "e_layers": 1,
    "n_heads": 8,
    "dropout": 0.0,
    "fc_dropout": 0.0,
    "num_experts": 4,
    "k": 4,
    "factor": 3,
    "CI": True,
    "norm": True, # If True, scaling will be applied.
    # --- These parameters are often fixed but required by the model config ---
    "enc_in": len(SERIES_ORDER),
    "dec_in": len(SERIES_ORDER),
    "c_out": len(SERIES_ORDER),
    "freq": "h", # Adjust if your data has a different frequency (e.g., 'd' for daily)
    "label_len": 144, # Typically seq_len / 2
    "output_attention": False,
    "lradj": "cosine_warmup",
    "loss": "MSE",
}
# ===================================================================

def load_and_prepare_data(csv_path: str, seq_len: int) -> pd.DataFrame:
    """
    Loads data from a long-format CSV, pivots it to wide format,
    and extracts the last `seq_len` timesteps needed for inference.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df_long = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {csv_path}")
        return None

    # Pivot from long format (date, value, series_name) to wide format (date, series1, series2, ...)
    print("Pivoting data from long to wide format...")
    df_wide = df_long.pivot(index=DATE_COL, columns=SERIES_NAME_COL, values=VALUE_COL)

    # Ensure columns are in the correct order
    df_wide = df_wide[SERIES_ORDER]

    # Convert index to datetime objects
    df_wide.index = pd.to_datetime(df_wide.index)

    # Check for missing values
    if df_wide.isnull().values.any():
        print("WARNING: Missing values detected. Filling with forward-fill.")
        df_wide.fillna(method='ffill', inplace=True)
        df_wide.fillna(method='bfill', inplace=True) # For any NaNs at the start

    # The model needs an input of length `seq_len`
    if len(df_wide) < seq_len:
        print(f"ERROR: Data has only {len(df_wide)} rows, but model requires `seq_len` of {seq_len}.")
        return None

    # Get the last `seq_len` data points as input for the forecast
    input_data = df_wide.iloc[-seq_len:]
    print(f"Using last {len(input_data)} data points for inference (from {input_data.index.min()} to {input_data.index.max()}).")
    return input_data

def main():
    """Main function to run the inference process."""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'. Please update the path.")
        return

    # 1. Load and prepare the input data
    config = TransformerConfig(**MODEL_HYPERPARAMS)
    input_df = load_and_prepare_data(INPUT_CSV_PATH, config.seq_len)
    if input_df is None:
        return

    # 2. Scale the data
    # IMPORTANT: For best results, you should use the scaler that was fitted on the
    # original training data. For this simple script, we fit a new scaler on the
    # input data. This is a simplification but works for a standalone example.
    scaler = StandardScaler()
    scaled_input_values = scaler.fit_transform(input_df.values)

    # 3. Initialize model and load checkpoint
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = DUETModel(config)
    print(f"Loading model checkpoint from {CHECKPOINT_PATH} to device '{device}'...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Perform inference
    print("Performing forecast...")
    input_tensor = torch.tensor(scaled_input_values, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        # The model returns (prediction, importance_loss)
        prediction_scaled_tensor, _ = model(input_tensor)

    # Get the numpy array from the tensor and remove the batch dimension
    prediction_scaled = prediction_scaled_tensor.squeeze(0).cpu().numpy()

    # 5. Inverse scale the prediction to get the actual values
    prediction_values = scaler.inverse_transform(prediction_scaled)

    # 6. Format and save the output
    # Create a future date range for the forecast
    last_timestamp = input_df.index[-1]
    forecast_index = pd.date_range(
        start=last_timestamp,
        periods=config.horizon + 1,
        freq=config.freq.upper()
    )[1:] # Exclude the start date itself

    # Create a DataFrame for the forecast
    df_forecast = pd.DataFrame(
        prediction_values,
        index=forecast_index,
        columns=[f"{col}_pred" for col in input_df.columns]
    )

    df_forecast.to_csv(OUTPUT_CSV_PATH)
    print(f"\nSuccess! Forecast saved to '{OUTPUT_CSV_PATH}'")
    print("Forecast preview:")
    print(df_forecast.head())

if __name__ == '__main__':
    main()
