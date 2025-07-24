import pandas as pd
import pickle
import base64
import os
import tarfile
import tempfile
import sys
import numpy as np
import json
from pathlib import Path

# ======================= CONFIGURATION =======================
TARGET_FOLDER = 'result/eisbach/DUET'
# The names of the time series features in the order they appear in the data.
# This MUST match the data generation process.
# Based on your description, the order is:
# 0: airtemp_shifted (the leaked feature)
# 1: wassertemp
# 2: airtemp (the target for the copy-task)
SERIES_NAMES = ['airtemp_shifted', 'wassertemp', 'airtemp']
# The index of the feature we want to analyze the "copy" performance for.
TARGET_FEATURE_INDEX = 2
LEAKED_FEATURE_INDEX = 0
# How many steps of the horizon to analyze.
ANALYSIS_HORIZON = 96
# How many random windows to print detailed stats for.
NUM_WINDOWS_TO_INSPECT = 3
# =============================================================

def decode_and_unpickle(encoded_data):
    """Decodes base64 and un-pickles the data."""
    if pd.isna(encoded_data): return None
    try:
        return pickle.loads(base64.b64decode(encoded_data))
    except Exception as e:
        print(f"Error during decoding: {e}")
        return None

def analyze_file(log_file_path: Path):
    """Reads a log file and performs a detailed analysis of the prediction quality."""
    print(f"\n--- Analyzing: {log_file_path.name} ---")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(log_file_path, "r:gz") as tar:
                csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
                if not csv_member: return
                tar.extract(csv_member, path=temp_dir)
                df = pd.read_csv(Path(temp_dir) / csv_member.name)

        plot_row_series = df.dropna(subset=['inference_data', 'actual_data'])
        if plot_row_series.empty:
            print("  -> No data rows found.")
            return
        plot_row = plot_row_series.iloc[0]

        # The evaluation horizon might be 336, but we only care about the first 96
        actual_data_collection = decode_and_unpickle(plot_row['actual_data'])
        predicted_data_collection = decode_and_unpickle(plot_row['inference_data'])

        if actual_data_collection is None or predicted_data_collection is None:
            print("  -> Failed to decode data.")
            return

        # --- SLICE TO THE ANALYSIS HORIZON ---
        # Shape is (num_windows, horizon, num_features)
        actual_data = actual_data_collection[:, :ANALYSIS_HORIZON, :]
        predicted_data = predicted_data_collection[:, :ANALYSIS_HORIZON, :]

        # Extract the specific feature we are interested in
        actual_target_feature = actual_data[:, :, TARGET_FEATURE_INDEX]
        predicted_target_feature = predicted_data[:, :, TARGET_FEATURE_INDEX]
        leaked_feature = actual_data[:, :, LEAKED_FEATURE_INDEX]

        # --- Quantitative Analysis ---
        # 1. Overall MAE for the target feature
        mae = np.mean(np.abs(actual_target_feature - predicted_target_feature))
        print(f"\n[Overall Quantitative Analysis for '{SERIES_NAMES[TARGET_FEATURE_INDEX]}']")
        print(f"  - Mean Absolute Error (MAE) over all windows: {mae:.6f}")

        # 2. Correlation
        # Flatten the arrays to calculate correlation across all windows and time steps
        correlation = np.corrcoef(actual_target_feature.flatten(), predicted_target_feature.flatten())[0, 1]
        print(f"  - Correlation between Actual and Predicted: {correlation:.6f}")

        # 3. Verify the data leak
        leak_check_mae = np.mean(np.abs(actual_target_feature - leaked_feature))
        print(f"  - Sanity Check: MAE between '{SERIES_NAMES[TARGET_FEATURE_INDEX]}' and '{SERIES_NAMES[LEAKED_FEATURE_INDEX]}': {leak_check_mae:.6f}")
        if leak_check_mae > 1e-5:
            print("  - WARNING: The data leak is not perfect. The target and leaked features are not identical.")
        else:
            print("  - OK: Data leak confirmed. Target and leaked features are identical.")

        # --- Detailed Inspection of Random Windows ---
        num_available_windows = actual_data.shape[0]
        window_indices = np.random.choice(num_available_windows, size=min(NUM_WINDOWS_TO_INSPECT, num_available_windows), replace=False)

        print(f"\n[Detailed Inspection of {len(window_indices)} Random Windows]")
        for i, window_idx in enumerate(window_indices):
            print(f"\n--- Window #{window_idx} ---")
            actual_slice = actual_target_feature[window_idx]
            predicted_slice = predicted_target_feature[window_idx]
            error_slice = actual_slice - predicted_slice

            # Create a small DataFrame for easy viewing
            comparison_df = pd.DataFrame({
                'Step': range(ANALYSIS_HORIZON),
                f'Actual_{SERIES_NAMES[TARGET_FEATURE_INDEX]}': actual_slice,
                f'Predicted_{SERIES_NAMES[TARGET_FEATURE_INDEX]}': predicted_slice,
                'Error': error_slice
            })
            print(f"  - MAE for this window: {np.mean(np.abs(error_slice)):.6f}")
            print("  - Sample data points:")
            # Show first 5, middle 5, last 5 steps
            sample_indices = list(range(5)) + list(range(ANALYSIS_HORIZON//2 - 2, ANALYSIS_HORIZON//2 + 3)) + list(range(ANALYSIS_HORIZON - 5, ANALYSIS_HORIZON))
            print(comparison_df.iloc[sample_indices].to_string(index=False))

    except Exception as e:
        print(f"An unexpected error occurred while processing {log_file_path.name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to find and analyze result files."""
    target_path = Path(TARGET_FOLDER)
    if not target_path.is_dir():
        print(f"ERROR: Target folder '{TARGET_FOLDER}' not found.")
        sys.exit(1)

    files_to_analyze = sorted(list(target_path.glob('*.csv.tar.gz')))
    if not files_to_analyze:
        print(f"No .csv.tar.gz files found in '{TARGET_FOLDER}'.")
        sys.exit(1)

    print(f"=== Starting Analysis for {len(files_to_analyze)} file(s) in '{TARGET_FOLDER}' ===")
    for file_path in files_to_analyze:
        analyze_file(file_path)
    print("\n=== Analysis Complete ===")

if __name__ == '__main__':
    main()

