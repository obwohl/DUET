import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import argparse

from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig

# ========================== KONFIGURATION ==========================
INPUT_CSV_PATH = "dataset/forecasting/eisbach_shifted_96_only.csv"
SERIES_ORDER = ['wassertemp', 'airtemp_96']
QUANTILES_TO_PLOT = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
DATE_COL, VALUE_COL, SERIES_NAME_COL = 'date', 'data', 'cols'

CI_COLORS = ['#08519c', '#6baed6', '#c6dbef'] 
CI_ALPHAS = [0.7, 0.5, 0.3]


def main():
    parser = argparse.ArgumentParser(description="Run inference and plotting for the DUET-Prob-SBP model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.pt).")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path) or not os.path.exists(INPUT_CSV_PATH):
        print(f"ERROR: Checkpoint '{checkpoint_path}' or data file '{INPUT_CSV_PATH}' not found.")
        return

    run_dir = Path(checkpoint_path).parent
    output_dir = run_dir / "inference_plots_after_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to '{output_dir.resolve()}'")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'config_dict' not in checkpoint:
            raise ValueError("Checkpoint is invalid or from an old version: missing 'config_dict'. Please retrain the model.")
        
        config_dict = checkpoint['config_dict']
        config = TransformerConfig(**config_dict)
        config.quantiles = QUANTILES_TO_PLOT
        
        print("\n--- Final Model Configuration (from checkpoint) ---")
        for k, v in sorted(config_dict.items()): print(f"  - {k:<12}: {v}")
        print("---------------------------------\n")

        model = DUETProbModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model and configuration from checkpoint.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load the model from checkpoint.")
        print(f"Specific error: {e}")
        return

    df_long = pd.read_csv(INPUT_CSV_PATH)
    df_full = df_long.pivot(index=DATE_COL, columns=SERIES_NAME_COL, values=VALUE_COL)[SERIES_ORDER]
    df_full.index = pd.to_datetime(df_full.index)
    df_full.ffill(inplace=True); df_full.bfill(inplace=True)

    model.to(device)
    model.eval()

    first_idx, last_idx = config.seq_len, len(df_full) - config.horizon
    valid_indices = list(range(first_idx, last_idx + 1))
    num_to_plot = min(20, len(valid_indices))
    selected_indices = random.sample(valid_indices, num_to_plot)
    print(f"\nRandomly selecting {num_to_plot} windows to plot.\n")
    
    for i, end_of_history_idx in enumerate(selected_indices):
        print(f"--- Processing Plot {i+1}/{num_to_plot} (using data up to index {end_of_history_idx}) ---")
        
        history_df = df_full.iloc[:end_of_history_idx]
        input_df = history_df.iloc[-config.seq_len:]
        actuals_df = df_full.iloc[end_of_history_idx : end_of_history_idx + config.horizon]

        input_tensor = torch.tensor(input_df.values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Das Modell gibt jetzt 6 Werte zurück (final_distr, base_distr, L_importance, gate_weights_linear, gate_weights_esn, selection_counts)
            distr, _, _, _, _, _ = model(input_tensor)
            
            quantile_forecasts = [distr.icdf(torch.tensor(q, device=device, dtype=torch.float32)) for q in QUANTILES_TO_PLOT]
            output_tensor = torch.stack(quantile_forecasts, dim=0)
            # --- END OF FIX ---

        prediction_final = output_tensor.squeeze(1).permute(2, 1, 0).cpu().numpy()

        output_path = output_dir / f"forecast_window_{i+1}_idx_{end_of_history_idx}.png"

        # --- FIX: Update the plot function call to remove NLL arguments ---
        plot_forecast(
            history_df,
            actuals_df,
            prediction_final,
            config.quantiles,
            output_path,
            f"Window {i+1}/{num_to_plot}"
        )
        # --- END OF FIX ---

    print("\n✅ All plots generated.")


# --- FIX: Update the plot function signature and title ---
def plot_forecast(
    history_df, actuals_df, prediction_array, quantiles,
    output_path, window_info
):
# --- END OF FIX ---
    print(f"  - Generating plot: {output_path.name}")
    num_vars = actuals_df.shape[1]
    fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(20, 7.5 * num_vars), sharex=True, squeeze=False)
    axes = axes.flatten()
    history_to_plot = history_df.iloc[-3 * len(actuals_df):]
    try:
        median_idx = quantiles.index(0.5)
    except (ValueError, AttributeError):
        median_idx = len(quantiles) // 2

    for i in range(num_vars):
        ax = axes[i]
        var_name = actuals_df.columns[i]
        
        ax.plot(history_to_plot.index, history_to_plot.iloc[:, i], label="History", color="gray", alpha=0.8)
        ax.plot(actuals_df.index, actuals_df.iloc[:, i], label="Actual", color="black", linewidth=2.5, zorder=10)
        num_ci_levels = len(quantiles) // 2
        for j in range(num_ci_levels):
            lower_q_idx, upper_q_idx = j, len(quantiles) - 1 - j
            lower_q, upper_q = quantiles[lower_q_idx], quantiles[upper_q_idx]
            color_idx = len(CI_COLORS) - 1 - j
            color = CI_COLORS[color_idx % len(CI_COLORS)]
            alpha = CI_ALPHAS[color_idx % len(CI_ALPHAS)]
            confidence_level = (upper_q - lower_q) * 100
            label = f"{confidence_level:.0f}% Confidence Interval"
            ax.fill_between(
                actuals_df.index, 
                prediction_array[:, i, lower_q_idx], 
                prediction_array[:, i, upper_q_idx], 
                color=color, alpha=alpha, label=label, linewidth=0
            )
        ax.plot(actuals_df.index, prediction_array[:, i, median_idx], label="Median Forecast (q=0.5)", color="#FF0000", linestyle='--', linewidth=2, zorder=11)
        
        # --- FIX: Simplify the plot title ---
        channel_title = f'Forecast for "{var_name}" - {window_info}'
        # --- END OF FIX ---

        ax.set_title(channel_title, fontsize=14, loc='left')
        ax.set_ylabel("Value")
        
        handles, labels = ax.get_legend_handles_labels()
        num_main_lines = 2
        reordered_handles = handles[:num_main_lines] + handles[num_main_lines+num_ci_levels-1:num_main_lines-1:-1] + handles[num_main_lines+num_ci_levels:]
        reordered_labels = labels[:num_main_lines] + labels[num_main_lines+num_ci_levels-1:num_main_lines-1:-1] + labels[num_main_lines+num_ci_levels:]
        ax.legend(handles=reordered_handles, labels=reordered_labels, loc='upper left')

        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    main()