import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re
import random

from sklearn.preprocessing import StandardScaler
# --- Important: Adjust these imports based on your project structure ---
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig

# ========================== CONFIGURATION ==========================
# --- Path Settings ---
CHECKPOINT_PATH = "/Users/friedrichsiemers/DUET/runs/DUET-Prob-SBP_1751982093/best_model_checkpoint.pt"
INPUT_CSV_PATH = "dataset/forecasting/eisbach_shifted_96_only.csv"
CONFIG_SHELL_SCRIPT_PATH = "scripts/multivariate_forecast/my_script/eisbach_proba.sh"
PLOT_OUTPUT_DIR = "inference_plots_detailed"

# --- Data Format Settings ---
DATE_COL = 'date'
VALUE_COL = 'data'
SERIES_NAME_COL = 'cols'
SERIES_ORDER = ['wassertemp', 'airtemp_96']

# --- NEW: Quantile & Plotting Configuration ---
# Define the quantiles we want to predict and plot.
# Must be sorted, with 0.5 in the middle.
QUANTILES_TO_PLOT = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]

# Define corresponding colors for the confidence intervals (from widest to narrowest)
CI_COLORS = ['#a1c9f4', '#8de5a1', '#ffb482'] # Light Blue, Green, Orange

# --- Model Hyperparameter Settings (Defaults, will be overridden) ---
MODEL_HYPERPARAMS = {
    "horizon": 96,
    "seq_len": 384,
    "lr": 0.0001,
    "lradj": "cosine_warmup",
    "d_model": 128,
    "d_ff": 128,
    "n_heads": 2,
    "e_layers": 2,
    "factor": 2,
    "dropout": 0.15,
    "fc_dropout": 0.05,
    "k": 1,
    "num_experts": 1,
    "quantiles": QUANTILES_TO_PLOT, # Use the new quantile list
    "freq": "h",
}
# ===================================================================

# (load_hyperparams_from_shell function remains unchanged)
def load_hyperparams_from_shell(script_path):
    # ... (no changes needed here)
    hyperparams = {}
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^#\s]+))")
    try:
        with open(script_path, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    key = match.group(1).lower()
                    value_str = next((g for g in match.groups()[2:] if g is not None), None)
                    if value_str is None: continue
                    try:
                        if '.' in value_str or 'e' in value_str.lower(): value = float(value_str)
                        else: value = int(value_str)
                    except (ValueError, TypeError):
                        if isinstance(value_str, str) and value_str.lower() == 'true': value = True
                        elif isinstance(value_str, str) and value_str.lower() == 'false': value = False
                        else: value = value_str
                    hyperparams[key] = value
    except Exception as e:
        print(f"WARNING: Could not parse shell script '{script_path}'. Error: {e}.")
    return hyperparams


def plot_forecast(history_df, actuals_df, prediction_array, quantiles, nll_per_channel, output_path, window_title):
    """
    Generates and saves a detailed plot with multiple confidence intervals.
    """
    print(f"  - Generating plot: {output_path.name}")
    num_vars = actuals_df.shape[1]
    
    fig, axes = plt.subplots(
        nrows=num_vars, ncols=1, figsize=(20, 7 * num_vars), sharex=True, squeeze=False
    )
    axes = axes.flatten()
    history_to_plot = history_df.iloc[-3 * len(actuals_df):]

    # Find the index of the median (q=0.5)
    try:
        median_idx = quantiles.index(0.5)
    except ValueError:
        raise ValueError("Quantile list must contain 0.5 for the median forecast.")

    for i in range(num_vars):
        ax = axes[i]
        var_name = actuals_df.columns[i]
        
        # Plot History and Actuals
        ax.plot(history_to_plot.index, history_to_plot.iloc[:, i], label="History", color="gray", alpha=0.8)
        ax.plot(actuals_df.index, actuals_df.iloc[:, i], label="Actual", color="black", linewidth=2.5, zorder=10)

        # === NEW: Plot multiple, nested confidence intervals ===
        num_ci_levels = len(quantiles) // 2
        for j in range(num_ci_levels):
            lower_q_idx = j
            upper_q_idx = len(quantiles) - 1 - j
            lower_q = quantiles[lower_q_idx]
            upper_q = quantiles[upper_q_idx]
            color = CI_COLORS[j % len(CI_COLORS)]
            
            ax.fill_between(
                actuals_df.index,
                prediction_array[:, i, lower_q_idx],
                prediction_array[:, i, upper_q_idx],
                color=color, alpha=0.3, label=f"CI ({lower_q:.3f}-{upper_q:.3f})"
            )
        
        # Plot Median Forecast on top
        ax.plot(actuals_df.index, prediction_array[:, i, median_idx], label=f"Median Forecast (q=0.5)", color="blue", linestyle='--', linewidth=1.5)
        
        # === NEW: Add NLL to the title ===
        channel_nll = nll_per_channel[i]
        ax.set_title(f'Forecast for "{var_name}" - {window_title}  (Window NLL: {channel_nll:.3f})', fontsize=16)
        
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.xlabel('Date')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    """Main function to run the inference and plotting process."""
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(INPUT_CSV_PATH):
        print("ERROR: Checkpoint or data file not found.")
        return

    output_dir = Path(PLOT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to '{output_dir.resolve()}'")

    # --- Load Config from Shell Script ---
    shell_params = load_hyperparams_from_shell(CONFIG_SHELL_SCRIPT_PATH)
    config_dict = MODEL_HYPERPARAMS.copy()
    for key, value in shell_params.items():
        if key in config_dict:
            config_dict[key] = value
    
    # Add derived parameters
    config_dict.update({
        'enc_in': len(SERIES_ORDER), 'dec_in': len(SERIES_ORDER),
        'c_out': len(SERIES_ORDER), 'label_len': config_dict['seq_len'] // 2,
        'CI': True, 'norm': True, 'quantiles': QUANTILES_TO_PLOT
    })
    
    print("\n--- Final Model Configuration ---")
    for k, v in sorted(config_dict.items()): print(f"  - {k:<12}: {v}")
    print("---------------------------------\n")

    # --- Data & Model Loading ---
    df_long = pd.read_csv(INPUT_CSV_PATH)
    df_full = df_long.pivot(index=DATE_COL, columns=SERIES_NAME_COL, values=VALUE_COL)[SERIES_ORDER]
    df_full.index = pd.to_datetime(df_full.index)
    df_full.ffill().bfill(inplace=True)

    config = TransformerConfig(**config_dict)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = DUETProbModel(config)
    
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # --- Scaler Loading ---
    try:
        scaler = checkpoint['scaler']
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model and scaler from new-format checkpoint.")
    except (KeyError, TypeError):
        print("WARNING: Old checkpoint format. Using fallback scaler.")
        model.load_state_dict(checkpoint)
        tv_ratio = 0.9
        train_len = int(len(df_full) * tv_ratio)
        train_valid_data = df_full.iloc[:train_len]
        scaler = StandardScaler().fit(train_valid_data.values)
        
    model.to(device)
    model.eval()

    # --- Random Window Selection ---
    first_idx = config.seq_len
    last_idx = len(df_full) - config.horizon
    valid_indices = list(range(first_idx, last_idx + 1))
    num_to_plot = min(10, len(valid_indices))
    selected_indices = random.sample(valid_indices, num_to_plot)
    print(f"\nRandomly selecting {num_to_plot} windows to plot.\n")
    
    # --- Main Inference & Plotting Loop ---
    for i, end_of_history_idx in enumerate(selected_indices):
        print(f"--- Processing Plot {i+1}/{num_to_plot} (using data up to index {end_of_history_idx}) ---")
        
        history_df = df_full.iloc[:end_of_history_idx]
        input_df = history_df.iloc[-config.seq_len:]
        actuals_df = df_full.iloc[end_of_history_idx : end_of_history_idx + config.horizon]

        scaled_input_values = scaler.transform(input_df.values)
        input_tensor = torch.tensor(scaled_input_values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Das Modell gibt jetzt 6 Werte zurück (final_distr, base_distr, L_importance, gate_weights_linear, gate_weights_esn, selection_counts)
            distr, _, _, _, _, _ = model(input_tensor)
            
            # --- NLL Calculation for this window ---
            scaled_actuals = scaler.transform(actuals_df.values)
            actuals_tensor = torch.tensor(scaled_actuals, dtype=torch.float32, device=device).unsqueeze(0)
            actuals_for_loss = actuals_tensor.permute(0, 2, 1) # Shape: [1, n_vars, horizon]
            
            log_probs = distr.log_prob(actuals_for_loss)
            nll_per_channel = -log_probs.mean(dim=(0, 2)).cpu().numpy()

            # --- Quantile Extraction ---
            quantile_forecasts = [distr.icdf(torch.tensor(q, device=device, dtype=torch.float32)) for q in config.quantiles]
            output_tensor = torch.stack(quantile_forecasts, dim=0)

        prediction_scaled = output_tensor.squeeze(1).permute(2, 1, 0).cpu().numpy()
        h, c, q = prediction_scaled.shape
        prediction_reshaped = np.transpose(prediction_scaled, (0, 2, 1)).reshape(h * q, c)
        denorm_reshaped = scaler.inverse_transform(prediction_reshaped)
        prediction_final = np.transpose(denorm_reshaped.reshape(h, q, c), (0, 2, 1))

        output_path = output_dir / f"forecast_window_{i+1}_idx_{end_of_history_idx}.png"
        window_title = f"Window {i+1}/{num_to_plot}"
        plot_forecast(history_df, actuals_df, prediction_final, config.quantiles, nll_per_channel, output_path, window_title)

    print("\n✅ All plots generated.")

if __name__ == '__main__':
    main()