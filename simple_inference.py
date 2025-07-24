import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
import inspect

# === IMPORTS FÜR MODELL UND LOSS ===
from ts_benchmark.baselines.duet.models.duet_prob_model import DUETProbModel
from ts_benchmark.baselines.duet.duet_prob import TransformerConfig
from ts_benchmark.baselines.duet.utils.crps import crps_loss

# --- DIAGNOSTIC: WELCHE DATEI WIRD VERWENDET? ---
print("--- STARTING FILE PATH DIAGNOSTIC ---")
print(f"The DUETProbModel class is being loaded from: {inspect.getfile(DUETProbModel)}")
print("--- ENDING FILE PATH DIAGNOSTIC ---")

# --- KONFIGURATION ---
SERIES_ORDER = ['wassertemp', 'airtemp_96', 'pressure_96']
QUANTILES_TO_PLOT = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
CI_COLORS = ['#c6dbef', '#6baed6', '#08519c'] 

def plot_forecast(history_df, actuals_df, prediction_array, quantiles, output_path, window_info):
    """
    Erzeugt und speichert einen Plot der Vorhersage im Vergleich zu den tatsächlichen Werten.
    """
    print(f"  -> Plotting window '{window_info}' to {output_path.name}")
    
    num_vars = actuals_df.shape[1]
    fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(20, 7 * num_vars), sharex=True, squeeze=False)
    axes = axes.flatten()
    
    history_to_plot = history_df.iloc[-3 * len(actuals_df):]
    
    try:
        median_idx = quantiles.index(0.5)
    except (ValueError, AttributeError):
        median_idx = len(quantiles) // 2

    for i in range(num_vars):
        ax = axes[i]
        var_name = actuals_df.columns[i]
        
        ax.plot(history_to_plot.index, history_to_plot.iloc[:, i], color="gray", label="History")
        ax.plot(actuals_df.index, actuals_df.iloc[:, i], color="black", linewidth=2.5, label="Actual Value", zorder=10)
        
        num_ci_levels = len(quantiles) // 2
        for j in range(num_ci_levels):
            lower_q_idx = j
            upper_q_idx = len(quantiles) - 1 - j
            color = CI_COLORS[j % len(CI_COLORS)]
            
            ax.fill_between(
                actuals_df.index, 
                prediction_array[:, i, lower_q_idx], # Erwartet H, V, Q
                prediction_array[:, i, upper_q_idx], 
                color=color,
                alpha=0.7,
                label=f"{(quantiles[upper_q_idx] - quantiles[lower_q_idx])*100:.0f}% CI",
                linewidth=0
            )

        ax.plot(actuals_df.index, prediction_array[:, i, median_idx], color="#FF0000", linestyle='--', label="Median Forecast")

        ax.set_title(f'Forecast for "{var_name}" - {window_info}', fontsize=14)
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    """
    Hauptfunktion zur Ausführung der Inferenz und des Plottings.
    """
    parser = argparse.ArgumentParser(description="Run inference and plotting for a trained DUET-Prob model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.pt).")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument('--num-plots', type=int, default=20, help="Number of random windows to plot.")
    args = parser.parse_args()

    # --- 1. Setup und Plausibilitätsprüfungen ---
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data_file)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Plots will be saved to: {output_dir.resolve()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- 2. Modell aus dem Checkpoint laden ---
    print(f"\n🔄 Loading model from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config_dict = checkpoint['config_dict']
        config = TransformerConfig(**config_dict)
        config.quantiles = QUANTILES_TO_PLOT
        model = DUETProbModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model. Reason: {e}")
        return

    # --- 3. Daten laden und vorbereiten ---
    print("\n🔄 Loading and preparing data...")
    df_long = pd.read_csv(data_path)
    df_wide = df_long.pivot(index='date', columns='cols', values='data')[SERIES_ORDER]
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide.ffill(inplace=True)
    df_wide.bfill(inplace=True)
    print("✅ Data prepared.")

    # --- 4. Hauptschleife für Inferenz und Plotting ---
    start_index = config.seq_len
    end_index = len(df_wide) - config.horizon
    if start_index >= end_index:
        print(f"❌ ERROR: Not enough data for a single forecast. Data length: {len(df_wide)}, required: {config.seq_len + config.horizon}.")
        return
    valid_indices = list(range(start_index, end_index))
    num_to_plot = min(args.num_plots, len(valid_indices))
    selected_indices = random.sample(valid_indices, num_to_plot)
    print(f"\n🔄 Generating {num_to_plot} forecast plots from random windows...")

    for i, forecast_start_idx in enumerate(selected_indices):
        history_end_idx = forecast_start_idx
        history_start_idx = history_end_idx - config.seq_len
        
        history_df = df_wide.iloc[:history_end_idx]
        input_df = df_wide.iloc[history_start_idx:history_end_idx]
        actuals_df = df_wide.iloc[forecast_start_idx : forecast_start_idx + config.horizon]
        
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Das Modell gibt jetzt 6 Werte zurück (final_distr, base_distr, L_importance, gate_weights_linear, gate_weights_esn, selection_counts)
            distr, _, _, _, _, _ = model(input_tensor)
            
            actuals_tensor = torch.tensor(actuals_df.values, dtype=torch.float32).unsqueeze(0).to(device)
            crps_values_per_point = crps_loss(distr, actuals_tensor)
            
            # === ANFANG DER ÄNDERUNG: CRPS pro Kanal berechnen ===
            # Aggregiere über Batch und Horizont, behalte die Kanal-Dimension
            crps_per_channel = crps_values_per_point.mean(dim=(0, 1)) # Ergibt einen Vektor der Länge N_VARS
            
            # Gesamt-CRPS für die Gesamtübersicht
            mean_crps_for_window = crps_per_channel.mean().item()

            print(f"  Window {i+1}/{num_to_plot}: Overall CRPS = {mean_crps_for_window:.4f}")

            # Erstelle einen String für die Anzeige im Plot-Titel
            crps_str_parts = [f"Overall: {mean_crps_for_window:.3f}"]
            for channel_idx, channel_name in enumerate(SERIES_ORDER):
                channel_crps = crps_per_channel[channel_idx].item()
                crps_str_parts.append(f"{channel_name}: {channel_crps:.3f}")
                print(f"    - CRPS for '{channel_name}': {channel_crps:.4f}")

            crps_details_str = " | ".join(crps_str_parts)
            # === ENDE DER ÄNDERUNG ===
            
            # Hole die Vorhersagen für die zu plottenden Quantile
            q_tensor = torch.tensor(QUANTILES_TO_PLOT, device=device, dtype=torch.float32)
            quantile_predictions_tensor = distr.icdf(q_tensor)

        # Form für das Plotting anpassen auf [H, V, Q]
        prediction_array = quantile_predictions_tensor.squeeze(0).cpu().numpy()

        # Generiere den Plot
        output_path = output_dir / f"forecast_plot_{i+1}_end_{history_end_idx}.png"
        
        # Füge die detaillierten CRPS-Scores zum Titel hinzu
        window_info_with_crps = f"Plot {i+1}/{num_to_plot} (CRPS: {crps_details_str})"
        
        plot_forecast(
            history_df=history_df,
            actuals_df=actuals_df,
            prediction_array=prediction_array,
            quantiles=QUANTILES_TO_PLOT,
            output_path=output_path,
            window_info=window_info_with_crps
        )

    print("\n🎉 All plots generated successfully!")


if __name__ == '__main__':
    main()