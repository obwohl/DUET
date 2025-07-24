import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import os
import tarfile
import tempfile
import sys
import numpy as np
import json
from pathlib import Path

# ======================= KONFIGURATION =======================
# 1. Der Ordner mit den Ergebnisdateien
TARGET_FOLDER = 'result/eisbach/DUET'

# NEU: Ein dedizierter Ordner für die erstellten Plots
PLOT_OUTPUT_FOLDER = os.path.join(TARGET_FOLDER, 'plots')

# 2. Wie viele zufällige Fenster sollen pro Datei geplottet werden?
NUM_RANDOM_WINDOWS_TO_PLOT = 10

# 3. HIER DIE ECHTEN NAMEN DEINER ZEITREIHEN EINTRAGEN
#    Die Reihenfolge muss mit der in der finalen CSV-Datei übereinstimmen.
SERIES_NAMES = ['airtemp_shifted', 'wassertemp', 'airtemp']

# NEU: Ein Seed für die Zufallszahlen, um reproduzierbare Ergebnisse zu erhalten
RANDOM_SEED = 1
# =============================================================

def decode_and_unpickle(encoded_data):
    """Dekodiert base64 und ent-pickelt die Daten."""
    if pd.isna(encoded_data): return None
    try:
        return pickle.loads(base64.b64decode(encoded_data))
    except Exception as e:
        print(f"Fehler beim Dekodieren: {e}")
        return None

def calculate_metrics(actual, predicted):
    """Calculates MSE, RMSE, MAE, and max absolute residual."""
    residuals = actual - predicted
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    max_residual = np.max(np.abs(residuals))
    return mse, rmse, mae, max_residual, residuals

def create_plot_for_window(actual_window, predicted_window, window_index, source_file_info, output_dir):
    """
    Erstellt und speichert einen detaillierten Plot mit Subplots für jedes Feature in einem Vorhersage-Fenster.
    """
    num_features = actual_window.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(20, 7 * num_features), sharex=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f"Multivariate Vorhersage-Analyse (Fenster #{window_index})\n(aus {source_file_info})", fontsize=20, y=1.0)

    for i in range(num_features):
        ax = axes[i]
        series_name = SERIES_NAMES[i] if i < len(SERIES_NAMES) else f'Serie {i}'
        actual_series = actual_window[:, i]
        predicted_series = predicted_window[:, i]
        mse, rmse, mae, max_res, residuals = calculate_metrics(actual_series, predicted_series)

        ax.plot(actual_series, label=f'Actual ({series_name})', color='C0', linewidth=2.5, marker='o', markersize=4, linestyle='-')
        ax.plot(predicted_series, label=f'Prediction ({series_name})', color='C1', linewidth=2, marker='x', markersize=4, linestyle='--')
        ax.set_ylabel("Wert (Actual/Prediction)", fontsize=12, color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2 = ax.twinx()
        ax2.plot(residuals, label='Residuals', color='C2', linewidth=1.5, alpha=0.7, marker='.', linestyle=':')
        ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=12, color='C2')
        ax2.tick_params(axis='y', labelcolor='C2')

        q05 = np.quantile(residuals, 0.05)
        q95 = np.quantile(residuals, 0.95)
        ax2.axhline(q05, color='C4', linestyle='--', linewidth=1.5, label=f'5% Quantil: {q05:.3f}')
        ax2.axhline(q95, color='C5', linestyle='--', linewidth=1.5, label=f'95% Quantil: {q95:.3f}')

        subplot_title = (
            f"Analyse für: {series_name}\n"
            f"RMSE: {rmse:.4f}  |  MSE: {mse:.4f}  |  MAE: {mae:.4f}  |  Max. Abs. Residual: {max_res:.4f}"
        )
        ax.set_title(subplot_title, fontsize=14, loc='left')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    axes[-1].set_xlabel(f"Zeitschritte im Vorhersage-Horizont (Länge: {actual_window.shape[0]})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    plot_filename_base = f"plot_analysis_{source_file_info.replace('.csv.tar.gz', '')}_window_{window_index}.png"
    full_plot_path = os.path.join(output_dir, plot_filename_base)
    fig.savefig(full_plot_path, bbox_inches='tight')
    plt.close(fig)
    return full_plot_path

def analyze_and_plot_file(log_file_path, plot_output_dir):
    """Liest eine Log-Datei und startet den Plot-Prozess für zufällige Fenster."""
    # Reset the random seed HERE to ensure every file gets the same random sequence
    np.random.seed(RANDOM_SEED)

    print(f"\n--- Verarbeite: {log_file_path.name} ---")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(log_file_path, "r:gz") as tar:
                csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
                if not csv_member: return
                tar.extract(csv_member, path=temp_dir)
                df = pd.read_csv(Path(temp_dir) / csv_member.name)

        plot_row_series = df.dropna(subset=['inference_data', 'actual_data'])
        if plot_row_series.empty: return
        plot_row = plot_row_series.iloc[0]

        try:
            strategy_args_str = plot_row['strategy_args'].replace("'", '"')
            strategy_args = json.loads(strategy_args_str)
            horizon = int(strategy_args.get('horizon'))
            if not horizon:
                raise ValueError("Horizon not found or is zero in strategy_args")
        except Exception as e:
            print(f"  -> Fehler: Konnte 'horizon' nicht aus 'strategy_args' extrahieren. Überspringe Datei. Fehler: {e}")
            return

        actual_data_collection = decode_and_unpickle(plot_row['actual_data'])
        predicted_data_collection = decode_and_unpickle(plot_row['inference_data'])

        if actual_data_collection is None or predicted_data_collection is None:
            return

        if actual_data_collection.ndim != 3 or predicted_data_collection.ndim != 3:
            print(f"  -> Fehler: Dekodierte Daten sind nicht 3-dimensional, wie erwartet. Überspringe.")
            return

        num_available_windows = actual_data_collection.shape[0]
        num_to_plot = min(NUM_RANDOM_WINDOWS_TO_PLOT, num_available_windows)
        window_indices = np.random.choice(num_available_windows, size=num_to_plot, replace=False)

        print(f"Random Seed auf {RANDOM_SEED} zurückgesetzt.")
        print(f"{num_available_windows} Fenster gefunden. Wähle zufällig {num_to_plot} davon aus: {window_indices}")

        for window_idx in window_indices:
            actual_window = actual_data_collection[window_idx, :, :]
            predicted_window = predicted_data_collection[window_idx, :, :]

            plot_filename = create_plot_for_window(actual_window, predicted_window, window_idx, log_file_path.name, plot_output_dir)
            print(f"Plot für Fenster #{window_idx} erfolgreich als '{plot_filename}' gespeichert.")

    except Exception as e:
        print(f"Ein unerwarteter Fehler ist bei der Verarbeitung von {log_file_path.name} aufgetreten: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Hauptfunktion zum Durchsuchen des Ordners und Starten der Plots."""
    target_path = Path(TARGET_FOLDER)
    if not target_path.is_dir():
        print(f"FEHLER: Der Zielordner '{TARGET_FOLDER}' wurde nicht gefunden.")
        sys.exit(1)

    files_to_plot = sorted(list(target_path.glob('*.csv.tar.gz')))
    if not files_to_plot:
        print(f"Keine .csv.tar.gz-Dateien im Ordner '{TARGET_FOLDER}' gefunden.")
        sys.exit(1)

    os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
    print(f"Plots werden im Ordner '{PLOT_OUTPUT_FOLDER}' gespeichert.")

    print(f"=== Starte Plot-Erstellung für {len(files_to_plot)} Datei(en) im Ordner '{TARGET_FOLDER}' ===")
    for file_path in files_to_plot:
        analyze_and_plot_file(file_path, PLOT_OUTPUT_FOLDER)
    print("\n=== Alle Plots wurden erstellt. ===")

if __name__ == '__main__':
    main()