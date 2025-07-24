import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import os
import tarfile
import tempfile
import numpy as np
import json
import sys
from pathlib import Path
from scipy.stats import kruskal, gaussian_kde
import scikit_posthocs as sp

# ======================= KONFIGURATION =======================
TARGET_FOLDER = 'result/eisbach/DUET'
TARGET_SERIES_NAME = 'wassertemp'
ALL_SERIES_NAMES = ['airtemp_shifted', 'wassertemp', 'airtemp']
NUM_RANDOM_WINDOWS = 200
RANDOM_SEED = 42
ALPHA = 0.05
# =============================================================

# ... (alle Funktionen von decode_and_unpickle bis get_num_windows bleiben unverändert) ...
def decode_and_unpickle(encoded_data):
    if pd.isna(encoded_data): return None
    try:
        return pickle.loads(base64.b64decode(encoded_data))
    except Exception as e:
        print(f"Fehler beim Dekodieren: {e}")
        return None

def get_run_name_from_path(path: Path) -> str:
    """Extrahiert einen beschreibenden Namen aus dem Dateipfad, indem Hyperparameter gelesen werden."""
    # Fallback-Name, falls etwas schiefgeht
    fallback_name = f"Run_{path.name.split('.')[1]}"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(path, "r:gz") as tar:
                csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
                if not csv_member: return fallback_name
                tar.extract(csv_member, path=temp_dir)
                df = pd.read_csv(Path(temp_dir) / csv_member.name)

        if 'model_hyper_params' not in df.columns or df['model_hyper_params'].isnull().all():
            return fallback_name

        params_str = df['model_hyper_params'].iloc[0]
        # HINWEIS: Python's str(dict) verwendet einfache Anführungszeichen, json.loads braucht doppelte.
        # Diese Ersetzung macht den String kompatibel.
        params_str = params_str.replace("'", '"')
        params = json.loads(params_str)
        
        # Erstelle einen beschreibenden Namen. Passe dies bei Bedarf an.
        # Hier verwenden wir 'dropout' als Schlüssel, da es in den Skripten variiert wird.
        dropout = params.get('dropout', 'NA')
        return f"dropout_{dropout}"

    except Exception as e:
        # Bei jedem Fehler wird der sichere Fallback-Name zurückgegeben.
        print(f"Warnung: Konnte Hyperparameter für {path.name} nicht lesen. Fehler: {e}. Fallback-Name wird verwendet.")
        return fallback_name

def calculate_aggregate_metrics(residuals):
    return {
        'RMSE': np.sqrt(np.mean(residuals ** 2)),
        'MAE': np.mean(np.abs(residuals)),
        'Median_Error': np.median(residuals),
        'Std_Error': np.std(residuals)
    }

def process_file_for_target_series(file_path: Path, window_indices: np.ndarray, target_series_idx: int):
    print(f"--- Verarbeite: {file_path.name} ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(file_path, "r:gz") as tar:
            csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
            if not csv_member:
                print(f"Warnung: Keine .csv-Datei in {file_path.name} gefunden.")
                return None
            tar.extract(csv_member, path=temp_dir)
            df = pd.read_csv(Path(temp_dir) / csv_member.name)

    data_rows = df.dropna(subset=['inference_data', 'actual_data'])
    if data_rows.empty:
        print(f"Warnung: Keine Datenzeilen in {file_path.name} gefunden. Lauf wird übersprungen.")
        return None
    plot_row = data_rows.iloc[0]

    actual_data = decode_and_unpickle(plot_row['actual_data'])
    predicted_data = decode_and_unpickle(plot_row['inference_data'])

    if actual_data is None or predicted_data is None:
        print(f"Warnung: Dekodierung der Daten in {file_path.name} fehlgeschlagen. Lauf wird übersprungen.")
        return None

    if max(window_indices) >= actual_data.shape[0]:
        print(f"Warnung: Nicht genügend Fenster in {file_path.name} für die angeforderten Indizes. Lauf wird übersprungen.")
        return None

    residuals_list = [
        (actual_data[idx, :, target_series_idx] - predicted_data[idx, :, target_series_idx])
        for idx in window_indices
    ]
    return np.concatenate(residuals_list)

def get_num_windows(file_path: Path) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(file_path, "r:gz") as tar:
            csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
            if not csv_member:
                print(f"Warnung: Keine .csv-Datei in {file_path.name} gefunden.")
                return 0
            tar.extract(csv_member, path=temp_dir)
            df = pd.read_csv(Path(temp_dir) / csv_member.name)

    data_rows = df.dropna(subset=['actual_data'])
    if data_rows.empty:
        print(f"Warnung: Keine Zeilen mit 'actual_data' in {file_path.name} gefunden. Möglicherweise ist dieser Lauf fehlgeschlagen.")
        return 0

    row = data_rows.iloc[0]
    data = decode_and_unpickle(row['actual_data'])
    return data.shape[0] if data is not None else 0


def main():
    target_path = Path(TARGET_FOLDER)
    if not target_path.is_dir():
        print(f"FEHLER: Ordner '{TARGET_FOLDER}' nicht gefunden.")
        sys.exit(1)

    all_files = sorted(list(target_path.glob('*.csv.tar.gz')))
    print(f"{len(all_files)} Ergebnisdateien gefunden.")

    # Filtere ungültige Läufe (ohne Daten) im Voraus heraus
    print("\nPrüfe Dateien auf Gültigkeit...")
    valid_files_with_windows = []
    for f in all_files:
        num_windows = get_num_windows(f)
        if num_windows > 0:
            valid_files_with_windows.append((f, num_windows))

    if len(valid_files_with_windows) < 2:
        print(f"\nFEHLER: Weniger als 2 gültige Ergebnisdateien für einen Vergleich gefunden (nur {len(valid_files_with_windows)}).")
        print("Einige Läufe sind möglicherweise fehlgeschlagen oder enthalten keine Daten.")
        sys.exit(1)
    
    print(f"-> {len(valid_files_with_windows)} von {len(all_files)} Dateien sind gültig und werden verglichen.")
    files_to_compare = [item[0] for item in valid_files_with_windows]

    try:
        target_series_idx = ALL_SERIES_NAMES.index(TARGET_SERIES_NAME)
    except ValueError:
        print(f"FEHLER: '{TARGET_SERIES_NAME}' nicht in ALL_SERIES_NAMES gefunden.")
        sys.exit(1)

    window_counts = [item[1] for item in valid_files_with_windows]
    min_windows = min(window_counts)
    
    num_windows_to_sample = NUM_RANDOM_WINDOWS
    if min_windows < NUM_RANDOM_WINDOWS:
        print(f"\nWARNUNG: Nicht alle gültigen Dateien haben die gewünschte Anzahl von {NUM_RANDOM_WINDOWS} Fenstern (Minimum: {min_windows}).")
        print(f"Die Anzahl der zu vergleichenden Fenster wird auf {min_windows} reduziert.")
        num_windows_to_sample = min_windows

    np.random.seed(RANDOM_SEED)
    window_indices = np.random.choice(min_windows, size=num_windows_to_sample, replace=False)
    print(f"\nVergleiche {len(files_to_compare)} Läufe anhand von {num_windows_to_sample} zufälligen Fenstern (Indizes: {sorted(window_indices.tolist())}).\n")
    
    all_residuals = {
        get_run_name_from_path(fp): process_file_for_target_series(fp, window_indices, target_series_idx)
        for fp in files_to_compare
    }
    all_residuals = {k: v for k, v in all_residuals.items() if v is not None}

    # --- 1. & 2. Metriken und Statistische Signifikanz (unverändert) ---
    # ... (Code für Metriken und Signifikanztest bleibt genau gleich) ...
    comparison_metrics = {name: calculate_aggregate_metrics(res) for name, res in all_residuals.items()}
    metrics_df = pd.DataFrame.from_dict(comparison_metrics, orient='index').sort_values(by='RMSE')
    print("\n\n" + "="*80)
    print(f" METRISCHER VERGLEICH für '{TARGET_SERIES_NAME}' (sortiert nach RMSE) ".center(80, "="))
    print("="*80)
    print(metrics_df.to_string())
    print("="*80)

    print("\n\n" + "="*80)
    print(f" STATISTISCHE SIGNIFIKANZ (ALPHA = {ALPHA}) ".center(80, "="))
    print("="*80)
    
    residuals_data = list(all_residuals.values())
    h_stat, p_value = kruskal(*residuals_data)
    print("Kruskal-Wallis H-Test (Vergleich aller Läufe):")
    print(f"  - H-Statistik: {h_stat:.4f}")
    print(f"  - p-Wert: {p_value:.4f}")

    if p_value < ALPHA:
        print(f"\n-> Ergebnis ist signifikant (p < {ALPHA}). Mindestens ein Lauf unterscheidet sich von den anderen.")
        print("Führe Dunn's Post-hoc-Test für Paarvergleiche durch:\n")
        
        dunn_results = sp.posthoc_dunn(residuals_data)
        dunn_results.columns = all_residuals.keys()
        dunn_results.index = all_residuals.keys()
        
        print("Dunn's Test p-Werte (ein Wert < 0.05 zeigt einen signifikanten Unterschied zwischen dem Paar):")
        # Format the DataFrame for better readability
        print(dunn_results.map(lambda x: f'{x:.4f}').to_string())
    else:
        print(f"\n-> Ergebnis ist NICHT signifikant (p >= {ALPHA}).")
        print("Es gibt keinen statistischen Nachweis für einen Unterschied zwischen den Läufen.")
    print("="*80)


    # --- 3. Ausgabe: Verteilungs-Plot (KDE) ---
    print("\nErstelle Kernel-Density-Plot der Fehlerverteilungen...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for run_name, residuals in all_residuals.items():
        # Erstelle ein KDE-Objekt
        kde = gaussian_kde(residuals)
        # Erstelle einen Wertebereich für den Plot
        x_range = np.linspace(residuals.min() - 1, residuals.max() + 1, 500)
        # Plotte die Dichtekurve
        ax.plot(x_range, kde(x_range), label=f'Dichte {run_name}', lw=2.5)

    ax.axvline(0, color='k', linestyle='--', linewidth=1.5, label='Perfekter Fehler (0)')
    ax.set_title(f"Dichteverteilung der Vorhersagefehler für '{TARGET_SERIES_NAME}'", fontsize=16, pad=20)
    ax.set_xlabel("Vorhersagefehler (Residuum: Actual - Predicted)", fontsize=12)
    ax.set_ylabel("Geschätzte Dichte", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    output_filename = f'vergleich_dichteplot_{TARGET_SERIES_NAME}.png'
    fig.savefig(output_filename)
    print(f"Plot erfolgreich als '{output_filename}' gespeichert.")
    plt.show()

    # --- 4. Ausgabe: Boxplot zum Vergleich der Fehlerverteilungen ---
    print("\nErstelle Boxplot zum Vergleich der Fehlerverteilungen...")
    fig_box, ax_box = plt.subplots(figsize=(14, 8))

    # Sortiere die Daten nach der Reihenfolge im Metrik-DataFrame (nach RMSE)
    sorted_run_names = metrics_df.index.tolist()
    sorted_residuals = [all_residuals[name] for name in sorted_run_names]

    ax_box.boxplot(sorted_residuals, tick_labels=sorted_run_names, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red', linewidth=2))

    ax_box.axhline(0, color='k', linestyle='--', linewidth=1.5, label='Perfekter Fehler (0)')
    ax_box.set_title(f"Vergleich der Vorhersagefehler für '{TARGET_SERIES_NAME}' (sortiert nach RMSE)", fontsize=16, pad=20)
    ax_box.set_ylabel("Vorhersagefehler (Residuum: Actual - Predicted)", fontsize=12)
    ax_box.set_xlabel("Experiment-Lauf", fontsize=12)
    plt.setp(ax_box.get_xticklabels(), rotation=30, ha="right")
    ax_box.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig_box.tight_layout()

    output_filename_box = f'vergleich_boxplot_{TARGET_SERIES_NAME}.png'
    fig_box.savefig(output_filename_box)
    print(f"Boxplot erfolgreich als '{output_filename_box}' gespeichert.")
    plt.show()

if __name__ == '__main__':
    main()