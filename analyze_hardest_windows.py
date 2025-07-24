import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import os
import tarfile
import tempfile
import numpy as np
import sys
from pathlib import Path
from scipy.stats import kruskal
import scikit_posthocs as sp
import re

# ======================= KONFIGURATION =======================
# Ordner, der die Ergebnisdateien (.csv.tar.gz) der verschiedenen Läufe enthält.
TARGET_FOLDER = 'result/result/eisbach/DUET_OPTIMIZATION'

# Name der Zeitreihe, die als Zielvariable für die Analyse verwendet wird.
TARGET_SERIES_NAME = 'wassertemp'

# Liste aller Zeitreihennamen in der korrekten Reihenfolge, wie sie im Modell verwendet wurden.
ALL_SERIES_NAMES = ['wassertemp', 'airtemp_96']

# Anzahl der "schwierigsten" Fenster, die für jede Methode identifiziert und analysiert werden sollen.
NUM_HARDEST_WINDOWS = 20

# Signifikanzniveau für die statistischen Tests (z.B. Dunn's Test).
ALPHA = 0.05
# =============================================================

def decode_and_unpickle(encoded_data: str):
    """Dekodiert einen base64-kodierten String und ent-pickelt die resultierenden Bytes."""
    if pd.isna(encoded_data):
        return None
    try:
        return pickle.loads(base64.b64decode(encoded_data))
    except (TypeError, pickle.UnpicklingError, base64.binascii.Error) as e:
        print(f"Fehler beim Dekodieren oder Ent-pickeln: {e}", file=sys.stderr)
        return None

def get_run_name_from_path(path: Path) -> str:
    """
    Extrahiert einen verständlichen und eindeutigen Namen für einen Lauf aus dem Dateipfad.
    Für die Optuna-Struktur wird der Name des Eltern-Ordners (z.B. 'trial_148') verwendet.
    """
    # Der Name des Elternordners ist eine gute Kennung für einen Optuna-Trial.
    return path.parent.name

def load_all_data(target_folder: Path, target_series_name: str, all_series_names: list) -> dict:
    """
    Lädt die Daten aller Läufe einmalig in eine zentrale Cache-Struktur im Speicher.
    Liest 'actual_data' nur einmal und 'inference_data' für jeden Lauf.
    """
    print("Starte das Laden aller Daten in den Cache...")

    files_to_process = sorted(list(target_folder.glob('**/*.csv.tar.gz')))

    if not files_to_process:
        print(f"FEHLER: Keine .csv.tar.gz Dateien im Ordner {target_folder} gefunden.", file=sys.stderr)
        sys.exit(1)

    try:
        target_series_idx = all_series_names.index(target_series_name)
    except ValueError:
        print(f"FEHLER: '{target_series_name}' nicht in ALL_SERIES_NAMES gefunden.", file=sys.stderr)
        sys.exit(1)

    data_cache = {}
    actual_data_cache = None

    for file_path in files_to_process:
        run_name = get_run_name_from_path(file_path)
        print(f"  -> Verarbeite: {run_name} ({file_path.name})")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(file_path, "r:gz") as tar:
                    csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
                    if not csv_member:
                        print(f"WARNUNG: Keine .csv-Datei im Archiv {file_path.name} gefunden.", file=sys.stderr)
                        continue
                    tar.extract(csv_member, path=temp_dir)
                    df = pd.read_csv(Path(temp_dir) / csv_member.name)

            row = df.dropna(subset=['actual_data', 'inference_data']).iloc[0]
            
            # Lade Actuals nur einmal, da sie für alle Läufe identisch sind
            if actual_data_cache is None:
                actual_data_full = decode_and_unpickle(row['actual_data'])
                if actual_data_full is None:
                    print("FEHLER: 'actual_data' konnte nicht geladen werden. Breche ab.", file=sys.stderr)
                    sys.exit(1)
                actual_data_cache = actual_data_full[:, :, target_series_idx]

            inference_data_full = decode_and_unpickle(row['inference_data'])
            if inference_data_full is None:
                print(f"WARNUNG: 'inference_data' für {run_name} konnte nicht geladen werden. Überspringe.", file=sys.stderr)
                continue
            
            predictions = inference_data_full[:, :, target_series_idx]

            data_cache[run_name] = {
                'actuals': actual_data_cache,
                'predictions': predictions
            }
        except Exception as e:
            print(f"WARNUNG: Kritischer Fehler beim Verarbeiten von {file_path.name}: {e}. Überspringe.", file=sys.stderr)

    if not data_cache:
        print("FEHLER: Es konnten keine Daten erfolgreich geladen werden. Überprüfe die Dateien.", file=sys.stderr)
        sys.exit(1)
    
    print("Daten-Cache wurde erfolgreich erstellt.")
    return data_cache

# --- Schritt 2: Drei Methoden zur Identifikation "schwieriger" Fenster ---

def find_hardest_by_actuals_variance(data_cache: dict, num_windows: int) -> list:
    """Findet Fenster mit der höchsten Varianz in den tatsächlichen Werten (Actuals)."""
    first_run_name = next(iter(data_cache))
    actuals = data_cache[first_run_name]['actuals']  # Shape: (n_windows, window_len)
    
    variances = np.var(actuals, axis=1)
    hardest_indices = np.argsort(variances)[-num_windows:]
    
    return sorted(hardest_indices.tolist())

def find_hardest_by_avg_prediction_error(data_cache: dict, num_windows: int) -> list:
    """Findet Fenster mit dem höchsten durchschnittlichen Vorhersagefehler (MSE) über alle Modelle."""
    num_total_windows = data_cache[next(iter(data_cache))]['actuals'].shape[0]
    avg_errors = []

    for window_idx in range(num_total_windows):
        window_errors = []
        for run_name in data_cache:
            actual = data_cache[run_name]['actuals'][window_idx]
            pred = data_cache[run_name]['predictions'][window_idx]
            mse = np.mean((actual - pred)**2)
            window_errors.append(mse)
        avg_errors.append(np.mean(window_errors))
        
    hardest_indices = np.argsort(avg_errors)[-num_windows:]
    return sorted(hardest_indices.tolist())

def find_hardest_by_prediction_disagreement(data_cache: dict, num_windows: int) -> list:
    """Findet Fenster, in denen sich die Vorhersagen der verschiedenen Modelle am meisten unterscheiden."""
    num_total_windows = data_cache[next(iter(data_cache))]['actuals'].shape[0]
    disagreement_scores = []

    for window_idx in range(num_total_windows):
        all_preds_for_window = np.array([
            data_cache[run_name]['predictions'][window_idx] for run_name in data_cache
        ])  # Shape: (n_runs, window_len)
        
        variance_over_time = np.var(all_preds_for_window, axis=0)
        avg_disagreement = np.mean(variance_over_time)
        disagreement_scores.append(avg_disagreement)
            
    hardest_indices = np.argsort(disagreement_scores)[-num_windows:]
    return sorted(hardest_indices.tolist())

# --- Schritt 3: Eine wiederverwendbare Analyse-Pipeline ---

def perform_analysis_and_plotting(method_name: str, window_indices: list, data_cache: dict):
    """Führt eine vollständige statistische Analyse und Visualisierung für eine gegebene Liste von Fenstern durch."""
    print(f"\n--- Starte Analyse für Methode: '{method_name}' ---")
    
    # --- 3.1 Metriken und Residuen berechnen ---
    all_residuals = {}
    metrics_data = []

    print("\nMetriken für die ausgewählten Fenster:")
    for run_name, data in data_cache.items():
        actuals_subset = data['actuals'][window_indices]
        preds_subset = data['predictions'][window_indices]
        
        residuals = actuals_subset - preds_subset
        all_residuals[run_name] = residuals.flatten()
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        metrics_data.append({'Lauf': run_name, 'RMSE': rmse, 'MAE': mae})

    metrics_df = pd.DataFrame(metrics_data).set_index('Lauf')
    print(metrics_df.to_string(float_format="%.4f"))

    # --- 3.2 Signifikanz testen ---
    print("\nStatistische Signifikanz der Residuen (Fehler):")
    residuals_list = [res for res in all_residuals.values()]
    
    if len(residuals_list) > 1:
        h_stat, p_kruskal = kruskal(*residuals_list)
        print(f"Kruskal-Wallis H-Test: H-Statistik = {h_stat:.4f}, p-Wert = {p_kruskal:.4f}")
        
        if p_kruskal < ALPHA:
            print(f"Das Ergebnis ist signifikant (p < {ALPHA}). Führe Dunn's Post-hoc-Test durch:")
            dunn_result = sp.posthoc_dunn(residuals_list, p_adjust='bonferroni')
            dunn_result.columns = all_residuals.keys()
            dunn_result.index = all_residuals.keys()
            print(dunn_result.to_string(float_format="%.4f"))
        else:
            print(f"Das Ergebnis ist nicht signifikant (p >= {ALPHA}). Kein signifikanter Unterschied zwischen den Fehlerverteilungen der Läufe.")
    else:
        print("Nur ein Lauf vorhanden, statistische Tests werden übersprungen.")

    # --- 3.3 Boxplot erstellen ---
    fig_box, ax_box = plt.subplots(figsize=(10, 7))
    ax_box.boxplot(all_residuals.values(), labels=all_residuals.keys(), vert=True, patch_artist=True)
    ax_box.set_title(f"Fehlerverteilung (Residuen) für '{method_name}'", fontsize=16)
    ax_box.set_ylabel("Vorhersagefehler (Actual - Predicted)")
    ax_box.set_xlabel("Modell-Lauf")
    ax_box.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    fig_box.tight_layout()
    filename_box = f"boxplot_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig_box.savefig(filename_box, dpi=300)
    print(f"\nBoxplot gespeichert als: {filename_box}")
    plt.close(fig_box)

    # --- 3.4 Detail-Plot erstellen ---
    cols = 3
    rows = int(np.ceil(len(window_indices) / cols))
    figsize = (cols * 8, rows * 4)
    fig_detail, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = axes.flatten()
    pred_colors = plt.cm.viridis(np.linspace(0, 1, len(data_cache)))
    run_names = list(data_cache.keys())

    for i, window_idx in enumerate(window_indices):
        ax = axes[i]
        actual_series = data_cache[run_names[0]]['actuals'][window_idx]
        ax.plot(actual_series, label='Actual', color='black', lw=2.5, zorder=10)
        
        for run_idx, run_name in enumerate(run_names):
            pred_series = data_cache[run_name]['predictions'][window_idx]
            ax.plot(pred_series, label=f'Pred: {run_name}', linestyle='--', alpha=0.9, color=pred_colors[run_idx], lw=2)

        ax.set_title(f"Fenster #{window_idx}", fontsize=12, loc='left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylabel("Wert")
        if i >= len(axes) - cols:
             ax.set_xlabel("Zeitschritte im Fenster")

    handles, labels = axes[0].get_legend_handles_labels()
    fig_detail.legend(handles, labels, loc='upper right', fontsize=12)
    
    for j in range(len(window_indices), len(axes)):
        axes[j].set_visible(False)

    fig_detail.suptitle(f"Detail-Analyse für '{method_name}' ({TARGET_SERIES_NAME})", fontsize=20, y=0.99)
    fig_detail.tight_layout(rect=[0, 0.03, 0.9, 0.96])
    filename_detail = f"detailplot_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig_detail.savefig(filename_detail, dpi=300)
    print(f"Detail-Plot gespeichert als: {filename_detail}")
    plt.close(fig_detail)


def main():
    """Orchestriert den gesamten Analyseprozess."""
    target_path = Path(TARGET_FOLDER)
    
    # Schritt 1: Alle Daten zentral laden
    data_cache = load_all_data(target_path, TARGET_SERIES_NAME, ALL_SERIES_NAMES)
    
    # Schritt 2: Definitionen der "Härte"-Methoden
    hardness_methods = {
        "Höchste Varianz der Actuals": find_hardest_by_actuals_variance,
        "Höchster durchschn. Vorhersagefehler": find_hardest_by_avg_prediction_error,
        "Höchste Modell-Uneinigkeit": find_hardest_by_prediction_disagreement
    }

    # Schritt 4: Iteration durch die Methoden und Ausführung der Analyse-Pipeline
    for name, method_func in hardness_methods.items():
        print(f"\n{'='*80}")
        print(f"BEGINNE ANALYSE FÜR METHODE: '{name}'")
        print(f"{'='*80}")
        
        # Schritt 2 (Aufruf): Schwierigste Fenster für die aktuelle Methode finden
        hardest_indices = method_func(data_cache, NUM_HARDEST_WINDOWS)
        print(f"\nDie {NUM_HARDEST_WINDOWS} schwierigsten Fenster nach dieser Methode sind: {hardest_indices}")
        
        # Schritt 3 (Aufruf): Analyse und Plotting für diese Fenster durchführen
        perform_analysis_and_plotting(name, hardest_indices, data_cache)

    print(f"\n{'='*80}")
    print("Alle Analysen wurden erfolgreich abgeschlossen.")
    print(f"{'='*80}")


if __name__ == '__main__':
    # Überprüfen, ob das Zielverzeichnis existiert
    if not os.path.isdir(TARGET_FOLDER):
        print(f"FEHLER: Das angegebene Verzeichnis '{TARGET_FOLDER}' existiert nicht.", file=sys.stderr)
        sys.exit(1)
    main()