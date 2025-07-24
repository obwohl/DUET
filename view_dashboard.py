import os
import sys
import pandas as pd
import json
import tarfile
import tempfile
import shutil
from ts_benchmark.report import report
from ts_benchmark.common.constant import ROOT_PATH

# Fügen Sie das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath('.'))

def view_combined_results_final():
    """
    Sammelt, modifiziert und zeigt Ergebnisse aus mehreren Benchmark-Läufen
    mit unterschiedlichen Konfigurationen an, indem alle inkonsistenten
    Argumente neutralisiert werden.
    """
    temp_dir = None
    try:
        # Der Ordner, in dem Ihre Ergebnisse gespeichert sind
        log_folder = os.path.join(ROOT_PATH, "result", "eisbach/DUET")

        # Findet alle komprimierten CSV-Ergebnisdateien
        all_files = os.listdir(log_folder)
        log_files = [os.path.join(log_folder, f) for f in all_files if f.endswith(".csv.tar.gz")]

        if not log_files:
            print(f"Keine Ergebnisdateien (.csv.tar.gz) im Ordner gefunden: {log_folder}")
            print("Bitte führen Sie zuerst das Skript 'run_all_benchmarks.sh' aus.")
            return

        print(f"Gefundene Ergebnisdateien: {len(log_files)}. Verarbeite...")

        # Erstellt ein temporäres Verzeichnis für modifizierte CSVs
        temp_dir = tempfile.mkdtemp()
        modified_log_files = []

        # Ein konsistenter Platzhalter für die problematischen Spalten
        CONSISTENT_ARGS_PLACEHOLDER = "{'status': 'neutralized for report'}"

        for log_file in log_files:
            # Extrahiert die .tar.gz-Datei
            try:
                with tarfile.open(log_file, "r:gz") as tar:
                    csv_member_info = tar.getmembers()[0]
                    tar.extractall(path=temp_dir)
                    csv_path = os.path.join(temp_dir, csv_member_info.name)
            except Exception as e:
                print(f"Warnung: Konnte die Datei {os.path.basename(log_file)} nicht entpacken. Überspringe... Fehler: {e}")
                continue

            # Liest die CSV-Datei und modifiziert sie
            df = pd.read_csv(csv_path)

            # --- KERN DER LÖSUNG ---
            # 1. Extrahiert den Horizont aus strategy_args zur Benennung
            try:
                strategy_args_str = df['strategy_args'].iloc[0].replace("'", '"')
                strategy_args = json.loads(strategy_args_str)
                horizon = strategy_args.get('horizon', 'N/A')
            except Exception:
                horizon = f"unknown_{len(modified_log_files)}" # Eindeutiger Fallback

            # 2. Erstellt einen eindeutigen Modellnamen mit dem Horizont
            original_model_name = df['model_name'].iloc[0]
            df['model_name'] = f'{original_model_name}_h{horizon}'

            # 3. Neutralisiert die Spalten, die auf Konsistenz geprüft werden
            df['strategy_args'] = CONSISTENT_ARGS_PLACEHOLDER
            if 'model_hyper_params' in df.columns:
                 df['model_hyper_params'] = CONSISTENT_ARGS_PLACEHOLDER
            # --- ENDE DER LÖSUNG ---

            # Speichert die geänderte Datei in einem neuen CSV
            modified_csv_path = os.path.join(temp_dir, f"modified_{os.path.basename(csv_path)}")
            df.to_csv(modified_csv_path, index=False)
            modified_log_files.append(modified_csv_path)

        if not modified_log_files:
            print("Konnte keine gültigen Ergebnisdateien verarbeiten.")
            return
            
        print("Verarbeitung abgeschlossen. Starte kombiniertes Dashboard...")

        # Konfiguration für den Report mit den modifizierten Dateien
        report_config = {
            "log_files_list": modified_log_files,
            "report_metrics": ["mae", "mse"],
            "aggregate_type": "mean",
            "fill_type": "mean_value",
            "null_value_threshold": 0.3,
            "save_path": "eisbach/DUET"
        }

        # Startet das interaktive Dashboard
        report(report_config, report_method="dash")

    except Exception as e:
        import traceback
        print(f"\nEin unerwarteter Fehler ist aufgetreten: {e}")
        traceback.print_exc()
    finally:
        # Räumt das temporäre Verzeichnis auf
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nTemporäres Verzeichnis aufgeräumt.")

if __name__ == "__main__":
    view_combined_results_final()