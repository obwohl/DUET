# check_data.py
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def check_time_series_data(file_path: Path):
    """
    Führt eine grundlegende Integritätsprüfung für eine Zeitreihen-CSV-Datei durch.
    """
    if not file_path.exists():
        print(f"❌ FEHLER: Datei nicht gefunden unter: {file_path}")
        return

    print(f"--- Starte Analyse für: {file_path.name} ---")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ FEHLER: Konnte CSV-Datei nicht lesen. Grund: {e}")
        return

    # --- Test 1: Grundlegende Informationen ---
    print("\n[1] Grundlegende Datei-Informationen:")
    print(f"  - Spalten: {df.columns.tolist()}")
    print(f"  - Zeilen: {len(df)}")
    # KORREKTUR: Der 'indent'-Parameter wird von Series.to_string() nicht unterstützt.
    # Wir rücken den String manuell ein.
    indented_dtypes = "    " + df.dtypes.to_string().replace('\n', '\n    ')
    print(f"  - Datentypen:\n{indented_dtypes}")

    # --- Test 2: Fehlende Werte (NaNs) ---
    print("\n[2] Prüfung auf fehlende Werte (NaN):")
    nan_counts = df.isnull().sum()
    if nan_counts.sum() == 0:
        print("  ✅ OK: Keine NaN-Werte gefunden.")
    else:
        indented_nans = "    " + nan_counts[nan_counts > 0].to_string().replace('\n', '\n    ')
        print(f"  🚨 WARNUNG: NaN-Werte gefunden!\n{indented_nans}")

    # --- Test 3: Prüfung auf Null-Werte (der Hauptverdächtige) ---
    print("\n[3] Prüfung auf Null-Werte (0.0):")
    value_col = 'data' # Annahme basierend auf deinen CSVs
    if value_col not in df.columns:
        print(f"  - INFO: Spalte '{value_col}' nicht gefunden. Überspringe Null-Prüfung.")
    else:
        zero_count = (df[value_col] == 0).sum()
        if zero_count == 0:
            print("  ✅ OK: Keine Null-Werte in der Datenspalte gefunden.")
        else:
            print(f"  🚨 ALARM: {zero_count} Null-Werte in '{value_col}' gefunden!")
            print("     Dies ist die wahrscheinlichste Ursache für das pathologische Modellverhalten.")

    # --- Test 4: Statistische Übersicht ---
    print("\n[4] Statistische Übersicht:")
    if value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        stats = df[value_col].describe()
        indented_stats = "  " + stats.to_string().replace('\n', '\n  ')
        print(indented_stats)
        if stats['min'] == 0:
            print("  🚨 ALARM: Der Minimalwert ist 0.0, was die Anwesenheit von Nullen bestätigt.")
    else:
        print("  - INFO: Keine numerische Datenspalte für Statistik gefunden.")

    print("\n--- Analyse abgeschlossen. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimalistisches Skript zur Überprüfung der Datenqualität einer Zeitreihen-CSV.")
    # Setze den Standardwert auf die von dir genannte Datei
    parser.add_argument('--file', type=str, default="dataset/forecasting/pressure_96.csv", help="Pfad zur zu überprüfenden CSV-Datei.")
    args = parser.parse_args()
    
    check_time_series_data(Path(args.file))
