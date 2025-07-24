import pandas as pd
from ydata_profiling import ProfileReport

# --- Konfiguration ---
input_csv_path = 'lorenz.csv'
output_wide_processed_csv = "df_wide_processed.csv" # Name angepasst
output_long_processed_csv = "df_long_processed.csv" # Name angepasst
output_profile_report_html = "mein_korrekter_timeseries_report.html"

# --- 1. Laden des Datensatzes ---
try:
    df = pd.read_csv(input_csv_path)
    print(f"'{input_csv_path}' erfolgreich geladen.")
except FileNotFoundError:
    print(f"Fehler: Die Datei '{input_csv_path}' wurde nicht gefunden.")
    exit()
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    exit()

# Sicherstellen, dass die 'date'-Spalte im korrekten Datumsformat ist
df['date'] = pd.to_datetime(df['date'])

# --- 2. Dynamische Erkennung der Kanalnamen (Werte in der 'cols'-Spalte) ---
channel_order = df['cols'].unique().tolist()
channel_order.sort() # Optional: Sortiere die Kanäle, z.B. 'Channel_0', 'Channel_1'
print(f"Dynamisch erkannte Kanäle (channels): {channel_order}")

# --- 3. Konvertierung ins Wide-Format ---
df_wide = df.pivot(index='date', columns='cols', values='data')

# Sicherstellen, dass die Spalten des df_wide der gewünschten Reihenfolge entsprechen
df_wide = df_wide.reindex(columns=channel_order)
print("Daten erfolgreich ins Wide-Format konvertiert.")

# --- Die gesamte 0- und NaN-Prüfung/Interpolation wurde hier entfernt ---
print("\nDie Prüfung und Interpolation von Nullen und NaNs wurde übersprungen, da die Daten als 'gesund' angenommen werden.")


# --- 4. Export der df_wide (ohne Interpolation, nur Transformation) ---
df_wide.to_csv(output_wide_processed_csv)
print(f"df_wide erfolgreich als '{output_wide_processed_csv}' gespeichert.")

# --- 5. Konvertierung zurück ins Long-Format ---
df_long_processed = df_wide.reset_index().melt(
    id_vars='date',
    value_vars=channel_order, # Verwendet die dynamisch ermittelte Kanalreihenfolge
    var_name='cols',
    value_name='data'
)

# Sicherstellen, dass die finalen Spalten in der exakten Reihenfolge 'date', 'data', 'cols' sind
final_long_column_order = ['date', 'data', 'cols'] 
df_long_processed = df_long_processed[final_long_column_order]

# --- 6. Export der df_long (ohne Interpolation, nur Transformation) ---
df_long_processed.to_csv(output_long_processed_csv, index=False)
print(f"df_long erfolgreich als '{output_long_processed_csv}' gespeichert.")

# --- 7. Erstellung des Profilberichts ---
report_title = f"Profiling von {len(channel_order)} Lorenz-Zeitreihen"
print(f"\nErstelle Profilbericht '{report_title}'...")
profile = ProfileReport(
    df_wide, # Verwenden Sie das Wide-Format für den Profilbericht
    tsmode=True,
    title=report_title
)
profile.to_file(output_profile_report_html)
print(f"Bericht erfolgreich als '{output_profile_report_html}' erstellt!")