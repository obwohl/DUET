import pandas as pd
import numpy as np

# 1. Parameter für den Datensatz festlegen
n_samples = 365 * 24  # 1 Jahr an stündlichen Daten
shift_hours = 24

# Parameter für die Signale
noise_level_cause = 0.3
trend_strength = 0.001
daily_period = 24
weekly_period = 24 * 7

# 2. Zeitstempel erstellen
# Wir erstellen mehr Samples, um die NaNs nach dem Verschieben auszugleichen
date_rng = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples + shift_hours, freq='H'))

# 3. Synthetische "cause"-Zeitreihe erstellen
time_steps = np.arange(len(date_rng))
trend = trend_strength * time_steps
daily_seasonality = np.sin(2 * np.pi * time_steps / daily_period)
weekly_seasonality = 0.5 * np.sin(2 * np.pi * time_steps / weekly_period)
cause_noise = np.random.normal(0, noise_level_cause, len(date_rng))

cause_data = trend + daily_seasonality + weekly_seasonality + cause_noise

# 4. DataFrame im Wide-Format erstellen
df_wide = pd.DataFrame(date_rng, columns=['date'])
df_wide['cause'] = cause_data

# 5. "effect"-Zeitreihe durch Verschieben der "cause"-Reihe erstellen
# This creates a perfect time-shifted relationship. The 'effect' channel is
# exactly the 'cause' channel, but 24 hours later. The Mahalanobis mask, which
# uses the magnitude of the FFT, is designed to be invariant to such time shifts.
# Therefore, it should calculate a very small distance between these two channels,
# leading to a high attention probability. We remove any additional noise or drift
# to make this relationship as clear as possible for the model to detect.
df_wide['effect'] = df_wide['cause'].shift(shift_hours)

# 6. NaNs entfernen, die durch das Verschieben entstanden sind
df_wide.dropna(inplace=True)

print("Vorschau des DataFrames im Wide-Format (nach dem Verschieben):")
print(df_wide.head())
print("\n")

# 7. DataFrame ins Long-Format umwandeln
df_long = pd.melt(
    df_wide,
    id_vars=['date'],
    value_vars=['cause', 'effect'],
    var_name='cols',
    value_name='data'
)

# 8. Spalten in die gewünschte Reihenfolge bringen und sortieren
df_long = df_long[['date', 'data', 'cols']]
df_long.sort_values(by=['cols', 'date'], inplace=True)
df_long.reset_index(drop=True, inplace=True)


print("Vorschau des finalen DataFrames im Long-Format:")
print(df_long.head())
print("...")
print(df_long.tail())
print("\n")

# 9. DataFrame als CSV-Datei speichern
output_filename = 'causal_1.csv'
df_long.to_csv(output_filename, index=False)

print(f"Datensatz erfolgreich in der Datei '{output_filename}' gespeichert.")
print(f"Anzahl der Zeilen: {len(df_long)}")
print(f"Anzahl der Einträge pro Kanal: {len(df_long) / 2}")