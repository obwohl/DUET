import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone


def fetch_brightsky_data(start_date: datetime, end_date: datetime, station_id: str) -> pd.DataFrame | None:
    TARGET_TIMEZONE = 'Europe/Berlin'
    start_utc = start_date.astimezone(timezone.utc) if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date.astimezone(timezone.utc) if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    start_str = start_utc.isoformat(timespec='seconds')
    end_str = end_utc.isoformat(timespec='seconds')
    params = {'dwd_station_id': station_id, 'date': start_str, 'last_date': end_str}
    
    print(f"Lade Wetterdaten von Bright Sky für den Zeitraum (in UTC): {start_str} bis {end_str}...")
    try:
        response = requests.get("https://api.brightsky.dev/weather", params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('weather', [])
        if not data:
            print("Keine Wetterdaten für den angefragten Zeitraum gefunden.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        print(f"Erfolgreich {len(df)} stündliche Wetter-Datenpunkte geladen.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Netzwerk- oder API-Fehler beim Abrufen der Wetterdaten: {e}")
        return None

def get_prepared_weather_data():
    TARGET_TIMEZONE = 'Europe/Berlin'
    TAGE_VERGANGENHEIT = 11*365+90
    TAGE_ZUKUNFT = 5
    now_local = datetime.now().astimezone()
    start_date = now_local - timedelta(days=TAGE_VERGANGENHEIT)
    end_date = now_local + timedelta(days=TAGE_ZUKUNFT)

    df_raw = fetch_brightsky_data(start_date, end_date, "03379")
    if df_raw is None or df_raw.empty:
        print("Download der Wetterdaten fehlgeschlagen. Überspringe Wetter-Integration.")
        return pd.DataFrame()

    print(f"\nVerarbeite Wetterdaten und konvertiere zu Zeitzone '{TARGET_TIMEZONE}'...")
    print(df_raw.columns)
    wetter_df = df_raw[['timestamp', 'temperature', 'pressure_msl']].copy()
    wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
    wetter_df.set_index('timestamp', inplace=True)
    wetter_df.index = wetter_df.index.tz_convert(TARGET_TIMEZONE)
    wetter_df.sort_index(inplace=True)
    wetter_df = wetter_df[~wetter_df.index.duplicated(keep='first')]
    wetter_df.rename(columns={'temperature': 'lufttemperatur_c', 'pressure_msl': 'luftdruck'}, inplace=True)
    
    # --- KORREKTUR: Fehlende Werte interpolieren, anstatt sie mit 0 zu füllen ---
    # Dies ist der entscheidende Fix. Wir behandeln Luftdruck genauso wie Temperatur.
    wetter_df['luftdruck'] = wetter_df['luftdruck'].interpolate(method='time')
    wetter_df['lufttemperatur_c'] = wetter_df['lufttemperatur_c'].interpolate(method='time')

    print("Resample Wetterdaten auf 1-Stunden-Intervall...")
    wetter_1h = wetter_df.resample('1h').agg({
        'lufttemperatur_c': 'mean',
        'luftdruck': 'mean'
    }).round(2)
    return wetter_1h


weather_wide = get_prepared_weather_data()
weather_wide.reset_index(inplace=True)

# --- KORRIGIERTE VERARBEITUNG FÜR MEHRERE WETTER-FEATURES ---
# 1. Umwandlung vom breiten ins lange Format mit pd.melt
weather = pd.melt(
    weather_wide,
    id_vars=['timestamp'],
    value_vars=['lufttemperatur_c', 'luftdruck'],
    var_name='cols',
    value_name='data'
)

# 2. Anwenden des 96h-Shifts auf ALLE Wetter-Features
# Der Shift um -96 ist für eine "geleakte" Zukunfts-Information für das Modell.
# .groupby('cols') stellt sicher, dass der Shift für jede Zeitreihe (Temperatur, Druck)
# separat und korrekt durchgeführt wird, ohne Daten zwischen ihnen zu mischen.
weather['data'] = weather.groupby('cols')['data'].shift(-96)

# 3. Spalten umbenennen und aufräumen
weather.rename(columns={'timestamp': 'date'}, inplace=True)
weather['cols'] = weather['cols'].replace({
        'lufttemperatur_c': 'airtemp_96',
        'luftdruck': 'pressure_96'
    })
weather.dropna(subset=['data'], inplace=True)

weather.to_csv("weather.csv")

def fetch_data_from_url(url, column_name):
    # <<< UNVERÄNDERT: Diese Funktion bleibt exakt wie im Original. >>>
    print(f"-> Processing URL for: {column_name}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Could not fetch URL {url}: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find("table", class_="tblsort")
    if not table:
        table = soup.find("table", class_="datentabelle")
        if not table:
            print(f"   [WARN] No table with class 'tblsort' or 'datentabelle' found on URL: {url}")
            return pd.DataFrame()
    try:
        headers = [header.get_text(strip=True) for header in table.find('thead').find_all("th")]
    except AttributeError:
        print(f"   [WARN] Could not find a 'thead' section in the table for URL: {url}")
        return pd.DataFrame()
    if any('Uhrzeit' in s for s in headers):
        df_headers = headers
    else:
        df_headers = ['Datum/Uhrzeit'] + headers[1:]
    rows = table.find('tbody').find_all("tr")
    if not rows:
        print(f"   [WARN] No data rows (tr) found in table body for URL: {url}")
        return pd.DataFrame()
    data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        row_data = {df_headers[i]: cell.get_text(strip=True) for i, cell in enumerate(cells) if i < len(df_headers)}
        data.append(row_data)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'Datum/Uhrzeit' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M', errors='coerce')
    elif 'Datum' in df.columns:
        if 'Uhrzeit' in df.columns:
            df['timestamp_str'] = df['Datum'] + ' ' + df['Uhrzeit']
            df["timestamp"] = pd.to_datetime(df['timestamp_str'], format="%d.%m.%Y %H:%M", errors='coerce')
        else:
            df["timestamp"] = pd.to_datetime(df['Datum'], format="%d.%m.%Y", errors='coerce')
    else:
        print(f"   [WARN] 'Datum' column not found. Cannot process timestamps.")
        return pd.DataFrame()
    df.dropna(subset=['timestamp'], inplace=True)
    if df.empty:
        print(f"   [WARN] No valid timestamps could be parsed.")
        return pd.DataFrame()
    target_header = column_name.split('_')[0]
    if target_header not in df.columns:
        print(f"   [ERROR] Expected header '{target_header}' not found in table. Available headers: {df.columns.tolist()}")
        return pd.DataFrame()
    df_final = df[["timestamp", target_header]].copy()
    df_final.rename(columns={target_header: column_name}, inplace=True)
    df_final[column_name] = pd.to_numeric(
        df_final[column_name].astype(str).str.replace("--", "", regex=False).str.replace(",", ".", regex=False),
        errors='coerce'
    )
    print(f"   [SUCCESS] Found and processed {len(df_final)} rows.")
    return df_final


end_date = datetime.now() - timedelta(hours=1)
start_date = end_date - timedelta(days=90)

end_date_str = end_date.strftime("%d.%m.%Y")
start_date_str = start_date.strftime("%d.%m.%Y")

print(f"--- Starting data fetch for period: {start_date_str} to {end_date_str} ---\n")

urls_and_columns = {
    f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date_str}&ende={end_date_str}": "Wassertemperatur [°C]_München Himmelreichbruecke"
}
all_dfs = []
for url, col_name in urls_and_columns.items():
    df = fetch_data_from_url(url, col_name)
    if not df.empty:
        all_dfs.append(df)
if all_dfs:
    print("\n--- All fetching complete. Merging data... ---")
    merged_data = all_dfs[0]
    for df_to_merge in all_dfs[1:]:
        merged_data = pd.merge(merged_data, df_to_merge, on='timestamp', how='outer')
    merged_data = merged_data.sort_values(by='timestamp', ascending=True)
    merged_data = merged_data.reset_index(drop=True)
    print(f"\nSUCCESS: Data processed successfully.")
    print("\nInfo on final DataFrame:")
    merged_data.info()
    print(f"isnas: {merged_data.isna().sum()}")
else:
    print("\n--- No data was successfully merged from any URL. Please check warnings above. ---")

import re
cols_to_keep = ['timestamp', 'item_id']
cols_to_rename = [col for col in merged_data.columns if col not in cols_to_keep]
new_names = {}
for col in cols_to_rename:
    new_col = col.lower()
    new_col = new_col.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    new_col = re.sub(r'\[.*?\]', '', new_col)
    new_col = re.sub(r'_+', '_', new_col)
    new_col = new_col.strip('_')
    new_names[col] = new_col
merged_data.rename(columns=new_names, inplace=True)

water = merged_data
water["timestamp"] = pd.to_datetime(water["timestamp"])
water = water.sort_values("timestamp")
water = water.set_index("timestamp").resample("1h").median()
water=water.reset_index()


water = water.set_index(
    pd.to_datetime(water['timestamp'])
      .dt.tz_localize('Europe/Berlin', nonexistent='NaT')
)

water.drop(columns=['timestamp'], inplace=True)
water.reset_index(inplace=True)
water.rename(columns={"timestamp": "date", "wassertemperatur _muenchen himmelreichbruecke": "data"}, inplace=True)
water["cols"]="wassertemp"
start_date = max(water['date'].min(), weather['date'].min())
end_date = min(water['date'].max(), weather['date'].max())

water_filtered = water[water['date'].between(start_date, end_date)]
weather_filtered = weather[weather['date'].between(start_date, end_date)]

combined_df = pd.concat([water_filtered, weather_filtered], ignore_index=True)

combined_df.to_csv("inference_live.csv", index=False)