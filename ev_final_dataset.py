import pandas as pd
import os
from datetime import timedelta

# ======== 1. Set base path ========
base_path = r"C:\Users\USER\Downloads\EV_Charging_Forecast"

# ======== 2. Define CSV files ========
files = {
    "ev": "ev_usage.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv"
}

# ======== 3. Load all files safely ========
data = {}
for key, file in files.items():
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        print(f"✅ Loading {file}")
        data[key] = pd.read_csv(file_path)
    else:
        print(f"⚠️ File not found: {file_path} (Skipping this file)")

# ======== 4. Convert date columns ========
for key, df in data.items():
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df['date'] = pd.to_datetime(df[date_cols[0]], errors='coerce', utc=True)
        print(f"✅ Converted '{date_cols[0]}' to datetime in {key}")
    else:
        print(f"⚠️ No date column found in {key}. Columns: {df.columns}")

# ======== 5. Create EV datetime from dayIndicator + connectionTime_decimal ========
if 'ev' in data:
    if 'traffic' in data and 'date' in data['traffic'].columns:
        first_date = pd.to_datetime(data['traffic']['date'].min()).normalize()
    else:
        first_date = pd.Timestamp.today().normalize()

    def ev_to_datetime(row):
        try:
            ev_date = first_date + timedelta(days=int(row['dayIndicator']) - 1)
            hours = int(row['connectionTime_decimal'])
            minutes = int((row['connectionTime_decimal'] - hours) * 60)
            return ev_date + timedelta(hours=hours, minutes=minutes)
        except Exception:
            return pd.NaT

    data['ev']['date'] = data['ev'].apply(ev_to_datetime, axis=1)

# ======== 6. Ensure every dataset has valid datetime ========
for key, df in data.items():
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
        print(f"⚙️ Creating synthetic datetime column for {key}")
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
    else:
        # Force type consistency
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ======== 7. Round all datasets safely to nearest hour ========
for key, df in data.items():
    try:
        df['date_hour'] = df['date'].dt.floor('h')
    except Exception:
        print(f"⚠️ Could not apply .dt.floor() to {key}, coercing now...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_hour'] = df['date'].dt.floor('h')

# ======== 8. Merge datasets ========
merged = data['ev']
if 'traffic' in data:
    merged = merged.merge(data['traffic'][['date_hour', 'Vehicles']], on='date_hour', how='left')
if 'weather' in data:
    # Keep only unique date_hour and numeric columns for weather
    weather_cols = ['date_hour'] + [c for c in data['weather'].columns if c != 'date_hour']
    merged = merged.merge(data['weather'][weather_cols], on='date_hour', how='left')

# ======== 9. Save final dataset ========
final_file_path = os.path.join(base_path, "ev_final_dataset.csv")
merged.to_csv(final_file_path, index=False)

# ======== 10. Summary ========
print(f"\n✅ Final dataset saved as '{final_file_path}'")
print(f"Rows: {len(merged)}, Columns: {len(merged.columns)}")
print("Columns:", list(merged.columns))
print("\nPreview:\n", merged.head())
