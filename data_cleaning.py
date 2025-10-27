# scripts/data_cleaning.py
import os
import pandas as pd
from utils import ensure_dirs

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'EV_Forecast_Output')
ensure_dirs(OUT_DIR)

def load_ev_usage(path):
    # Expect ev_usage to have a timestamp column (name could be 'timestamp' or 'datetime') and a 'demand' or 'usage' column
    df = pd.read_csv(path)
    # try common names
    for col in ['timestamp','time','datetime']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    if 'datetime' not in df.columns:
        raise ValueError("ev_usage.csv must contain a datetime/timestamp column")
    # unify demand column name
    if 'usage' in df.columns:
        df['demand'] = df['usage']
    elif 'demand' in df.columns:
        pass
    elif 'count' in df.columns:
        df['demand'] = df['count']
    else:
        # If no demand column, assume second column is demand
        cols = [c for c in df.columns if c != 'datetime']
        if len(cols)==0:
            raise ValueError("No demand column found")
        df['demand'] = df[cols[0]]
    df = df[['datetime','demand']]
    df = df.sort_values('datetime').drop_duplicates(subset='datetime')
    df = df.set_index('datetime')
    return df

def load_weather(path):
    df = pd.read_csv(path)
    # unify datetime
    for col in ['timestamp','time','datetime','date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    df = df.set_index('datetime').sort_index()
    # keep common features if present
    keep = [c for c in ['temperature','temp','rain','precipitation','wind_speed','humidity','weather_desc'] if c in df.columns]
    if not keep:
        # keep all except columns that are non-numeric strings
        keep = df.columns.tolist()
    return df[keep]

def load_traffic(path):
    df = pd.read_csv(path)
    for col in ['timestamp','time','datetime','date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    df = df.set_index('datetime').sort_index()
    # common columns: 'traffic_volume', 'traffic_count'
    if 'traffic_volume' in df.columns:
        df = df[['traffic_volume']]
    elif 'traffic_count' in df.columns:
        df = df[['traffic_count']].rename(columns={'traffic_count':'traffic_volume'})
    else:
        # take first numeric column
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols)==0:
            raise ValueError("No numeric traffic column found")
        df = df[[numeric_cols[0]]].rename(columns={numeric_cols[0]:'traffic_volume'})
    return df

def merge_all(ev_path, weather_path, traffic_path, resample_rule='H'):
    ev = load_ev_usage(ev_path)
    weather = load_weather(weather_path)
    traffic = load_traffic(traffic_path)

    # Resample to hourly (or given rule)
    ev = ev.resample(resample_rule).sum(min_count=1)  # sum demand in hour
    weather = weather.resample(resample_rule).mean()
    traffic = traffic.resample(resample_rule).mean()

    # Merge
    df = ev.join(weather, how='left').join(traffic, how='left')

    # Feature engineering
    df = df.reset_index()
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df = df.set_index('datetime').sort_index()

    # Lags and rolling features for demand
    for lag in [1,2,3,6,24,168]:   # 1h, 2h, 3h, 6h, 24h, 7d
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    df['demand_roll_3'] = df['demand'].rolling(window=3, min_periods=1).mean()
    df['demand_roll_24'] = df['demand'].rolling(window=24, min_periods=1).mean()

    # simple missing value handling
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df

if __name__ == "__main__":
    ev_path = os.path.join(DATA_DIR, 'ev_usage.csv')
    weather_path = os.path.join(DATA_DIR, 'weather.csv')
    traffic_path = os.path.join(DATA_DIR, 'traffic.csv')

    df = merge_all(ev_path, weather_path, traffic_path, resample_rule='H')
    out_file = os.path.join(OUT_DIR, 'ev_final_dataset.csv')
    df.to_csv(out_file)
    print(f"Saved merged dataset to: {out_file}")
