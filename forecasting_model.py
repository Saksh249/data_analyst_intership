import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import timedelta

# === 1. Load dataset ===
file_path = r"C:\Users\USER\Downloads\EV_Charging_Forecast\ev_final_dataset.csv"

# Fix dtype & mixed data warning
df = pd.read_csv(file_path, low_memory=False)

# === 2. Clean and prepare time-series data ===
df['kWhDelivered'] = pd.to_numeric(df['kWhDelivered'], errors='coerce')
df['date_hour'] = pd.to_datetime(df['date_hour'], errors='coerce')
df = df.sort_values('date_hour')
df['kWhDelivered'] = df['kWhDelivered'].ffill()  # forward fill
df = df.dropna(subset=['date_hour', 'kWhDelivered'])

# Set datetime index
ts = df.set_index('date_hour')['kWhDelivered']

# === 3. Plot original time series ===
plt.figure(figsize=(12, 5))
plt.plot(ts, color='teal', linewidth=1.8)
plt.title('⚡ EV Charging Demand Over Time', fontsize=14)
plt.xlabel('Date Hour', fontsize=11)
plt.ylabel('kWh Delivered', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 4. Build ARIMA model ===
print("\n🔹 Training ARIMA model...")
model_arima = ARIMA(ts, order=(2, 1, 2))
results_arima = model_arima.fit()

# Forecast next 24 hours
forecast_steps = 24
forecast_arima = results_arima.forecast(steps=forecast_steps)
future_dates = [ts.index[-1] + timedelta(hours=i) for i in range(1, forecast_steps + 1)]
forecast_df_arima = pd.DataFrame({'date_hour': future_dates, 'Forecast_ARIMA': forecast_arima})

# === 5. Build Prophet model ===
print("\n🔹 Training Prophet model...")
prophet_df = df[['date_hour', 'kWhDelivered']].rename(columns={'date_hour': 'ds', 'kWhDelivered': 'y'})
model_prophet = Prophet(interval_width=0.95, daily_seasonality=True)
model_prophet.fit(prophet_df)

# Forecast 24 hours ahead
future = model_prophet.make_future_dataframe(periods=24, freq='H')
forecast_prophet = model_prophet.predict(future)

# === 6. Prophet Forecast Plot ===
fig1 = model_prophet.plot(forecast_prophet, figsize=(12, 5))
plt.title("🔮 Prophet Forecast (Next 24 Hours)", fontsize=14)
plt.xlabel("Years", fontsize=11)
plt.ylabel("Predicted kWh Delivered", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 7. Comparison Plot ===
plt.figure(figsize=(12, 5))
plt.plot(ts[-100:], label='Actual Demand', color='black', linewidth=2)
plt.plot(forecast_df_arima['date_hour'], forecast_df_arima['Forecast_ARIMA'],
         label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
plt.plot(forecast_prophet['ds'][-24:], forecast_prophet['yhat'][-24:],
         label='Prophet Forecast', color='blue', linestyle='--', linewidth=2)
plt.title("📈 EV Charging Demand Forecast Comparison", fontsize=14)
plt.xlabel("Time", fontsize=11)
plt.ylabel("kWh Delivered", fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 8. Save forecast results ===
output_path = r"C:\Users\USER\Downloads\EV_Charging_Forecast\ev_forecast_results.csv"
final_forecast = pd.DataFrame({
    'datetime': forecast_df_arima['date_hour'],
    'ARIMA_forecast_kWh': forecast_df_arima['Forecast_ARIMA'].values,
    'Prophet_forecast_kWh': forecast_prophet['yhat'][-24:].values
})
final_forecast.to_csv(output_path, index=False)

print(f"\n✅ Forecast results saved to: {output_path}")
