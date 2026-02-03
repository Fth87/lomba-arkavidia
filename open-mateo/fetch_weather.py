import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


# --- CONFIGURATION ---
FOLDER_NAME = "open-mateo"
START_DATE = "2025-09-01"
END_DATE = "2025-11-30"

LOCATIONS = {
    "DKI1": {"lat": -6.1927, "lon": 106.8222},
    "DKI2": {"lat": -6.1557, "lon": 106.8973},
    "DKI3": {"lat": -6.3533, "lon": 106.8285},
    "DKI4": {"lat": -6.2878, "lon": 106.9100},
    "DKI5": {"lat": -6.1901, "lon": 106.7628}
}

# Variable mapping untuk pengambilan data Hourly
# Kita ambil hourly lalu aggregate ke daily agar bisa dapat MIN/MAX/MEAN lengkap
HOURLY_VARS = [
    "temperature_2m", "precipitation", "wind_speed_10m", 
    "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation", 
    "relative_humidity_2m", "cloud_cover", "surface_pressure"
]

# --- SETUP CLIENT ---
# Setup cache & retry (Best Practice)
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data(name, coords):
    """Fetch hourly ERA5 data and aggregate to daily statistics."""
    
    # Gunakan Historical Forecast API untuk "Forecast" (bukan Real/Reanalysis ERA5)
    # Sesuai requirement: "forecasting, bukan real"
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Jakarta"
    }

    try:
        print(f"⏳ Mengambil data untuk {name}...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # 1. Process Hourly Data
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Asia/Jakarta"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Jakarta"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        for i, col in enumerate(HOURLY_VARS):
            hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data=hourly_data)
        df.set_index("date", inplace=True)

        # 2. Aggregation Logic (Resample Daily)
        # Dictionary comprehension untuk mendefinisikan rules aggregasi
        agg_rules = {
            "temperature_2m": ["max", "min", "mean"],
            "precipitation": ["sum"], # hours calculated separately
            "wind_speed_10m": ["max", "mean", "min"],
            "wind_gusts_10m": ["max", "mean", "min"],
            "wind_gusts_10m": ["max", "mean", "min"],
            # shortwave_radiation (W/m2) -> sum gives W/m2*hours count? 
            # We will handle solar separately to convert to MJ/m2
            "shortwave_radiation": ["sum"],
            "relative_humidity_2m": ["mean", "max", "min"],
            "cloud_cover": ["mean", "max", "min"],
            "surface_pressure": ["mean", "max", "min"]
        }

        # Lakukan resampling dasar
        df_daily = df.resample('D').agg(agg_rules)

        # Hitung Wind Direction Dominant (Mode/Most Frequent)
        # Mode tidak bisa langsung di agg(), jadi kita pakai apply lambda
        wind_dir = df['wind_direction_10m'].resample('D').apply(lambda x: x.mode()[0] if not x.mode().empty else x.mean())
        
        # Hitung Precipitation Hours (Jam dengan hujan > 0.1mm)
        precip_hours = df['precipitation'].resample('D').apply(lambda x: (x > 0.1).sum())

        # Koreksi Unit Solar Radiation: W/m² (Rata-rata per jam) -> Sum per hari
        # API hourly returns "instantaneous" or averaged W/m² for that hour.
        # Summing them gives "Watt-hours per m²".
        # Standard Open-Meteo Daily is MJ/m².
        # 1 Wh = 3600 Joules. 1 MJ = 1,000,000 Joules.
        # Conversion: Sum(W/m²) * 3600 / 1,000,000 = MJ/m²
        # Apply conversion to the 'sum' result in the df_daily if needed, OR recalculate here.
        # df_daily aggregates 'shortwave_radiation' -> 'sum'
        # We'll fix it in the df_final assignment or inplace.
        
        # 3. Flatten & Formatting Columns
        # Gabungkan semua hasil
        df_final = df_daily.copy()
        df_final.columns = [f"{col[0]}_{col[1]}" for col in df_final.columns] # Flatten MultiIndex
        
        # Tambahkan kolom custom yang dihitung terpisah
        df_final["wind_direction_10m_dominant"] = wind_dir
        df_final["precipitation_hours"] = precip_hours
        
        # Rename date index to tanggal (as per feature/final_feature.csv)
        df_final.index.name = "tanggal"
        # Ensure index is datetime and format to YYYY-MM-DD string
        df_final.index = df_final.index.strftime('%Y-%m-%d')
        
        
        # Fix Unit Solar Radiation (W/m² sum -> MJ/m²)
        if "shortwave_radiation_sum" in df_final.columns:
            df_final["shortwave_radiation_sum"] = df_final["shortwave_radiation_sum"] * 3600 / 1_000_000

        # Create duplicate wind direction column to match target dataset
        # In final_feature.csv both 'wind_direction_10m_dominant' and 'winddirection_10m_dominant' exist
        df_final["winddirection_10m_dominant"] = df_final["wind_direction_10m_dominant"]

        # Rename columns to match feature/final_feature.csv (No units in names)
        # We ensure names are clean. 
        # (The default aggregation produced 'col_stat', e.g., 'temperature_2m_max'. This is already correct for most.)
        # We just need to make sure we don't add units like we did in the previous step.
        
        # Explicit mapping to ensure safety (even if identity)
        column_mapping = {
            "temperature_2m_max": "temperature_2m_max",
            "temperature_2m_min": "temperature_2m_min",
            "temperature_2m_mean": "temperature_2m_mean",
            "precipitation_sum": "precipitation_sum",
            "precipitation_hours": "precipitation_hours",
            "wind_speed_10m_max": "wind_speed_10m_max",
            "wind_speed_10m_mean": "wind_speed_10m_mean",
            "wind_speed_10m_min": "wind_speed_10m_min",
            "wind_direction_10m_dominant": "wind_direction_10m_dominant",
            "winddirection_10m_dominant": "winddirection_10m_dominant",
            "wind_gusts_10m_max": "wind_gusts_10m_max",
            "wind_gusts_10m_mean": "wind_gusts_10m_mean",
            "wind_gusts_10m_min": "wind_gusts_10m_min",
            "shortwave_radiation_sum": "shortwave_radiation_sum",
            "relative_humidity_2m_mean": "relative_humidity_2m_mean",
            "relative_humidity_2m_max": "relative_humidity_2m_max",
            "relative_humidity_2m_min": "relative_humidity_2m_min",
            "cloud_cover_mean": "cloud_cover_mean",
            "cloud_cover_max": "cloud_cover_max",
            "cloud_cover_min": "cloud_cover_min",
            "surface_pressure_mean": "surface_pressure_mean",
            "surface_pressure_max": "surface_pressure_max",
            "surface_pressure_min": "surface_pressure_min"
        }
        
        df_final.rename(columns=column_mapping, inplace=True)

        # Reorder columns to match feature/final_feature.csv weather columns order
        target_order = [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_hours",
            "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "temperature_2m_mean", "relative_humidity_2m_mean", "cloud_cover_mean", "surface_pressure_mean",
            "wind_gusts_10m_max", "winddirection_10m_dominant", "relative_humidity_2m_max", "relative_humidity_2m_min",
            "cloud_cover_max", "cloud_cover_min", "wind_gusts_10m_mean", "wind_speed_10m_mean",
            "wind_gusts_10m_min", "wind_speed_10m_min", "surface_pressure_max", "surface_pressure_min"
        ]
        
        # Select columns that exist
        cols_to_use = [c for c in target_order if c in df_final.columns]
        df_final = df_final[cols_to_use]

        # 4. Save to CSV
        filename = f"weather_{name.lower()}_sep_nov_2025.csv"
        filepath = os.path.join(FOLDER_NAME, filename)
        df_final.to_csv(filepath)
        print(f"✅ Saved: {filepath} ({df_final.shape[0]} days)")

    except Exception as e:
        print(f"❌ Error {name}: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Buat folder jika belum ada
    os.makedirs(FOLDER_NAME, exist_ok=True)
    
    print("🚀 Memulai pengambilan data dari Historical Forecast API... (Forecasting, bukan Real)")
    for loc_name, loc_coords in LOCATIONS.items():
        get_weather_data(loc_name, loc_coords)
    print("🎉 Selesai!")