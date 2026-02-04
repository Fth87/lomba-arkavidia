"""
Build Complete Feature Set for Sep-Nov 2025 Predictions
Opsi 2: Weather (real) + Calendar (real) + Historical Avg Pollutants
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===========================
# 1. LOAD EXISTING DATA
# ===========================

print("="*80)
print("🔧 BUILDING FORECAST FEATURES (Sep-Nov 2025)")
print("="*80)

# Load historical data for reference
print("\n1️⃣  Loading historical data...")
df_hist = pd.read_csv('../feature/final_feature.csv')
df_hist['tanggal'] = pd.to_datetime(df_hist['tanggal'])
print(f"   Historical data: {df_hist.shape}")

# Load weather data (5 stasiun)
print("\n2️⃣  Loading weather data...")
weather_dfs = []
for station in ['dki1', 'dki2', 'dki3', 'dki4', 'dki5']:
    station_name = station.upper()
    df_weather = pd.read_csv(f'../open-mateo/weather_{station}_sep_nov_2025.csv')
    df_weather['stasiun'] = station_name
    df_weather['tanggal'] = pd.to_datetime(df_weather['tanggal'])
    weather_dfs.append(df_weather)
    print(f"   {station_name}: {len(df_weather)} days")

df_weather_all = pd.concat(weather_dfs, ignore_index=True)
print(f"   Total weather data: {df_weather_all.shape}")

# Load calendar data
print("\n3️⃣  Loading calendar data...")
df_calendar = pd.read_csv('../dataset/libur-nasional/dataset-libur-nasional-dan-weekend.csv')
df_calendar['tanggal'] = pd.to_datetime(df_calendar['tanggal'])
df_calendar = df_calendar[(df_calendar['tanggal'] >= '2025-09-01') & 
                          (df_calendar['tanggal'] <= '2025-11-30')]
print(f"   Calendar data: {len(df_calendar)} days")

# ===========================
# 2. MERGE WEATHER + CALENDAR
# ===========================

print("\n4️⃣  Merging weather + calendar...")
df_base = df_weather_all.merge(df_calendar, on='tanggal', how='left')
print(f"   Merged data: {df_base.shape}")

# ===========================
# 3. CALCULATE ENGINEERED FEATURES
# ===========================

print("\n5️⃣  Calculating engineered features...")

# Rainy day
df_base['rainy_day'] = (df_base['precipitation_sum'] > 0).astype(int)

# Washout effect
df_base['washout_effect'] = df_base['precipitation_sum'] * df_base['wind_speed_10m_max']

# Ventilation proxy
df_base['ventilation_proxy'] = (df_base['wind_speed_10m_max'] * 
                                 df_base['surface_pressure_mean'] / 
                                 df_base['relative_humidity_2m_mean'])

print(f"   Added: rainy_day, washout_effect, ventilation_proxy")

# ===========================
# 4. TEMPORAL FEATURES
# ===========================

print("\n6️⃣  Generating temporal features...")

df_base['dayofweek'] = df_base['tanggal'].dt.dayofweek
df_base['quarter'] = df_base['tanggal'].dt.quarter
df_base['month_feat'] = df_base['tanggal'].dt.month
df_base['year_feat'] = df_base['tanggal'].dt.year
df_base['dayofyear'] = df_base['tanggal'].dt.dayofyear
df_base['month'] = df_base['tanggal'].dt.month
df_base['year'] = df_base['tanggal'].dt.year
df_base['day_of_week'] = df_base['tanggal'].dt.dayofweek

# is_working_day
df_base['is_working_day'] = ((df_base['is_weekend'] == 0) & 
                             (df_base['is_holiday_nasional'] == 0)).astype(int)

print(f"   Added temporal features")

# ===========================
# 5. HISTORICAL AVG POLLUTANTS
# ===========================

print("\n7️⃣  Calculating historical avg pollutants (2010-2024, Sep-Nov)...")

# Filter historical data for Sep-Nov only
df_hist_sep_nov = df_hist[df_hist['tanggal'].dt.month.isin([9, 10, 11])].copy()
df_hist_sep_nov = df_hist_sep_nov[df_hist_sep_nov['tanggal'].dt.year < 2025]

pollutant_cols = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']

# Calculate average per station and month
pollutant_avg = df_hist_sep_nov.groupby(['stasiun', df_hist_sep_nov['tanggal'].dt.month])[pollutant_cols].mean().reset_index()
pollutant_avg.rename(columns={'tanggal': 'month'}, inplace=True)

print(f"   Historical averages calculated:")
for station in ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']:
    station_avg = pollutant_avg[pollutant_avg['stasiun'] == station]
    if len(station_avg) > 0:
        pm10_avg = station_avg['pm10'].mean()
        print(f"   {station}: PM10 avg = {pm10_avg:.1f}")

# Merge pollutant averages
df_base = df_base.merge(pollutant_avg, on=['stasiun', 'month'], how='left')

# If any missing, fill with overall average
for col in pollutant_cols:
    if df_base[col].isnull().sum() > 0:
        overall_avg = df_hist_sep_nov[col].mean()
        df_base[col].fillna(overall_avg, inplace=True)

print(f"   Pollutants added: {', '.join(pollutant_cols)}")

# ===========================
# 6. LAG FEATURES
# ===========================

print("\n8️⃣  Generating lag features...")

# For each station, generate lags
for station in ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']:
    station_mask = df_base['stasiun'] == station
    
    # Get last known pm10 from historical data (Aug 31, 2025)
    last_historical = df_hist[(df_hist['stasiun'] == station) & 
                              (df_hist['tanggal'] == '2025-08-31')]
    
    if len(last_historical) > 0:
        last_pm10 = last_historical['pm10'].iloc[0]
    else:
        # Fallback to historical average
        last_pm10 = df_hist[df_hist['stasiun'] == station]['pm10'].mean()
    
    # For forecast, use historical average as baseline for lags
    station_avg_pm10 = df_base.loc[station_mask, 'pm10'].iloc[0] if station_mask.sum() > 0 else 50
    
    df_base.loc[station_mask, 'lag_1'] = station_avg_pm10
    df_base.loc[station_mask, 'lag_7'] = station_avg_pm10
    df_base.loc[station_mask, 'lag_30'] = station_avg_pm10
    df_base.loc[station_mask, 'rolling_mean_7'] = station_avg_pm10
    df_base.loc[station_mask, 'rolling_std_7'] = df_hist[df_hist['stasiun'] == station]['pm10'].std()
    df_base.loc[station_mask, 'rolling_mean_30'] = station_avg_pm10

print(f"   Lag features added: lag_1, lag_7, lag_30, rolling_mean_7, rolling_std_7, rolling_mean_30")

# ===========================
# 7. STATIC FEATURES
# ===========================

print("\n9️⃣  Adding static features...")

# Get static features from historical data
static_features = df_hist.groupby('stasiun')[['Total_Penduduk', 'ndvi']].first().reset_index()

df_base = df_base.merge(static_features, on='stasiun', how='left')

print(f"   Static features added: Total_Penduduk, ndvi")

# ===========================
# 8. FINAL CLEANUP
# ===========================

print("\n🔟 Final cleanup...")

# Ensure all required columns exist
required_cols = [
    'tanggal', 'stasiun',
    # Pollutants
    'pm10', 'pm25', 'so2', 'co', 'o3', 'no2',
    # Weather
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'precipitation_sum', 'precipitation_hours',
    'wind_speed_10m_max', 'wind_speed_10m_mean', 'wind_speed_10m_min',
    'wind_direction_10m_dominant',
    'wind_gusts_10m_max', 'wind_gusts_10m_mean', 'wind_gusts_10m_min',
    'shortwave_radiation_sum',
    'relative_humidity_2m_mean', 'relative_humidity_2m_max', 'relative_humidity_2m_min',
    'cloud_cover_mean', 'cloud_cover_max', 'cloud_cover_min',
    'surface_pressure_mean', 'surface_pressure_max', 'surface_pressure_min',
    'washout_effect', 'ventilation_proxy', 'rainy_day',
    # Calendar
    'is_holiday_nasional', 'is_weekend', 'is_working_day', 'day_name',
    # Temporal
    'dayofweek', 'quarter', 'month_feat', 'year_feat', 'dayofyear',
    # Lags
    'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30',
    # Static
    'Total_Penduduk', 'ndvi'
]

# Check missing columns
missing_cols = [col for col in required_cols if col not in df_base.columns]
if missing_cols:
    print(f"   ⚠️  Missing columns: {missing_cols}")
else:
    print(f"   ✅ All required columns present")

# Add nama_libur if missing (for compatibility)
if 'nama_libur' not in df_base.columns:
    df_base['nama_libur'] = df_base['is_holiday_nasional'].apply(lambda x: 'Holiday' if x == 1 else 'Hari Kerja')

# Reorder columns to match historical data
df_base = df_base[required_cols + ['nama_libur']]

# Fill any remaining NaNs
for col in df_base.columns:
    if df_base[col].dtype in ['float64', 'int64']:
        if df_base[col].isnull().sum() > 0:
            df_base[col].fillna(df_base[col].median(), inplace=True)

# ===========================
# 9. SAVE
# ===========================

print("\n💾 Saving forecast features...")

output_file = '../feature/forecast_features_sep_nov_2025.csv'
df_base.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"✅ SUCCESS!")
print(f"{'='*80}")
print(f"Forecast features saved: {output_file}")
print(f"Shape: {df_base.shape}")
print(f"Date range: {df_base['tanggal'].min().date()} to {df_base['tanggal'].max().date()}")
print(f"Stations: {sorted(df_base['stasiun'].unique())}")
print(f"\nSample data:")
print(df_base.head())
print(f"\nReady for XGBoost prediction! 🚀")
