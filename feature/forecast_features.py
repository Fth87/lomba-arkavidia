
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
WEATHER_PATH_TEMPLATE = 'open-mateo/weather_{}_sep_nov_2025.csv'
CALENDAR_PATH = 'dataset/libur-nasional/dataset-libur-nasional-dan-weekend.csv'
HISTORICAL_PATH = 'feature/final_feature.csv'
OUTPUT_PATH = 'feature/forecast_features_sep_nov_2025.csv'

STATIONS = ['dki1', 'dki2', 'dki3', 'dki4', 'dki5']
MONTHS_INTEREST = [9, 10, 11] # Sep, Oct, Nov

# Fallback Data (if history missing)
FALLBACK_POPULATION = {
    'DKI1': 40000, 'DKI2': 45000, 'DKI3': 35000, 'DKI4': 38000, 'DKI5': 42000
}
# Fallback Monthly NDVI (Sep, Oct, Nov)
FALLBACK_NDVI = {
    'DKI1': {9: 0.20, 10: 0.21, 11: 0.23}, # Urban
    'DKI2': {9: 0.15, 10: 0.16, 11: 0.18},
    'DKI3': {9: 0.30, 10: 0.32, 11: 0.35}, # Greener
    'DKI4': {9: 0.25, 10: 0.26, 11: 0.28},
    'DKI5': {9: 0.18, 10: 0.19, 11: 0.21}
}

def resolve_path(path_str):
    """Resolves path whether running from root or subdir."""
    if os.path.exists(path_str): return path_str
    if os.path.exists(f"../{path_str}"): return f"../{path_str}"
    return None

def load_weather_data():
    """Loads and combines weather forecast data for all stations."""
    dfs = []
    print("   Loading weather data...")
    for st in STATIONS:
        path = resolve_path(WEATHER_PATH_TEMPLATE.format(st))
        if path:
            df = pd.read_csv(path)
            df['stasiun'] = st.upper()
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError("No weather files found.")
    return pd.concat(dfs, ignore_index=True)

def load_calendar_data():
    """Loads calendar data for Sep-Nov 2025."""
    path = resolve_path(CALENDAR_PATH)
    if not path:
        print("   ⚠️ Calendar file missing. Using defaults.")
        return None
        
    df = pd.read_csv(path)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    return df[(df['tanggal'] >= '2025-09-01') & (df['tanggal'] <= '2025-11-30')]

def get_historical_ndvi_population():
    """Calculates monthly average NDVI and static Population from history."""
    path = resolve_path(HISTORICAL_PATH)
    if not path: return None, None
    
    df_hist = pd.read_csv(path, parse_dates=['tanggal'])
    
    # Static Population (constant per station)
    pop_df = df_hist.groupby('stasiun')['Total_Penduduk'].median().reset_index()
    
    # Dynamic Monthly NDVI (Avg per Station-Month)
    df_hist['month'] = df_hist['tanggal'].dt.month
    ndvi_df = df_hist[df_hist['month'].isin(MONTHS_INTEREST)].groupby(['stasiun', 'month'])['ndvi'].mean().reset_index()
    
    return pop_df, ndvi_df

def main():
    print("="*60)
    print("🚀 Building Forecast Features (Future Covariates Only)")
    print("="*60)

    # 1. Load & Validation
    df_base = load_weather_data()
    df_cal = load_calendar_data()

    # 2. Merge Calendar
    if df_cal is not None:
        df_base = df_base.merge(df_cal, on='tanggal', how='left')
    else:
        # Minimal defaults
        df_base['is_holiday_nasional'] = 0
        df_base['is_weekend'] = df_base['tanggal'].dt.dayofweek.isin([5, 6]).astype(int)

    # 3. Feature Engineering
    print("   Engineering features...")
    
    # Temporal
    df_base['dayofweek'] = df_base['tanggal'].dt.dayofweek
    df_base['month'] = df_base['tanggal'].dt.month
    df_base['dayofyear'] = df_base['tanggal'].dt.dayofyear
    df_base['quarter'] = df_base['tanggal'].dt.quarter
    
    # Weather Derived
    if {'precipitation_sum', 'wind_speed_10m_max'}.issubset(df_base.columns):
        df_base['rainy_day'] = (df_base['precipitation_sum'] > 0).astype(int)
        df_base['washout_effect'] = df_base['precipitation_sum'] * df_base['wind_speed_10m_max']
    
    if {'wind_speed_10m_max', 'surface_pressure_mean', 'relative_humidity_2m_mean'}.issubset(df_base.columns):
        df_base['ventilation_proxy'] = (
            df_base['wind_speed_10m_max'] * df_base['surface_pressure_mean'] / 
            df_base['relative_humidity_2m_mean'].replace(0, 1)
        )

    # 4. Static & Dynamic Covariates (Population & NDVI)
    print("   Integrating Static (Population) & Dynamic (NDVI) features...")
    
    pop_hist, ndvi_hist = get_historical_ndvi_population()
    
    # A. Population (Static)
    if pop_hist is not None:
        df_base = df_base.merge(pop_hist, on='stasiun', how='left')
    else:
        df_base['Total_Penduduk'] = df_base['stasiun'].map(FALLBACK_POPULATION)

    # B. NDVI (Dynamic Monthly)
    # Why? Vegetation changes seasonally. We use historical monthly averages.
    if ndvi_hist is not None:
        df_base = df_base.merge(ndvi_hist, on=['stasiun', 'month'], how='left')
    else:
        # Apply manual fallback logic
        df_base['ndvi'] = df_base.apply(
            lambda x: FALLBACK_NDVI.get(x['stasiun'], {}).get(x['month'], 0.2), axis=1
        )
    
    # Fill gaps if history didn't cover specific station-months
    if df_base['ndvi'].isnull().any():
        df_base['ndvi'] = df_base['ndvi'].fillna(
            df_base.apply(lambda x: FALLBACK_NDVI.get(x['stasiun'], {}).get(x['month'], 0.2), axis=1)
        )
    
    # 5. Final Cleanup
    df_base['nama_libur'] = df_base.get('nama_libur', 'Hari Kerja').fillna('Hari Kerja')
    
    # Select final columns (exclude pollutants)
    final_cols = [
        col for col in df_base.columns 
        if col not in ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    ]
    df_final = df_base[final_cols].copy()
    
    # Fill remaining NaNs (simple median for robustness)
    df_final = df_final.fillna(df_final.median(numeric_only=True))

    # Save
    out_path = resolve_path(OUTPUT_PATH) or OUTPUT_PATH
    df_final.to_csv(out_path, index=False)
    
    print(f"\n✅ Done! Saved to: {out_path}")
    print(f"   Shape: {df_final.shape}")
    print(f"   Columns: {len(df_final.columns)}")

if __name__ == "__main__":
    main()
