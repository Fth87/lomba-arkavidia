
import pandas as pd
import numpy as np
import warnings
import torch
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.models import LightGBMModel
from ispu_calculator import calculate_ispu_for_dataframe, map_to_3_categories

def create_series_per_station(df, value_cols, station):
    df_st = df[df['stasiun'] == station].copy()
    df_st = df_st.sort_values('tanggal')
    
    # Handle duplicate dates by taking mean
    df_st = df_st.groupby('tanggal')[value_cols].mean()
    
    # Resample to daily frequency and fill missing values
    df_st = df_st.asfreq('D')
    df_st = df_st.ffill().bfill()
    return TimeSeries.from_dataframe(df_st, fill_missing_dates=True, freq='D')

def main():
    print("="*60)
    print("Running Annual Forecasting Pipeline with Fixed ISPU Calculator")
    print("="*60)

    # 1. Setup
    POLLUTANTS = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    WEATHER_FEATURES = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'precipitation_sum', 'wind_speed_10m_max', 'wind_speed_10m_mean',
        'relative_humidity_2m_mean', 'cloud_cover_mean', 'surface_pressure_mean'
    ]
    STATIONS = ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']
    
    # Hardware check
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        ACCELERATOR = 'gpu'
    else:
        print("⚠️ GPU not available, using CPU")
        ACCELERATOR = 'cpu'

    # 2. Load Data
    print("\nLoading data...")
    try:
        df_train = pd.read_csv('../feature/final_feature.csv', parse_dates=['tanggal'])
        df_forecast = pd.read_csv('../feature/forecast_features_sep_nov_2025.csv', parse_dates=['tanggal'])
        
        df_train = df_train[df_train['stasiun'].isin(STATIONS)].copy()
        df_train = df_train.dropna(subset=POLLUTANTS)
        print(f"Train data shape: {df_train.shape}")
    except FileNotFoundError:
        # Try absolute paths if relative fail
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df_train = pd.read_csv(os.path.join(base_dir, 'feature/final_feature.csv'), parse_dates=['tanggal'])
        df_forecast = pd.read_csv(os.path.join(base_dir, 'feature/forecast_features_sep_nov_2025.csv'), parse_dates=['tanggal'])
        print(f"Train data shape (loaded with abs path): {df_train.shape}")

    # 3. Create TimeSeries
    print("\nCreating Series...")
    target_series = {}
    cov_series = {}
    future_cov_series = {}
    
    for st in STATIONS:
        target_series[st] = create_series_per_station(df_train, POLLUTANTS, st)
        cov_series[st] = create_series_per_station(df_train, WEATHER_FEATURES, st)
        future_cov_series[st] = create_series_per_station(df_forecast, WEATHER_FEATURES, st)

    # 4. Prepare Training Data
    INPUT_CHUNK = 21
    OUTPUT_CHUNK = 14
    FORECAST_HORIZON = 91
    
    full_train_list = [target_series[st] for st in STATIONS]
    full_cov_list = []
    for st in STATIONS:
        combined = cov_series[st].append(future_cov_series[st])
        full_cov_list.append(combined)

    # 5. Train Model (LightGBM)
    print("\nTraining LightGBM Model (Best Model)...")
    # Note: LightGBM in Darts doesn't use 'accelerator' arg directly like PyTorch models, 
    # but we keep the config consistent with notebook
    model = LightGBMModel(
        lags=INPUT_CHUNK,
        lags_future_covariates=(INPUT_CHUNK, OUTPUT_CHUNK),
        output_chunk_length=OUTPUT_CHUNK,
        verbose=-1
    )
    
    model.fit(series=full_train_list, future_covariates=full_cov_list)
    
    # 6. Predict
    print("\nForecasting...")
    final_preds = model.predict(n=FORECAST_HORIZON, series=full_train_list, future_covariates=full_cov_list)

    # 7. Generate Submission with Fixed ISPU
    print(f"\n{'='*60}")
    print(f"Generating Submission with REVERTED ISPU Calculator (Dataset-Compatible)")
    print(f"{'='*60}\n")

    submissions = []
    pollutant_predictions = []

    for i, st in enumerate(STATIONS):
        # Convert TimeSeries to DataFrame (Robust method)
        # pred_df = final_preds[i].pd_dataframe() # Failed
        pred_df = pd.DataFrame(
            final_preds[i].values(), 
            index=final_preds[i].time_index,
            columns=final_preds[i].components
        )
        
        # Ensure non-negative values
        pred_df = pred_df.clip(lower=0)
        
        # Save raw pollutant predictions
        pollutant_pred_df = pred_df.copy()
        pollutant_pred_df['stasiun'] = st
        pollutant_pred_df['tanggal'] = pollutant_pred_df.index
        pollutant_predictions.append(pollutant_pred_df)
        
        # ✅ Use REVERTED ISPU calculation (dataset standard)
        pred_df = calculate_ispu_for_dataframe(
            pred_df, 
            pollutant_cols=POLLUTANTS
        )
        
        # Map to 3 categories
        pred_df['category_3class'] = pred_df['category'].apply(map_to_3_categories)
        
        # Create submission ID
        pred_df['id'] = pred_df.index.strftime('%Y-%m-%d') + '_' + st
        
        submissions.append(pred_df[['id', 'category', 'category_3class', 'max_ispu', 'critical_parameter']])

    submission_df = pd.concat(submissions, ignore_index=True)

    # Show category distribution
    print("\nCategory Distribution (3-Class):")
    print(submission_df['category_3class'].value_counts())
    print("\nCritical Parameter Distribution:")
    print(submission_df['critical_parameter'].value_counts())

    # Save files
    submission_df[['id', 'category']].to_csv('submission_darts_5class.csv', index=False)
    submission_df[['id', 'category_3class']].rename(columns={'category_3class': 'category'}).to_csv('submission_darts_3class.csv', index=False)

    pollutant_predictions_df = pd.concat(pollutant_predictions, ignore_index=True)
    pollutant_predictions_df = pollutant_predictions_df[['tanggal', 'stasiun'] + POLLUTANTS]
    pollutant_predictions_df.to_csv('predictions_pollutants.csv', index=False)

    print(f"\nSubmissions saved:")
    print(f"  - submission_darts_5class.csv")
    print(f"  - submission_darts_3class.csv")
    print(f"  - predictions_pollutants.csv")
    
    print("\nPreview:")
    print(submission_df.head())

if __name__ == "__main__":
    main()
