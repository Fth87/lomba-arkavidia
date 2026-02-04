import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Import ISPU calculator
sys.path.insert(0, str(Path().resolve().parent))
try:
    from forecasting.ispu_calculator import calculate_ispu, get_ispu_category
except ImportError:
    # Fallback if running from root
    sys.path.insert(0, str(Path().resolve()))
    from forecasting.ispu_calculator import calculate_ispu, get_ispu_category

# Config
INPUT_FILE = 'dataset_processing/dataset/merged_data_v4_complete.csv'
OUTPUT_FILE = 'dataset/final_data_full_imputed_v3.csv'
RANDOM_STATE = 42
TARGET_COL = 'pm25'

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle datetime
    if 'tanggal' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['tanggal'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    df = df.sort_values(by=['stasiun', 'datetime']).reset_index(drop=True)
    print(f"Shape: {df.shape}")
    print(f"Missing PM2.5: {df[TARGET_COL].isna().sum()}")
    return df

def clean_weather_data(df):
    print("Cleaning weather data (interpolation)...")
    weather_cols = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
        'wind_speed_10m_max', 'relative_humidity_2m_mean', 'surface_pressure_mean',
        'temperature_2m_mean', 'wind_speed_10m_mean'
    ]
    present_cols = [c for c in weather_cols if c in df.columns]
    
    # Apply strictly numeric fill to preserve 'stasiun' column if using transform
    for col in present_cols:
        df[col] = df.groupby('stasiun')[col].transform(lambda x: x.interpolate(method='linear', limit=6).bfill().ffill())
        
    return df

def create_time_features(df):
    df = df.copy()
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclical
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    return df

def create_lag_features(df, weather_cols, lags=[1, 3, 24]):
    df = df.sort_values(['stasiun', 'datetime'])
    for col in weather_cols:
        if col not in df.columns: continue
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('stasiun')[col].shift(lag)
    return df

def create_rolling_features(df, weather_cols, windows=[24]):
    df = df.sort_values(['stasiun', 'datetime'])
    for col in weather_cols:
        if col not in df.columns: continue
        for window in windows:
            df[f'{col}_roll_mean_{window}h'] = df.groupby('stasiun')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'{col}_roll_std_{window}h'] = df.groupby('stasiun')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
    return df

def create_interactions(df):
    if 'temperature_2m_mean' in df.columns and 'relative_humidity_2m_mean' in df.columns:
        df['temp_humidity'] = df['temperature_2m_mean'] * df['relative_humidity_2m_mean']
    if 'wind_speed_10m_mean' in df.columns and 'precipitation_sum' in df.columns:
        df['wind_rain'] = df['wind_speed_10m_mean'] * (1 + df['precipitation_sum'])
    return df

def feature_engineering(df):
    print("Engineering Features...")
    df = create_time_features(df)
    
    weather_cols = ['temperature_2m_mean', 'wind_speed_10m_mean', 'precipitation_sum', 'relative_humidity_2m_mean', 'surface_pressure_mean']
    cols = [c for c in weather_cols if c in df.columns]
    
    df = create_lag_features(df, cols, lags=[1, 3, 24])
    df = create_rolling_features(df, cols, windows=[24])
    df = create_interactions(df)
    
    if 'stasiun' in df.columns:
        le = LabelEncoder()
        df['stasiun_encoded'] = le.fit_transform(df['stasiun'])
    
    # Fill initial NaNs from lags
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Safe fill per station
    if 'stasiun' in df.columns:
        df[num_cols] = df.groupby('stasiun')[num_cols].transform(lambda x: x.bfill().ffill())
    else:
        df[num_cols] = df[num_cols].bfill().ffill()
    
    return df

def spatial_imputation(df):
    print("Running Spatial Imputation...")
    # remove duplicates
    df_dedup = df.drop_duplicates(subset=['datetime', 'stasiun'])
    
    pivot = df_dedup.pivot(index='datetime', columns='stasiun', values=TARGET_COL)
    
    # Sklearn requirement: string feature names
    pivot.columns = pivot.columns.astype(str)
    
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=RANDOM_STATE, keep_empty_features=True)
    imputed = imputer.fit_transform(pivot)
    
    pivot_imp = pd.DataFrame(imputed, index=pivot.index, columns=pivot.columns)
    imputed_long = pivot_imp.reset_index().melt(id_vars='datetime', var_name='stasiun', value_name='pm25_spatial')
    
    df = pd.merge(df, imputed_long, on=['datetime', 'stasiun'], how='left')
    df.loc[df['pm25_spatial'] < 0, 'pm25_spatial'] = 0
    
    # Primary fill
    df['pm25_filled_l1'] = df[TARGET_COL].fillna(df['pm25_spatial'])
    print(f"Missing after Spatial: {df['pm25_filled_l1'].isna().sum()}")
    return df

def model_imputation(df):
    missing_mask = df['pm25_filled_l1'].isna()
    if missing_mask.sum() == 0:
        print("No missing values. Skipping Layer 2.")
        df['pm25_final'] = df['pm25_filled_l1']
        return df
    
    print(f"Running LightGBM for {missing_mask.sum()} missing values...")
    
    # Select features
    exclude = ['datetime', 'stasiun', 'pm25', 'pm25_spatial', 'pm25_filled_l1', 'pm25_final', 'tanggal', 'ispu', 'target_kategori', 'categori_pm25', 'ispu_pm25_recalc']
    # Ensure features are numeric and exist
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # LightGBM handles NaNs, so we don't strictly need to drop them
    # but strictly speaking we should have cleaned them. 
    # If standard features like 'temperature' have NaNs, we should fix that,
    # but for now let's allow them so we don't end up with 0 features.
    
    print(f"Using {len(features)} features.")
    
    train_df = df[df['pm25_filled_l1'].notna()].copy()
    predict_df = df[df['pm25_filled_l1'].isna()].copy()
    
    X = train_df[features]
    y = train_df['pm25_filled_l1']
    
    # Optuna
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    print(f"Best params: {study.best_params}")
    
    # Final Train
    final_model = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE)
    final_model.fit(X, y)
    
    # Predict
    preds = final_model.predict(predict_df[features])
    df.loc[missing_mask, 'pm25_filled_l1'] = preds
    df['pm25_final'] = df['pm25_filled_l1']
    return df

def run():
    df = load_data(INPUT_FILE)
    df = clean_weather_data(df)
    df = feature_engineering(df)
    df = spatial_imputation(df)
    df = model_imputation(df)
    
    print("Recalculating ISPU...")
    df['ispu'] = df['pm25_final'].apply(lambda x: calculate_ispu('pm25', x))
    df['target_kategori'] = df['ispu'].apply(get_ispu_category)
    
    df_save = df.copy()
    df_save[TARGET_COL] = df_save['pm25_final']
    
    # Keep original schema mostly, remove temp cols
    drop_cols = ['pm25_spatial', 'pm25_filled_l1', 'pm25_final', 'stasiun_encoded']
    df_save = df_save.drop(columns=[c for c in drop_cols if c in df_save.columns])
    
    print(f"Saving to {OUTPUT_FILE}... Shape: {df_save.shape}")
    df_save.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    run()
