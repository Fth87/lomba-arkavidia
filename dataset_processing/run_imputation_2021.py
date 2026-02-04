import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# Config
INPUT_FILE = 'feature/final_feature_2021.csv'
OUTPUT_FILE = 'dataset/final_data_2021_imputed.csv'
RANDOM_STATE = 42
TARGET_COL = 'pm25'

def load_and_filter_data(filepath):
    """Load data and filter for Year >= 2021"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle datetime
    if 'tanggal' in df.columns:
        df['datetime'] = pd.to_datetime(df['tanggal'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("No datetime column found")
        
    df['year'] = df['datetime'].dt.year
    
    # Filter 2021+
    df_filtered = df[df['year'] >= 2021].copy()
    print(f"Original shape: {df.shape}")
    print(f"Filtered 2021+ shape: {df_filtered.shape}")
    
    # Drop rows without station (if any)
    df_filtered = df_filtered.dropna(subset=['stasiun'])
    
    return df_filtered.sort_values(by=['stasiun', 'datetime']).reset_index(drop=True)

def clean_weather_data(df):
    """Interpolate small gaps in weather data"""
    print("Cleaning weather data...")
    weather_cols = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
        'wind_speed_10m_max', 'relative_humidity_2m_mean', 'surface_pressure_mean'
    ]
    
    # Check present columns
    present_cols = [c for c in weather_cols if c in df.columns]
    
    def interpolate_group(group):
        group[present_cols] = group[present_cols].interpolate(method='time', limit=3)
        return group.fillna(method='bfill').fillna(method='ffill')
    
    try:
        # Ensure proper index handling
        if 'stasiun' in df.columns and 'datetime' in df.columns:
            # Set multi-index for proper groupby apply return structure
            df = df.set_index(['stasiun', 'datetime'])
            
        # Apply interpolation
        # Groupby on level 0 (stasiun)
        df = df.groupby(level=0).apply(interpolate_group)
        
        # Reset index to get both stasiun and datetime back as columns
        df = df.reset_index()
        
    except Exception as e:
        print(f"Warning in weather cleaning: {e}")
        # Fallback if complex indexing fails: ensure we have columns back
        df = df.reset_index() # Try to recover whatever is in index
        # Fallback simple fill on original columns if needed (though df might be messed up here, usually robust enough)
        
    return df

# --- Enhanced Feature Engineering (Aligned with forecast_features_enhanced.py) ---

def create_time_features(df):
    """Create comprehensive time-based features with cyclical encoding."""
    df = df.copy()
    
    # Basic time components
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['quarter'] = df['datetime'].dt.quarter
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Categorical time features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Dry season in Indonesia roughly May-Oct
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    
    return df

def create_lag_features(df, weather_cols, lags=[1, 3, 24]):
    """Create lag features (Shifted within station)."""
    df = df.copy()
    # Ensure sorted order
    df = df.sort_values(['stasiun', 'datetime'])
    
    for col in weather_cols:
        if col not in df.columns: continue
        for lag in lags:
            # Shift within each station group
            df[f'{col}_lag{lag}'] = df.groupby('stasiun')[col].shift(lag)
            
    return df

def create_rolling_features(df, weather_cols, windows=[24]):
    """Create rolling window features."""
    df = df.copy()
    df = df.sort_values(['stasiun', 'datetime'])
    
    for col in weather_cols:
        if col not in df.columns: continue
        for window in windows:
            # Rolling mean/std within station
            df[f'{col}_roll_mean_{window}h'] = df.groupby('stasiun')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'{col}_roll_std_{window}h'] = df.groupby('stasiun')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
    return df

def create_weather_interactions(df):
    """Create weather interaction features."""
    df = df.copy()
    
    # Temperature-Humidity interaction
    if 'temperature_2m_mean' in df.columns and 'relative_humidity_2m_mean' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_2m_mean'] * df['relative_humidity_2m_mean']
    
    # Wind-Rain interaction
    if 'wind_speed_10m_mean' in df.columns and 'precipitation_sum' in df.columns:
        df['wind_rain_interaction'] = df['wind_speed_10m_mean'] * (1 + df['precipitation_sum'])
    
    # Pressure anomaly (by station)
    if 'surface_pressure_mean' in df.columns:
        df['pressure_anomaly'] = df.groupby('stasiun')['surface_pressure_mean'].transform(
            lambda x: x - x.mean()
        )
    
    # Rainy day indicator
    if 'precipitation_sum' in df.columns:
        df['rainy_day'] = (df['precipitation_sum'] > 0).astype(int)
    
    return df

def feature_engineering(df):
    """Create features for Layer 2 Imputation with Enhanced Logic"""
    print("Engineering Features with Enhanced Logic...")
    df = df.copy()
    
    # 1. Time Features
    df = create_time_features(df)
    
    # 2. Weather Lags & Rolling
    # We depend on cleaner weather data for this
    weather_cols = ['temperature_2m_mean', 'wind_speed_10m_mean', 'precipitation_sum', 'relative_humidity_2m_mean', 'surface_pressure_mean']
    # Filter only available columns
    avail_weather_cols = [c for c in weather_cols if c in df.columns]
    
    print(f"Creating lags and rolling windows for: {avail_weather_cols}")
    df = create_lag_features(df, avail_weather_cols, lags=[1, 3, 24])
    df = create_rolling_features(df, avail_weather_cols, windows=[24])
    
    # 3. Interactions
    df = create_weather_interactions(df)
    
    # 4. Station Encoding
    le = LabelEncoder()
    df['stasiun_encoded'] = le.fit_transform(df['stasiun'])
    
    # 5. Fallback Fill (for initial lags created by shifting)
    # Important for imputation: we don't want to drop rows, so backfill initial missing
    # We apply fill only to relevant columns to avoid losing 'stasiun' or other metadata
    cols_to_fill = [c for c in df.columns if c not in ['stasiun', 'datetime', 'tanggal']]
    
    # Define a fill function
    def fill_group(group):
        return group.bfill().ffill()
        
    # Apply to grouped object and assign back to ensure alignment
    # Note: simple bfill() on the whole df might bleed across stations if not careful,
    # but since we sorted by station, it's safer. However, explicit group fill is best.
    
    # Safe approach: fill numeric cols mostly 
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df.groupby('stasiun')[num_cols].transform(lambda x: x.bfill().ffill())
    
    # Double check for object cols if any (excluding stasiun)
    # For this dataset, most features are numeric. stasiun is object.
    
    
    return df, le

def spatial_imputation(df):
    """Layer 1: Spatial Imputation using IterativeImputer"""
    print("Running Layer 1: Spatial Imputation...")
    
    # Pivot to Wide Format
    # Drop duplicates if any
    df_dedup = df.drop_duplicates(subset=['datetime', 'stasiun'])
    
    pivot_target = df_dedup.pivot(index='datetime', columns='stasiun', values=TARGET_COL)
    
    # Impute
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=RANDOM_STATE)
    imputed_matrix = imputer.fit_transform(pivot_target)
    
    pivot_imputed = pd.DataFrame(
        imputed_matrix, 
        index=pivot_target.index, 
        columns=pivot_target.columns
    )
    
    # Melt back to long format
    imputed_long = pivot_imputed.reset_index().melt(
        id_vars='datetime', 
        var_name='stasiun', 
        value_name='pm25_spatial'
    )
    
    # Merge back
    df = pd.merge(df, imputed_long, on=['datetime', 'stasiun'], how='left')
    
    # Count how many filled
    filled_count = df[df[TARGET_COL].isna() & df['pm25_spatial'].notna()].shape[0]
    print(f"Layer 1 filled {filled_count} values via Spatial correlation")
    
    # Mask negative values from spatial imputation
    df.loc[df['pm25_spatial'] < 0, 'pm25_spatial'] = 0
    
    return df

def model_imputation(df):
    """Layer 2: LightGBM for remaining gaps"""
    print("Running Layer 2: LightGBM Imputation...")
    
    # Define Source for Training
    # We train on: Rows where Original PM2.5 is present OR Spatially Imputed is present
    # We use Spatially imputed values as ground truth for training Layer 2? 
    # Better: Train on valid Original Data, Use Weighted approach if including Spatial.
    # Given 2021+ is clean, we stick to training on Original Data for highest quality.
    
    # Current State of Target
    # We want to fill gaps in 'pm25' using 'pm25_spatial' first
    df['pm25_filled_l1'] = df[TARGET_COL].fillna(df['pm25_spatial'])
    
    # Identify what's still missing
    missing_mask = df['pm25_filled_l1'].isna()
    if missing_mask.sum() == 0:
        print("No missing values left after Spatial Imputation. Skipping Layer 2.")
        df['pm25_final'] = df['pm25_filled_l1']
        return df
    
    print(f"Remaining missing after Layer 1: {missing_mask.sum()}")
    
    # Features for Model
    features = [
        'stasiun_encoded', 
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
        'wind_speed_10m_max', 'relative_humidity_2m_mean', 'surface_pressure_mean'
    ]
    # Add other pollutants if available and not missing? 
    # For now sticking to weather + time to avoid circular dependency issues in simple script
    valid_features = [f for f in features if f in df.columns]
    
    # Train/Test Split
    train_df = df[df['pm25_filled_l1'].notna()].copy()
    train_df = train_df.dropna(subset=valid_features) # Ensure features are clean
    
    predict_df = df[df['pm25_filled_l1'].isna()].copy()
    
    if len(train_df) == 0:
        print("Error: No training data available for Layer 2")
        return df
        
    X = train_df[valid_features]
    y = train_df['pm25_filled_l1']
    
    # Optuna Optimization (Fast)
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
        }
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, show_progress_bar=False) # Quick optimization
    
    print(f"Best params: {study.best_params}")
    
    # Retrain final model
    final_model = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE)
    final_model.fit(X, y)
    
    # Predict
    if len(predict_df) > 0:
        # Interpolate features if still missing (last resort)
        X_pred = predict_df[valid_features].fillna(method='ffill').fillna(method='bfill')
        preds = final_model.predict(X_pred)
        
        # Assign
        df.loc[missing_mask, 'pm25_filled_l1'] = preds
    
    df['pm25_final'] = df['pm25_filled_l1']
    return df

def run():
    # 1. Load
    df = load_and_filter_data(INPUT_FILE)
    
    # 2. Clean Weather
    df = clean_weather_data(df)
    
    # 3. Feature Eng
    df, le = feature_engineering(df)
    
    # 4. Spatial Imputation
    df = spatial_imputation(df)
    
    # 5. Model Imputation
    df = model_imputation(df)
    
    # 6. Finalize
    # Overwrite pm25 with final values
    df[TARGET_COL] = df['pm25_final']
    
    # Drop temp cols
    cols_to_drop = ['pm25_spatial', 'pm25_filled_l1', 'stasiun_encoded', 'pm25_final', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    df_final = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Done.")
    
    # Optional Plotting
    plt.figure(figsize=(15, 6))
    subset = df[df['stasiun'] == 'DKI1 (Bunderan HI)'].sort_values('datetime')
    plt.plot(subset['datetime'], subset[TARGET_COL], label='Final Imputed', alpha=0.7)
    plt.title('PM2.5 Imputation 2021+ (DKI1)')
    plt.legend()
    plt.savefig('dataset_processing/imputation_2021_check.png')
    print("Plot saved to dataset_processing/imputation_2021_check.png")

if __name__ == "__main__":
    run()
