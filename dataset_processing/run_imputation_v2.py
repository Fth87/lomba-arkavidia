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

# Config
INPUT_FILE = 'dataset_processing/dataset/merged_data_v4_complete.csv'
OUTPUT_FILE = 'dataset_processing/dataset/final_dataset_imputed_v2.csv'
PLOT_FILE = 'dataset_processing/imputation_check.png'
RANDOM_STATE = 42

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# Rename tanggal to datetime for consistency if needed, or just use it
if 'tanggal' in df.columns:
    df.rename(columns={'tanggal': 'datetime'}, inplace=True)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by=['stasiun', 'datetime']).reset_index(drop=True)
df = df.drop_duplicates(subset=['stasiun', 'datetime'])

# Weather Cleaning
print("Cleaning weather data...")
weather_cols = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_mean', 'so2', 'co', 'no2', 'o3']

def clean_weather(group):
    group[weather_cols] = group[weather_cols].interpolate(method='linear', limit=6)
    group[weather_cols] = group[weather_cols].ffill().bfill()
    return group

df = df.groupby('stasiun', group_keys=False).apply(clean_weather)
print("Missing PM2.5 before:", df['pm25'].isna().sum())

# Feature Engineering
print("Engineering features...")
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

le = LabelEncoder()
df['stasiun_id'] = le.fit_transform(df['stasiun'])

df = df.sort_values(['stasiun', 'datetime'])
lag_cols = ['temperature_2m_mean', 'wind_speed_10m_mean', 'precipitation_sum']
shifts = [1, 3, 24]

for col in lag_cols:
    for s in shifts:
        df[f'{col}_lag{s}'] = df.groupby('stasiun')[col].shift(s)

for col in lag_cols:
    # rolling mean
    df[f'{col}_roll_mean_24h'] = df.groupby('stasiun')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())

df = df.fillna(method='bfill') # Fill initial lags

# Layer 1: Spatial Imputation
print("Layer 1: Spatial Imputation...")
# Ensure no duplicates before pivoting
df = df.drop_duplicates(subset=['stasiun', 'datetime'])
pivot_pm25 = df.pivot_table(index='datetime', columns='stasiun', values='pm25', aggfunc='mean')
imputer_spatial = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=RANDOM_STATE)
imputed_values = imputer_spatial.fit_transform(pivot_pm25)
pivot_imputed = pd.DataFrame(imputed_values, index=pivot_pm25.index, columns=pivot_pm25.columns)

df_spatial = pivot_imputed.reset_index().melt(id_vars='datetime', var_name='stasiun', value_name='pm25_spatial')
df_combined = df.merge(df_spatial, on=['datetime', 'stasiun'], how='left')

df_combined['source'] = 'original'
df_combined.loc[df_combined['pm25'].isna() & df_combined['pm25_spatial'].notna(), 'source'] = 'spatial_imputed'
df_combined['pm25_filled_l1'] = df_combined['pm25'].combine_first(df_combined['pm25_spatial'])
print("Missing after Spatial:", df_combined['pm25_filled_l1'].isna().sum())

# Layer 2: Weather Imputation
print("Layer 2: Weather Imputation (LightGBM)...")
features = [
    'stasiun_id', 'temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_mean',
    'so2', 'co', 'no2', 'o3', 
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 
    'temperature_2m_mean_lag1', 'wind_speed_10m_mean_lag1', 'precipitation_sum_lag1', 
    'temperature_2m_mean_roll_mean_24h'
]

train_mask = df_combined['pm25_filled_l1'].notna()
predict_mask = df_combined['pm25_filled_l1'].isna()

if predict_mask.sum() > 0:
    X_train = df_combined.loc[train_mask, features]
    y_train = df_combined.loc[train_mask, 'pm25_filled_l1']
    weights = df_combined.loc[train_mask, 'source'].map({'original': 1.0, 'spatial_imputed': 0.8})

    def objective(trial):
        param = {
            'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500), # Reduced for speed
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
        }
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, weights, test_size=0.2, random_state=RANDOM_STATE)
        model = lgb.LGBMRegressor(**param, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val], callbacks=[lgb.early_stopping(stopping_rounds=30)])
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 10 trials for speed
    
    print("Best params:", study.best_params)
    final_model = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train, sample_weight=weights)
    
    X_missing = df_combined.loc[predict_mask, features]
    df_combined.loc[predict_mask, 'pm25_filled_l1'] = final_model.predict(X_missing)
    df_combined.loc[predict_mask, 'source'] = 'model_imputed'

df_combined['pm25_final'] = df_combined['pm25_filled_l1']
print("Final Missing:", df_combined['pm25_final'].isna().sum())

# Recalculate ISPU/Category
print("Recalculating ISPU...")
def calculate_ispu_pm25(pm_val):
    if pm_val < 0: pm_val = 0
    if pm_val <= 15.5: Ia, Ib, Xa, Xb = 50, 0, 15.5, 0
    elif pm_val <= 55.4: Ia, Ib, Xa, Xb = 100, 51, 55.4, 15.6
    elif pm_val <= 150.4: Ia, Ib, Xa, Xb = 200, 101, 150.4, 55.5
    elif pm_val <= 250.4: Ia, Ib, Xa, Xb = 300, 201, 250.4, 150.5
    else: Ia, Ib, Xa, Xb = 400, 301, 500, 250.5
    return int(round(((Ia - Ib) / (Xa - Xb)) * (pm_val - Xb) + Ib))

def get_category(ispu):
    if ispu <= 50: return 'BAIK'
    elif ispu <= 100: return 'SEDANG'
    elif ispu <= 200: return 'TIDAK SEHAT'
    elif ispu <= 300: return 'SANGAT TIDAK SEHAT'
    else: return 'BERBAHAYA'

df_combined['ispu_pm25_recalc'] = df_combined['pm25_final'].apply(calculate_ispu_pm25)
df_combined['category_recalc'] = df_combined['ispu_pm25_recalc'].apply(get_category)
df_combined['pm25'] = df_combined['pm25_final']

# Save
df_save = df_combined.drop(columns=['pm25_spatial', 'pm25_filled_l1', 'pm25_final'])
df_save.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")

# Visual Check
print("Generating plot...")
plt.figure(figsize=(15, 6))
# Plot for DKI1 specifically as an example
subset = df_combined[df_combined['stasiun'] == 'DKI1 (Bunderan HI)'].sort_values('datetime')
plt.plot(subset['datetime'], subset['pm25'], label='Imputed', color='orange', alpha=0.7)
original = df[df['stasiun'] == 'DKI1 (Bunderan HI)'].sort_values('datetime')
plt.plot(original['datetime'], original['pm25'], label='Original', color='blue', alpha=0.6)
plt.title('PM2.5 Imputation Result (DKI1)')
plt.legend()
plt.savefig(PLOT_FILE)
print(f"Plot saved to {PLOT_FILE}")
