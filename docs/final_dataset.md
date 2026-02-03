# Final Dataset Documentation

**File Path**: `dataset/final_dataset_ready_for_modeling.csv`  
**Last Updated**: 2026-02-03  
**Status**: Ready for Modeling

## 1. Dataset Overview

This dataset consolidates Air Quality Index (ISPU) data, meteorological observations (ERA5 Reanalysis), and socio-economic indicators for Jakarta, Indonesia. It serves as the primary input for the Arkavidia ITB Datavidia competition (ISPU Classification).

*   **Total Rows**: 15,412
*   **Total Columns**: 47
*   **Time Period**: January 1, 2010 – August 31, 2025 (Daily resolution)
*   **Spatial Coverage**: 5 Monitoring Stations in Jakarta (DKI1 - DKI5)
*   **Missing Values**: 
    *   Pollutants: 0 missing (Imputed)
    *   Weather/Metadata: ~329 rows missing (~2%)
    *   Dates: 327 rows missing `tanggal`

## 2. Feature Data Dictionary

### A. Identification & Temporal
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `tanggal` | `datetime` | Date of observation (YYYY-MM-DD). **Note**: 327 rows are missing this value. |
| `stasiun` | `object` | Monitoring Station code (DKI1, DKI2, DKI3, DKI4, DKI5). |
| `year` | `float` | Year extracted from date. |
| `month` | `float` | Month (1-12). |
| `day_of_week` | `float` | Day of week (0=Monday, 6=Sunday). |
| `day_name` | `object` | Name of the day (e.g., 'Friday'). |
| `is_weekend` | `bool` | 1 if Saturday/Sunday, 0 otherwise. |
| `is_working_day` | `bool` | 1 if Monday-Friday and not a holiday, 0 otherwise. |
| `is_holiday_nasional` | `bool` | 1 if it is a national holiday. |
| `nama_libur` | `object` | Specific name of the holiday (e.g., "New Year's Day"). High cardinality of NULLs. |

### B. Air Quality (Pollutants & Targets)
All pollutant values are daily averages (or max, depending on sensor) in ISPU units or concentration (needs verification based on scale, max values ~300 suggest ISPU or µg/m³).

| Column Name | Type | Description | Stats (Min/Max) |
| :--- | :--- | :--- | :--- |
| `pm10` | `float` | Particulate Matter < 10µm. | 0 - 187 |
| `pm25` | `float` | Particulate Matter < 2.5µm. (Critical for health) | 0 - 287 |
| `so2` | `float` | Sulfur Dioxide. | 0 - 112 |
| `co` | `float` | Carbon Monoxide. | 0 - 134 |
| `o3` | `float` | Ozone (Surface). | 0 - 314 |
| `no2` | `float` | Nitrogen Dioxide. | 0 - 202 |
| `max` | `float` | **Highest** pollutant value for the day (Basis for ISPU). | 0 - 314 |
| `critical_parameter` | `object` | The pollutant responsible for the `max` value (e.g., 'PM25', 'O3'). | - |
| `target_kategori` | `object` | **TARGET VARIABLE**. ISPU Category. <br>Values: `BAIK`, `SEDANG`, `TIDAK SEHAT`, `SANGAT TIDAK SEHAT`, `BERBAHAYA`. | - |

### C. Meteorological Features (Weather)
High-resolution weather data likely sourced from ERA5 or local sensors.

| Column Name | Description |
| :--- | :--- |
| **Wind** | |
| `wind_speed_10m_mean` | Average wind speed at 10m height (m/s). |
| `wind_speed_10m_max` / `_min` | Max/Min wind speed. |
| `wind_direction_10m_dominant` | Dominant wind direction (0-360 degrees). |
| `wind_gusts_10m_mean` | Average wind gust speed. |
| **Temperature** | |
| `temperature_2m_mean` | Average temperature at 2m height (°C). |
| `temperature_2m_max` / `_min` | Daily Max/Min temperature. |
| **Atmosphere** | |
| `surface_pressure_mean` | Avg atmospheric pressure at surface (hPa). |
| `relative_humidity_2m_mean` | Avg Relative Humidity (%). |
| `cloud_cover_mean` | Avg Cloud Cover (0-100%). |
| `shortwave_radiation_sum` | Total solar radiation (Energy). Important for O3 formation. |
| **Precipitation** | |
| `precipitation_sum` | Total daily rainfall (mm). |
| `precipitation_hours` | Number of hours with rain. |
| `rainy_day` | Binary flag (1 if rained, 0 otherwise). |

### D. Engineered Interaction Features
Physics-guided features created to capture atmospheric dynamics.

| Column Name | Formula / Logic | Purpose |
| :--- | :--- | :--- |
| `washout_effect` | `precipitation_sum * wind_speed` | Captures pollutant removal by rain/wind (wet deposition). |
| `ventilation_proxy` | `wind_speed * boundary_layer_height` (approx) | Estimates atmosphere's ability to disperse pollutants. |

### E. Socio-Economic & External
| Column Name | Description |
| :--- | :--- |
| `Total_Penduduk` | Population count (Annual resolution, interpolated). Proxy for anthropogenic activity. |
| `ndvi` | Normalized Difference Vegetation Index. Green space density (0-1). Higher values = more vegetation. |

## 3. Statistical Insights

### Target Distribution (`target_kategori`)
The dataset is **imbalanced**, dominated by the 'SEDANG' (Moderate) category.
1.  **SEDANG**: ~10,448 (67.8%)
2.  **TIDAK SEHAT**: ~2,421 (15.7%)
3.  **BAIK**: ~2,342 (15.2%)
4.  **SANGAT TIDAK SEHAT**: ~200 (1.3%)
5.  **BERBAHAYA**: 1 (Extremely Rare)

### Key Observations
*   **PM2.5 Dominance**: `PM25` is the `critical_parameter` in 5,574 cases, followed closely by `O3` (5,471). This suggests Jakarta's air quality is primarily driven by fine particles and photochemical smog.
*   **Station Variability**: Stations have roughly equal data counts (~3,000 dates each), indicating good spatial coverage across the city divisions.
*   **Data Health**:
    *   Pollutants are fully available (likely pre-imputed).
    *   Weather columns share the exact same missingness pattern (329 rows), suggesting a systemic gap in the weather data source for a specific period.

## 4. Recommendations for Modeling
1.  **Handle Missing Dates**: The 327 rows with null `tanggal` should likely be dropped or investigated, as time-series models (like Chronos) strictly require valid timestamps.
2.  **Imbalance Handling**: Use Class Weights, Focal Loss, or Resampling (SMOTE) to handle the rarity of 'SANGAT TIDAK SEHAT' and 'BERBAHAYA'. Merging categories (Ordinal Regression) is also a strong strategy.
3.  **Feature Selection**: `critical_parameter` and `max` are **leaky features** (derived directly from the target components) and MUST be removed from `X` (input features) during training.
4.  **Weather Imputation**: The ~329 missing weather rows can be imputed using interpolation or forward-fill given the time-series nature.
