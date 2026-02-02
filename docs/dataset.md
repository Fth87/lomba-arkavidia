# DATASET SPECIFICATION: DATAVIDIA 10.0 (ARKAVIDIA)

## 1. OBJECTIVE

Building a Machine Learning/Deep Learning classification model to predict daily air quality category (ISPU) in DKI Jakarta [1, 2].

- **Target Variable:** `kategori`
- **Evaluation Metric:** F1 Score (Macro Average) [2, 3].
- **Problem Type:** Multi-class Classification on Imbalanced Dataset [4].

## 2. DATASET SCHEMA

There are 6 main file categories/tables that are interrelated with a total of 30+ CSV files:

### A. Main Dataset: ISPU (Air Quality) - 16 Files

Historical air pollution measurement data (2010-2025). This is the main table (train/test) [5, 6].

**⚠️ IMPORTANT: There are 3 Different Schema Structures Across Years**

#### **A.1. Old Format (2010-2014) - 5 Files**

**Files:** `indeks-standar-pencemaran-udara-(ispu)-tahun-{2010-2014}-komponen-data.csv`

- **Number of Columns:** 11
- **Data Range:** 2010-2014 (full year coverage, ~1827 rows/year)
- **Schema:**
  - `periode_data` (integer): YYYYMM format (example: 201001)
  - `tanggal` (date): YYYY-MM-DD format
  - `stasiun` (string): Full station name (example: "DKI2 (Kelapa Gading)")
  - `pm10` (numeric/string): PM₁₀ concentration, "---" for missing data
  - `so2` (numeric/string): Sulfur Dioxide concentration, "---" for missing data
  - `co` (numeric/string): Carbon Monoxide concentration, "---" for missing data
  - `o3` (numeric/string): Ozone concentration, "---" for missing data
  - `no2` (numeric/string): Nitrogen Dioxide concentration, "---" for missing data
  - `max` (numeric): Maximum pollution index value daily
  - `critical` (string): Critical pollutant parameter (often empty in this period)
  - `categori` (string): **TARGET LABEL** - values: "TIDAK ADA DATA", "BAIK", "SEDANG", "TIDAK SEHAT"
- **Characteristics:**
  - Many missing values marked with "---"
  - "TIDAK ADA DATA" category for days without measurement
  - Column names use lowercase (pm10, categori)

#### **A.2. Transition Format (2015-2021) - 7 Files**

**Files:** `indeks-standar-pencemaran-udara-(ispu)-tahun-{2015-2021}-komponen-data.csv`

- **Number of Columns:** 11
- **Data Range:** 2015-2021 (partial coverage, ~367 rows/year)
- **Schema:**
  - `periode_data` (integer): YYYYMM format
  - `tanggal` (date): YYYY-MM-DD format
  - `pm10` (numeric): PM₁₀ concentration (only numeric values, no "---")
  - `so2` (numeric): SO₂ concentration
  - `co` (numeric): CO concentration
  - `o3` (numeric): O₃ concentration
  - `no2` (numeric): NO₂ concentration
  - `max` (numeric): Maximum index value
  - `critical` (string): Critical pollutant abbreviation (example: "O3", "PM10", "SO2")
  - `categori` (string): **TARGET LABEL** - values: "BAIK", "SEDANG", "TIDAK SEHAT"
  - `lokasi_spku` (string): Station code (example: "DKI3", "DKI4", "DKI5")
- **Characteristics:**
  - Only numeric values (no missing markers)
  - Partial coverage per year
  - Addition of `lokasi_spku` column for station codes

#### **A.3. 2022 Format (Transition to PM2.5)**

**File:** `indeks-standar-pencemaran-udara-(ispu)-tahun-2022-komponen-data.csv`

- **Number of Columns:** 12
- **Data Range:** 2022 (367 rows)
- **Schema:**
  - `periode_data` (integer)
  - `tanggal` (date/mixed): **⚠️ ATTENTION:** Some date parsing errors (values: 44926.625 instead of date)
  - `pm_10` (numeric): PM₁₀ concentration (column name changed with underscore)
  - `pm_duakomalima` (numeric): **NEW!** PM₂.₅ concentration starts being recorded
  - `so2`, `co`, `o3`, `no2` (numeric)
  - `max` (numeric)
  - `critical` (string)
  - `categori` (string): **TARGET LABEL**
  - `lokasi_spku` (string): Station code
- **Characteristics:**
  - **First year with PM2.5 measurement**
  - Some date parsing errors need cleaning
  - `pm_10` column name (not `pm10`)

#### **A.4. Modern Format (2023-2025) - 3 Files**

**Files:**

- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-2023-komponen-data.csv` (1827 rows)
- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-komponen-data-2024.csv` (1832 rows)
- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-komponen-data-2025.csv` (1217 rows)

**2023 Schema (12 columns):**

- `periode_data` (integer): YYYYMM format
- `tanggal` (date): YYYY-MM-DD
- `stasiun` (string): Full station name with location (example: "DKI5 Kebon Jeruk Jakarta Barat")
- `pm_sepuluh` (numeric/string): PM₁₀, "-" or "---" for missing
- `pm_duakomalima` (numeric/string): PM₂.₅, "-" for missing
- `sulfur_dioksida` (numeric/string): SO₂, "---" for missing
- `karbon_monoksida` (numeric/string): CO, "---" for missing
- `ozon` (numeric/string): O₃, "---" for missing
- `nitrogen_dioksida` (numeric/string): NO₂, "---" for missing
- `max` (numeric): Maximum index value
- `parameter_pencemar_kritis` (string): Complete critical parameter (example: "PM10", "O3", or sometimes numeric values like "3")
- `kategori` (string): **TARGET LABEL** - values: "BAIK", "SEDANG", "TIDAK SEHAT"

**2024-2025 Schema (13 columns):**
Same as 2023, **plus:**

- `bulan` (integer): Month number (1-12) as separate column

**Time Range:**

- 2023: February - November 2023
- 2024: January 2024 - December 2024
- 2025: April 2025 - August 2025

**Characteristics:**

- Column names use full Indonesian language (`pm_sepuluh`, `sulfur_dioksida`)
- Mix of "-" and "---" for missing values
- Float values with decimal precision
- "TIDAK ADA DATA" category still appears for days without data
- **Most complete & recent data for modeling**

**📌 Station Mapping:**

- **DKI1:** Bundaran HI (Bundaran Hotel Indonesia) - Jakarta Pusat
- **DKI2:** Kelapa Gading - Jakarta Utara
- **DKI3:** Jagakarsa - Jakarta Selatan
- **DKI4:** Lubang Buaya - Jakarta Timur
- **DKI5:** Kebon Jeruk - Jakarta Barat

**📊 ISPU Categories (Target Variable):**

- **BAIK:** Clean air, no health effects
- **SEDANG:** Acceptable, sensitive may be affected
- **TIDAK SEHAT:** Starting to be dangerous for sensitive groups
- **SANGAT TIDAK SEHAT:** Dangerous for all population
- **BERBAHAYA:** Emergency health condition (very rare)
- **TIDAK ADA DATA:** Measurement not available

### B. Supporting Dataset: Cuaca Harian (Weather) - 5 Files

Meteorological factors affecting pollutant dispersion [9, 10]. Complete and consistent data for all stations.

**Files (5 stations):**

- `cuaca-harian-dki1-bundaranhi.csv` (5724 rows)
- `cuaca-harian-dki2-kelapagading.csv` (5724 rows)
- `cuaca-harian-dki3-jagakarsa.csv` (5724 rows)
- `cuaca-harian-dki4-lubangbuaya.csv` (5724 rows)
- `cuaca-harian-dki5-kebonjeruk.csv` (5724 rows)

**Data Range:** 2010-01-01 to 2025-08-31 (complete daily data, NO missing values)

**Schema (24 columns, all numeric except time):**

**1. Temporal:**

- `time` (date): Measurement date (YYYY-MM-DD format) - **Join Key with ISPU**

**2. Temperature (3 columns):**

- `temperature_2m_max` (float): Maximum daily temperature (°C)
- `temperature_2m_min` (float): Minimum daily temperature (°C)
- `temperature_2m_mean` (float): Mean daily temperature (°C)

**3. Precipitation (2 columns):**

- `precipitation_sum` (float): Total daily rainfall (mm) - **Important for pollutant washout**
- `precipitation_hours` (float): Rainfall duration (hours)

**4. Wind (8 columns):**

- `wind_speed_10m_max` (float): Maximum wind speed at 10m height (km/h)
- `wind_speed_10m_min` (float): Minimum wind speed (km/h)
- `wind_speed_10m_mean` (float): Mean wind speed (km/h) - **Important for pollutant dispersion**
- `wind_direction_10m_dominant` (float): Dominant wind direction (degrees, 0-360°) - **Determines pollutant movement direction**
- `winddirection_10m_dominant` (float): Duplicate wind direction column (identical to above)
- `wind_gusts_10m_max` (float): Maximum wind gust speed (km/h)
- `wind_gusts_10m_min` (float): Minimum wind gust speed (km/h)
- `wind_gusts_10m_mean` (float): Mean wind gust speed (km/h)

**5. Humidity (3 columns):**

- `relative_humidity_2m_mean` (float): Mean relative humidity (%) - **Affects particle formation**
- `relative_humidity_2m_max` (float): Maximum relative humidity (%)
- `relative_humidity_2m_min` (float): Minimum relative humidity (%)

**6. Cloud Cover (3 columns):**

- `cloud_cover_mean` (float): Mean cloud cover (%)
- `cloud_cover_max` (float): Maximum cloud cover (%)
- `cloud_cover_min` (float): Minimum cloud cover (%)

**7. Air Pressure (3 columns):**

- `surface_pressure_mean` (float): Mean surface pressure (hPa)
- `surface_pressure_max` (float): Maximum surface pressure (hPa)
- `surface_pressure_min` (float): Minimum surface pressure (hPa)
- **Note:** Pressure values slightly different across stations (different location altitudes)

**8. Solar Radiation (1 column):**

- `shortwave_radiation_sum` (float): Total daily shortwave radiation (MJ/m²) - **Catalyst for photochemical ozone formation**

**Characteristics:**

- ✅ **Most complete data:** No missing values whatsoever
- ✅ **Longest coverage:** 2010-2025 (15+ years)
- ✅ **High consistency:** Identical schema for all 5 stations
- 📍 **Location-specific:** Values slightly different per station (local microclimate)
- 🔗 **Perfect for merging:** Can be directly joined with ISPU via `time = tanggal`

### C. Supporting Dataset: NDVI (Vegetation) - 1 File

Greenness index of the area (-1 to +1). Vegetation absorbs pollutants [8, 11].

**File:** `indeks-ndvi-jakarta.csv`

- **Number of Rows:** 1,812
- **Data Range:** 2009-12-19 to 2025-08-29
- **Frequency:** Irregular (~every 16 days) - depends on satellite cycle

**Schema (3 columns):**

- `tanggal` (date): Satellite data acquisition date (format: YYYY-MM-DD) - **Join Key**
- `stasiun_id` (string): Monitoring station code (values: "DKI1", "DKI2", "DKI3", "DKI4", "DKI5")
- `ndvi` (float): Normalized Difference Vegetation Index value (range: ~0.2-0.6)
  - **0.0 - 0.2:** Non-vegetated areas (urban, bare soil)
  - **0.2 - 0.4:** Sparse/moderate vegetation
  - **0.4 - 0.6:** Dense vegetation (parks, urban forests)
  - **0.6 - 1.0:** Very dense vegetation (rare in Jakarta)

**Characteristics:**

- 📡 **Satellite-based data:** Depends on satellite (Landsat/Sentinel)
- ⏰ **Irregular intervals:** Not daily, usually every 8-16 days
- 🌱 **Vegetation indicator:** Measures plant health and density
- 🔄 **Temporal interpolation needed:** Requires interpolation for matching with daily ISPU data
- ⚖️ **Unbalanced station coverage:** Not all stations have equal coverage
- 🌳 **Pollutant absorber:** High NDVI → more vegetation → lower pollution

**Important Notes:**

- Data starts from 2009 (1 year before ISPU begins)
- For joining with daily ISPU, strategy needed:
  - Forward-fill / Backward-fill
  - Linear interpolation
  - Or take nearest NDVI value within ±7 day window

### D. Supporting Dataset: National Holidays & Weekends - 1 File

Temporal data for human mobility patterns (_Weekend Effect_) [12, 13].

**File:** `dataset-libur-nasional-dan-weekend.csv`

- **Number of Rows:** 5,846
- **Data Range:** 2010-01-01 to 2025-12-31 (complete 16-year calendar)

**Schema (5 columns):**

- `tanggal` (date): Calendar date (format: YYYY-MM-DD) - **Join Key with ISPU**
- `is_holiday_nasional` (binary): National holiday indicator
  - `1`: National holiday (example: Eid, Christmas, New Year)
  - `0`: Not a national holiday
- `nama_libur` (string): Holiday name in English (example: "New Year's Day", "Eid al-Fitr")
  - **Empty/null** if not a holiday
- `is_weekend` (binary): Weekend indicator
  - `1`: Saturday or Sunday
  - `0`: Monday-Friday (weekdays)
- `day_name` (string): Day name in English (values: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

**Characteristics:**

- 📅 **Complete calendar:** Every day from 2010-2025 recorded
- 🚗 **Mobility indicator:** Weekend & holiday → lower traffic → potential pollution decrease
- 🏭 **Industrial activity:** Weekdays → high activity → more emissions
- 🔗 **Easy join:** Direct merge with ISPU via `tanggal`
- 🎉 **National holidays:** Includes Eid al-Fitr, Eid al-Adha, Christmas, New Year, etc.

**Use Cases:**

1. **Feature engineering:** Create feature `is_workday = NOT (is_holiday OR is_weekend)`
2. **Seasonality:** Detect monthly patterns (Ramadan effect, long holidays)
3. **Lag features:** Pollution day after long holidays (post-holiday surge)
4. **Cyclical encoding:** Day of week as sin/cos transform

### E. Supporting Dataset: Population Count - 1 File

Indicator of anthropogenic activity and emission sources [14, 15].

**File:** `data-jumlah-penduduk-provinsi-dki-jakarta-berdasarkan-kelompok-usia-dan-jenis-kelamin-tahun-2013-2021-komponen-data.csv`

- **Number of Rows:** 34,178
- **Data Range:** Years 2013-2021 (annual data)
- **Granularity:** Up to kelurahan (sub-district) level

**Schema (9 columns):**

- `periode_data` (integer): Recording year (values: 2013-2021)
- `tahun` (integer): Year (duplicate of `periode_data`)
- `nama_provinsi` (string): Always "PROVINSI DKI JAKARTA"
- `nama_kabupaten_kota` (string): Administrative city area of Jakarta
  - Values: "JAKARTA TIMUR", "JAKARTA SELATAN", "JAKARTA UTARA", "JAKARTA BARAT", "JAKARTA PUSAT", "KEPULAUAN SERIBU"
- `nama_kecamatan` (string): District name
  - Example: "TANAH ABANG", "MENTENG", "TEBET", "CAKUNG", etc.
- `nama_kelurahan` (string): Sub-district name (smallest level)
  - Example: "PETOJO UTARA", "KEBON MELATI", "KAMPUNG BALI", etc.
- `usia` (string): Age bracket
  - Values: "0-4", "5-9", "10-14", "15-19", "20-24", ..., "70-74", "75+"
  - Total: ~16 age groups
- `jenis_kelamin` (string): Gender
  - Values: "Laki-laki", "Laki laki" (typo inconsistency), "Perempuan"
  - ⚠️ **Data quality issue:** Inconsistent spelling "Laki-laki" vs "Laki laki"
- `jumlah_penduduk` (integer): Population count in that category

**Characteristics:**

- 🏙️ **Granular:** Data down to kelurahan × age × gender level
- 👥 **Demographic detail:** Can calculate total population per district/city
- 📍 **Spatial matching challenge:** Need mapping district → ISPU station (geographic location)
- 📅 **Annual data:** No monthly/daily variation (assume constant population per year)
- 🚗 **Emission proxy:** High population → more vehicles, energy consumption → higher emissions

**Usage Methods:**

1. **Aggregate per area:** Sum `jumlah_penduduk` by `nama_kabupaten_kota` or `nama_kecamatan`
2. **Mapping to stations:**
   - DKI1 (Bundaran HI) → Jakarta Pusat
   - DKI2 (Kelapa Gading) → Jakarta Utara
   - DKI3 (Jagakarsa) → Jakarta Selatan
   - DKI4 (Lubang Buaya) → Jakarta Timur
   - DKI5 (Kebon Jeruk) → Jakarta Barat
3. **Age demographics:** Can calculate working-age population proportion (15-64 years) → economic activity
4. **Temporal join:** For years 2010-2012, use 2013 data; for 2022+, use 2021 data

**⚠️ Data Quality Issues:**

- Inconsistent gender spelling needs cleaning
- Coverage only 2013-2021 (needs extrapolation for 2010-2012 and 2022-2025)

### F. Supporting Dataset: River Water Quality - 1 File

Indicator of regional environmental burden [16, 17].

**File:** `data-kualitas-air-sungai-komponen-data.csv`

- **Number of Rows:** 14,402
- **Data Range:** Year 2024 (focused on May)
- **Format:** Long-format (one row per parameter per sampling point)

**Schema (12 columns):**

- `periode_data` (integer): Recording year (value: 2024)
- `periode_pemantauan` (string): Monitoring period within year
  - Values: "Periode 1", "Periode 2", "Periode 3", etc.
- `bulan_sampling` (integer): Sampling month (1-12)
  - Data sample seen: mostly month 5 (May)
- `titik_sampel` (string): Sampling point code
  - Example: "KLT 3", "SKR 2", "PSR 2", "CKR 3", etc.
- `nama_sungai` (string): Monitored river name
  - Example: "Kalibaru Timur", "Sekertaris", "Pesanggrahan", "Cakung Drain", etc.
- `alamat` (string): Detailed sampling location
  - Example: "Jl. Inspeksi Kalimalang", "Jl. Raya Bogor KM 26", etc.
- `latitude` (float): GPS latitude coordinate
  - Range: ~ -6.1 to -6.3 (Jakarta region)
- `longitude` (float): GPS longitude coordinate
  - Range: ~ 106.7 to 106.9 (Jakarta region)
- `jenis_parameter` (string): Measured parameter type
  - Main value: "Kimia" (chemical parameters)
- `parameter` (string): Measured water quality parameter
  - **Chemical:** pH, BOD, COD, DO (Dissolved Oxygen), TSS (Total Suspended Solids)
  - **Nutrients:** Nitrat, Nitrit, Total P (Phosphorus)
  - **Pollutants:** F (Fluoride), H2S (Hydrogen Sulfide), Minyak & Lemak, Deterjen (MBAS)
  - **Heavy Metals:** Cd (Cadmium), Cu (Copper), Pb (Lead), Zn (Zinc), Cr (Chromium)
  - **Toxins:** Fenol, Sianida
- `baku_mutu` (float): Established quality standard/threshold
- `hasil_pengukuran` (float): Actual field measurement result

**Characteristics:**

- 🌊 **Multi-parameter:** 15+ chemical parameters per sampling point
- 📍 **Georeferenced:** Has GPS coordinates for spatial analysis
- 🏭 **Environmental burden:** Poor water quality → indicator of high industrial/domestic pollution
- 🔗 **Spatial join needed:** Need mapping river locations to nearest ISPU stations
- 📊 **Long format:** Each row = 1 measurement of 1 parameter (need pivot for wide format)
- ⏰ **Limited temporal coverage:** Only 2024, not suitable for long time series

**Important Parameters for Air Quality Context:**

1. **High BOD/COD** → Organic pollution → high anthropogenic activity
2. **Heavy metals (Pb, Cd)** → Industrial/vehicle pollution
3. **High Nitrate/Nitrite** → Agricultural fertilizer runoff/domestic waste
4. **Extreme pH** → Uncontrolled industrial waste
5. **Low DO** → Polluted water, disturbed ecosystem

**Usage Methods:**

1. **Aggregate per river:** Average parameters per river name
2. **Exceedance rate:** Calculate % of samples exceeding quality standards
3. **Spatial matching:** Join with ISPU stations based on distance (latitude/longitude)
4. **Composite index:** Create "Water Pollution Index" from multiple parameters
5. **Temporal limitation:** Data only 2024 → can be used as static feature or constant assumption

**⚠️ Limitations:**

- Limited coverage (2024 only) → cannot be used for trend analysis
- Not all months fully covered
- Requires domain knowledge for chemical parameter interpretation
- Different spatial resolution from ISPU stations (needs interpolation/distance-based weighting)

### G. Submission File - 1 File

Template for model prediction submission.

**File:** `sample_submission.csv`

- **Number of Rows:** 457
- **Prediction Period:** 2025-09-01 to 2025-11-29 (~3 months)
- **Coverage:** 5 stations × ~91 days = 455-457 entries

**Schema (2 columns):**

- `id` (string): Unique identifier for each prediction
  - Format: `YYYY-MM-DD_STATIONCODE`
  - Example: "2025-09-01_DKI1", "2025-09-01_DKI2", ..., "2025-11-29_DKI5"
- `category` (string): Placeholder for ISPU category prediction
  - Default value: "NULL" (must be filled with model predictions)
  - Valid values: "BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"

**Characteristics:**

- 🎯 **Target submission:** This file is the final submission template
- 📅 **Future dates:** Sep-Nov 2025 period (data doesn't exist yet, needs prediction)
- 🔢 **Daily predictions:** Every day for each station must have a prediction
- ✅ **Validation:** Ensure all ids are covered and no duplicates

**Submission Format:**

```csv
id,category
2025-09-01_DKI1,SEDANG
2025-09-01_DKI2,BAIK
2025-09-01_DKI3,SEDANG
...
2025-11-29_DKI5,BAIK
```

**Important Notes:**

- Categories must **exact match** with categories in training data
- No missing predictions allowed (complete 457 rows)
- Case-sensitive: use uppercase for categories

---

## 2.1. DATA FILES SUMMARY (Total: 30+ Files)

| Category        | File Count | Total Rows | Time Range    | Frequency | Completeness            |
| --------------- | ---------- | ---------- | ------------- | --------- | ----------------------- |
| **ISPU (Main)** | 16         | ~20,000+   | 2010-2025     | Daily     | ⚠️ Variable (improving) |
| **Weather**     | 5          | 28,620     | 2010-2025     | Daily     | ✅ Complete (100%)      |
| **NDVI**        | 1          | 1,812      | 2009-2025     | ~16 days  | ⚠️ Sparse (satellite)   |
| **Holidays**    | 1          | 5,846      | 2010-2025     | Daily     | ✅ Complete (100%)      |
| **Population**  | 1          | 34,178     | 2013-2021     | Annual    | ⚠️ Partial years        |
| **River Water** | 1          | 14,402     | 2024          | Irregular | ⚠️ Limited (2024 only)  |
| **Submission**  | 1          | 457        | 2025 (future) | Daily     | Target for prediction   |

---

## 3. DATA CHARACTERISTICS & CHALLENGES

### 3.1. Data Quality Issues

1. **Missing Values Patterns:**
   - **ISPU files:** Multiple representations ("---", "-", empty string, "TIDAK ADA DATA")
   - **Strategy:** Need standardization of missing value handling
   - **Impact:** Some pollutants not measured in early years

2. **Schema Inconsistency:**
   - **Column names change across years:**
     - `pm10` (2010-2021) → `pm_10` (2022) → `pm_sepuluh` (2023-2025)
     - `categori` (2010-2021) → `kategori` (2023-2025)
     - `critical` → `parameter_pencemar_kritis`
   - **Solution:** Need mapping & column name standardization

3. **Data Type Issues:**
   - **ISPU 2022:** Date parsing errors (values: 44926.625 instead of date)
   - **Population:** Gender spelling inconsistency ("Laki-laki" vs "Laki laki")
   - **Solution:** Data cleaning & validation pipeline

4. **Temporal Gaps:**
   - **NDVI:** Sparse (16-day intervals), not all stations have equal coverage
   - **Population:** 2013-2021 only (needs extrapolation for 2010-2012, 2022-2025)
   - **River Water:** 2024 only (insufficient for time series)

### 3.2. Problem Characteristics

1. **Imbalanced Data:**
   - Categories like "Berbahaya" and "Sangat Tidak Sehat" are very rare
   - Majority data: "BAIK" and "SEDANG"
   - **Solution:** Model must be optimized for F1-Macro, not Accuracy [3, 4]
   - **Techniques:** SMOTE, class weights, focal loss, ensemble methods

2. **Multivariate Complexity:**
   - Air quality affected by complex interactions between:
     - **Emissions:** Vehicle traffic, industry (proxy: population, workdays)
     - **Meteorology:** Rain (wash-out), wind (dispersion), temperature, humidity
     - **Environment:** Vegetation (NDVI), river conditions
   - **Feature interactions matter:** Rain + strong wind → drastic pollution decrease [18]

3. **Temporal Dependencies:**
   - **Seasonality:** Dry vs rainy season
   - **Weekly patterns:** Weekday (high) vs weekend (low)
   - **Holiday effects:** Long holidays → traffic down → pollution down
   - **Lag effects:** Today's pollution affected by conditions 1-3 days prior

4. **Spatial Factors:**
   - **Location matters:** 5 stations have different characteristics
     - DKI1 (Bundaran HI): City center, high traffic
     - DKI2 (Kelapa Gading): Residential, near sea
     - DKI3 (Jagakarsa): Suburban, greener
     - DKI4 (Lubang Buaya): East Jakarta, near industry
     - DKI5 (Kebon Jeruk): West Jakarta, near airport
   - **Microclimate:** Weather slightly different per location

### 3.3. Modeling Considerations

1. **Feature Engineering Priorities:**
   - ✅ **Temporal:** Day of week, month, season, is_workday, lag features (1-7 days)
   - ✅ **Weather:** Interaction terms (rain × wind), moving averages
   - ✅ **Cyclical encoding:** Sin/cos transform for day, month
   - ✅ **NDVI interpolation:** Forward/backward fill or linear interpolation
   - ✅ **Spatial:** Station-specific features, distance to city center

2. **Data Preparation Challenges:**
   - **Schema standardization** across 16 ISPU files
   - **Merging strategy** for 6 data sources with different frequencies
   - **Missing value imputation** appropriate for time series
   - **Outlier handling** on sensor readings (possible sensor malfunction)

3. **Train/Test Split Strategy:**
   - ⚠️ **Don't use random split!** (time series data)
   - ✅ **Time-based split:** 2010-2024 for training, 2025 (Sep-Nov) for prediction
   - ✅ **Consider validation:** Use 2024 Q3-Q4 as validation set
   - ✅ **Cross-validation:** Time series CV (rolling window or blocked CV)

---

## 5. EVALUATION FORMULA

Evaluation uses **F1 Score (Macro Average)**:

```python
from sklearn.metrics import f1_score
# y_true: actual labels, y_pred: model predictions
score = f1_score(y_true, y_pred, average='macro')
```
