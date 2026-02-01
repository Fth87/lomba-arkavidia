# AGENT.md

## Agent Role

This agent acts as an **ML experiment runner and reviewer** for the notebook `pengerjaan.ipynb`. Its responsibility is to ensure that the modeling pipeline complies with competition rules, follows clean ML practices, and produces reproducible results.

The agent **does not manage external configuration files**. All experiment parameters must be defined explicitly at the very top of the notebook.

## Scope of Responsibilities

* Understand the context of Jakarta air quality (ISPU) datasets
* Execute and validate the notebook `pengerjaan.ipynb`
* Enforce time-based train and test separation
* Evaluate models using Macro F1-Score
* Generate `submission.csv` in the required competition format

## Notebook Contract

Primary notebook:

* `pengerjaan.ipynb`

Mandatory rules:

* All configuration variables are declared in the first cell
* No model fitting on data dated **>= 2025-09-01**
* No hard-coded paths except `DATA_ROOT`
* All outputs must be deterministic under `RANDOM_SEED`

## Dataset Reference

* Dataset schema, metadata, and descriptions are defined in `docs/dataset.md`
* Core and supporting datasets include:

  * Daily ISPU measurements
  * Daily weather data
  * NDVI (vegetation index)
  * National holidays and weekends
* External datasets are allowed only if relevant and properly documented

## Supported Models

The agent supports and validates the following models:

* CatBoostClassifier
* XGBoostClassifier
* LightGBMClassifier

Model characteristics:

* Multiclass classification
* Designed for tabular and imbalanced data
* No unnecessary preprocessing or feature leakage

## Evaluation Protocol

* Primary metric: **Macro F1-Score**
* Validation strategy: TimeSeriesSplit
* Scores are computed per fold and averaged

## Competition Rules Enforcement

The agent must strictly enforce:

* No training on data after `2025-09-01`
* No fitting or tuning on the test set
* No usage of future information unavailable at prediction time
* Submission format must match `dataset/sample_submission.csv`

## Implementation Principles

The agent verifies adherence to:

* **KISS**: simple, auditable pipelines
* **DRY**: no duplicated evaluation logic
* **Clean Code**: short functions, explicit naming
* **Reproducibility**: fixed seeds and declared dependencies

## Output Artifacts

* `submission.csv`
* Validation Macro F1 score
* Fully executable notebook without errors

## Dependencies

Runtime dependencies are declared in `requirements.txt`:

* pandas
* numpy
* scikit-learn
* catboost
* xgboost
* lightgbm

## Prohibited Actions

The agent must not:

* Modify configuration outside the notebook
* Add unnecessary pipeline complexity
* Perform heavy hyperparameter search
* Fit or refit models on the test set

## Final Objective

Ensure the notebook produces a valid, clean, competitive ISPU classification model that is ready for submission and aligned with professional data science standards.

## Dataset Overview

This project integrates six correlated data domains to improve air quality prediction accuracy. All datasets are temporally and/or spatially aligned at the monitoring-station and daily level unless otherwise stated.

## Dataset Specification

### 1. Air Quality Index (ISPU)

Primary target-driving dataset describing daily air pollution conditions in Jakarta.

**Fields**

* `periode_data`: Reporting time range
* `tanggal`: Sampling date
* `stasiun`: Air quality monitoring station (SPKU) ID or location
* `pm_sepuluh`: PM10 concentration (<10 μm particles)
* `pm_duakomalima`: PM2.5 concentration (<2.5 μm fine particles)
* `sulfur_dioksida`: SO₂ concentration from sulfur-containing fuel combustion
* `karbon_monoksida`: CO concentration from incomplete combustion (mainly traffic)
* `ozon`: Ground-level O₃ concentration (secondary pollutant)
* `nitrogen_dioksida`: NO₂ concentration from high-temperature combustion
* `max`: Maximum pollutant index value across parameters
* `parameter_pencemar_kritis`: Pollutant determining the `max` value
* `kategori`: Target label (dependent variable)

### 2. Vegetation Index (NDVI)

Spatial environmental context around monitoring stations.

**Fields**

* `tanggal`: NDVI observation date
* `stasiun_id`: Corresponding air quality station ID
* `ndvi`: Normalized Difference Vegetation Index value

### 3. Daily Weather Data

Meteorological drivers influencing pollutant dispersion and formation.

**Fields**

* `time`: Weather observation date (aligned with ISPU)
* `temperature_2m_(max|min|mean)`: Air temperature at 2 m (°C)
* `precipitation_(sum|hours)`: Daily rainfall (mm) and duration (hours)
* `wind_speed_10m_(max|mean|min)`: Wind speed at 10 m (km/h)
* `wind_direction_10m_dominant`: Dominant wind direction (degrees)
* `wind_gusts_10m_(max|mean|min)`: Wind gust speed
* `shortwave_radiation_sum`: Solar radiation (MJ/m²)
* `relative_humidity_2m_(mean|max|min)`: Relative humidity (%)
* `cloud_cover_(mean|max|min)`: Cloud coverage (%)
* `surface_pressure_(mean|max|min)`: Surface air pressure (hPa)

### 4. Population Data

Demographic pressure as a long-term pollution proxy.

**Fields**

* `periode_data`, `tahun`: Census or reporting period
* `nama_provinsi`, `kabupaten_kota`, `kecamatan`, `kelurahan`: Administrative hierarchy
* `usia`: Age group
* `jenis_kelamin`: Gender
* `jumlah_penduduk`: Population count

### 5. River Water Quality

Environmental degradation signal used as auxiliary contextual feature.

**Fields**

* `periode_data`: Reporting year
* `periode_pemantauan`: Monitoring cycle within year
* `bulan_sampling`: Sampling month
* `titik_sampel`: River sampling point ID
* `nama_sungai`: River name
* `alamat`: Physical sampling location
* `latitude`, `longitude`: Geographic coordinates
* `jenis_parameter`: Test category (chemical, physical, biological)
* `parameter`: Measured element
* `baku_mutu`: Regulatory quality threshold
* `hasil_pengukuran`: Observed measurement value

### 6. National Holidays

Human activity modulation signal.

**Fields**

* `tanggal`: Calendar date
* `is_holiday_nasional`: Binary national holiday indicator
* `nama_libur`: Official holiday name
* `is_weekend`: Binary weekend indicator
* `day_name`: Day of week (extended explanation in `docs/dataset.md`)

**Notes**

* ISPU `kategori` is the sole prediction target.
* Other datasets are auxiliary features.
* Temporal joins use daily granularity; spatial joins use station or administrative proximity.
