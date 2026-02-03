# Plan: Chronos-2 Direct Pollutant Forecasting

Build production-ready Jupyter notebook for ISPU air quality forecasting (Sep-Nov 2025) using **Chronos-2** (`amazon/chronos-2`) to directly forecast pollutant concentrations, then apply deterministic ISPU thresholds for category prediction. Reference: Context7 `/amazon-science/chronos-forecasting`.

## Core Strategy

**Why Direct Pollutant Forecasting?**

- Pollutants (`pm10`, `so2`, etc.) are **highest feature importance** (per EDA)
- ISPU categories are **deterministic** (threshold-based, not probabilistic)
- Avoids compounding errors (weather forecast → pollutant prediction)
- Chronos-2 excels at autoregressive patterns (PM10 today → PM10 tomorrow)
- Simpler pipeline: `Historical Pollutant → Chronos-2 → Forecasted Pollutant → ISPU Rule → Category`

**Chronos-2 Advantages (from Context7):**

- Pretrained foundation model on massive time-series corpus
- Supports univariate/multivariate forecasting with covariates
- Probabilistic forecasts (quantile predictions: p10, p50, p90)
- Zero-shot generalization to unseen data (Sep-Nov 2025)

## Steps

1. **Setup & data preparation** — Import `final_dataset_ready_for_modeling.csv`, install `chronos` package (`pip install git+https://github.com/amazon-science/chronos-forecasting.git`), check GPU availability (`torch.cuda.device_count()` should return 2 for P100s), load `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto")` to auto-distribute model across 2 GPUs, parse dates (2010-01-01 to 2025-08-31), verify 5 stations (DKI1-DKI5), handle missing values (forward-fill for pollutants), set seed=42, torch.manual_seed(42), torch.cuda.manual_seed_all(42).

2. **Direct pollutant forecasting with Chronos-2** — For each station, prepare univariate inputs: `context = torch.tensor([station_data['pm10'].values[-512:]], dtype=torch.float32).cuda()` (shape: `[1, 1, 512]`). Call `pipeline.predict(inputs=context, prediction_length=91, batch_size=256)` with large batch size to saturate 2x P100 GPUs (~32GB combined VRAM). Extract median (p50) from quantile predictions (shape: `[1, 9, 91]`). Process all 30 forecasts (6 pollutants × 5 stations: **PM10, PM2.5, SO2, CO, O3, NO2**) in parallel batches where possible. Apply constraints: clip negatives to 0, cap at 99th percentile from training data per pollutant. Expected runtime: ~3-5 min total on 2x P100 vs ~40 min on CPU.

3. **ISPU category mapping (Permen LHK 14/2020)** — For each forecasted day, calculate ISPU index per pollutant using **linear interpolation formula**: `I = ((I_hi - I_lo) / (BP_hi - BP_lo)) * (C_p - BP_lo) + I_lo`, where `C_p` is forecasted concentration. Use official breakpoints (24h avg for PM10/PM2.5/SO2/NO2, 8h for CO, 1h for O3):
   - **PM10**: BAIK (0-50 µg/m³ → 0-50 ISPU), SEDANG (51-150 → 51-100), TIDAK SEHAT (151-350 → 101-200)
   - **PM2.5**: BAIK (0-15.5 → 0-50), SEDANG (15.6-55.4 → 51-100), TIDAK SEHAT (55.5-150.4 → 101-200)
   - **SO2**: BAIK (0-52 → 0-50), SEDANG (53-180 → 51-100), TIDAK SEHAT (181-400 → 101-200)
   - **CO**: BAIK (0-4000 → 0-50), SEDANG (4001-8000 → 51-100), TIDAK SEHAT (8001-15000 → 101-200)
   - **O3**: BAIK (0-120 → 0-50), SEDANG (121-235 → 51-100), TIDAK SEHAT (236-400 → 101-200)
   - **NO2**: BAIK (0-80 → 0-50), SEDANG (81-200 → 51-100), TIDAK SEHAT (201-1130 → 101-200)

   Take `max(pm10_ispu, pm25_ispu, so2_ispu, co_ispu, o3_ispu, no2_ispu)` as dominant ISPU, map to 3 categories: BAIK (0-50), SEDANG (51-100), TIDAK SEHAT (>100). Historical categories SANGAT TIDAK SEHAT (201-300) and BERBAHAYA (>300) collapse to TIDAK SEHAT for submission.

4. **Validation on historical Sep-Nov** — Test pipeline on Sep-Nov periods from 2020-2024 (5 validation folds). For each fold: filter training data up to Aug 31 of validation year, run Chronos-2 forecast for Sep-Nov, compare predicted categories vs actual `target_kategori` (map historical 5 categories to 3: BAIK→BAIK, SEDANG→SEDANG, [TIDAK SEHAT|SANGAT TIDAK SEHAT|BERBAHAYA]→TIDAK SEHAT). Compute F1-Score (Macro) using `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`. Analyze per-station errors and per-pollutant MAPE to identify weaknesses.

5. **Submission generation** — Run final Chronos-2 forecast using full training data (2010-01-01 to 2025-08-31). Generate 456 predictions: 91 days × 5 stations, starting from 2025-09-01. Format DataFrame with columns `id` (format: `YYYY-MM-DD_STATION`, e.g., `2025-09-01_DKI1`), `category` (one of: BAIK, SEDANG, TIDAK SEHAT). Verify: no missing rows, no invalid categories, correct date range (Sep 1 - Nov 30). Save as `predictions.csv` without index. Document model source in `SOURCES.md`: "Chronos-2 pretrained model from amazon/chronos-2 (HuggingFace), accessed via Context7 /amazon-science/chronos-forecasting".

## Further Considerations

1. **Context window optimization** — Chronos-2 accepts variable-length inputs. Options: Last 365 days (1 seasonal cycle, faster), last 512 days (default, captures 1.4 years), or last 730 days (2 full years, more stable but slower). Recommend **512 days** per Context7 examples—balances seasonal patterns with computational efficiency. With 2x P100 GPUs, can handle up to 1024 context length without memory issues if needed.

2. **Quantile selection strategy** — Chronos-2 outputs 9 quantiles (0.1, 0.2, ..., 0.9). For safety-critical air quality: use **p50 (median)** for unbiased central tendency. Alternative: conservative approach with p75 (upper quartile) to avoid underestimating pollution, but may inflate false alarms. Recommend p50 for F1-Score optimization, log p10/p90 intervals for post-analysis uncertainty.

3. **Multivariate vs univariate inference** — Context7 shows Chronos-2 supports multivariate (all 6 pollutants jointly in shape `[batch, 6, 512]`). Multivariate captures cross-pollutant correlations but requires consistent data quality across all series. Recommend **univariate** (6 separate forecasts) for robustness—handles missing data better, and pollutants have different physical timescales (O3 photochemical vs PM10 mechanical). GPU parallelism makes 30 univariate forecasts nearly as fast as 1 multivariate.

4. **GPU utilization optimization** — With 2x P100 (15GB each), leverage data parallelism: process multiple stations simultaneously (`batch_size=256` to keep GPUs saturated). Use `device_map="auto"` for automatic model sharding across GPUs, or manual split: GPU:0 for model layers 0-12, GPU:1 for layers 13-24. Monitor with `nvidia-smi` during inference. Expected throughput: ~10-15 forecasts/min vs ~0.5 forecasts/min on CPU.

5. **Covariate integration possibility** — Chronos-2 supports `future_covariates` (e.g., known temperature, wind from climatology). Could add historical Sep-Nov weather averages as weak future signal. Per Context7: `inputs = [{"target": pm10_series, "future_covariates": {"temp": avg_sept_temp}}]`. Recommend **skip for v1**—adds complexity, and direct pollutant autoregression already captures weather effects implicitly (PM10 today reflects yesterday's wind).
