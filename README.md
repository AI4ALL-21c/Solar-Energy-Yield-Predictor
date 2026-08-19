# Solar Energy Yield Predictor

![Python](https://img.shields.io/badge/python-3.12-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

A physics-informed machine learning pipeline that forecasts solar power output from weather data, built to support grid stability decisions.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Model Evaluation](#model-evaluation)
- [Impact and Bias](#impact-and-bias)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Next Steps](#next-steps)
- [Streamlit App](#streamlit-app)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Documentation & Citations](#documentation--citations)

---

## Overview

Solar output is clean but volatile — clouds, wind, and temperature all shift generation minute to minute, and grid operators need advance notice of expected output to plan reserves. This project predicts real AC power output from a solar plant by combining a physics model (module temperature and theoretical AC power from irradiance) with an XGBoost model trained to correct the gap between that physics estimate and what the plant actually produced.

The project was built in three phases over the program: data ingestion and cleaning (fixing unit mismatches, filling time gaps, merging 22 inverters into one plant-level total), physics-based feature engineering, and a unified, reproducible notebook pipeline that ties both together into one final prediction.

## Algorithm

| | |
|---|---|
| **Model type** | XGBoost regression (supervised learning), layered on top of a physics-based baseline |
| **Output** | Predicted power gap per kWp, added to theoretical AC power for the final forecast (kW) |

**Inputs (6 features):**

| Feature | Description |
|---|---|
| `IRRADIATION` | Solar irradiation (W/m²) |
| `AMBIENT_TEMPERATURE` | Ambient temperature (°C) |
| `WIND_SPEED` | Wind speed (m/s) |
| `CLOUD_COVER` | Cloud cover (%) |
| `TIME_OF_DAY_SIN/COS` | Time-of-day, cyclically encoded |
| `DAY_OF_YEAR_SIN/COS` | Day-of-year, cyclically encoded |

**Why XGBoost:** compared against two baselines on the same features, XGBoost had the highest R² by a wide margin and captured nonlinear weather-to-yield relationships the simpler models couldn't.

| Model | R² |
|---|---:|
| Linear Regression | 0.696 |
| Random Forest | 0.795 |
| **XGBoost (final model)** | **0.963** |

## Model Evaluation

Official metrics for the retrained XGBoost gap model using `solar_plant_data_cleaned.csv`:

| Evaluation | RMSE | MAE | R² |
|---|---:|---:|---:|
| 80/20 Holdout | 0.0488 gap/kWp | 0.0177 gap/kWp | 0.7910 |
| 5-Fold Cross-Validation | 0.0466 ± 0.0014 gap/kWp | — | — |
| AC Power Backtest | 1245.20 kW | 431.84 kW | 0.9607 |

> **What these numbers mean:** on the AC power backtest, the model explains about 96% of the real variance in plant output (R² ≈ 0.96), and a typical prediction is off by roughly 432 kW (MAE) — small relative to the plant's ~25 MW capacity. RMSE is higher than MAE because it penalizes large misses more heavily, such as sudden overcast swings the physics-only baseline can't anticipate. The 5-fold cross-validation result stays consistent across folds, indicating the model isn't overfitting to one slice of the data.

Feature importances pulled directly from the trained model:

| Feature | Importance |
|---|---:|
| Irradiation | 34% |
| Day of year (sin) | 15% |
| Day of year (cos) | 14% |
| Time of day (cos) | 11% |
| Cloud cover | 7% |
| Ambient temperature | 7% |
| Time of day (sin) | 7% |
| Wind speed | 6% |

Irradiation dominates, as expected. But the seasonal and time-of-day cyclical features rank ahead of cloud cover, temperature, and wind — the model isn't just reacting to instantaneous weather, it's also learned slower seasonal effects (like panel soiling) that pure physics doesn't capture.

## Impact and Bias

**Positive impact**
- Advance forecasts support grid stability, letting operators schedule reserves instead of reacting to sudden drops
- The residual (predicted vs. actual gap) supports predictive maintenance — flagging an underperforming inverter before a manual inspection would catch it
- Better forecasts reduce curtailment and reliance on fossil-fuel peaker plants as backup

**Negative impact / risks**
- Automating inspection and monitoring may reduce demand for manual inspection crews
- Operators may over-trust the forecast during rare weather events the model has never seen (automation complacency)
- Smaller or independent operators without similar telemetry and ML infrastructure could fall behind larger utilities

**How this could amplify or mitigate bias**

Published PV forecasting research confirms a model trained on one solar site typically doesn't transfer directly to a different site with different specifications and geography — accuracy degrades without local recalibration (Tang, Y., Yang, K., Zhang, S., & Zhang, Z. (2022). Photovoltaic power forecasting: a hybrid deep learning model incorporating transfer learning strategy. *[Renewable and Sustainable Energy Reviews, 162](https://ideas.repec.org/a/eee/rensus/v162y2022ics1364032122003781.html)*). Our training data covers a single plant across one ~5-week monsoon window (May 15–Jun 17, 2020), so the model has effectively learned this plant's weather and season, not weather in general.

Embedding physics-based features gives the model a more universal anchor than a purely data-driven model, since the underlying thermodynamics don't change by location — but this doesn't fully solve the transfer problem. Research shows models still typically need a small amount of real local data (as little as two weeks) to recalibrate at a new site (Zhang, L., Wilson, R., Sumner, M., & Wu, Y. (2025). Transfer learning in very-short-term solar forecasting: bridging single site data to diverse geographical applications. *[Applied Energy, 377](https://ideas.repec.org/a/eee/appene/v377y2025ipcs0306261924017367.html)*). We have not yet tested our model on a second plant, so we can't claim the physics-informed approach solves this bias — only that it's a defensible first step.

## Monitoring and Maintenance

If this model were deployed in production, real operation would require:
- Monitoring prediction drift by season and weather pattern
- Regularly checking input data quality and sensor anomalies
- Re-evaluating MAE and R² after every model update
- Versioning the dataset, model, and feature definitions so every prediction is traceable

## Next Steps

1. **Test on a second, cooler-climate plant** — collect and train on a plant outside the monsoon window to directly test the single-site bias described above. `Target: within 2 weeks of program end`
2. **Fine-tune if accuracy drops** — recalibrate on roughly two weeks of the new site's data, matching what the research shows is needed to transfer. `Target: following the test above`
3. **Add automated drift monitoring** — re-check accuracy and feature importances as new data comes in, to catch the model going stale before it affects a forecast. `Ongoing after deployment`
4. **Keep this repository and documentation current** — full pipeline, model, and analysis, kept reproducible and up to date.

## Streamlit App

```bash
streamlit run solarapp.py
```

**Live demo:** https://solar-energy-yield-predictor-b2somffs8gmotjcdksczlr.streamlit.app/

The UI lets users:
- Enter date and time
- Adjust irradiation and temperature values
- Generate a DC power prediction
- Inspect holdout results and feature importance

## Getting Started

```bash
git clone https://github.com/AI4ALL-21c/Solar-Energy-Yield-Predictor.git
cd Solar-Energy-Yield-Predictor
pip install -r requirements.txt
streamlit run solarapp.py
```

**Requirements:**

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
xgboost>=2.0
scikit-learn>=1.3
joblib>=1.3
requests>=2.31
plotly>=5.18
```

## Project Structure

```
Solar-Energy-Yield-Predictor/
├── solarapp.py                  # Streamlit app entry point
├── tab1.py ... tab4.py          # App tab views
├── modelutil.py                 # Model loading / inference helpers
├── pipeline_functions.py        # Feature engineering pipeline
├── physicsCalc.py, physicsutil.py  # Physics baseline model
├── weatherapi.py                # Open-Meteo API integration
├── solar_gap_model.joblib       # Trained XGBoost gap model
├── solar_gap_model_metadata.json
├── SOLAR_PLANT_DATA(...).csv    # Cleaned plant + weather dataset
├── Model.ipynb                  # Model development notebook
├── SolarPlantData.ipynb         # Data cleaning notebook
├── Visualization.ipynb          # Chart generation
├── deployment_notes.md          # Monitoring/maintenance plan
└── requirements.txt
```

## Documentation & Citations

**Data sources**

| Source | Link |
|---|---|
| Anikannal, *Solar Power Generation Data* (Kaggle) | [kaggle.com/datasets/anikannal/solar-power-generation-data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) |
| Open-Meteo — Historical Weather API | [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api) |
| Open-Meteo — Geocoding API | [open-meteo.com/en/docs/geocoding-api](https://open-meteo.com/en/docs/geocoding-api) |

**Citations**

1. Faiman, D. — PV performance model. Sandia PVPMC. [pvpmc.sandia.gov/modeling-guide/2-dc-module-iv](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/)
2. Dobos, A. P. (2014). *PVWatts Version 5 Manual.* NREL/TP-6A20-62641. [docs.nrel.gov/docs/fy14osti/62641.pdf](https://docs.nrel.gov/docs/fy14osti/62641.pdf)
3. Asante-Okyere, S., et al. (2024). Machine learning forecasting of solar PV production using single and hybrid models over different time horizons. *PMC/NCBI.* [pmc.ncbi.nlm.nih.gov/articles/PMC11002275](https://pmc.ncbi.nlm.nih.gov/articles/PMC11002275/)
4. Nature Communications (2020). Impacts of solar intermittency on future photovoltaic reliability. [nature.com/articles/s41467-020-18602-6](https://www.nature.com/articles/s41467-020-18602-6)
5. IEEE Xplore (2022). Grid integration challenges and solution strategies for solar PV systems: a review. [ieeexplore.ieee.org/document/9773105](https://ieeexplore.ieee.org/document/9773105/)
6. Pombo, D. V., et al. (2022). Increasing the accuracy of hourly multi-output solar power forecast with physics-informed machine learning. *Sensors, 22*(3), 749. [pmc.ncbi.nlm.nih.gov/articles/PMC8839153](https://pmc.ncbi.nlm.nih.gov/articles/PMC8839153/)
7. de Oliveira Santos, L., et al. (2024). Photovoltaic power estimation and forecast models integrating physics and machine learning: a review. *Solar Energy, 284.* [sciencedirect.com/science/article/pii/S0038092X24007394](https://www.sciencedirect.com/science/article/pii/S0038092X24007394)

---

## Final Presentation Polish Plan

- [ ] Confirm the app runs cleanly in the target environment
- [ ] Add sample input presets for sunny, cloudy, and low-light cases
- [ ] Save a serialized model artifact to speed startup
- [ ] Improve layout spacing and labels in the Streamlit UI
- [ ] Capture screenshots and prepare a short demo script

---
