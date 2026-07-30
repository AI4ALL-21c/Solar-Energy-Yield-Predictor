# Solar-Energy-Yield-Predictor

## Overview
This project predicts solar DC power from weather and time inputs. The notebook evaluation favored **XGBoost** as the final model because it outperformed the baseline models.

## Official Model Metrics

These are the official metrics for the retrained XGBoost gap model using `solar_plant_data_cleaned.csv`.

| Evaluation | RMSE | MAE | R² |
|---|---:|---:|---:|
| 80/20 Holdout | 0.0488 gap/kWp | 0.0177 gap/kWp | 0.7910 |
| 5-Fold Cross-Validation | 0.0466 ± 0.0014 gap/kWp | — | — |
| AC Power Backtest | 1245.20 kW | 431.84 kW | 0.9607 |

## Streamlit App
Run the interactive app with:

```bash
streamlit run app.py
```

The UI lets users:
- Enter date and time
- Adjust irradiation and temperature values
- Generate a DC power prediction
- Inspect holdout results and feature importance

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Monitoring and Maintenance
- Track prediction drift by season and weather pattern
- Check input data quality and sensor anomalies regularly
- Re-evaluate MAE and R² after model updates
- Version the dataset, model, and feature definitions

## Final Presentation Polish Plan
1. Confirm the app runs cleanly in the target environment.
2. Add sample input presets for sunny, cloudy, and low-light cases.
3. Save a serialized model artifact to speed startup.
4. Improve layout spacing and labels in the Streamlit UI.
5. Capture screenshots and prepare a short demo script.
