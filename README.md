# Solar-Energy-Yield-Predictor

## Overview
This project predicts solar DC power from weather and time inputs. The notebook evaluation favored **XGBoost** as the final model because it outperformed the baseline models.

## Evaluation Summary
- **Linear Regression:** R² ≈ 0.6957, RMSE ≈ 3340.30 kW
- **Random Forest:** R² ≈ 0.7945, RMSE ≈ 2745.00 kW
- **XGBoost:** R² ≈ 0.9628, MAE ≈ 496.15 kW

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
