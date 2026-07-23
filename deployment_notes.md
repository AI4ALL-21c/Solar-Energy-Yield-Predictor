# Deployment Notes

## Evaluation Summary
The notebook results support selecting **XGBoost** as the final model.

- **Linear Regression:** R² ≈ 0.6957, RMSE ≈ 3340.30 kW
- **Random Forest:** R² ≈ 0.7945, RMSE ≈ 2745.00 kW
- **XGBoost:** R² ≈ 0.9628, MAE ≈ 496.15 kW

## Why XGBoost Was Chosen
- Highest R² in the notebook evaluation
- Much lower error than the baseline models
- Better at capturing nonlinear weather-to-yield relationships

## Streamlit UI
The app includes:
- Weather and time input controls
- A single-click prediction button
- A prediction result panel
- A holdout-data preview and feature-importance view

## Monitoring and Maintenance
If this were in production, I would monitor:
- Prediction drift across seasons and plant locations
- Input data quality, missing values, and sensor anomalies
- Error metrics over time against actual plant output
- Retraining frequency after weather or equipment changes

Governance would also matter:
- Version the training data and model artifacts
- Log who deployed each model version
- Document assumptions, feature definitions, and metric thresholds
- Review for fairness and safety if the model influenced operational decisions

## Refinement Plan Before Presentation
1. Verify the Streamlit app runs cleanly in the target environment.
2. Tune the visual design and add clearer input labels.
3. Add sample scenarios for sunny, cloudy, and low-light conditions.
4. Save the trained model artifact so startup is faster.
5. Add a short model-card style summary to the app.
6. Test edge cases such as zero irradiation and extreme temperatures.
7. Prepare screenshots and a 1-minute demo script.
