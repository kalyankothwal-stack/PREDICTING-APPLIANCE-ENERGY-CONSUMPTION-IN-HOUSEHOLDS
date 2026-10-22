# Predicting Appliance Energy Consumption in Households

I built this project during my Data Science internship at Spinnaker Analytics. The goal is to predict how much energy a household's appliances will use in the next 10 minutes, using the time of day and the most recent readings.

## Data

- Household readings every 10 minutes, from January to May 2016
- 19,735 rows, no gaps in the timeline
- Target: `Appliances` (appliance energy use per 10-minute reading)
- Other columns: indoor and outdoor temperature and humidity, weather, and lights
- The data file is not included in this repo. To run the notebook, put `energydata_complete.xlsx` in the same folder as the notebook.

## How the data was split

The split is by time, not random, because the model must predict the future from the past:

| Part | Rows |
|---|---|
| Train (70%) | 13,814 |
| Validation (15%) | 2,960 |
| Test (15%) | 2,961 |

The test set was used once, at the end. The scaler was fitted on the training data only.

## What I did

1. Explored the data (distributions, outliers, correlations, daily and weekly patterns)
2. Built features from the time and from earlier readings (lags and rolling averages, all using only past readings)
3. Compared a previous-reading baseline, linear models (Linear, Ridge, Lasso), tree models, ARIMA and an LSTM
4. Tuned the best models with GridSearchCV and TimeSeriesSplit
5. Checked stability with a weekly backtest (8 weeks, using train and validation data only)
6. Checked which features matter with permutation importance
7. Scored the final model once on the test set

## Results (test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Previous-reading baseline | 66.84 | 26.738 | 0.459 |
| Lasso (final model) | 60.07 | 26.795 | 0.563 |

- Lasso is better than the baseline on RMSE and R². The MAE is about the same, which means it helps mostly on larger errors.
- In the weekly backtest, Lasso beat ARIMA in 7 of 8 weeks, but only by a small margin (mean RMSE 63.19 vs 63.73).
- The recent readings carry almost all of the predictive power. Weather, room temperatures and lights add very little.
- Large sudden spikes are hard to predict and are usually predicted too low.
- I do not cap or remove the high readings, so these scores are harder than ones from a capped target, and they cannot be compared with scores that do.

## Limits

- Only five months of data (January to May 2016), so other seasons are not covered
- A short-term forecast: it predicts the next 10 minutes, not days ahead
- Spikes are underestimated

## The app

**Live app:** [Add your link here](https://your-app-link-here)

`app.py` is a small Streamlit app. You enter the day, the hour and the last 12 readings (oldest first), and it predicts the next reading. It uses a smaller Lasso model saved in `app_model.pkl` (16 inputs). Its validation RMSE is 59.8 Wh, against 59.4 Wh for the full Lasso. The scores the app shows are validation scores, not test scores.

To run it:

```
pip install -r requirements.txt
streamlit run app.py
```

`app.py` and `app_model.pkl` must be in the same folder.

## Files

- `PREDICTING-APPLIANCE-ENERGY-CONSUMPTION-PROJECT-BY-KALYAN.ipynb`: the full analysis, from exploration to the final test
- `app.py`: the Streamlit app
- `final_model.pkl`: the full Lasso model from the notebook, scored once on the test set
- `app_model.pkl`: the small model used by the app
- `requirements.txt`: the few packages the app needs (used by Streamlit Cloud)
- `requirements-notebook.txt`: all the packages needed to run the notebook

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, TensorFlow/Keras, Plotly, Streamlit

Built by Sai Kalyan