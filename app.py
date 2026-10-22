"""Household energy forecast (Streamlit).

Predicts the next 10-minute appliance energy reading from the time and the
last 12 readings (2 hours). The model is the small Lasso saved by the notebook
in app_model.pkl.
"""
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Household Energy Forecast', page_icon='⚡')

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_model.pkl')


@st.cache_resource
def load_package():
    return joblib.load(MODEL_PATH)


package = load_package()
model = package['model']
scaler = package['scaler']
feature_names = package['feature_names']
scores = package['metrics']          # validation scores saved by the notebook

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
EXAMPLE = '60, 60, 70, 60, 60, 50, 60, 70, 80, 90, 80, 70'


def parse_readings(text):
    """Turn the pasted text into 12 numbers (oldest first)."""
    parts = [p for p in re.split(r'[,\s;]+', text.strip()) if p]
    if len(parts) != 12:
        raise ValueError(f'Please enter exactly 12 readings (you entered {len(parts)}).')
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError('The readings must be numbers, separated by commas or spaces.')
    if not np.all(np.isfinite(values)):
        raise ValueError('The readings must be ordinary numbers.')
    if min(values) < 0:
        raise ValueError('The readings cannot be negative.')
    return values


def build_features(readings, day_index, hour):
    """Build the 16 model inputs the same way the notebook does.

    readings are oldest first, so readings[-1] is the reading just before the
    one being predicted. The rolling features use only those earlier readings.
    """
    r = np.asarray(readings, dtype=float)
    row = {
        'hour': hour,
        'dayofweek': day_index,                       # Monday = 0
        'is_weekend': int(day_index in (5, 6)),
        'is_night': int(hour >= 22 or hour <= 5),
        'is_peak_hour': int(17 <= hour <= 21),
        'lag_1': r[-1], 'lag_2': r[-2], 'lag_3': r[-3], 'lag_6': r[-6], 'lag_12': r[-12],
        'rolling_mean_3': r[-3:].mean(),
        'rolling_mean_6': r[-6:].mean(),
        'rolling_mean_12': r.mean(),
        'rolling_std_6': r[-6:].std(ddof=1),          # same as pandas
        'rolling_max_6': r[-6:].max(),
        'rolling_min_6': r[-6:].min(),
    }
    return pd.DataFrame([row])[feature_names]         # error if a name does not match


def forecast(readings, day_index, hour):
    features = build_features(readings, day_index, hour)
    X = pd.DataFrame(scaler.transform(features), columns=feature_names)
    return max(0.0, float(model.predict(X)[0]))



ABOUT = """
### How it works
The app forecasts the next 10-minute appliance energy reading from the time (day of the week and hour)
and the last 12 readings, taken every 10 minutes. In the project's analysis, the recent readings were
almost everything the model relied on. Weather, room temperatures and lights added almost nothing, so
they are not asked for.

### Limits
- The model was trained on data from January to May 2016 only, so other seasons are not covered.
- Large sudden spikes are hard to predict and are usually predicted too low. In the notebook's test of
  the main model, readings above 300 Wh were predicted about 178 Wh too low on average.
- The scores shown are validation scores of this small model, not test scores.
- It is a short-term forecast and an estimate, not a measurement.

Built by Sai Kalyan.
"""

st.title('Household Energy Forecast')
st.write('Forecast the next 10-minute appliance energy reading from the time and the last 12 readings.')

now = datetime.now()
with st.form('inputs'):
    day = st.selectbox('Day of the week', DAYS, index=now.weekday())
    hour = st.slider('Hour of the reading to predict (0-23)', 0, 23, now.hour)
    readings_text = st.text_area(
        'Last 12 readings (Wh), oldest first', value=EXAMPLE, height=80,
        help='One reading every 10 minutes, separated by commas. The last number is the '
             'reading just before the one you want to predict.')
    submitted = st.form_submit_button('Predict', type='primary')

if submitted:
    try:
        readings = parse_readings(readings_text)
    except ValueError as error:
        st.error(str(error))
    else:
        prediction = forecast(readings, DAYS.index(day), int(hour))
        st.subheader(f'Predicted energy use: {prediction:.0f} Wh')
        st.write(f'For the reading at hour {int(hour)} on a {day}.')
        st.write(f'- If the next reading simply equals the last one, the guess would be {readings[-1]:.0f} Wh.')
        st.write(f'- On the validation data this model missed by {scores["MAE"]:.0f} Wh on average '
                 f'(RMSE {scores["RMSE"]:.0f} Wh).')
        if max(readings) > 1100:
            st.warning('Some readings are higher than the highest readings in the training data '
                       '(about 1,100 Wh), so this prediction is less reliable.')

st.markdown(ABOUT)