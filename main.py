import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.write("""
# Cricket Player's Average Forecasting App
""")

data = pd.read_csv("Virat Kohli - One Day Internationals - Performance Analysis by Year - Sheet1.csv")

st.write("""
### One Day Internationals - Performance Analysis by Year
""")
st.table(data)

#setting up the dataset
df = data[:-1]

# create data frame with needed columns
df = df[['Year', 'Avg', 'S/R']]

df['Year'] = pd.to_datetime(df['Year'])

df = df.rename(columns={'Avg': 'y', 'Year': 'ds'})

df['y_orig'] = df['y'] # to save a copy of the original data.
df['y'] = np.log(df['y'])

# Reads in saved player S/R prophet model
SR_model = pickle.load(open('SR_model.pkl', 'rb'))
# Apply SR model to make SR_future_data
SR_future_data = SR_model.make_future_dataframe(periods=1, freq='Y')
# Apply model to make predictions
SR_forecast_data = SR_model.predict(SR_future_data)
# Find the predicted S/R
predicted_SR = np.exp(list(SR_forecast_data['yhat'])[-1])

# Reads in saved player prophet model
load_model = pickle.load(open('player_model.pkl', 'rb'))
# Apply model to make future_data
future_data = load_model.make_future_dataframe(periods=1, freq='Y')

# Prepare dataset for forecasting
fut_df = df.copy()
fut_df.drop('y', inplace=True, axis=1)
fut_df.drop('y_orig', inplace=True, axis=1)

# Adding new row based on forcasted S/R
new_row = {'ds':'2022-12-31', 'S/R':predicted_SR} #S/R had been forecasted
new_fut_df = fut_df.append(new_row, ignore_index=True)

new_fut_df['ds'] = pd.to_datetime(new_fut_df['ds'])

# Apply model to make predictions
forecast_data = load_model.predict(new_fut_df)

#print(forecast_data.iloc[-1]['yhat'])
st.write("""
### Virat Kohli's upcoming year Average: 
""")
avg = np.exp(forecast_data.iloc[-1]['yhat'])
st.write(np.exp(forecast_data.iloc[-1]['yhat']))

# Player score for next 20 ODI matches
score_20 = round(avg*20)
st.write('Virat Kohli will score ', score_20, ' runs in next 20 ODI matches' )