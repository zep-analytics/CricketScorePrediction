import pandas as pd
import numpy as np
import pickle
import streamlit as st
from prophet import Prophet
from SR_forecast_model_building import initiate_SR_forecast_model
from player_model_building import initiate_player_model

st.write("""
# Cricket Player's Average Forecasting App
""")

player_lst = ['Virat Kohli', 'Rohit Sharma']

st.sidebar.header('User Selected Player')
selected_player = st.sidebar.selectbox('Player', player_lst)
print(selected_player)

if selected_player == 'Virat Kohli':
    data = pd.read_csv("Virat Kohli - One Day Internationals - Performance Analysis by Year - Sheet1.csv")
else:
    data = pd.read_csv("Rohit Sharma - Sheet1.csv")

st.write("""
### One Day Internationals - Performance Analysis by Year
""")
st.table(data)

#setting up the dataset
df = data[:-1]

# create data frame with needed columns
df = df[['Year', 'Avg', 'S/R']]
SR_df = df[['Year', 'Avg', 'S/R']]

df['Year'] = pd.to_datetime(df['Year'])
SR_df['Year'] = pd.to_datetime(SR_df['Year'])

df = df.rename(columns={'Avg': 'y', 'Year': 'ds'})
SR_df = SR_df.rename(columns={'Avg': 'y', 'Year': 'ds'})

df['y_orig'] = df['y'] # to save a copy of the original data.
df['y'] = np.log(df['y'])

# Initiate the model to predict the Player's S/R
SR_model = initiate_SR_forecast_model(data=data)
# Apply SR model to make SR_future_data
SR_future_data = SR_model.make_future_dataframe(periods=1, freq='Y')
# Apply model to make predictions
SR_forecast_data = SR_model.predict(SR_future_data)
# Find the predicted S/R
predicted_SR = np.exp(list(SR_forecast_data['yhat'])[-1])

# Initiate the model to predict the Player's Avg using S/R as the add_regressor
model = initiate_player_model(data=data)
# Apply model to make future_data
future_data = model.make_future_dataframe(periods=1, freq='Y')

# Prepare dataset for forecasting
fut_df = df.copy()
fut_df.drop('y', inplace=True, axis=1)
fut_df.drop('y_orig', inplace=True, axis=1)

# Create the new row as its own dataframe
new_row = pd.DataFrame({ 'ds': ['2022-12-31'], 'S/R': [predicted_SR] })
new_fut_df = pd.concat([fut_df, new_row])

new_fut_df['ds'] = pd.to_datetime(new_fut_df['ds'])

# Apply model to make predictions
forecast_data = model.predict(new_fut_df)

#print(forecast_data.iloc[-1]['yhat'])
st.write(selected_player, "'s upcoming year Average:")
avg = np.exp(forecast_data.iloc[-1]['yhat'])
st.write(np.exp(forecast_data.iloc[-1]['yhat']))

# Player score for next 20 ODI matches
score_20 = round(avg*20)
st.write(selected_player, ' will score ', score_20, ' runs in next 20 ODI matches' )