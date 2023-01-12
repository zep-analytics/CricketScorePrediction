import pandas as pd
import numpy as np
from prophet import Prophet

def initiate_SR_forecast_model(data):
    # Removing the last row from data table
    df = data[:-1]

    # create data frame with needed columns
    df = df[['Year', 'S/R']]

    # converting the Year column into datetime
    df['Year'] = pd.to_datetime(df['Year'])

    # rename the column names
    df = df.rename(columns={'S/R': 'y', 'Year': 'ds'})

    df['y_orig'] = df['y'] # to save a copy of the original data.
    df['y'] = np.log(df['y'])

    model = Prophet() #instantiate Prophet
    model.fit(df)
    return model
