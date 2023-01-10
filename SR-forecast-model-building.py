import pandas as pd
import numpy as np
from prophet import Prophet

data = pd.read_csv('Virat Kohli - One Day Internationals - Performance Analysis by Year - Sheet1.csv')

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

# Saving the model
import pickle
pickle.dump(model, open('SR_model.pkl', 'wb'))