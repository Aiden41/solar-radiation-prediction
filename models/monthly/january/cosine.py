import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json


dataset = pd.read_csv('data_utc/data_utc.csv')
mask = (dataset['Year'] <= 2020) & ((dataset['Month'] == 1) | (dataset['Month'] == 12) | (dataset['Month'] == 2))
train_dataset = dataset[mask].copy()
mask = (dataset['Year'] == 2021) & ((dataset['Month'] == 1) | (dataset['Month'] == 12) | (dataset['Month'] == 2))
test_dataset = dataset[mask].copy()

# create day of year column 
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Year'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute', 'Year'])
y_test = test_dataset[['GHI']]

# reorder columns so weight graphs are aligned
column_order = ['Day', 'Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation', 'DayOfYear', 'Hour', 'Month']
x_train = x_train[column_order]
x_test = x_test[column_order]

x_columns = column_order
y_columns = list(y_train)

# scale all input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

def cosine_similarity(A, B):
    similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

dict = {}
# for hour in validation
for validation_index, validation_hour in enumerate(x_test):
    minimum_similarity = 1.0
    minimum_training_index = None 
    for training_index, training_hour in enumerate(x_train):
        similarity = cosine_similarity(validation_hour, training_hour)
        if abs(similarity) < minimum_similarity:
            minimum_similarity = similarity
            minimum_training_index = training_index
    dict[validation_index] = (minimum_similarity, minimum_training_index)
    print(validation_index, minimum_similarity)

with open('similarity.json', 'w') as file:
    json.dump(dict, file)
    