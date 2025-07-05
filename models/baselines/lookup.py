import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os

# set flags for showing graphs and checkpointing
show_graph = False
weight_graphs = False
resume = False
overwrite_save = True
loss_graphs = True
output_graphs = False
verbose = False
utc_data = True
location = 0
offset = 1 # number of hours ahead to predict
n_neighbors = 1

# get baseline results to plot on graphs
try:
    # with open("results/baseline_results/baseline_results.txt", 'r') as file:
    #     baseline_loss = float(file.readline())
    with open("results/baseline_results/baseline_night_results.txt", 'r') as file:
        baseline_night_loss = float(file.readline())
except:
    print("Could not get baseline results")

# read in data
# if utc_data == False:
#     train_files = ['data/2017.csv', 'data/2018.csv', 'data/2019.csv', 'data/2020.csv']
#     train_dataset = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True)
#     test_dataset = pd.read_csv('data/2021.csv')
# else:
#     train_files = ['data_utc/2017.csv', 'data_utc/2018.csv', 'data_utc/2019.csv', 'data_utc/2020.csv']
#     train_dataset = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True)
#     test_dataset = pd.read_csv('data_utc/2021.csv')

if location == 0:
    if utc_data == True:
        dataset = pd.read_csv('data_utc/Seattle/data_utc.csv')
    else:
        dataset = pd.read_csv('data/data.csv')
else:
    if utc_data == True:
        dataset = pd.read_csv('data_utc/Aurora/data_utc.csv')
    else:
        dataset = pd.read_csv('data/data.csv')
mask = dataset['Year'] <= 2020
train_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2021
test_dataset = dataset[mask].copy()

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

# create cyclical columns
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
train_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear'])) 
train_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear']))
test_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear'])) 
test_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear']))
# train_dataset['Sin_Hour'] = np.sin(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour'])) 
# train_dataset['Cos_Hour'] = np.cos(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour']))
# test_dataset['Sin_Hour'] = np.sin(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour'])) 
# test_dataset['Cos_Hour'] = np.cos(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour']))
# train_dataset['Sin_Month'] = np.sin(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month'])) 
# train_dataset['Cos_Month'] = np.cos(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month']))
# test_dataset['Sin_Month'] = np.sin(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month'])) 
# test_dataset['Cos_Month'] = np.cos(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month']))

last_index = len(train_dataset)
drop_rows = range(last_index-offset, last_index)
train_dataset['GHI'] = train_dataset['GHI'].shift(-offset)
train_dataset.drop(drop_rows, inplace=True)

last_index = len(test_dataset)
drop_rows = range(last_index-offset, last_index)
test_dataset['GHI'] = test_dataset['GHI'].shift(-offset)
test_dataset.drop(drop_rows, inplace=True)

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['Day', 'GHI', 'Minute', 'DayOfYear', 'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['Day', 'GHI', 'Minute', 'DayOfYear', 'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle'])
y_test = test_dataset[['GHI']]
y_mean = y_test.mean()['GHI']

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# scale label column
# y_scaler = ColumnTransformer([("scaler", StandardScaler(), y_columns)], remainder='passthrough')
# y_train = y_scaler.fit_transform(y_train)
# y_train = y_train.reshape(1, -1)[0]
# y_test = y_scaler.transform(y_test)
# y_test = y_test.reshape(1, -1)[0]
# inverse_y_scaler = y_scaler.transformers_[0][1]

# transform data for look ahead
new_x_train = []
for index, row in enumerate(x_train):
    if index < 24:
        continue
    t = row
    t_prev_day = x_train[index-24]
    t_prev_sr = y_train[index-24]
    new_row = np.array([t, t_prev_day]).flatten()
    new_row = np.append(new_row, t_prev_sr)
    new_x_train.append(new_row)
x_train = np.array(new_x_train)

new_x_test = []
for index, row in enumerate(x_test):
    if index < 24:
        continue
    t = row
    t_prev_day = x_test[index-24]
    t_prev_sr = y_test[index-24]
    new_row = np.array([t, t_prev_day]).flatten()
    new_row = np.append(new_row, t_prev_sr)
    new_x_test.append(new_row)
x_test = np.array(new_x_test)

y_train = np.delete(y_train, slice(0,24))
y_test = np.delete(y_test, slice(0,24))

model = KNeighborsRegressor(n_neighbors=n_neighbors)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
nrmse = rmse/y_mean
print(f"NRMSE: {nrmse}")

# save result for other model graphs
with open("results/baseline_results/lookup.txt", 'w') as file:
    file.write(f"RMSE: {rmse}")
    file.write(f"NRMSE: {nrmse}")