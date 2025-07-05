import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import KNeighborsRegressor
# set flags for showing graphs and checkpointing
show_graph = False
weight_graphs = False
resume = False
overwrite_save = False
loss_graphs = True
output_graphs = False
verbose = False
utc_data = True
location = 0
offset = 1 # number of hours ahead to predict

# get baseline results to plot on graphs
try:
    # with open("results/baseline_results/baseline_results.txt", 'r') as file:
    #     baseline_loss = float(file.readline())
    with open("results/baseline_results/baseline_night_results.txt", 'r') as file:
        baseline_night_loss = float(file.readline())
except:
    print("Could not get baseline results")

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
dataset = pd.get_dummies(dataset, columns=['Cloud Type'], dtype=int)
mask = (dataset['Year'] == 2014)
train_dataset = dataset[mask].copy()
mask = (dataset['Year'] != 2014) & (dataset['Year'] != 2022)
valid_dataset = dataset[mask].copy()

mask = (dataset['Year'] == 2014)
dataset_2014 = dataset[mask].copy()
mask = (dataset['Year'] == 2015)
dataset_2015 = dataset[mask].copy()
mask = (dataset['Year'] == 2016) 
dataset_2016 = dataset[mask].copy()
mask = (dataset['Year'] == 2017)
dataset_2017 = dataset[mask].copy()
mask = (dataset['Year'] == 2018) 
dataset_2018 = dataset[mask].copy()
mask = (dataset['Year'] == 2019)
dataset_2019 = dataset[mask].copy()
mask = (dataset['Year'] == 2020)
dataset_2020 = dataset[mask].copy()
mask = (dataset['Year'] == 2021)   | (dataset['Year'] == 2014)  
dataset_2021 = dataset[mask].copy()

mask = dataset['Year'] == 2022
test_dataset = dataset[mask].copy()

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
valid_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)
dataset_2014.reset_index(drop=True, inplace=True)
dataset_2015.reset_index(drop=True, inplace=True)
dataset_2016.reset_index(drop=True, inplace=True)
dataset_2017.reset_index(drop=True, inplace=True)
dataset_2018.reset_index(drop=True, inplace=True)
dataset_2019.reset_index(drop=True, inplace=True)
dataset_2020.reset_index(drop=True, inplace=True)
dataset_2021.reset_index(drop=True, inplace=True)

# create cyclical columns
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
valid_dataset['DayOfYear'] = pd.to_datetime(valid_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2014['DayOfYear'] = pd.to_datetime(dataset_2014[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2015['DayOfYear'] = pd.to_datetime(dataset_2015[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2016['DayOfYear'] = pd.to_datetime(dataset_2016[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2017['DayOfYear'] = pd.to_datetime(dataset_2017[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2018['DayOfYear'] = pd.to_datetime(dataset_2018[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2019['DayOfYear'] = pd.to_datetime(dataset_2019[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2020['DayOfYear'] = pd.to_datetime(dataset_2020[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
dataset_2021['DayOfYear'] = pd.to_datetime(dataset_2021[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

train_dataset['DayOfYear'] = train_dataset['DayOfYear'].mask((train_dataset['DayOfYear'] >= 60) & ((train_dataset['Year'] == 2016) | (train_dataset['Year'] == 2020)), train_dataset['DayOfYear']-1)
dataset_2016['DayOfYear'] = dataset_2016['DayOfYear'].mask((dataset_2016['DayOfYear'] >= 60), dataset_2016['DayOfYear']-1)
dataset_2020['DayOfYear'] = dataset_2020['DayOfYear'].mask((dataset_2020['DayOfYear'] >= 60), dataset_2020['DayOfYear']-1)
train_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear'])) 
train_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear']))
valid_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * valid_dataset['DayOfYear'] / max(valid_dataset['DayOfYear'])) 
valid_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * valid_dataset['DayOfYear'] / max(valid_dataset['DayOfYear']))
test_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear'])) 
test_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear']))
dataset_2014['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2014['DayOfYear'] / max(dataset_2014['DayOfYear'])) 
dataset_2014['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2014['DayOfYear'] / max(dataset_2014['DayOfYear']))
dataset_2015['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2015['DayOfYear'] / max(dataset_2015['DayOfYear'])) 
dataset_2015['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2015['DayOfYear'] / max(dataset_2015['DayOfYear']))
dataset_2016['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2016['DayOfYear'] / max(dataset_2016['DayOfYear'])) 
dataset_2016['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2016['DayOfYear'] / max(dataset_2016['DayOfYear']))
dataset_2017['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2017['DayOfYear'] / max(dataset_2017['DayOfYear'])) 
dataset_2017['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2017['DayOfYear'] / max(dataset_2017['DayOfYear']))
dataset_2018['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2018['DayOfYear'] / max(dataset_2018['DayOfYear'])) 
dataset_2018['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2018['DayOfYear'] / max(dataset_2018['DayOfYear']))
dataset_2019['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2019['DayOfYear'] / max(dataset_2019['DayOfYear'])) 
dataset_2019['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2019['DayOfYear'] / max(dataset_2019['DayOfYear']))
dataset_2020['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2020['DayOfYear'] / max(dataset_2020['DayOfYear'])) 
dataset_2020['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2020['DayOfYear'] / max(dataset_2020['DayOfYear']))
dataset_2021['DayOfYear_Sin'] = np.sin(2 * np.pi * dataset_2021['DayOfYear'] / max(dataset_2021['DayOfYear'])) 
dataset_2021['DayOfYear_Cos'] = np.cos(2 * np.pi * dataset_2021['DayOfYear'] / max(dataset_2021['DayOfYear']))

train_dataset['Sin_Hour'] = np.sin(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour'])) 
train_dataset['Cos_Hour'] = np.cos(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour']))
valid_dataset['Sin_Hour'] = np.sin(2 * np.pi * valid_dataset['Hour'] / max(valid_dataset['Hour'])) 
valid_dataset['Cos_Hour'] = np.cos(2 * np.pi * valid_dataset['Hour'] / max(valid_dataset['Hour']))
test_dataset['Sin_Hour'] = np.sin(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour'])) 
test_dataset['Cos_Hour'] = np.cos(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour']))
dataset_2014['Sin_Hour'] = np.sin(2 * np.pi * dataset_2014['Hour'] / max(dataset_2014['Hour'])) 
dataset_2014['Cos_Hour'] = np.cos(2 * np.pi * dataset_2014['Hour'] / max(dataset_2014['Hour']))
dataset_2015['Sin_Hour'] = np.sin(2 * np.pi * dataset_2015['Hour'] / max(dataset_2015['Hour'])) 
dataset_2015['Cos_Hour'] = np.cos(2 * np.pi * dataset_2015['Hour'] / max(dataset_2015['Hour']))
dataset_2016['Sin_Hour'] = np.sin(2 * np.pi * dataset_2016['Hour'] / max(dataset_2016['Hour'])) 
dataset_2016['Cos_Hour'] = np.cos(2 * np.pi * dataset_2016['Hour'] / max(dataset_2016['Hour']))
dataset_2017['Sin_Hour'] = np.sin(2 * np.pi * dataset_2017['Hour'] / max(dataset_2017['Hour'])) 
dataset_2017['Cos_Hour'] = np.cos(2 * np.pi * dataset_2017['Hour'] / max(dataset_2017['Hour']))
dataset_2018['Sin_Hour'] = np.sin(2 * np.pi * dataset_2018['Hour'] / max(dataset_2018['Hour'])) 
dataset_2018['Cos_Hour'] = np.cos(2 * np.pi * dataset_2018['Hour'] / max(dataset_2018['Hour']))
dataset_2019['Sin_Hour'] = np.sin(2 * np.pi * dataset_2019['Hour'] / max(dataset_2019['Hour'])) 
dataset_2019['Cos_Hour'] = np.cos(2 * np.pi * dataset_2019['Hour'] / max(dataset_2019['Hour']))
dataset_2020['Sin_Hour'] = np.sin(2 * np.pi * dataset_2020['Hour'] / max(dataset_2020['Hour'])) 
dataset_2020['Cos_Hour'] = np.cos(2 * np.pi * dataset_2020['Hour'] / max(dataset_2020['Hour']))
dataset_2021['Sin_Hour'] = np.sin(2 * np.pi * dataset_2021['Hour'] / max(dataset_2021['Hour'])) 
dataset_2021['Cos_Hour'] = np.cos(2 * np.pi * dataset_2021['Hour'] / max(dataset_2021['Hour']))

train_dataset['Sin_Month'] = np.sin(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month'])) 
train_dataset['Cos_Month'] = np.cos(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month']))
valid_dataset['Sin_Month'] = np.sin(2 * np.pi * valid_dataset['Month'] / max(valid_dataset['Month'])) 
valid_dataset['Cos_Month'] = np.cos(2 * np.pi * valid_dataset['Month'] / max(valid_dataset['Month']))
test_dataset['Sin_Month'] = np.sin(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month'])) 
test_dataset['Cos_Month'] = np.cos(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month']))
dataset_2014['Sin_Month'] = np.sin(2 * np.pi * dataset_2014['Month'] / max(dataset_2014['Month'])) 
dataset_2014['Cos_Month'] = np.cos(2 * np.pi * dataset_2014['Month'] / max(dataset_2014['Month']))
dataset_2015['Sin_Month'] = np.sin(2 * np.pi * dataset_2015['Month'] / max(dataset_2015['Month'])) 
dataset_2015['Cos_Month'] = np.cos(2 * np.pi * dataset_2015['Month'] / max(dataset_2015['Month']))
dataset_2016['Sin_Month'] = np.sin(2 * np.pi * dataset_2016['Month'] / max(dataset_2016['Month'])) 
dataset_2016['Cos_Month'] = np.cos(2 * np.pi * dataset_2016['Month'] / max(dataset_2016['Month']))
dataset_2017['Sin_Month'] = np.sin(2 * np.pi * dataset_2017['Month'] / max(dataset_2017['Month'])) 
dataset_2017['Cos_Month'] = np.cos(2 * np.pi * dataset_2017['Month'] / max(dataset_2017['Month']))
dataset_2018['Sin_Month'] = np.sin(2 * np.pi * dataset_2018['Month'] / max(dataset_2018['Month'])) 
dataset_2018['Cos_Month'] = np.cos(2 * np.pi * dataset_2018['Month'] / max(dataset_2018['Month']))
dataset_2019['Sin_Month'] = np.sin(2 * np.pi * dataset_2019['Month'] / max(dataset_2019['Month'])) 
dataset_2019['Cos_Month'] = np.cos(2 * np.pi * dataset_2019['Month'] / max(dataset_2019['Month']))
dataset_2020['Sin_Month'] = np.sin(2 * np.pi * dataset_2020['Month'] / max(dataset_2020['Month'])) 
dataset_2020['Cos_Month'] = np.cos(2 * np.pi * dataset_2020['Month'] / max(dataset_2020['Month']))
dataset_2021['Sin_Month'] = np.sin(2 * np.pi * dataset_2021['Month'] / max(dataset_2021['Month'])) 
dataset_2021['Cos_Month'] = np.cos(2 * np.pi * dataset_2021['Month'] / max(dataset_2021['Month']))



last_index = len(train_dataset)
drop_rows = range(last_index-offset, last_index)
train_dataset['oldGHI'] = train_dataset['GHI']
train_dataset['GHI'] = train_dataset['GHI'].shift(-offset)
train_dataset.drop(drop_rows, inplace=True)

last_index = len(valid_dataset)
drop_rows = range(last_index-offset, last_index)
valid_dataset['oldGHI'] = valid_dataset['GHI']
valid_dataset['GHI'] = valid_dataset['GHI'].shift(-offset)
valid_dataset.drop(drop_rows, inplace=True)

last_index = len(test_dataset)
drop_rows = range(last_index-offset, last_index)
test_dataset['oldGHI'] = test_dataset['GHI']
test_dataset['GHI'] = test_dataset['GHI'].shift(-offset)
test_dataset.drop(drop_rows, inplace=True)

last_index = len(dataset_2014)
drop_rows = range(last_index-offset, last_index)
dataset_2014['oldGHI'] = dataset_2014['GHI']
dataset_2014['GHI'] = dataset_2014['GHI'].shift(-offset)
dataset_2014.drop(drop_rows, inplace=True)
last_index = len(dataset_2015)
drop_rows = range(last_index-offset, last_index)
dataset_2015['oldGHI'] = dataset_2015['GHI']
dataset_2015['GHI'] = dataset_2015['GHI'].shift(-offset)
dataset_2015.drop(drop_rows, inplace=True)
last_index = len(dataset_2016)
drop_rows = range(last_index-offset, last_index)
dataset_2016['oldGHI'] = dataset_2016['GHI']
dataset_2016['GHI'] = dataset_2016['GHI'].shift(-offset)
dataset_2016.drop(drop_rows, inplace=True)
last_index = len(dataset_2017)
drop_rows = range(last_index-offset, last_index)
dataset_2017['oldGHI'] = dataset_2017['GHI']
dataset_2017['GHI'] = dataset_2017['GHI'].shift(-offset)
dataset_2017.drop(drop_rows, inplace=True)
last_index = len(dataset_2018)
drop_rows = range(last_index-offset, last_index)
dataset_2018['oldGHI'] = dataset_2018['GHI']
dataset_2018['GHI'] = dataset_2018['GHI'].shift(-offset)
dataset_2018.drop(drop_rows, inplace=True)
last_index = len(dataset_2019)
drop_rows = range(last_index-offset, last_index)
dataset_2019['oldGHI'] = dataset_2019['GHI']
dataset_2019['GHI'] = dataset_2019['GHI'].shift(-offset)
dataset_2019.drop(drop_rows, inplace=True)
last_index = len(dataset_2020)
drop_rows = range(last_index-offset, last_index)
dataset_2020['oldGHI'] = dataset_2020['GHI']
dataset_2020['GHI'] = dataset_2020['GHI'].shift(-offset)
dataset_2020.drop(drop_rows, inplace=True)
last_index = len(dataset_2021)
drop_rows = range(last_index-offset, last_index)
dataset_2021['oldGHI'] = dataset_2021['GHI']
dataset_2021['GHI'] = dataset_2021['GHI'].shift(-offset)
dataset_2021.drop(drop_rows, inplace=True)

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
valid_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)
dataset_2014.reset_index(drop=True, inplace=True)
dataset_2015.reset_index(drop=True, inplace=True)
dataset_2016.reset_index(drop=True, inplace=True)
dataset_2017.reset_index(drop=True, inplace=True)
dataset_2018.reset_index(drop=True, inplace=True)
dataset_2019.reset_index(drop=True, inplace=True)
dataset_2020.reset_index(drop=True, inplace=True)
dataset_2021.reset_index(drop=True, inplace=True)

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_train_oldGHI = train_dataset[['oldGHI']]
y_train = train_dataset[['GHI']]

x_valid = valid_dataset.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_valid = valid_dataset[['GHI']]
y_valid_oldGHI = valid_dataset[['oldGHI']]

x_test = test_dataset.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_test = test_dataset[['GHI']]
y_test_oldGHI = test_dataset[['oldGHI']]

x_2014 = dataset_2014.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2014 = dataset_2014[['GHI']]
y_2014_oldGHI = dataset_2014[['oldGHI']]

x_2015 = dataset_2015.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2015 = dataset_2015[['GHI']]
y_2015_oldGHI = dataset_2015[['oldGHI']]

x_2016 = dataset_2016.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2016 = dataset_2016[['GHI']]
y_2016_oldGHI = dataset_2016[['oldGHI']]

x_2017 = dataset_2017.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2017 = dataset_2017[['GHI']]
y_2017_oldGHI = dataset_2017[['oldGHI']]

x_2018 = dataset_2018.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2018 = dataset_2018[['GHI']]
y_2018_oldGHI = dataset_2018[['oldGHI']]

x_2019 = dataset_2019.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2019 = dataset_2019[['GHI']]
y_2019_oldGHI = dataset_2019[['oldGHI']]

x_2020 = dataset_2020.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2020 = dataset_2020[['GHI']]
y_2020_oldGHI = dataset_2020[['oldGHI']]

x_2021 = dataset_2021.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_2021 = dataset_2021[['GHI']]
y_2021_oldGHI = dataset_2021[['oldGHI']]

train_y_mean = y_train.mean()['GHI']
valid_y_mean = y_valid.mean()['GHI']
test_y_mean = y_test.mean()['GHI']
y_mean_2014 = y_2014.mean()['GHI']
y_mean_2015 = y_2015.mean()['GHI']
y_mean_2016 = y_2016.mean()['GHI']
y_mean_2017 = y_2017.mean()['GHI']
y_mean_2018 = y_2018.mean()['GHI']
y_mean_2019 = y_2019.mean()['GHI']
y_mean_2020 = y_2020.mean()['GHI']
y_mean_2021 = y_2021.mean()['GHI']

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation', 'Surface Albedo']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_valid = x_scaler.transform(x_valid)
x_test = x_scaler.transform(x_test)
x_2014 = x_scaler.transform(x_2014)
x_2015 = x_scaler.transform(x_2015)
x_2016 = x_scaler.transform(x_2016)
x_2017 = x_scaler.transform(x_2017)
x_2018 = x_scaler.transform(x_2018)
x_2019 = x_scaler.transform(x_2019)
x_2020 = x_scaler.transform(x_2020)
x_2021 = x_scaler.transform(x_2021)

y_train = y_train.to_numpy()
y_train_oldGHI = y_train_oldGHI.to_numpy()
y_valid = y_valid.to_numpy()
y_valid_oldGHI = y_valid_oldGHI.to_numpy()
y_test = y_test.to_numpy()
y_test_oldGHI = y_test_oldGHI.to_numpy()
y_2014 = y_2014.to_numpy()
y_2014_oldGHI = y_2014_oldGHI.to_numpy()
y_2015 = y_2015.to_numpy()
y_2015_oldGHI = y_2015_oldGHI.to_numpy()
y_2016 = y_2016.to_numpy()
y_2016_oldGHI = y_2016_oldGHI.to_numpy()
y_2017 = y_2017.to_numpy()
y_2017_oldGHI = y_2017_oldGHI.to_numpy()
y_2018 = y_2018.to_numpy()
y_2018_oldGHI = y_2018_oldGHI.to_numpy()
y_2019 = y_2019.to_numpy()
y_2019_oldGHI = y_2019_oldGHI.to_numpy()
y_2020 = y_2020.to_numpy()
y_2020_oldGHI = y_2020_oldGHI.to_numpy()
y_2021 = y_2021.to_numpy()
y_2021_oldGHI = y_2021_oldGHI.to_numpy()

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
    if index < 25:
        continue
    t = row
    t_prev_hour = x_train[index-1]
    t_prev_hour_sr = y_train[index-1]
    t_prev_day = x_train[index-24]
    t_prev_sr = y_train[index-24]
    t_prev_day2 = x_train[index-25]
    t_prev_sr2 = y_train[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_train.append(new_row)
x_train = np.array(new_x_train)

new_x_valid = []
for index, row in enumerate(x_valid):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_valid[index-1]
    t_prev_hour_sr = y_valid[index-1]
    t_prev_day = x_valid[index-24]
    t_prev_sr = y_valid[index-24]
    t_prev_day2 = x_valid[index-25]
    t_prev_sr2 = y_valid[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_valid.append(new_row)
x_valid = np.array(new_x_valid)

new_x_test = []
for index, row in enumerate(x_test):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_test[index-1]
    t_prev_hour_sr = y_test[index-1]
    t_prev_day = x_test[index-24]
    t_prev_sr = y_test[index-24]
    t_prev_day2 = x_test[index-25]
    t_prev_sr2 = y_test[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_test.append(new_row)
x_test = np.array(new_x_test)

new_x_2014 = []
for index, row in enumerate(x_2014):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2014[index-1]
    t_prev_hour_sr = y_2014[index-1]
    t_prev_day = x_2014[index-24]
    t_prev_sr = y_2014[index-24]
    t_prev_day2 = x_2014[index-25]
    t_prev_sr2 = y_2014[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2014.append(new_row)
x_2014 = np.array(new_x_2014)

new_x_2015 = []
for index, row in enumerate(x_2015):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2015[index-1]
    t_prev_hour_sr = y_2015[index-1]
    t_prev_day = x_2015[index-24]
    t_prev_sr = y_2015[index-24]
    t_prev_day2 = x_2015[index-25]
    t_prev_sr2 = y_2015[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2015.append(new_row)
x_2015 = np.array(new_x_2015)

new_x_2016 = []
for index, row in enumerate(x_2016):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2016[index-1]
    t_prev_hour_sr = y_2016[index-1]
    t_prev_day = x_2016[index-24]
    t_prev_sr = y_2016[index-24]
    t_prev_day2 = x_2016[index-25]
    t_prev_sr2 = y_2016[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2016.append(new_row)
x_2016 = np.array(new_x_2016)

new_x_2017 = []
for index, row in enumerate(x_2017):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2017[index-1]
    t_prev_hour_sr = y_2017[index-1]
    t_prev_day = x_2017[index-24]
    t_prev_sr = y_2017[index-24]
    t_prev_day2 = x_2017[index-25]
    t_prev_sr2 = y_2017[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2017.append(new_row)
x_2017 = np.array(new_x_2017)

new_x_2018 = []
for index, row in enumerate(x_2018):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2018[index-1]
    t_prev_hour_sr = y_2018[index-1]
    t_prev_day = x_2018[index-24]
    t_prev_sr = y_2018[index-24]
    t_prev_day2 = x_2018[index-25]
    t_prev_sr2 = y_2018[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2018.append(new_row)
x_2018 = np.array(new_x_2018)

new_x_2019 = []
for index, row in enumerate(x_2019):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2019[index-1]
    t_prev_hour_sr = y_2019[index-1]
    t_prev_day = x_2019[index-24]
    t_prev_sr = y_2019[index-24]
    t_prev_day2 = x_2019[index-25]
    t_prev_sr2 = y_2019[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2019.append(new_row)
x_2019 = np.array(new_x_2019)

new_x_2020 = []
for index, row in enumerate(x_2020):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2020[index-1]
    t_prev_hour_sr = y_2020[index-1]
    t_prev_day = x_2020[index-24]
    t_prev_sr = y_2020[index-24]
    t_prev_day2 = x_2020[index-25]
    t_prev_sr2 = y_2020[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2020.append(new_row)
x_2020 = np.array(new_x_2020)

new_x_2021 = []
for index, row in enumerate(x_2021):
    if index < 25:
        continue
    t = row
    t_prev_hour = x_2021[index-1]
    t_prev_hour_sr = y_2021[index-1]
    t_prev_day = x_2021[index-24]
    t_prev_sr = y_2021[index-24]
    t_prev_day2 = x_2021[index-25]
    t_prev_sr2 = y_2021[index-25]
    new_row = np.append(t, t_prev_hour)
    new_row = np.append(new_row, t_prev_hour_sr)
    new_row = np.append(new_row, t_prev_day2)
    new_row = np.append(new_row, t_prev_sr2)
    new_row = np.append(new_row, t_prev_day)
    new_row = np.append(new_row, t_prev_sr)
    new_x_2021.append(new_row)
x_2021 = np.array(new_x_2021)

y_train = np.delete(y_train, slice(0,25))
y_valid = np.delete(y_valid, slice(0,25))
y_test = np.delete(y_test, slice(0,25))
y_2014 = np.delete(y_2014, slice(0,25))
y_2015 = np.delete(y_2015, slice(0,25))
y_2016 = np.delete(y_2016, slice(0,25))
y_2017 = np.delete(y_2017, slice(0,25))
y_2018 = np.delete(y_2018, slice(0,25))
y_2019 = np.delete(y_2019, slice(0,25))
y_2020 = np.delete(y_2020, slice(0,25))
y_2021 = np.delete(y_2021, slice(0,25))

x_months = np.array_split(x_test, 12)
y_months = np.array_split(y_test, 12)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(x_2021, y_2021)

run_num = 1
folder_path = f"results/model_results/ahead/KNN_{offset}/run_"
while os.path.exists(folder_path+str(run_num)):
    run_num+=1
folder_path = folder_path+str(run_num)
os.makedirs(folder_path)

pred_2014 = model.predict(x_2014)
loss_2014 = np.sqrt(mean_squared_error(y_2014, pred_2014))
pred_2015 = model.predict(x_2015)
loss_2015 = np.sqrt(mean_squared_error(y_2015, pred_2015))
pred_2016 = model.predict(x_2016)
loss_2016 = np.sqrt(mean_squared_error(y_2016, pred_2016))
pred_2017 = model.predict(x_2017)
loss_2017 = np.sqrt(mean_squared_error(y_2017, pred_2017))
pred_2018 = model.predict(x_2018)
loss_2018 = np.sqrt(mean_squared_error(y_2018, pred_2018))
pred_2019 = model.predict(x_2019)
loss_2019 = np.sqrt(mean_squared_error(y_2019, pred_2019))
pred_2020 = model.predict(x_2020)
loss_2020 = np.sqrt(mean_squared_error(y_2020, pred_2020))
pred_2021 = model.predict(x_2021)
loss_2021 = np.sqrt(mean_squared_error(y_2021, pred_2021))

# best_month_losses = []
# for index, month in enumerate(x_months):
#     month_pred = model.predict(month)
#     best_month_losses.append(np.sqrt(mean_squared_error(y_months[index], month_pred)))

# print lowest loss and epoch
print(str(loss_2014))
#print("2014 NRMSE: " + str(loss_2014/y_mean_2014))
print(str(loss_2015))
#print("2015 NRMSE: " + str(loss_2015/y_mean_2015))
print(str(loss_2016))
#print("2016 NRMSE: " + str(loss_2016/y_mean_2016))
print(str(loss_2017))
#print("2017 NRMSE: " + str(loss_2017/y_mean_2017))
print(str(loss_2018))
#print("2018 NRMSE: " + str(loss_2018/y_mean_2018))
print(str(loss_2019))
#print("2019 NRMSE: " + str(loss_2019/y_mean_2019))
print(str(loss_2020))
#print("2020 NRMSE: " + str(loss_2020/y_mean_2020))
print(str(loss_2021))
#print("2021 NRMSE: " + str(loss_2021/y_mean_2021))
# print("Best Month Losses: " + str(best_month_losses))

with open(f"{folder_path}/results.txt", 'w') as file:
    # file.write("Training Loss: " + str(train_loss) + "\n")
    # file.write("Validation Loss: " + str(valid_loss) + "\n")
    # file.write("Testing Loss: " + str(test_loss) + "\n")
    # file.write("Training NRMSE: " + str(train_loss/train_y_mean) + "\n")
    # file.write("Validation NRMSE: " + str(valid_loss/valid_y_mean) + "\n")
    # file.write("Testing NRMSE: " + str(test_loss/test_y_mean) + "\n")
    file.write("2014 Loss: " + str(loss_2014)+ "\n")
    #file.write("2014 NRMSE: " + str(loss_2014/y_mean_2014)+ "\n")
    file.write("2015 Loss: " + str(loss_2015)+ "\n")
    #file.write("2015 NRMSE: " + str(loss_2015/y_mean_2015)+ "\n")
    file.write("2016 Loss: " + str(loss_2016)+ "\n")
    #file.write("2016 NRMSE: " + str(loss_2016/y_mean_2016)+ "\n")
    file.write("2017 Loss: " + str(loss_2017)+ "\n")
    #file.write("2017 NRMSE: " + str(loss_2017/y_mean_2017)+ "\n")
    file.write("2018 Loss: " + str(loss_2018)+ "\n")
    #file.write("2018 NRMSE: " + str(loss_2018/y_mean_2018)+ "\n")
    file.write("2019 Loss: " + str(loss_2019)+ "\n")
    #file.write("2019 NRMSE: " + str(loss_2019/y_mean_2019)+ "\n")
    file.write("2020 Loss: " + str(loss_2020)+ "\n")
    #file.write("2020 NRMSE: " + str(loss_2020/y_mean_2020)+ "\n")
    file.write("2021 Loss: " + str(loss_2021)+ "\n")
    #file.write("2021 NRMSE: " + str(loss_2021/y_mean_2021)+ "\n")
    # file.write("Best Month Losses: " + str(best_month_losses) + "\n")

# load a checkpoint for the model
# try:
#     if resume:
#         state = torch.load("saves/cyc_ahead_wide_saved_state_1")
#         model.load_state_dict(state['model_state'])
#         optimizer.load_state_dict(state['optimizer_state'])
#         current_epoch = state['epoch_num']
#         min_loss = state['min_loss']
#         best_epoch = state['best_epoch']
#         losses = state['losses']
#         valid_losses = state['valid_losses']
#         best_month_losses = state['best_month_losses']
# except:
#     print("Error loading saved model, starting from beginning")

# graph initial model weight distributions
# if weight_graphs:
#     weight_path = folder_path+"/weight_distributions"
#     os.makedirs(weight_path)
#     initial_weights = model[0].weight.detach().cpu().numpy()
#     weight_sums = [0] * 14
#     weight_squared_sums = [0] * 14
#     for node in initial_weights:
#         for i, item in enumerate(node):
#             weight_sums[i] += abs(item)
#             weight_squared_sums[i] += (item**2)

#     plt.bar(range(1,15), weight_sums, width=1)
#     plt.ylabel("Weight")
#     plt.xlabel("Node")
#     plt.title("Cyclical Model Initial Weight Distribution")
#     plt.savefig(f"{weight_path}/cyc_init_weight_dist.pdf")
#     plt.show(block=show_graph)
#     plt.close()

#     plt.bar(range(1,15), weight_squared_sums, width=1)
#     plt.ylabel("Weight")
#     plt.xlabel("Node")
#     plt.title("Cyclical Model Initial Squared Weight Distribution")
#     plt.savefig(f"{weight_path}/cyc_init_squared_weight_dist.pdf")
#     plt.show(block=show_graph)
#     plt.close()

# if output_graphs:
#     months = ["Jan", "Apr", "Jul", "Oct"]
#     output_path = folder_path+"/outputs"
#     os.makedirs(output_path)

# def compare_outputs(sample_list, expected_list, months, epoch_num):
#     model.eval()
#     for index, month in enumerate(months):
#         size = len(sample_list[index])
#         predicted = model(sample_list[index])
#         loss = criterion(predicted.squeeze(), expected_list[index]).cpu()
#         loss = np.sqrt(loss.detach().numpy())
#         predicted = predicted.cpu()
#         expected = expected_list[index].cpu()
#         # predicted = predicted.cpu().detach().numpy().reshape(-1, 1)

#         # expected = inverse_y_scaler.inverse_transform(expected).reshape(1, -1)[0]
#         # predicted = inverse_y_scaler.inverse_transform(predicted).reshape(1, -1)[0]

#         # plot expected vs actual values
#         plt.plot(range(size), expected, label='Actual')
#         plt.plot(range(size), predicted, label='Prediction')
#         plt.ylabel("GHI")
#         plt.xlabel("Hour")
#         plt.legend(loc="upper right")
#         plt.title(f"Cyclical Model Ouputs for First 72 Hours of {month} at Epoch {epoch_num}\nLoss: {loss} RMSE")
#         figure_name = f"{output_path}/{month}_{epoch_num}"
#         plt.savefig(figure_name + ".pdf")
#         plt.show(block=show_graph)
#         plt.close()

# if loss_graphs:
#     # create numpy arrays for graphing
#     losses = np.array(losses)
#     valid_losses = np.array(valid_losses)
#     # baseline_loss = np.array([baseline_loss]*(epoch+current_epoch))
#     baseline_night_loss = np.array([baseline_night_loss]*(epoch+current_epoch))

#     # plot the results
#     plt.plot(range(epoch+current_epoch), losses, label='Training Loss')
#     plt.plot(range(epoch+current_epoch), valid_losses, label='Validation Loss')
#     try:
#         # plt.plot(range(epoch+current_epoch), baseline_loss, linestyle='dashed', label='Baseline')
#         plt.plot(range(epoch+current_epoch), baseline_night_loss, linestyle='dashed', label='Night Baseline')
#     except:
#         pass
#     for var in (losses, valid_losses, baseline_night_loss):
#         plt.annotate('%0.4f' % var.min(), xy=(1, var.min()), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
#     plt.ylabel("RMSE")
#     plt.xlabel("Epoch")
#     plt.legend(loc="upper right")
#     plt.title(f"Cyclical Model Losses w/ {learning_rate} Learning Rate")
#     fig_id = 1
#     figure_name = f"{folder_path}/lr{learning_rate}_epoch{epoch+current_epoch}"
#     plt.savefig(figure_name + ".pdf")
#     plt.show(block=show_graph)
#     plt.close()