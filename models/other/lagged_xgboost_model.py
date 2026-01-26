import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

offset = 1 # number of hours ahead to predict
lagged_hours = 1 # number of hours behind to append to the row

# read in data
dataset = pd.read_csv('data/target/data.csv')
dataset = pd.get_dummies(dataset, columns=['Cloud Type'], dtype=int)
mask = dataset['Year'] <= 2022
train_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2023
valid_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2024
test_dataset = dataset[mask].copy()

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
valid_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)

# create day of year column
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
# fix leap year day count
train_dataset['DayOfYear'] = train_dataset['DayOfYear'].mask((train_dataset['DayOfYear'] >= 60) & ((train_dataset['Year'] == 2020)), train_dataset['DayOfYear']-1)

valid_dataset['DayOfYear'] = pd.to_datetime(valid_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

# same for test set
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = test_dataset['DayOfYear'].mask((test_dataset['DayOfYear'] >= 60) & ((test_dataset['Year'] == 2024)), test_dataset['DayOfYear']-1)

# create cyclical columns
train_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear'])) 
train_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear']))
valid_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * valid_dataset['DayOfYear'] / max(valid_dataset['DayOfYear'])) 
valid_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * valid_dataset['DayOfYear'] / max(valid_dataset['DayOfYear']))
test_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear'])) 
test_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear']))

train_dataset['Sin_Hour'] = np.sin(2 * np.pi * train_dataset['Hour'] / (max(train_dataset['Hour'])+1)) 
train_dataset['Cos_Hour'] = np.cos(2 * np.pi * train_dataset['Hour'] / (max(train_dataset['Hour'])+1))
valid_dataset['Sin_Hour'] = np.sin(2 * np.pi * valid_dataset['Hour'] / (max(valid_dataset['Hour'])+1))
valid_dataset['Cos_Hour'] = np.cos(2 * np.pi * valid_dataset['Hour'] / (max(valid_dataset['Hour'])+1))
test_dataset['Sin_Hour'] = np.sin(2 * np.pi * test_dataset['Hour'] / (max(test_dataset['Hour'])+1))
test_dataset['Cos_Hour'] = np.cos(2 * np.pi * test_dataset['Hour'] / (max(test_dataset['Hour'])+1))

train_dataset['Sin_Month'] = np.sin(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month'])) 
train_dataset['Cos_Month'] = np.cos(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month']))
valid_dataset['Sin_Month'] = np.sin(2 * np.pi * valid_dataset['Month'] / max(valid_dataset['Month'])) 
valid_dataset['Cos_Month'] = np.cos(2 * np.pi * valid_dataset['Month'] / max(valid_dataset['Month']))
test_dataset['Sin_Month'] = np.sin(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month']))
test_dataset['Cos_Month'] = np.cos(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month']))

# Calculate clear sky index for each row
train_dataset['CSI'] = train_dataset['GHI'] / train_dataset['Clearsky GHI']
valid_dataset['CSI'] = valid_dataset['GHI'] / valid_dataset['Clearsky GHI']
test_dataset['CSI'] = test_dataset['GHI'] / test_dataset['Clearsky GHI']
train_dataset['CSI'] = train_dataset['CSI'].fillna(0.0)
valid_dataset['CSI'] = valid_dataset['CSI'].fillna(0.0)
test_dataset['CSI'] = test_dataset['CSI'].fillna(0.0)
mask = train_dataset['Solar Zenith Angle'] >= 90
train_dataset.loc[mask, 'CSI'] = 0.0
mask = valid_dataset['Solar Zenith Angle'] >= 90
valid_dataset.loc[mask, 'CSI'] = 0.0
mask = test_dataset['Solar Zenith Angle'] >= 90
test_dataset.loc[mask, 'CSI'] = 0.0

# move up deterministic columns and place into new columns
train_dataset["Future_SZA"] = train_dataset["Solar Zenith Angle"].shift(-1)
valid_dataset["Future_SZA"] = valid_dataset["Solar Zenith Angle"].shift(-1)
test_dataset["Future_SZA"] = test_dataset["Solar Zenith Angle"].shift(-1)

train_dataset["Future_CS_GHI"] = train_dataset["Clearsky GHI"].shift(-1)
valid_dataset["Future_CS_GHI"] = valid_dataset["Clearsky GHI"].shift(-1)
test_dataset["Future_CS_GHI"] = test_dataset["Clearsky GHI"].shift(-1)

train_dataset["Future_CS_DNI"] = train_dataset["Clearsky DNI"].shift(-1)
valid_dataset["Future_CS_DNI"] = valid_dataset["Clearsky DNI"].shift(-1)
test_dataset["Future_CS_DNI"] = test_dataset["Clearsky DNI"].shift(-1)

train_dataset["Future_CS_DHI"] = train_dataset["Clearsky DHI"].shift(-1)
valid_dataset["Future_CS_DHI"] = valid_dataset["Clearsky DHI"].shift(-1)
test_dataset["Future_CS_DHI"] = test_dataset["Clearsky DHI"].shift(-1)

train_dataset["Future_GHI"] = train_dataset["GHI"].shift(-1)
valid_dataset["Future_GHI"] = valid_dataset["GHI"].shift(-1)
test_dataset["Future_GHI"] = test_dataset["GHI"].shift(-1)

train_dataset["Future_CSI"] = train_dataset["CSI"].shift(-1)
valid_dataset["Future_CSI"] = valid_dataset["CSI"].shift(-1)
test_dataset["Future_CSI"] = test_dataset["CSI"].shift(-1)

train_dataset = train_dataset.iloc[:-1]
valid_dataset = valid_dataset.iloc[:-1]
test_dataset = test_dataset.iloc[:-1]

# all_columns = list(train_dataset.columns)

future_columns = ['Future_SZA', 'Future_CS_GHI', 'Future_CS_DNI', 'Future_CS_DHI', 'Future_GHI', 'Future_CSI']
drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Solar Zenith Angle', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = drop_columns)
y_train = train_dataset[['Future_CSI']]
y_train_ghi = train_dataset[['Future_GHI']]

x_valid = valid_dataset.drop(columns = drop_columns)
y_valid = valid_dataset[['Future_CSI']]
y_valid_ghi = valid_dataset[['Future_GHI']]

x_test = test_dataset.drop(columns = drop_columns)
y_test = test_dataset[['Future_CSI']]
y_test_ghi = test_dataset[['Future_GHI']]

y_train_ghi = y_train_ghi.to_numpy().flatten()
y_valid_ghi = y_valid_ghi.to_numpy().flatten()
y_test_ghi = y_test_ghi.to_numpy().flatten()

remaining_columns = list(x_train.columns)
# print(remaining_columns)

# create a mask of daytime hours to generate averages
train_mask = (train_dataset['Solar Zenith Angle'] < 90)
valid_mask = (valid_dataset['Solar Zenith Angle'] < 90)
test_mask = (test_dataset['Solar Zenith Angle'] < 90)
daytime_average = np.mean(train_dataset['CSI'][train_mask])
train_average_GHI = np.mean(train_dataset['GHI'][train_mask])
valid_average = np.mean(valid_dataset['CSI'][valid_mask])
valid_average_GHI = np.mean(valid_dataset['GHI'][valid_mask])
test_average = np.mean(test_dataset['CSI'][test_mask])
test_average_GHI = np.mean(test_dataset['GHI'][test_mask])

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_valid = x_scaler.transform(x_valid)
x_test = x_scaler.transform(x_test)

y_train = y_train.to_numpy()
y_valid = y_valid.to_numpy()
y_test = y_test.to_numpy()

train_weights = np.where(train_dataset['Solar Zenith Angle'] >= 90, 0.0, 1.0)
valid_weights = np.where(valid_dataset['Solar Zenith Angle'] >= 90, 0.0, 1.0)

def stack_lagged_rows(X, lagged_hours):
    T, D = X.shape
    stacked = []
    for t in range(lagged_hours, T):
        window = [X[t]]
        for lag in range(1, lagged_hours + 1):
            window.append(X[t - lag])
        stacked.append(np.concatenate(window))
    return np.vstack(stacked)

x_train = stack_lagged_rows(x_train, lagged_hours)
x_valid = stack_lagged_rows(x_valid, lagged_hours)
x_test = stack_lagged_rows(x_test, lagged_hours)

future_cs_ghi_train = train_dataset['Future_CS_GHI'].to_numpy()[lagged_hours:]
future_cs_ghi_valid = valid_dataset['Future_CS_GHI'].to_numpy()[lagged_hours:]
future_cs_ghi_test = test_dataset['Future_CS_GHI'].to_numpy()[lagged_hours:]

y_train_ghi = y_train_ghi[lagged_hours:]
y_valid_ghi = y_valid_ghi[lagged_hours:]
y_test_ghi = y_test_ghi[lagged_hours:]

y_train = y_train[lagged_hours:]
y_valid = y_valid[lagged_hours:]
y_test = y_test[lagged_hours:]

train_weights = train_weights[lagged_hours:]
valid_weights = valid_weights[lagged_hours:]

model = XGBRegressor(n_estimators=1000, eval_metric='rmse', early_stopping_rounds=100, eta=0.05)
model.fit(x_train, y_train, sample_weight=train_weights, eval_set=[(x_train, y_train), (x_valid, y_valid)], sample_weight_eval_set=[train_weights, valid_weights], verbose=False)
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

train_pred = model.predict(x_train)
valid_pred = model.predict(x_valid)
test_pred = model.predict(x_test)

train_mask_sza = (train_dataset['Future_SZA'].to_numpy()[lagged_hours:] >= 90)
valid_mask_sza = (valid_dataset['Future_SZA'].to_numpy()[lagged_hours:] >= 90)
test_mask_sza = (test_dataset['Future_SZA'].to_numpy()[lagged_hours:] >= 90)

train_pred[train_mask_sza] = 0
valid_pred[valid_mask_sza] = 0
test_pred[test_mask_sza] = 0

train_pred_ghi = train_pred * future_cs_ghi_train
valid_pred_ghi = valid_pred * future_cs_ghi_valid
test_pred_ghi = test_pred * future_cs_ghi_valid

# MSE
train_mse = mean_squared_error(y_train_ghi, train_pred_ghi)
valid_mse = mean_squared_error(y_valid_ghi, valid_pred_ghi)
test_mse = mean_squared_error(y_test_ghi, test_pred_ghi)

# RMSE
train_rmse = np.sqrt(train_mse)
valid_rmse = np.sqrt(valid_mse)
test_rmse = np.sqrt(test_mse)

# NRMSE
train_nrmse = train_rmse / train_average_GHI
valid_nrmse = valid_rmse / valid_average_GHI
test_nrmse = test_rmse / test_average_GHI

# MAE
train_mae = mean_absolute_error(y_train_ghi, train_pred_ghi)
valid_mae = mean_absolute_error(y_valid_ghi, valid_pred_ghi)
test_mae = mean_absolute_error(y_test_ghi, test_pred_ghi)

# MBE
def mbe(y_true, y_pred): 
    return np.mean(y_pred - y_true)

train_mbe = mbe(y_train_ghi, train_pred_ghi)
valid_mbe = mbe(y_valid_ghi, valid_pred_ghi)
test_mbe = mbe(y_test_ghi, test_pred_ghi)

# sMAPE
def smape(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = den > 1e-6

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / den[mask])

train_smape = smape(y_train, train_pred)
valid_smape = smape(y_valid, valid_pred)
test_smape = smape(y_test, test_pred)

# R^2
train_r2 = r2_score(y_train_ghi, train_pred_ghi)
valid_r2 = r2_score(y_valid_ghi, valid_pred_ghi)
test_r2 = r2_score(y_test_ghi, test_pred_ghi)

# print results
print("Training Error")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("NRMSE:", train_nrmse)
print("MAE:", train_mae)
print("MBE:", train_mbe)
print("sMAPE:", train_smape)
print("R^2:", train_r2)

print("\nValidation Error")
print("MSE:", valid_mse)
print("RMSE:", valid_rmse)
print("NRMSE:", valid_nrmse)
print("MAE:", valid_mae)
print("MBE:", valid_mbe)
print("sMAPE:", valid_smape)
print("R^2:", test_r2)

print("\nTesting Error")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("NRMSE:", test_nrmse)
print("MAE:", test_mae)
print("MBE:", test_mbe)
print("sMAPE:", test_smape)
print("R^2:", test_r2)

# save results
with open(f"results/other/lagged_xgboost/lagged_xgboost_{lagged_hours}.txt", 'w') as file:
    file.write("Training Error\n")
    file.write("MSE: " + str(train_mse) + "\n")
    file.write("RMSE: " + str(train_rmse) + "\n")
    file.write("NRMSE: " + str(train_nrmse) + "\n")
    file.write("MAE: " +str(train_mae) + "\n")
    file.write("MBE: " + str(train_mbe) + "\n")
    file.write("sMAPE: " + str(train_smape) + "\n")
    file.write("R^2: " + str(train_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("MSE: " + str(valid_mse) + "\n")
    file.write("RMSE: " + str(valid_rmse) + "\n")
    file.write("NRMSE: " + str(valid_nrmse) + "\n")
    file.write("MAE: " +str(valid_mae) + "\n")
    file.write("MBE: " + str(valid_mbe) + "\n")
    file.write("sMAPE: " + str(valid_smape) + "\n")
    file.write("R^2: " + str(valid_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("MSE: " + str(test_mse) + "\n")
    file.write("RMSE: " + str(test_rmse) + "\n")
    file.write("NRMSE: " + str(test_nrmse) + "\n")
    file.write("MAE: " +str(test_mae) + "\n")
    file.write("MBE: " + str(test_mbe) + "\n")
    file.write("sMAPE: " + str(test_smape) + "\n")
    file.write("R^2: " + str(test_r2))

# plot the results
plt.plot(range(72), y_test_ghi[:72], label="Actual")
plt.plot(range(72), test_pred_ghi[:72], label="Predicted")
plt.title("XGBoost GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig(f"results/other/lagged_xgboost/lagged_xgboost_{lagged_hours}.pdf")
plt.show(block=False)