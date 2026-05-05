import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

offset = 1 # number of rows ahead to predict
horizon = 12 # number of values to predict

target = 'CSI' # GHI or CSI

# read in data
dataset = pd.read_csv('data/5min/row4/4/data.csv', dtype=np.float32)
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
for h in range(horizon):
    shift = offset + h
    train_dataset[f"Future_GHI_{h}"] = train_dataset["GHI"].shift(-shift)
    valid_dataset[f"Future_GHI_{h}"] = valid_dataset["GHI"].shift(-shift)
    test_dataset[f"Future_GHI_{h}"] = test_dataset["GHI"].shift(-shift)

    train_dataset[f"Future_CSI_{h}"] = train_dataset["CSI"].shift(-shift)
    valid_dataset[f"Future_CSI_{h}"] = valid_dataset["CSI"].shift(-shift)
    test_dataset[f"Future_CSI_{h}"] = test_dataset["CSI"].shift(-shift)

    train_dataset[f"Future_CS_GHI_{h}"] = train_dataset["Clearsky GHI"].shift(-shift)
    valid_dataset[f"Future_CS_GHI_{h}"] = valid_dataset["Clearsky GHI"].shift(-shift)
    test_dataset[f"Future_CS_GHI_{h}"] = test_dataset["Clearsky GHI"].shift(-shift)

    train_dataset[f"Future_SZA_{h}"] = train_dataset["Solar Zenith Angle"].shift(-shift)
    valid_dataset[f"Future_SZA_{h}"] = valid_dataset["Solar Zenith Angle"].shift(-shift)
    test_dataset[f"Future_SZA_{h}"] = test_dataset["Solar Zenith Angle"].shift(-shift)

cut = offset + horizon
train_dataset = train_dataset.iloc[:-cut]
valid_dataset = valid_dataset.iloc[:-cut]
test_dataset = test_dataset.iloc[:-cut]

# all_columns = list(train_dataset.columns)

future_columns = []
for h in range(horizon):
    future_columns += [f"Future_SZA_{h}", f"Future_GHI_{h}", f"Future_CSI_{h}", f"Future_CS_GHI_{h}"]

drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = drop_columns)
x_valid = valid_dataset.drop(columns = drop_columns)
x_test = test_dataset.drop(columns = drop_columns)

if target == "CSI":
    future_csi_cols = [f"Future_CSI_{h}" for h in range(horizon)]
    y_train = train_dataset[future_csi_cols].to_numpy().astype(np.float32)
    y_valid = valid_dataset[future_csi_cols].to_numpy().astype(np.float32)
    y_test = test_dataset[future_csi_cols].to_numpy().astype(np.float32)

    future_csghi_cols = [f"Future_CS_GHI_{h}" for h in range(horizon)]
    train_csghi = train_dataset[future_csghi_cols].to_numpy().astype(np.float32)
    valid_csghi = valid_dataset[future_csghi_cols].to_numpy().astype(np.float32)
    test_csghi = test_dataset[future_csghi_cols].to_numpy().astype(np.float32)
else:
    future_ghi_cols = [f"Future_GHI_{h}" for h in range(horizon)]
    y_train = train_dataset[future_ghi_cols].to_numpy().astype(np.float32)
    y_valid = valid_dataset[future_ghi_cols].to_numpy().astype(np.float32)
    y_test = test_dataset[future_ghi_cols].to_numpy().astype(np.float32)

# remaining_columns = list(x_train.columns)
# print(remaining_columns)

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train).astype(np.float32)
x_valid = x_scaler.transform(x_valid).astype(np.float32)
x_test = x_scaler.transform(x_test).astype(np.float32)

sza_train = train_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_valid = valid_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_test = test_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()

train_pred = np.zeros_like(y_train)
valid_pred = np.zeros_like(y_valid)
test_pred = np.zeros_like(y_test)

for h in range(horizon):
    train_weights = (sza_train[:, h] < 90).astype(np.float32)
    valid_weights = (sza_valid[:, h] < 90).astype(np.float32)

    model = XGBRegressor(n_estimators=1000, eval_metric="rmse", objective="reg:pseudohubererror", early_stopping_rounds=100, eta=0.05)

    model.fit(x_train, y_train[:, h], sample_weight=train_weights, eval_set=[(x_valid, y_valid[:, h])], 
    sample_weight_eval_set=[valid_weights], verbose=False)

    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # feature importance
    # booster = model.get_booster()

    # importance_gain = booster.get_score(importance_type='gain')
    # importance_cover = booster.get_score(importance_type='cover')
    # importance_weight = booster.get_score(importance_type='weight')

    # importance = pd.DataFrame({
    #     'feature': list(importance_gain.keys()),
    #     'gain': list(importance_gain.values()),
    #     'cover': [importance_cover.get(f, 0) for f in importance_gain.keys()],
    #     'weight': [importance_weight.get(f, 0) for f in importance_gain.keys()]
    # })

    # importance.sort_values('gain', ascending=False)

    # print(importance)

    train_pred[:, h] = model.predict(x_train)
    valid_pred[:, h] = model.predict(x_valid)
    test_pred[:, h] = model.predict(x_test)

if target == 'CSI':
    train_pred = train_pred * train_csghi
    valid_pred = valid_pred * valid_csghi
    test_pred = test_pred * test_csghi

train_true = train_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()
valid_true = valid_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()
test_true = test_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()

train_mask = sza_train < 90
valid_mask = sza_valid < 90
test_mask = sza_test < 90

test_pred = np.maximum(test_pred, 0)
valid_pred = np.maximum(valid_pred, 0)
train_pred = np.maximum(train_pred, 0)

train_pred[sza_train >= 90] = 0
valid_pred[sza_valid >= 90] = 0
test_pred[sza_test >= 90] = 0

# mask real values to daytime only
train_true_day = train_true[train_mask]
valid_true_day = valid_true[valid_mask]
test_true_day = test_true[test_mask]

# mask predictions to daytime only
train_pred_day = train_pred[train_mask]
valid_pred_day = valid_pred[valid_mask]
test_pred_day = test_pred[test_mask]

# MSE
train_mse = mean_squared_error(train_true_day, train_pred_day)
valid_mse = mean_squared_error(valid_true_day, valid_pred_day)
test_mse = mean_squared_error(test_true_day, test_pred_day)

# RMSE
train_rmse = np.sqrt(train_mse)
valid_rmse = np.sqrt(valid_mse)
test_rmse = np.sqrt(test_mse)

train_average_GHI = np.mean(train_true_day)
valid_average_GHI = np.mean(valid_true_day)
test_average_GHI = np.mean(test_true_day)

# NRMSE (normalize by daytime mean GHI)
train_nrmse = train_rmse / train_average_GHI
valid_nrmse = valid_rmse / valid_average_GHI
test_nrmse = test_rmse / test_average_GHI

# MAE
train_mae = mean_absolute_error(train_true_day, train_pred_day)
valid_mae = mean_absolute_error(valid_true_day, valid_pred_day)
test_mae = mean_absolute_error(test_true_day, test_pred_day)

# MBE
def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

train_mbe = mbe(train_true_day, train_pred_day)
valid_mbe = mbe(valid_true_day, valid_pred_day)
test_mbe = mbe(test_true_day, test_pred_day)

# sMAPE
def smape(y_true, y_pred):
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = den > 1e-6
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / den[mask])

train_smape = smape(train_true_day, train_pred_day)
valid_smape = smape(valid_true_day, valid_pred_day)
test_smape = smape(test_true_day, test_pred_day)

# R^2
train_r2 = r2_score(train_true_day, train_pred_day)
valid_r2 = r2_score(valid_true_day, valid_pred_day)
test_r2 = r2_score(test_true_day, test_pred_day)

# print results
print("GHI METRICS")
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
print("R^2:", valid_r2)

print("\nTesting Error")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("NRMSE:", test_nrmse)
print("MAE:", test_mae)
print("MBE:", test_mbe)
print("sMAPE:", test_smape)
print("R^2:", test_r2)

train_actual_energy = train_true.sum(axis=1) * (5/60)
train_pred_energy = train_pred.sum(axis=1) * (5/60)

valid_actual_energy = valid_true.sum(axis=1) * (5/60)
valid_pred_energy = valid_pred.sum(axis=1) * (5/60)

test_actual_energy = test_true.sum(axis=1) * (5/60)
test_pred_energy = test_pred.sum(axis=1) * (5/60)

train_energy_mask = (sza_train < 90).any(axis=1)
valid_energy_mask = (sza_valid < 90).any(axis=1)
test_energy_mask = (sza_test < 90).any(axis=1)

train_actual_energy_day = train_actual_energy[train_energy_mask]
train_pred_energy_day = train_pred_energy[train_energy_mask]

valid_actual_energy_day = valid_actual_energy[valid_energy_mask]
valid_pred_energy_day = valid_pred_energy[valid_energy_mask]

test_actual_energy_day = test_actual_energy[test_energy_mask]
test_pred_energy_day = test_pred_energy[test_energy_mask]

train_energy_mse = mean_squared_error(train_actual_energy_day, train_pred_energy_day)
train_energy_rmse = np.sqrt(train_energy_mse)
train_energy_nrmse = train_energy_rmse / np.mean(train_actual_energy_day)
train_energy_mae = mean_absolute_error(train_actual_energy_day, train_pred_energy_day)
train_energy_mbe = np.mean(train_pred_energy_day - train_actual_energy_day)
train_energy_smape = smape(train_actual_energy_day, train_pred_energy_day)
train_energy_r2 = r2_score(train_actual_energy_day, train_pred_energy_day)

valid_energy_mse = mean_squared_error(valid_actual_energy_day, valid_pred_energy_day)
valid_energy_rmse = np.sqrt(valid_energy_mse)
valid_energy_nrmse = valid_energy_rmse / np.mean(valid_actual_energy_day)
valid_energy_mae = mean_absolute_error(valid_actual_energy_day, valid_pred_energy_day)
valid_energy_mbe = np.mean(valid_pred_energy_day - valid_actual_energy_day)
valid_energy_smape = smape(valid_actual_energy_day, valid_pred_energy_day)
valid_energy_r2 = r2_score(valid_actual_energy_day, valid_pred_energy_day)

test_energy_mse = mean_squared_error(test_actual_energy_day, test_pred_energy_day)
test_energy_rmse = np.sqrt(test_energy_mse)
test_energy_nrmse = test_energy_rmse / np.mean(test_actual_energy_day)
test_energy_mae = mean_absolute_error(test_actual_energy_day, test_pred_energy_day)
test_energy_mbe = np.mean(test_pred_energy_day - test_actual_energy_day)
test_energy_smape = smape(test_actual_energy_day, test_pred_energy_day)
test_energy_r2 = r2_score(test_actual_energy_day, test_pred_energy_day)

print("\nENERGY METRICS")
print("Training Error")
print("MSE:", train_energy_mse)
print("RMSE:", train_energy_rmse)
print("NRMSE:", train_energy_nrmse)
print("MAE:", train_energy_mae)
print("MBE:", train_energy_mbe)
print("sMAPE:", train_energy_smape)
print("R^2:", train_energy_r2)

print("\nValidation Error")
print("MSE:", valid_energy_mse)
print("RMSE:", valid_energy_rmse)
print("NRMSE:", valid_energy_nrmse)
print("MAE:", valid_energy_mae)
print("MBE:", valid_energy_mbe)
print("sMAPE:", valid_energy_smape)
print("R^2:", valid_energy_r2)

print("\nTesting Error")
print("MSE:", test_energy_mse)
print("RMSE:", test_energy_rmse)
print("NRMSE:", test_energy_nrmse)
print("MAE:", test_energy_mae)
print("MBE:", test_energy_mbe)
print("sMAPE:", test_energy_smape)
print("R^2:", test_energy_r2)

# save results
with open(f"results/point_results/xgboost_{target.lower()}.txt", 'w') as file:
    file.write("GHI METRICS\n")
    file.write("Training Error\n")
    file.write("MSE: " + str(train_mse) + "\n")
    file.write("RMSE: " + str(train_rmse) + "\n")
    file.write("NRMSE: " + str(train_nrmse) + "\n")
    file.write("MAE: " + str(train_mae) + "\n")
    file.write("MBE: " + str(train_mbe) + "\n")
    file.write("sMAPE: " + str(train_smape) + "\n")
    file.write("R^2: " + str(train_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("MSE: " + str(valid_mse) + "\n")
    file.write("RMSE: " + str(valid_rmse) + "\n")
    file.write("NRMSE: " + str(valid_nrmse) + "\n")
    file.write("MAE: " + str(valid_mae) + "\n")
    file.write("MBE: " + str(valid_mbe) + "\n")
    file.write("sMAPE: " + str(valid_smape) + "\n")
    file.write("R^2: " + str(valid_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("MSE: " + str(test_mse) + "\n")
    file.write("RMSE: " + str(test_rmse) + "\n")
    file.write("NRMSE: " + str(test_nrmse) + "\n")
    file.write("MAE: " + str(test_mae) + "\n")
    file.write("MBE: " + str(test_mbe) + "\n")
    file.write("sMAPE: " + str(test_smape) + "\n")
    file.write("R^2: " + str(test_r2) + "\n")

    file.write("\nENERGY METRICS\n")
    file.write("Training Error\n")
    file.write("MSE: " + str(train_energy_mse) + "\n")
    file.write("RMSE: " + str(train_energy_rmse) + "\n")
    file.write("NRMSE: " + str(train_energy_nrmse) + "\n")
    file.write("MAE: " + str(train_energy_mae) + "\n")
    file.write("MBE: " + str(train_energy_mbe) + "\n")
    file.write("sMAPE: " + str(train_energy_smape) + "\n")
    file.write("R^2: " + str(train_energy_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("MSE: " + str(valid_energy_mse) + "\n")
    file.write("RMSE: " + str(valid_energy_rmse) + "\n")
    file.write("NRMSE: " + str(valid_energy_nrmse) + "\n")
    file.write("MAE: " + str(valid_energy_mae) + "\n")
    file.write("MBE: " + str(valid_energy_mbe) + "\n")
    file.write("sMAPE: " + str(valid_energy_smape) + "\n")
    file.write("R^2: " + str(valid_energy_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("MSE: " + str(test_energy_mse) + "\n")
    file.write("RMSE: " + str(test_energy_rmse) + "\n")
    file.write("NRMSE: " + str(test_energy_nrmse) + "\n")
    file.write("MAE: " + str(test_energy_mae) + "\n")
    file.write("MBE: " + str(test_energy_mbe) + "\n")
    file.write("sMAPE: " + str(test_energy_smape) + "\n")
    file.write("R^2: " + str(test_energy_r2) + "\n")

# plot the results
hours = np.arange(864) * (5/60) # 5 minutes to hours
plt.plot(hours, test_actual_energy[:864], label="Actual")
plt.plot(hours, test_pred_energy[:864], label="Predicted")
plt.title("XGBoost Hourly Energy Pred vs Actual")
plt.legend()
plt.ylabel("Energy (Wh/m\u00b2)")
plt.xlabel("Hour")
plt.savefig(f"results/point_results/xgboost_{target.lower()}.pdf")
plt.show(block=False)