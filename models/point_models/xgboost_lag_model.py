import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

offset = 12 # number of rows ahead to predict
horizon = 12 # number of values to predict

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

lag_vars = [
    "GHI", "DNI", "DHI", "CSI",
    "Wind Speed", "Wind Direction",
    "Relative Humidity", "Precipitable Water"
]

lags = range(1, 12)

def build_lag_df(df, vars_to_lag):
    lagged = {}
    for var in vars_to_lag:
        for L in lags:
            lagged[f"{var}_t{L}"] = df[var].shift(L)
    return pd.DataFrame(lagged)

train_lags = build_lag_df(train_dataset, lag_vars)
valid_lags = build_lag_df(valid_dataset, lag_vars)
test_lags = build_lag_df(test_dataset, lag_vars)

train_dataset = pd.concat([train_dataset, train_lags], axis=1)
valid_dataset = pd.concat([valid_dataset, valid_lags], axis=1)
test_dataset = pd.concat([test_dataset, test_lags], axis=1)

max_lag = max(lags)
train_dataset = train_dataset.iloc[max_lag:].reset_index(drop=True)
valid_dataset = valid_dataset.iloc[max_lag:].reset_index(drop=True)
test_dataset = test_dataset.iloc[max_lag:].reset_index(drop=True)

# move up deterministic columns and place into new columns
for h in range(horizon):
    step = offset + h
    train_dataset[f"Future_GHI_{h}"] = train_dataset["GHI"].shift(-step)
    valid_dataset[f"Future_GHI_{h}"] = valid_dataset["GHI"].shift(-step)
    test_dataset[f"Future_GHI_{h}"] = test_dataset["GHI"].shift(-step)

    train_dataset[f"Future_SZA_{h}"] = train_dataset["Solar Zenith Angle"].shift(-step)
    valid_dataset[f"Future_SZA_{h}"] = valid_dataset["Solar Zenith Angle"].shift(-step)
    test_dataset[f"Future_SZA_{h}"] = test_dataset["Solar Zenith Angle"].shift(-step)

cut = offset + horizon
train_dataset = train_dataset.iloc[:-cut]
valid_dataset = valid_dataset.iloc[:-cut]
test_dataset = test_dataset.iloc[:-cut]

# all_columns = list(train_dataset.columns)

future_columns = []
for h in range(horizon):
    future_columns += [f"Future_SZA_{h}", f"Future_GHI_{h}"]

drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Solar Zenith Angle', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = drop_columns)
x_valid = valid_dataset.drop(columns = drop_columns)
x_test = test_dataset.drop(columns = drop_columns)

future_ghi_cols = [f"Future_GHI_{h}" for h in range(horizon)]
y_train = train_dataset[future_ghi_cols].to_numpy().astype(np.float32)
y_valid = valid_dataset[future_ghi_cols].to_numpy().astype(np.float32)
y_test = test_dataset[future_ghi_cols].to_numpy().astype(np.float32)

remaining_columns = list(x_train.columns)
# print(remaining_columns)

# create a mask of daytime hours to generate averages
train_mask = (train_dataset['Solar Zenith Angle'] < 90)
valid_mask = (valid_dataset['Solar Zenith Angle'] < 90)
test_mask = (test_dataset['Solar Zenith Angle'] < 90)

train_average_GHI = np.mean(train_dataset['GHI'][train_mask])
valid_average_GHI = np.mean(valid_dataset['GHI'][valid_mask])
test_average_GHI = np.mean(test_dataset['GHI'][test_mask])

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train).astype(np.float32)
x_valid = x_scaler.transform(x_valid).astype(np.float32)
x_test = x_scaler.transform(x_test).astype(np.float32)

train_weights = (train_dataset['Solar Zenith Angle'] < 90).to_numpy().astype(np.float32)
valid_weights = (valid_dataset['Solar Zenith Angle'] < 90).to_numpy().astype(np.float32)

train_pred = np.zeros_like(y_train)
valid_pred = np.zeros_like(y_valid)
test_pred = np.zeros_like(y_test)

for h in range(horizon):
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

sza_train = train_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_valid = valid_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_test = test_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()

train_true = train_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()
valid_true = valid_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()
test_true = test_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()

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

# save results
with open(f"results/point_results/xgboost_lag_{max(lags)+1}.txt", 'w') as file:
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

# plot the results
hours = np.arange(864) * (5/60) # 5 minutes to hours
actual_energy = test_true.sum(axis=1) * (5/60)
pred_energy = test_pred.sum(axis=1) * (5/60)
plt.plot(hours, actual_energy[:864], label="Actual")
plt.plot(hours, pred_energy[:864], label="Predicted")
plt.title(f"XGBoost ({max(lags)+1} Rows) Hourly Energy Pred vs Actual")
plt.legend()
plt.ylabel("Energy (Wh/m\u00b2)")
plt.xlabel("Hour")
plt.savefig(f"results/point_results/xgboost_lag_{max(lags)+1}.pdf")
plt.show(block=False)