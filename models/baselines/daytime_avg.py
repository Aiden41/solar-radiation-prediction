import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset

# read in data
dataset = pd.read_csv('data/target/data.csv')
mask = dataset['Year'] <= 2022
train_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2023
valid_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2024
test_dataset = dataset[mask].copy()

# initialize train and batch sizes
train_size = len(train_dataset)
batch_size = 1
num_of_batches = train_size / batch_size

test_size = len(test_dataset)

# create day of year column
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
# fix leap year day count
train_dataset['DayOfYear'] = train_dataset['DayOfYear'].mask((train_dataset['DayOfYear'] >= 60) & ((train_dataset['Year'] == 2020)), train_dataset['DayOfYear']-1)

valid_dataset['DayOfYear'] = pd.to_datetime(valid_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

# same for test set
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = test_dataset['DayOfYear'].mask((test_dataset['DayOfYear'] >= 60) & ((test_dataset['Year'] == 2024)), test_dataset['DayOfYear']-1)

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

# drop unused columns and get values out of dataframe
x_train = train_dataset[['Future_SZA', 'Future_CS_GHI']]
y_train = train_dataset[['Future_CSI']]
y_train_ghi = train_dataset[['Future_GHI']]
x_valid = valid_dataset[['Future_SZA', 'Future_CS_GHI']]
y_valid = valid_dataset[['Future_CSI']]
y_valid_ghi = valid_dataset[['Future_GHI']]
x_test = test_dataset[['Future_SZA', 'Future_CS_GHI']]
y_test = test_dataset[['Future_CSI']]
y_test_ghi = test_dataset[['Future_GHI']]

# create a mask of daytime hours to generate averages
train_mask = (train_dataset['Solar Zenith Angle'] < 90)
valid_mask = (valid_dataset['Solar Zenith Angle'] < 90)
test_mask = (test_dataset['Solar Zenith Angle'] < 90)
train_average = np.mean(train_dataset['CSI'][train_mask])
# print("Daytime Average CSI: " + str(daytime_average))
valid_average = np.mean(valid_dataset['CSI'][valid_mask])
test_average = np.mean(test_dataset['CSI'][test_mask])

x_train = x_train.to_numpy()
x_valid = x_valid.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_valid = y_valid.to_numpy()
y_test = y_test.to_numpy()

zero = [0.0]

# turn data into tensors
x_train = torch.FloatTensor(x_train)
x_valid = torch.FloatTensor(x_valid)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_valid = torch.FloatTensor(y_valid)
y_test = torch.FloatTensor(y_test)
zero = torch.FloatTensor(np.array(zero))
average = torch.FloatTensor(np.array([train_average]))

# create training and validation datasets 
dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

# training validation testing loops
train_preds = []
valid_preds = []
test_preds = []
train_preds_day = []
valid_preds_day = []
test_preds_day = []
train_targets = []
valid_targets = [] 
test_targets = []

train_se = 0.0
valid_se = 0.0
test_se = 0.0

train_ae = 0.0 
valid_ae = 0.0
test_ae = 0.0

train_bias_sum = 0.0
valid_bias_sum = 0.0
test_bias_sum = 0.0

train_smape_sum = 0.0
valid_smape_sum = 0.0 
test_smape_sum = 0.0

train_smape_count = 0
valid_smape_count = 0
test_smape_count = 0

train_count = 0
valid_count = 0
test_count = 0

# training loop
for id_batch, (x_batch, y_batch) in enumerate(dataset):
    if x_batch[0] < 90:
        csi_pred = average
        pred = csi_pred * x_batch[1]
        csi_y = y_batch.item()
        y = csi_y * x_batch[1].item()
        err = pred.item() - y

        # MSE / MAE
        train_se += err**2
        train_ae += abs(err)
        train_count += 1

        # MBE
        train_bias_sum += (pred.item() - y)

        # sMAPE
        denom = abs(y) + abs(pred.item())
        if denom != 0:
            train_smape_sum += 2 * abs(err) / denom
            train_smape_count += 1

        train_preds_day.append(pred.item())
        train_targets.append(y)

    else:
        pred = zero
    
    train_preds.append(pred.item())

# validation loop
for id_batch, (x_batch, y_batch) in enumerate(valid_dataset):
    if x_batch[0] < 90:
        csi_pred = average
        pred = csi_pred * x_batch[1]
        csi_y = y_batch.item()
        y = csi_y * x_batch[1].item()
        err = pred.item() - y

        # MSE / MAE
        valid_se += err**2
        valid_ae += abs(err)
        valid_count += 1

        # MBE
        valid_bias_sum += (pred.item() - y)

        # sMAPE
        denom = abs(y) + abs(pred.item())
        if denom != 0:
            valid_smape_sum += 2 * abs(err) / denom
            valid_smape_count += 1

        valid_preds_day.append(pred.item())
        valid_targets.append(y)

    else:
        pred = zero
    
    valid_preds.append(pred.item())

# testing loop
for id_batch, (x_batch, y_batch) in enumerate(test_dataset):
    if x_batch[0] < 90:
        csi_pred = average
        pred = csi_pred * x_batch[1]
        csi_y = y_batch.item()
        y = csi_y * x_batch[1].item()
        err = pred.item() - y

        # MSE / MAE
        test_se += err**2
        test_ae += abs(err)
        test_count += 1

        # MBE
        test_bias_sum += (pred.item() - y)

        #sMAPE
        denom = abs(y) + abs(pred.item())
        if denom != 0:
            test_smape_sum += 2 * abs(err) / denom
            test_smape_count += 1

        test_preds_day.append(pred.item())
        test_targets.append(y)

    else:
        pred = zero
    
    test_preds.append(pred.item())

train_mse = train_se / train_count
valid_mse = valid_se / valid_count
test_mse = test_se / test_count

train_rmse = train_mse**0.5
valid_rmse = valid_mse**0.5
test_rmse = test_mse**0.5

train_mae = train_ae / train_count
valid_mae = valid_ae / valid_count
test_mae = test_ae / test_count

train_mbe = train_bias_sum / train_count
valid_mbe = valid_bias_sum / valid_count
test_mbe = test_bias_sum / test_count

train_smape = train_smape_sum / train_smape_count
valid_smape = valid_smape_sum / valid_smape_count
test_smape = test_smape_sum / test_smape_count

train_targets = np.asarray(train_targets)
train_preds_day = np.asarray(train_preds_day)

valid_targets = np.asarray(valid_targets)
valid_preds_day = np.asarray(valid_preds_day)

test_targets  = np.asarray(test_targets)
test_preds_day = np.asarray(test_preds_day)

train_mean_y = np.mean(train_targets)
valid_mean_y = np.mean(valid_targets)
test_mean_y = np.mean(test_targets)

train_nrmse = train_rmse / train_mean_y
valid_nrmse = valid_rmse / valid_mean_y
test_nrmse = test_rmse / test_mean_y

train_r2 = 1 - np.sum((train_targets - train_preds_day)**2) / np.sum((train_targets - train_mean_y)**2)
valid_r2 = 1 - np.sum((valid_targets - valid_preds_day)**2) / np.sum((valid_targets - valid_mean_y)**2)
test_r2 = 1 - np.sum((test_targets - test_preds_day)**2) / np.sum((test_targets - test_mean_y)**2)

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
with open("results/baseline_results/daytime_avg.txt", 'w') as file:
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
plt.plot(range(72), test_preds[:72], label="Predicted")
plt.title("Daytime Average GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/baseline_results/daytime_avg.pdf")
plt.show(block=False)