import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset

# read in data
dataset = pd.read_csv('data/target/data.csv')
mask = dataset['Year'] <= 2022
train_dataset = dataset[mask].copy()
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

# same for test set
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = test_dataset['DayOfYear'].mask((test_dataset['DayOfYear'] >= 60) & ((test_dataset['Year'] == 2024)), test_dataset['DayOfYear']-1)

# Calculate clear sky index for each row
train_dataset['CSI'] = train_dataset['GHI'] / train_dataset['Clearsky GHI']
test_dataset['CSI'] = test_dataset['GHI'] / test_dataset['Clearsky GHI']
train_dataset['CSI'] = train_dataset['CSI'].fillna(0.0)
test_dataset['CSI'] = test_dataset['CSI'].fillna(0.0)
mask = train_dataset['Solar Zenith Angle'] >= 90
train_dataset.loc[mask, 'CSI'] = 0.0
mask = test_dataset['Solar Zenith Angle'] >= 90
test_dataset.loc[mask, 'CSI'] = 0.0

# generate csi averages for each hour of each day of the year
averages = {}
for day in range(1,366):
    hours_in_day = []
    for hour in range(0,24):
        hour_to_guess = hour+1
        if hour == 23:
            hour_to_guess=0
        mask = (train_dataset['Hour'] == hour_to_guess) & (train_dataset['DayOfYear'] == day) & (train_dataset['Solar Zenith Angle'] < 90)
        csi_vals = train_dataset['CSI'][mask]
        avg = np.mean(csi_vals)
        if len(csi_vals) == 0:
            avg = 0.0
        hours_in_day.append(torch.FloatTensor([avg]))
    averages[day] = hours_in_day

# move up deterministic columns and place into new columns
train_dataset["Future_SZA"] = train_dataset["Solar Zenith Angle"].shift(-1)
test_dataset["Future_SZA"] = test_dataset["Solar Zenith Angle"].shift(-1)
train_dataset["Future_CS_GHI"] = train_dataset["Clearsky GHI"].shift(-1)
test_dataset["Future_CS_GHI"] = test_dataset["Clearsky GHI"].shift(-1)
train_dataset["Future_CS_DNI"] = train_dataset["Clearsky DNI"].shift(-1)
test_dataset["Future_CS_DNI"] = test_dataset["Clearsky DNI"].shift(-1)
train_dataset["Future_CS_DHI"] = train_dataset["Clearsky DHI"].shift(-1)
test_dataset["Future_CS_DHI"] = test_dataset["Clearsky DHI"].shift(-1)
train_dataset["Future_GHI"] = train_dataset["GHI"].shift(-1)
test_dataset["Future_GHI"] = test_dataset["GHI"].shift(-1)
train_dataset["Future_CSI"] = train_dataset["CSI"].shift(-1)
test_dataset["Future_CSI"] = test_dataset["CSI"].shift(-1)
train_dataset["Future_DOY"] = train_dataset["DayOfYear"].shift(-1)
test_dataset["Future_DOY"] = test_dataset["DayOfYear"].shift(-1)
train_dataset = train_dataset.iloc[:-1]
test_dataset = test_dataset.iloc[:-1]

# drop unused columns and get values out of dataframe
x_train = train_dataset[['Future_SZA', 'Future_CS_GHI', 'Future_DOY', 'Hour']]
y_train = train_dataset[['Future_CSI']]
y_train_ghi = train_dataset[['Future_GHI']]
x_test = test_dataset[['Future_SZA', 'Future_CS_GHI', 'Future_DOY', 'Hour']]
y_test = test_dataset[['Future_CSI']]
y_test_ghi = test_dataset[['Future_GHI']]

# create a mask of daytime hours to generate averages
train_mask = (train_dataset['Solar Zenith Angle'] < 90)
test_mask = (test_dataset['Solar Zenith Angle'] < 90)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

zero = [0.0]

# turn data into tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
zero = torch.FloatTensor(np.array(zero))

# create training and validation datasets 
dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# training and validation loop
train_preds = []
test_preds = []
train_targets = [] 
test_targets = []

train_se = 0.0
test_se = 0.0

train_ae = 0.0 
test_ae = 0.0

train_bias_sum = 0.0
test_bias_sum = 0.0

train_smape_sum = 0.0 
test_smape_sum = 0.0

train_smape_count = 0 
test_smape_count = 0

train_count = 0
test_count = 0

# training loop
for id_batch, (x_batch, y_batch) in enumerate(dataset):
    if x_batch[0] < 90:
        csi_pred = averages[int(x_batch[2].item())][int(x_batch[3].item())]
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

        train_targets.append(y)

    else:
        pred = zero
    
    train_preds.append(pred)

# testing loop
for id_batch, (x_batch, y_batch) in enumerate(test_dataset):
    if x_batch[0] < 90:
        csi_pred = averages[int(x_batch[2].item())][int(x_batch[3].item())]
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

        test_targets.append(y)

    else:
        pred = zero
    
    test_preds.append(pred)

train_mse = train_se / train_count
test_mse = test_se / test_count

train_rmse = train_mse**0.5
test_rmse = test_mse**0.5

train_mae = train_ae / train_count
test_mae = test_ae / test_count

train_mbe = train_bias_sum / train_count
test_mbe = test_bias_sum / test_count

train_smape = train_smape_sum / train_smape_count
test_smape = test_smape_sum / test_smape_count

train_targets_t = torch.tensor(train_targets)
train_preds_t = torch.tensor(train_preds[:len(train_targets)])
test_targets_t = torch.tensor(test_targets)
test_preds_t = torch.tensor(test_preds[:len(test_targets)])

train_mean_y = train_targets_t.mean().item()
test_mean_y = test_targets_t.mean().item()
train_nrmse = train_rmse / train_mean_y
test_nrmse = test_rmse / test_mean_y

train_r2 = 1 - torch.sum((train_targets_t - train_preds_t)**2) / torch.sum((train_targets_t - train_targets_t.mean())**2) 
test_r2 = 1 - torch.sum((test_targets_t - test_preds_t)**2) / torch.sum((test_targets_t - test_targets_t.mean())**2)

# print results
print("Training Error")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("NRMSE:", train_nrmse)
print("MAE:", train_mae)
print("MBE:", train_mbe)
print("sMAPE:", train_smape)
print("R^2:", train_r2.item())

print("\nTesting Error")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("NRMSE:", test_nrmse)
print("MAE:", test_mae)
print("MBE:", test_mbe)
print("sMAPE:", test_smape)
print("R^2:", test_r2.item())

# save results
with open("results/baseline_results/doy_hourly_avg.txt", 'w') as file:
    file.write("Training Error\n")
    file.write("MSE: " + str(train_mse) + "\n")
    file.write("RMSE: " + str(train_rmse) + "\n")
    file.write("NRMSE: " + str(train_nrmse) + "\n")
    file.write("MAE: " +str(train_mae) + "\n")
    file.write("MBE: " + str(train_mbe) + "\n")
    file.write("sMAPE: " + str(train_smape) + "\n")
    file.write("R^2: " + str(train_r2.item()) + "\n")

    file.write("\nTesting Error\n")
    file.write("MSE: " + str(test_mse) + "\n")
    file.write("RMSE: " + str(test_rmse) + "\n")
    file.write("NRMSE: " + str(test_nrmse) + "\n")
    file.write("MAE: " +str(test_mae) + "\n")
    file.write("MBE: " + str(test_mbe) + "\n")
    file.write("sMAPE: " + str(test_smape) + "\n")
    file.write("R^2: " + str(test_r2.item()))


# plot the results
plt.plot(range(72), y_test_ghi[:72], label="Actual")
plt.plot(range(72), test_preds[:72], label="Predicted")
plt.title("Day of Year Hourly Average GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/baseline_results/doy_hourly_avg.pdf")
plt.show(block=False)