import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset

# read in data
dataset = pd.read_csv('data_utc/Seattle/data_utc.csv')
mask = dataset['Year'] <= 2020
train_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2022
test_dataset = dataset[mask].copy()

# initialize sample and batch sizes
sample_size = len(train_dataset)
batch_size = 1
num_of_batches = sample_size / batch_size

valid_size = len(test_dataset)

# create day of year column
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
train_dataset['DayOfYear'] = train_dataset['DayOfYear'].mask((train_dataset['DayOfYear'] >= 60) & ((train_dataset['Year'] == 2016) | (train_dataset['Year'] == 2020)), train_dataset['DayOfYear']-1)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_test = test_dataset[['GHI']]

x_columns = []

# scale all input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

# create scaled instances of night to use as comparison
# night_start = np.array([[0,0,0,4,0,0,0,0,0,0,0]])
# night_end = np.array([[0,0,0,20,0,0,0,0,0,0,0]])
night_start = 3
night_end = 12

# create a mask of 9pm - 4am and get average of everything else
mask = (train_dataset['Hour'] >= 12) | (train_dataset['Hour'] <= 3)
test_average = np.mean(y_test)
the_average = np.mean(y_train[mask])
print("Mean: " + str(the_average))
mean = np.mean(y_train)
print("Overall Mean: " + str(mean))

# scale label columns
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# y_scaler = StandardScaler()
# y_train = y_scaler.fit_transform(y_train)
# y_train = y_train.reshape(1, -1)[0]
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# print(y_train)
# scaled_average = np.mean(y_train[mask])
# print("Mean of Scaled Values: " + str(scaled_average))

# y_test = y_scaler.transform(y_test)
# y_test = y_test.reshape(1, -1)[0]
# average = y_scaler.transform(average.reshape(-1,1))
# average = average.reshape(1,-1)[0][0]
# old_average = y_scaler.inverse_transform(average.reshape(-1,1))
# old_average = old_average.reshape(1,-1)[0]
# print("Scaled Mean: " + str(average))
# zero = y_scaler.transform(np.array(0).reshape(-1,1))
# zero = zero.reshape(1,-1)[0][0]
# old_zero = y_scaler.inverse_transform(zero.reshape(-1,1))
# old_zero = zero.reshape(1,-1)[0]

zero = [0]

# turn data into tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
zero = torch.FloatTensor(np.array(zero))
# old_zero = torch.FloatTensor(np.array(old_zero))
average = torch.FloatTensor(np.array([the_average]))
# print(average)
# input()
# old_average = torch.FloatTensor(np.array(old_average))

# create training and validation datasets 
dataset = TensorDataset(x_train, y_train)
validation_dataset = TensorDataset(x_test, y_test)

# set model criterion
criterion = nn.MSELoss()

# training and validation loop
epochs = 1
losses = []
valid_losses = []
guesses = []
for epoch in range(epochs):
    print("Epoch: " + str(epoch))
    # training loop
    loss_total = 0.0
    for id_batch, (x_batch, y_batch) in enumerate(dataset):
        # unscaled_y_batch = y_scaler.inverse_transform(y_batch.reshape(-1,1))
        # unscaled_y_batch = torch.FloatTensor(unscaled_y_batch.reshape(1,-1)[0])
        if x_batch[1] <= night_start or x_batch[1] >= night_end:
            loss = criterion(average, y_batch)
            guesses.append(average.item())
        else:
            guesses.append(zero.item())
            loss = criterion(zero, y_batch)
        loss_total += np.sqrt(loss.detach().numpy())
    #print("Loss: " + str(loss_total/num_of_batches))
    losses.append(loss_total/num_of_batches)

    # validation loop
    loss_total = 0.0
    for id_batch, (x_batch, y_batch) in enumerate(validation_dataset):
        # unscaled_y_batch = y_scaler.inverse_transform(y_batch.reshape(-1,1))
        # unscaled_y_batch = torch.FloatTensor(unscaled_y_batch.reshape(1,-1)[0])
        if x_batch[1] <= night_start or x_batch[1] >= night_end:
            loss = criterion(average, y_batch)
        else:
            loss = criterion(zero, y_batch)
        loss_total += np.sqrt(loss.detach().numpy())
    curr_loss = loss_total/valid_size
    valid_losses.append(curr_loss)
    print("Loss: " + str(curr_loss))
    print("NRMSE: " + str(curr_loss/test_average))

# plot the results
# plt.plot(range(epochs), losses)
# plt.plot(range(epochs), valid_losses)
# plt.ylabel("RMSE")
# plt.xlabel("Epoch")
# plt.savefig("results/baseline_results/baseline_night.pdf")
# plt.show(block=False)
# plt.close()

og_average = [150.16012720156556] * 72
new_average = [225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328, 225.2401885986328]

plt.plot(range(72), og_average)
plt.plot(range(72), y_test[:72])
plt.plot(range(72), guesses[:72])
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/baseline_results/baseline_night_avp.pdf")
plt.show(block=False)

# save result for other model graphs
# with open("results/baseline_results/baseline_night_results.txt", 'w') as file:
#     file.write(str(valid_losses[0]) + "\n")
#     file.write(str(valid_losses[0]/mean))