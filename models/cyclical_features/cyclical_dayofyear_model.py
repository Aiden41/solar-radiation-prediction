import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

# set flags for showing graphs and checkpointing
show_graph = False
resume = False
overwrite_fig = False
overwrite_save = True

# get baseline results to plot on graphs
try:
    # with open("results/baseline_results/baseline_results.txt", 'r') as file:
    #     baseline_loss = float(file.readline())
    with open("results/baseline_results/baseline_night_results.txt", 'r') as file:
        baseline_night_loss = float(file.readline())
except:
    print("Could not get baseline results")

# read in data
train_files = ['data/2017.csv', 'data/2018.csv', 'data/2019.csv', 'data/2020.csv']
train_dataset = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True)
test_dataset = pd.read_csv('data/2021.csv')

# create cyclical day of year columns
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
train_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear'])) 
train_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * train_dataset['DayOfYear'] / max(train_dataset['DayOfYear']))
test_dataset['DayOfYear_Sin'] = np.sin(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear'])) 
test_dataset['DayOfYear_Cos'] = np.cos(2 * np.pi * test_dataset['DayOfYear'] / max(test_dataset['DayOfYear']))

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'DayOfYear'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute', 'DayOfYear'])
y_test = test_dataset[['GHI']]

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Year', 'Day', 'Hour', 'Month', 'Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

# scale label column
y_scaler = ColumnTransformer([("scaler", StandardScaler(), y_columns)], remainder='passthrough')
y_train = y_scaler.fit_transform(y_train)
y_train = y_train.reshape(1, -1)[0]
y_test = y_scaler.transform(y_test)
y_test = y_test.reshape(1, -1)[0]

# turn data into tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# set tensors to gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# create training and validation datasets, use dataloaders for batching
batch_size = 8760 # this is one year in hours
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validation_dataset = TensorDataset(x_test, y_test)
validloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# define model and set to gpu if available
model = nn.Sequential(
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
model.to(device)

# set other various parameters
criterion = nn.MSELoss()
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
current_epoch = 1
min_loss = float('inf')
best_epoch = 0
epochs = 200
losses = []
valid_losses = []

# load a checkpoint for the model
try:
    if resume:
        state = torch.load("saves/dayofyear_saved_state")
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        current_epoch = state['epoch_num']
        min_loss = state['min_loss']
        best_epoch = state['best_epoch']
        losses = state['losses']
        valid_losses = state['valid_losses']
except:
    print("Error loading saved model, starting from beginning")

# training and validation loop
for epoch in range(epochs):
    print("Epoch: " + str(epoch+current_epoch))
    # training loop
    loss_total = 0.0
    train_count = 0
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
        y_pred = model.forward(x_batch)
        loss = criterion(y_pred.squeeze(), y_batch).cpu()
        loss_total += np.sqrt(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_count += 1
    #print("Loss: " + str(loss_total/train_count))
    losses.append(loss_total/train_count)

    # validation loop
    with torch.no_grad():
        loss_total = 0.0
        valid_count = 0
        for id_batch, (x_batch, y_batch) in enumerate(validloader):
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch).cpu()
            loss_total += np.sqrt(loss.detach().numpy())
            valid_count += 1
        curr_loss = loss_total/valid_count
        valid_losses.append(curr_loss)
        print("Loss: " + str(curr_loss))
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_epoch = epoch+current_epoch

print("Minimum Validation Loss: " + str(min_loss))
print("Best Epoch: " + str(best_epoch))

# save the model
if resume:
    state = {
        'epoch_num': epoch+current_epoch+1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'min_loss': min_loss,
        'best_epoch': best_epoch,
        'losses': losses,
        'valid_losses': valid_losses
    }
    if overwrite_save:
        torch.save(state, "saves/dayofyear_saved_state")
    else:
        save_id = 1
        while os.path.exists("saves/dayofyear_saved_state" + f"_{save_id}.pdf"):
            save_id += 1
        plt.savefig("saves/dayofyear_saved_state" + f"_{save_id}.pdf")

# create numpy arrays for graphing
losses = np.array(losses)
valid_losses = np.array(valid_losses)
# baseline_loss = np.array([baseline_loss]*(epoch+current_epoch))
baseline_night_loss = np.array([baseline_night_loss]*(epoch+current_epoch))

# plot the results
plt.plot(range(epoch+current_epoch), losses, label='Training Loss')
plt.plot(range(epoch+current_epoch), valid_losses, label='Validation Loss')
try:
    # plt.plot(range(epoch+current_epoch), baseline_loss, linestyle='dashed', label='Baseline')
    plt.plot(range(epoch+current_epoch), baseline_night_loss, linestyle='dashed', label='Night Baseline')
except:
    pass
for var in (losses, valid_losses, baseline_night_loss):
    plt.annotate('%0.4f' % var.min(), xy=(1, var.min()), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.title(f"Cyclical Day of Year Model Losses w/ {learning_rate} Learning Rate")
fig_id = 1
figure_name = f"results/model_results/dayofyear_lr{learning_rate}_epoch{epoch+current_epoch}"
if overwrite_fig or not os.path.exists(figure_name + ".pdf"):
    plt.savefig(figure_name + ".pdf")
else:
    while os.path.exists(figure_name + f"_{fig_id}.pdf"):
        fig_id += 1
    plt.savefig(figure_name + f"_{fig_id}.pdf")
plt.show(block=show_graph)
plt.close()