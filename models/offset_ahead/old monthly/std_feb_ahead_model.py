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
weight_graphs = False
resume = False
overwrite_save = True
loss_graphs = True
output_graphs = True
verbose = False
utc_data = True
offset = 1 # number of hours ahead to predict

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

if utc_data == True:
    dataset = pd.read_csv('data_utc/data_utc.csv')
else:
    dataset = pd.read_csv('data/data.csv')
mask = (dataset['Year'] <= 2020) & ((dataset['Month'] == 12) | (dataset['Month'] == 1) | (dataset['Month'] == 2))
train_dataset = dataset[mask].copy()
mask = (dataset['Year'] == 2021) & (dataset['Month'] == 2)
test_dataset = dataset[mask].copy()
train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

# create day of year column 
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

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
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Year'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute', 'Year'])
y_test = test_dataset[['GHI']]
y_mean = y_test.mean()['GHI']

# reorder columns so weight graphs are aligned
column_order = ['Day', 'Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation', 'DayOfYear', 'Hour', 'Month']
x_train = x_train[column_order]
x_test = x_test[column_order]

# get column titles for ColumnTransformer
x_columns = column_order
y_columns = list(y_train)

# scale all input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# transform data for look ahead
new_x_train = []
for index, row in enumerate(x_train):
    if index < 24:
        continue
    t = row
    t_prev_day = x_train[index-24]
    t_prev_offset = y_train[index-24]
    new_row = np.array([t, t_prev_day]).flatten()
    new_row = np.append(new_row, t_prev_offset)
    new_x_train.append(new_row)
x_train = np.array(new_x_train)

new_x_test = []
for index, row in enumerate(x_test):
    if index < 24:
        continue
    t = row
    t_prev_day = x_test[index-24]
    t_prev_offset = y_test[index-24]
    new_row = np.array([t, t_prev_day]).flatten()
    new_row = np.append(new_row, t_prev_offset)
    new_x_test.append(new_row)
x_test = np.array(new_x_test)

y_train = np.delete(y_train, slice(0,24))
y_test = np.delete(y_test, slice(0,24))

# scale label column 
# y_scaler = ColumnTransformer([("scaler", StandardScaler(), y_columns)], remainder='passthrough')
# y_train = y_scaler.fit_transform(y_train)
# y_train = y_train.reshape(1, -1)[0]
# y_test = y_scaler.transform(y_test)
# y_test = y_test.reshape(1, -1)[0]
# inverse_y_scaler = y_scaler.transformers_[0][1]

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

# jan_sample_1 = x_test[:72]
# jan_expected_1 = y_test[:72]
# jan_sample_2 = x_test[72:144]
# jan_expected_2 = y_test[72:144]
# jan_sample_3 = x_test[144:216]
# jan_expected_3 = y_test[144:216]
# jan_sample_4 = x_test[216:288]
# jan_expected_4 = y_test[216:288]
# jan_sample_5 = x_test[288:360]
# jan_expected_5 = y_test[288:360]
# jan_sample_6 = x_test[360:432]
# jan_expected_6 = y_test[360:432]
# jan_sample_7 = x_test[432:504]
# jan_expected_7 = y_test[432:504]
# jan_sample_8 = x_test[504:576]
# jan_expected_8 = y_test[504:576]
# jan_sample_9 = x_test[576:648]
# jan_expected_9 = y_test[576:648]
# jan_sample_10 = x_test[648:720]
# jan_expected_10 = y_test[648:720]
# validation_sample_list = [jan_sample_1, jan_sample_2, jan_sample_3, jan_sample_4, jan_sample_5, jan_sample_6, jan_sample_7, jan_sample_8, jan_sample_9, jan_sample_10]
# validation_expected_list = [jan_expected_1, jan_expected_2, jan_expected_3, jan_expected_4, jan_expected_5, jan_expected_6, jan_expected_7, jan_expected_8, jan_expected_9, jan_expected_10]

feb_sample_1 = x_train[:72]
feb_expected_1 = y_train[:72]
feb_sample_2 = x_train[72:144]
feb_expected_2 = y_train[72:144]
feb_sample_3 = x_train[144:216]
feb_expected_3 = y_train[144:216]
feb_sample_4 = x_train[216:288]
feb_expected_4 = y_train[216:288]
feb_sample_5 = x_train[288:360]
feb_expected_5 = y_train[288:360]
feb_sample_6 = x_train[360:432]
feb_expected_6 = y_train[360:432]
feb_sample_7 = x_train[432:504]
feb_expected_7 = y_train[432:504]
feb_sample_8 = x_train[504:576]
feb_expected_8 = y_train[504:576]
feb_sample_9 = x_train[576:648]
feb_expected_9 = y_train[576:648]
feb_sample_10 = x_train[648:720]
feb_expected_10 = y_train[648:720]
training_sample_list = [feb_sample_1, feb_sample_2, feb_sample_3, feb_sample_4, feb_sample_5, feb_sample_6, feb_sample_7, feb_sample_8, feb_sample_9, feb_sample_10]
training_expected_list = [feb_expected_1, feb_expected_2, feb_expected_3, feb_expected_4, feb_expected_5, feb_expected_6, feb_expected_7, feb_expected_8, feb_expected_9, feb_expected_10]
# for index, hour in enumerate(x_test):
#     if index % 72 != 0 or index==0:
#         curr_sample.append(hour)
#         curr_expected.append(y_test[index])
#     else:
#         curr_sample = torch.FloatTensor(np.array(curr_sample))
#         curr_expected = torch.FloatTensor(np.array(curr_expected))
#         sample_list.append(curr_sample)
#         expected_list.append(curr_expected)
#         curr_sample = [hour]
#         curr_expected = [y_test[index]]

# create training and validation datasets, use dataloaders for batching
batch_size = 744 # this is one month in hours
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
validation_dataset = TensorDataset(x_test, y_test)
validloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# define model and set to gpu if available
model = nn.Sequential(
    nn.Linear(21, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
model = model.to(device)

# set other various parameters
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
current_epoch = 1
min_loss = float('inf')
best_epoch = 0
epochs = 800
losses = []
valid_losses = []
best_month_losses = []

# load a checkpoint for the model
try:
    if resume:
        state = torch.load("saves/feb_ahead_state")
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        current_epoch = state['epoch_num']
        min_loss = state['min_loss']
        best_epoch = state['best_epoch']
        losses = state['losses']
        valid_losses = state['valid_losses']
        best_month_losses = state['best_month_losses']
except:
    print("Error loading saved model, starting from beginning")

run_num = 1
folder_path = f"results/model_results/ahead/feb_ahead_{offset}/run_"
while os.path.exists(folder_path+str(run_num)):
    run_num+=1
folder_path = folder_path+str(run_num)
os.makedirs(folder_path)

# graph initial model weight distributions
if weight_graphs:
    weight_path = folder_path+"/weight_distributions"
    os.makedirs(weight_path)
    initial_weights = model[0].weight.detach().cpu().numpy()
    weight_sums = [0] * 11
    weight_squared_sums = [0] * 11
    for node in initial_weights:
        for i, item in enumerate(node):
            weight_sums[i] += abs(item)
            weight_squared_sums[i] += (item**2)

    plt.bar(range(1,12), weight_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Standard Model Initial Weight Distribution")
    plt.savefig(f"{weight_path}/std_init_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

    plt.bar(range(1,12), weight_squared_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Standard Model Initial Squared Weight Distribution")
    plt.savefig(f"{weight_path}/std_init_squared_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

if output_graphs:
    month = "Feb"
    output_path = folder_path+"/outputs"
    os.makedirs(output_path)

# def compare_outputs(sample_list, expected_list, months, epoch_num):
#     model.eval()
#     for index, month in enumerate(months):
#         size = len(sample_list[index])
#         predicted = model(sample_list[index])
#         loss = criterion(predicted.squeeze(), expected_list[index]).cpu()
#         loss = np.sqrt(loss.detach().numpy())
#         expected = expected_list[index].cpu().reshape(-1, 1)
#         predicted = predicted.cpu().detach().numpy().reshape(-1, 1)

#         expected = inverse_y_scaler.inverse_transform(expected).reshape(1, -1)[0]
#         predicted = inverse_y_scaler.inverse_transform(predicted).reshape(1, -1)[0]

#         # plot expected vs actual values
#         plt.plot(range(size), expected, label='Actual')
#         plt.plot(range(size), predicted, label='Prediction')
#         plt.ylabel("GHI")
#         plt.xlabel("Hour")
#         plt.legend(loc="upper right")
#         plt.title(f"Standard Model Ouputs for First 72 Hours of {month} at Epoch {epoch_num}\nLoss: {loss} RMSE")
#         figure_name = f"{output_path}/{month}_{epoch_num}"
#         plt.savefig(figure_name + ".pdf")
#         plt.show(block=show_graph)
#         plt.close()

def compare_outputs(sample_list, expected_list, month, epoch_num, mode):
    model.eval()
    for list_num, list in enumerate(sample_list):
        os.makedirs(f"{output_path}/{mode}/{month}_{epoch_num}", exist_ok=True)
        size = len(list)
        predicted = model(list)
        # loss = criterion(predicted, expected_list[list_num]).cpu()
        # loss = np.sqrt(loss.detach().numpy())
        
        expected = expected_list[list_num].cpu()
        predicted = predicted.cpu()
        # predicted = predicted.cpu().detach().numpy().reshape(-1, 1)

        # expected = torch.FloatTensor(inverse_y_scaler.inverse_transform(expected).reshape(1, -1)[0])
        # predicted = torch.FloatTensor(inverse_y_scaler.inverse_transform(predicted).reshape(1, -1)[0])
        loss = criterion(predicted.squeeze(), expected)
        loss = np.sqrt(loss.detach().numpy())

        # plot expected vs actual values
        plt.plot(range(size), expected, label='Actual')
        plt.plot(range(size), predicted, label='Prediction')
        plt.ylabel("GHI")
        plt.xlabel("Hour")
        plt.legend(loc="upper right")
        plt.title(f"Standard Model Ouputs for 72 Hours {list_num} of {month} at Epoch {epoch_num}\nLoss: {loss} RMSE")
        figure_name = f"{output_path}/{mode}/{month}_{epoch_num}/{month}_{epoch_num}_{list_num}"
        plt.savefig(figure_name + ".pdf")
        plt.show(block=show_graph)
        plt.close()

# training and validation loop
for epoch in range(epochs):
    if verbose:
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
        month_losses = []
        valid_count = 0
        for id_batch, (x_batch, y_batch) in enumerate(validloader):
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch).cpu()
            loss = np.sqrt(loss.detach().numpy())
            loss_total += loss
            valid_count += 1
            month_losses.append(loss)
        curr_loss = loss_total/valid_count
        valid_losses.append(curr_loss)
        if verbose:
            print("Loss: " + str(curr_loss))
        if curr_loss < min_loss:
            best_month_losses = month_losses
            min_loss = curr_loss
            best_epoch = epoch+current_epoch
        if output_graphs and ((epoch+current_epoch)%40) == 0:
            #compare_outputs(validation_sample_list, validation_expected_list, month, epoch+current_epoch, "validation")
            compare_outputs(training_sample_list, training_expected_list, month, epoch+current_epoch, "training")
        
# print lowest loss and epoch
print("Minimum Validation Loss: " + str(min_loss))
print("Minimum NRMSE: " + str(min_loss/y_mean))
print("Best Epoch: " + str(best_epoch))
print("End Validation Loss: " + str(curr_loss))
print("Best Month Losses: " + str(best_month_losses))

with open(f"{folder_path}/results.txt", 'w') as file:
    file.write("Minimum Validation Loss: " + str(min_loss) + "\n")
    file.write("Minimum NRMSE: " + str(min_loss/y_mean) + "\n")
    file.write("Best Epoch: " + str(best_epoch) + "\n")
    file.write("End Validation Loss: " + str(curr_loss) + "\n")
    file.write("Best Month Losses: " + str(best_month_losses) + "\n")

# graph end model weight distributions
if weight_graphs:
    end_weights = model[0].weight.detach().cpu().numpy()
    weight_sums = [0] * 11
    weight_squared_sums = [0] * 11
    for node in end_weights:
        for i, item in enumerate(node):
            weight_sums[i] += abs(item)
            weight_squared_sums[i] += (item**2)

    plt.bar(range(1,12), weight_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Standard Model End Weight Distribution")
    plt.savefig(f"{weight_path}/std_end_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

    plt.bar(range(1,12), weight_squared_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Standard Model End Squared Weight Distribution")
    plt.savefig(f"{weight_path}/std_end_squared_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

# save the model
if resume:
    state = {
        'epoch_num': epoch+current_epoch+1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'min_loss': min_loss,
        'best_epoch': best_epoch,
        'losses': losses,
        'valid_losses': valid_losses,
        'best_month_losses': best_month_losses
    }
    if overwrite_save:
        torch.save(state, "saves/feb_ahead_state")
    else:
        save_id = 1
        while os.path.exists("saves/feb_ahead_state" + f"_{save_id}"):
            save_id += 1
        plt.savefig("saves/feb_ahead_state" + f"_{save_id}")

if loss_graphs:
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
    plt.title(f"Standard Model Losses w/ {learning_rate} Learning Rate")
    fig_id = 1
    figure_name = f"{folder_path}/lr{learning_rate}_epoch{epoch+current_epoch}"
    plt.savefig(figure_name + ".pdf")
    plt.show(block=show_graph)
    plt.close()