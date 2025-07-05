import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import captum
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from matplotlib.colors import LinearSegmentedColormap

# set flags for showing graphs and checkpointing
show_graph = False
weight_graphs = False
resume = True
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
train_dataset['Sin_Hour'] = np.sin(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour'])) 
train_dataset['Cos_Hour'] = np.cos(2 * np.pi * train_dataset['Hour'] / max(train_dataset['Hour']))
test_dataset['Sin_Hour'] = np.sin(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour'])) 
test_dataset['Cos_Hour'] = np.cos(2 * np.pi * test_dataset['Hour'] / max(test_dataset['Hour']))
train_dataset['Sin_Month'] = np.sin(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month'])) 
train_dataset['Cos_Month'] = np.cos(2 * np.pi * train_dataset['Month'] / max(train_dataset['Month']))
test_dataset['Sin_Month'] = np.sin(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month'])) 
test_dataset['Cos_Month'] = np.cos(2 * np.pi * test_dataset['Month'] / max(test_dataset['Month']))

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
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_test = test_dataset[['GHI']]
y_mean = y_test.mean()['GHI']

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation']
x_columns_test = ['Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation', 'DayOfYear_Sin', 'DayOfYear_Cos', 'Sin_Hour', 'Cos_Hour', 'Sin_Month', 'Cos_Month', 'Temperature_prev', 'Dew Point_prev', 'Relative Humidity_prev', 'Pressure_prev', 'Wind Speed_prev', 'Precipitation_prev', 'DayOfYear_Sin_prev', 'DayOfYear_Cos_prev', 'Sin_Hour_prev', 'Cos_Hour_prev', 'Sin_Month_prev', 'Cos_Month_prev', 'GHI_prev']
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

# jan_x_test = x_test[0:744]
# july_x_test = x_test[4368:5112]

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
# jan_x_test = torch.FloatTensor(jan_x_test).to(device)
# july_x_test = torch.FloatTensor(july_x_test).to(device)

jan_sample = x_test[:72]
jan_expected = y_test[:72]
apr_sample = x_test[2160:2232]
apr_expected = y_test[2160:2232]
jul_sample = x_test[4344:4416]
jul_expected = y_test[4344:4416]
oct_sample = x_test[6552:6624]
oct_expected = y_test[6552:6624]
sample_list = [jan_sample, apr_sample, jul_sample, oct_sample]
expected_list = [jan_expected, apr_expected, jul_expected, oct_expected]

# create training and validation datasets, use dataloaders for batching
batch_size = 8760 # this is one year in hours
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validation_dataset = TensorDataset(x_test, y_test)
validloader = DataLoader(validation_dataset, batch_size=int(batch_size/12), shuffle=False)

# define model and set to gpu if available
model = nn.Sequential(
    nn.Linear(25, 150),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(150, 250),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(100, 1)
)
model.to(device)

# set other various parameters
criterion = nn.MSELoss()
learning_rate = 0.005
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
        state = torch.load("saves/cyc_ahead_wide_dropout_saved_state_2")
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
folder_path = f"results/model_results/ahead/cyc_ahead_wide_dropout_{offset}_results/run_"
while os.path.exists(folder_path+str(run_num)):
    run_num+=1
folder_path = folder_path+str(run_num)
os.makedirs(folder_path)

# graph initial model weight distributions
if weight_graphs:
    weight_path = folder_path+"/weight_distributions"
    os.makedirs(weight_path)
    initial_weights = model[0].weight.detach().cpu().numpy()
    weight_sums = [0] * 14
    weight_squared_sums = [0] * 14
    for node in initial_weights:
        for i, item in enumerate(node):
            weight_sums[i] += abs(item)
            weight_squared_sums[i] += (item**2)

    plt.bar(range(1,15), weight_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Cyclical Model Initial Weight Distribution")
    plt.savefig(f"{weight_path}/cyc_init_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

    plt.bar(range(1,15), weight_squared_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Cyclical Model Initial Squared Weight Distribution")
    plt.savefig(f"{weight_path}/cyc_init_squared_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

if output_graphs:
    months = ["Jan", "Apr", "Jul", "Oct"]
    output_path = folder_path+"/outputs"
    os.makedirs(output_path)

def compare_outputs(sample_list, expected_list, months, epoch_num):
    model.eval()
    for index, month in enumerate(months):
        size = len(sample_list[index])
        predicted = model(sample_list[index])
        loss = criterion(predicted.squeeze(), expected_list[index]).cpu()
        loss = np.sqrt(loss.detach().numpy())
        predicted = predicted.cpu()
        expected = expected_list[index].cpu()
        # predicted = predicted.cpu().detach().numpy().reshape(-1, 1)

        # expected = inverse_y_scaler.inverse_transform(expected).reshape(1, -1)[0]
        # predicted = inverse_y_scaler.inverse_transform(predicted).reshape(1, -1)[0]

        # plot expected vs actual values
        plt.plot(range(size), expected, label='Actual')
        plt.plot(range(size), predicted, label='Prediction')
        plt.ylabel("GHI")
        plt.xlabel("Hour")
        plt.legend(loc="upper right")
        plt.title(f"Cyclical Model Ouputs for First 72 Hours of {month} at Epoch {epoch_num}\nLoss: {loss} RMSE")
        figure_name = f"{output_path}/{month}_{epoch_num}"
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
            compare_outputs(sample_list, expected_list, months, epoch+current_epoch)

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

# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# #gs = GradientShap(model)
# fa = FeatureAblation(model)

# ig_attr_test = ig.attribute(jan_x_test, n_steps=50)
# ig_nt_attr_test = ig_nt.attribute(jan_x_test)
# dl_attr_test = dl.attribute(jan_x_test)
# #gs_attr_test = gs.attribute(jan_x_test, x_train)
# fa_attr_test = fa.attribute(jan_x_test)

# x_axis_data = np.arange(x_test.shape[1])
# x_axis_data_labels = list(map(lambda idx: x_columns_test[idx], x_axis_data))

# ig_attr_test_sum = ig_attr_test.detach().cpu().numpy().sum(0)
# ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

# ig_nt_attr_test_sum = ig_nt_attr_test.detach().cpu().numpy().sum(0)
# ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

# dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
# dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

# # gs_attr_test_sum = gs_attr_test.detach().cpu().numpy().sum(0)
# # gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

# fa_attr_test_sum = fa_attr_test.detach().cpu().numpy().sum(0)
# fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

# # lin_weight = model.lin1.weight[0].detach().cpu().numpy()
# # y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

# width = 0.14
# legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

# plt.figure(figsize=(20, 10))

# ax = plt.subplot()
# ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
# ax.set_ylabel('Attributions')

# FONT_SIZE = 16
# plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
# plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
# plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
# plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

# ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
# ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
# # ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
# ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# # ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
# ax.autoscale_view()
# plt.tight_layout()

# ax.set_xticks(x_axis_data + 0.5)
# ax.set_xticklabels(x_axis_data_labels)

# plt.legend(legends, loc=3)
# plt.show()

# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# #gs = GradientShap(model)
# fa = FeatureAblation(model)

# ig_attr_test = ig.attribute(july_x_test, n_steps=50)
# ig_nt_attr_test = ig_nt.attribute(july_x_test)
# dl_attr_test = dl.attribute(july_x_test)
# #gs_attr_test = gs.attribute(jan_x_test, x_train)
# fa_attr_test = fa.attribute(july_x_test)

# x_axis_data = np.arange(x_test.shape[1])
# x_axis_data_labels = list(map(lambda idx: x_columns_test[idx], x_axis_data))

# ig_attr_test_sum = ig_attr_test.detach().cpu().numpy().sum(0)
# ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

# ig_nt_attr_test_sum = ig_nt_attr_test.detach().cpu().numpy().sum(0)
# ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

# dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
# dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

# # gs_attr_test_sum = gs_attr_test.detach().cpu().numpy().sum(0)
# # gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

# fa_attr_test_sum = fa_attr_test.detach().cpu().numpy().sum(0)
# fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

# # lin_weight = model.lin1.weight[0].detach().cpu().numpy()
# # y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

# width = 0.14
# legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

# plt.figure(figsize=(20, 10))

# ax = plt.subplot()
# ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
# ax.set_ylabel('Attributions')

# FONT_SIZE = 16
# plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
# plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
# plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
# plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

# ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
# ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
# # ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
# ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# # ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
# ax.autoscale_view()
# plt.tight_layout()

# ax.set_xticks(x_axis_data + 0.5)
# ax.set_xticklabels(x_axis_data_labels)

# plt.legend(legends, loc=3)
# plt.show()

# graph end model weight distributions
if weight_graphs:
    end_weights = model[0].weight.detach().cpu().numpy()
    weight_sums = [0] * 14
    weight_squared_sums = [0] * 14
    for node in end_weights:
        for i, item in enumerate(node):
            weight_sums[i] += abs(item)
            weight_squared_sums[i] += (item**2)

    plt.bar(range(1,15), weight_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Cyclical Model End Weight Distribution")
    plt.savefig(f"{weight_path}/cyc_end_weight_dist.pdf")
    plt.show(block=show_graph)
    plt.close()

    plt.bar(range(1,15), weight_squared_sums, width=1)
    plt.ylabel("Weight")
    plt.xlabel("Node")
    plt.title("Cyclical Model End Squared Weight Distribution")
    plt.savefig(f"{weight_path}/cyc_end_squared_weight_dist.pdf")
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
        torch.save(state, "saves/cyc_ahead_wide_dropout_saved_state")
    else:
        save_id = 1
        while os.path.exists("saves/cyc_ahead_wide_dropout_saved_state" + f"_{save_id}"):
            save_id += 1
        torch.save(state, "saves/cyc_ahead_wide_dropout_saved_state" + f"_{save_id}")

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
    plt.title(f"Cyclical Model Losses w/ {learning_rate} Learning Rate")
    fig_id = 1
    figure_name = f"{folder_path}/lr{learning_rate}_epoch{epoch+current_epoch}"
    plt.savefig(figure_name + ".pdf")
    plt.show(block=show_graph)
    plt.close()