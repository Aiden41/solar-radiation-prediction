import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os
from xgboost import XGBRegressor

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
mask = (dataset['Year'] != 2021) & (dataset['Year'] != 2022)
train_dataset = dataset[mask].copy()
mask = dataset['Year'] == 2021
test_dataset = dataset[mask].copy()

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

# create cyclical columns
train_dataset['DayOfYear'] = pd.to_datetime(train_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
test_dataset['DayOfYear'] = pd.to_datetime(test_dataset[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
train_dataset['DayOfYear'] = train_dataset['DayOfYear'].mask((train_dataset['DayOfYear'] >= 60) & ((train_dataset['Year'] == 2016) | (train_dataset['Year'] == 2020)), train_dataset['DayOfYear']-1)
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

mask = (train_dataset['Month'] == 10) | (train_dataset['Month'] == 11) | (train_dataset['Month'] == 12)
train_dataset = train_dataset[mask].copy()
mask =(test_dataset['Month'] == 11)
test_dataset = test_dataset[mask].copy()
train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

last_index = len(train_dataset)
drop_rows = range(last_index-offset, last_index)
train_dataset['oldGHI'] = train_dataset['GHI']
train_dataset['GHI'] = train_dataset['GHI'].shift(-offset)
train_dataset.drop(drop_rows, inplace=True)

last_index = len(test_dataset)
drop_rows = range(last_index-offset, last_index)
test_dataset['oldGHI'] = test_dataset['GHI']
test_dataset['GHI'] = test_dataset['GHI'].shift(-offset)
test_dataset.drop(drop_rows, inplace=True)

train_dataset.reset_index(drop=True, inplace=True) # set first index to 0
test_dataset.reset_index(drop=True, inplace=True)

# drop unused columns and get values out of dataframe
x_train = train_dataset.drop(columns = ['GHI', 'Minute', 'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_train_oldGHI = train_dataset[['oldGHI']]
y_train = train_dataset[['GHI']]
x_test = test_dataset.drop(columns = ['GHI', 'Minute',  'Month', 'Hour', 'Year', 'DNI', 'DHI', 'Solar Zenith Angle', 'Day', 'DayOfYear'])
y_test = test_dataset[['GHI']]
y_test_oldGHI = test_dataset[['oldGHI']]
train_y_mean = y_train.mean()['GHI']
valid_y_mean = y_test.mean()['GHI']

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Temperature', 'Dew Point', 'Relative Humidity', 'Pressure', 'Wind Speed', 'Precipitation', 'Surface Albedo']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

y_train = y_train.to_numpy()
y_train_oldGHI = y_train_oldGHI.to_numpy()
y_test = y_test.to_numpy()
y_test_oldGHI = y_test_oldGHI.to_numpy()

# scale label column
# y_scaler = ColumnTransformer([("scaler", StandardScaler(), y_columns)], remainder='passthrough')
# y_train = y_scaler.fit_transform(y_train)
# y_train = y_train.reshape(1, -1)[0]
# y_test = y_scaler.transform(y_test)
# y_test = y_test.reshape(1, -1)[0]
# inverse_y_scaler = y_scaler.transformers_[0][1]

new_x_train = []
for index, row in enumerate(x_train):
    if index < 24:
        continue
    t = row
    t_prev_day = x_train[index-24:index].flatten()
    t_prev_sr = y_train_oldGHI[index-24:index].flatten()
    new_row = np.concatenate([t, t_prev_day, t_prev_sr]).flatten()
    #new_row = np.append(t, t_prev_sr)
    new_x_train.append(new_row)
x_train = np.array(new_x_train)

new_x_test = []
for index, row in enumerate(x_test):
    if index < 24:
        continue
    t = row
    t_prev_day = x_test[index-24:index].flatten()
    t_prev_sr = y_test_oldGHI[index-24:index].flatten()
    new_row = np.concatenate([t, t_prev_day, t_prev_sr]).flatten()
    #new_row = np.append(t, t_prev_sr)
    new_x_test.append(new_row)
x_test = np.array(new_x_test)

y_train = np.delete(y_train, slice(0,24))
y_test = np.delete(y_test, slice(0,24))

x_months = np.array_split(x_test, 12)
y_months = np.array_split(y_test, 12)

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

model = XGBRegressor(n_estimators=500, eval_metric='rmse', early_stopping_rounds=100)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

run_num = 1
folder_path = f"results/model_results/ahead/xgboost_{offset}/run_"
while os.path.exists(folder_path+str(run_num)):
    run_num+=1
folder_path = folder_path+str(run_num)
os.makedirs(folder_path)

plt.figure()
plt.plot(x_axis, results['validation_0']['rmse'], label='Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title(f'XGBoost Model with Offset {offset}')
figure_name = f"{folder_path}/Results"
plt.savefig(figure_name + ".pdf")
plt.show(block=show_graph)
plt.close()

train_pred = model.predict(x_train)
valid_pred = model.predict(x_test)
train_loss = np.sqrt(mean_squared_error(y_train, train_pred))
valid_loss = np.sqrt(mean_squared_error(y_test, valid_pred))

# best_month_losses = []
# for index, month in enumerate(x_months):
#     month_pred = model.predict(month)
#     best_month_losses.append(np.sqrt(mean_squared_error(y_months[index], month_pred)))

# print lowest loss and epoch
print("Training Loss: " + str(train_loss))
print("Validation Loss: " + str(valid_loss))
print("Training NRMSE: " + str(train_loss/train_y_mean))
print("Validation NRMSE: " + str(valid_loss/valid_y_mean))
# print("Best Month Losses: " + str(best_month_losses))

with open(f"{folder_path}/results.txt", 'w') as file:
    file.write("Training Loss: " + str(train_loss) + "\n")
    file.write("Validation Loss: " + str(valid_loss) + "\n")
    file.write("Training NRMSE: " + str(train_loss/train_y_mean) + "\n")
    file.write("Validation NRMSE: " + str(valid_loss/valid_y_mean) + "\n")
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