import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

num_epochs = 100
batch_size = 1024

early_stopping = True
patience = 8

offset = 12 # number of rows ahead to predict

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
train_dataset["Future_SZA"] = train_dataset["Solar Zenith Angle"].shift(-offset)
valid_dataset["Future_SZA"] = valid_dataset["Solar Zenith Angle"].shift(-offset)
test_dataset["Future_SZA"] = test_dataset["Solar Zenith Angle"].shift(-offset)

train_dataset["Future_CS_GHI"] = train_dataset["Clearsky GHI"].shift(-offset)
valid_dataset["Future_CS_GHI"] = valid_dataset["Clearsky GHI"].shift(-offset)
test_dataset["Future_CS_GHI"] = test_dataset["Clearsky GHI"].shift(-offset)

train_dataset["Future_CS_DNI"] = train_dataset["Clearsky DNI"].shift(-offset)
valid_dataset["Future_CS_DNI"] = valid_dataset["Clearsky DNI"].shift(-offset)
test_dataset["Future_CS_DNI"] = test_dataset["Clearsky DNI"].shift(-offset)

train_dataset["Future_CS_DHI"] = train_dataset["Clearsky DHI"].shift(-offset)
valid_dataset["Future_CS_DHI"] = valid_dataset["Clearsky DHI"].shift(-offset)
test_dataset["Future_CS_DHI"] = test_dataset["Clearsky DHI"].shift(-offset)

train_dataset["Future_GHI"] = train_dataset["GHI"].shift(-offset)
valid_dataset["Future_GHI"] = valid_dataset["GHI"].shift(-offset)
test_dataset["Future_GHI"] = test_dataset["GHI"].shift(-offset)

train_dataset["Future_CSI"] = train_dataset["CSI"].shift(-offset)
valid_dataset["Future_CSI"] = valid_dataset["CSI"].shift(-offset)
test_dataset["Future_CSI"] = test_dataset["CSI"].shift(-offset)

train_dataset = train_dataset.iloc[:-offset]
valid_dataset = valid_dataset.iloc[:-offset]
test_dataset = test_dataset.iloc[:-offset]

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

y_train_ghi = y_train_ghi.to_numpy().astype(np.float32).flatten()
y_valid_ghi = y_valid_ghi.to_numpy().astype(np.float32).flatten()
y_test_ghi = y_test_ghi.to_numpy().astype(np.float32).flatten()

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

train_dataset = train_dataset.astype(np.float32)
valid_dataset = valid_dataset.astype(np.float32)
test_dataset = test_dataset.astype(np.float32)

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']
y_columns = list(y_train)

# scale all specified input columns 
x_scaler = ColumnTransformer([("scaler", StandardScaler(), x_columns)], remainder='passthrough')
x_train = x_scaler.fit_transform(x_train).astype(np.float32)
x_valid = x_scaler.transform(x_valid).astype(np.float32)
x_test = x_scaler.transform(x_test).astype(np.float32)

y_train = y_train.to_numpy().astype(np.float32).flatten()
y_valid = y_valid.to_numpy().astype(np.float32).flatten()
y_test = y_test.to_numpy().astype(np.float32).flatten()

train_weights = (train_dataset['Solar Zenith Angle'] < 90).to_numpy().astype(np.float32)
valid_weights = (valid_dataset['Solar Zenith Angle'] < 90).to_numpy().astype(np.float32)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP(input_dim=x_train.shape[1]).to(device)

# set other various parameters
criterion = nn.SmoothL1Loss(beta=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
train_targets = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(train_tensor, train_targets),
    batch_size=batch_size,
    shuffle=True
)

valid_tensor = torch.tensor(x_valid, dtype=torch.float32, device=device)
valid_targets = torch.tensor(y_valid, dtype=torch.float32, device=device).unsqueeze(1)

valid_loader = DataLoader(
    TensorDataset(valid_tensor, valid_targets),
    batch_size=batch_size,
    shuffle=False
)

best_val_loss = float('inf')
since_improvement = 0
best_state_dict = None
best_epoch = 0
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs+1):
    model.train()
    train_epoch_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item() * xb.size(0)

    train_epoch_loss /= len(train_tensor)
    train_losses.append(train_epoch_loss)

    model.eval()
    valid_epoch_loss = 0.0

    with torch.no_grad():
        for xb, yb in valid_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            valid_epoch_loss += loss.item() * xb.size(0)

    valid_epoch_loss /= len(valid_tensor)
    valid_losses.append(valid_epoch_loss)
    
    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}")

    if early_stopping:
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            since_improvement = 0
            best_state_dict = model.state_dict()
            best_epoch = epoch
        else:
            since_improvement += 1
            if since_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}, restoring model from epoch {best_epoch}")
                break
    
    scheduler.step()

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

print("\n-------------------\n")

model.eval()
with torch.no_grad():
    train_pred = model(train_tensor).cpu().numpy().flatten()
    valid_pred = model(valid_tensor).cpu().numpy().flatten()
    test_pred = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy().flatten()

train_pred[train_dataset['Future_SZA'] >= 90] = 0
valid_pred[valid_dataset['Future_SZA'] >= 90] = 0
test_pred[test_dataset['Future_SZA'] >= 90] = 0

train_pred_ghi = train_pred * train_dataset['Future_CS_GHI'].to_numpy()
valid_pred_ghi = valid_pred * valid_dataset['Future_CS_GHI'].to_numpy() 
test_pred_ghi = test_pred * test_dataset['Future_CS_GHI'].to_numpy()

train_mask = train_dataset['Future_SZA'] < 90
valid_mask = valid_dataset['Future_SZA'] < 90
test_mask = test_dataset['Future_SZA'] < 90

train_true = y_train_ghi[train_mask]
valid_true = y_valid_ghi[valid_mask]
test_true = y_test_ghi[test_mask]

train_pred_day = train_pred_ghi[train_mask]
valid_pred_day = valid_pred_ghi[valid_mask]
test_pred_day = test_pred_ghi[test_mask]

# MSE
train_mse = mean_squared_error(train_true, train_pred_day)
valid_mse = mean_squared_error(valid_true, valid_pred_day)
test_mse = mean_squared_error(test_true, test_pred_day)

# RMSE
train_rmse = np.sqrt(train_mse)
valid_rmse = np.sqrt(valid_mse)
test_rmse = np.sqrt(test_mse)

# NRMSE (normalize by daytime mean GHI)
train_nrmse = train_rmse / train_average_GHI
valid_nrmse = valid_rmse / valid_average_GHI
test_nrmse = test_rmse / test_average_GHI

# MAE
train_mae = mean_absolute_error(train_true, train_pred_day)
valid_mae = mean_absolute_error(valid_true, valid_pred_day)
test_mae = mean_absolute_error(test_true, test_pred_day)

# MBE
def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

train_mbe = mbe(train_true, train_pred_day)
valid_mbe = mbe(valid_true, valid_pred_day)
test_mbe = mbe(test_true, test_pred_day)

# sMAPE
def smape(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = den > 1e-6
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / den[mask])

train_smape = smape(train_true, train_pred_day)
valid_smape = smape(valid_true, valid_pred_day)
test_smape = smape(test_true, test_pred_day)

# R^2
train_r2 = r2_score(train_true, train_pred_day)
valid_r2 = r2_score(valid_true, valid_pred_day)
test_r2 = r2_score(test_true, test_pred_day)

# print results
print("GHI-SPACE METRICS")
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

train_true_csi = y_train[train_mask].flatten()
valid_true_csi = y_valid[valid_mask].flatten()
test_true_csi = y_test[test_mask].flatten()

train_pred_csi = train_pred[train_mask]
valid_pred_csi = valid_pred[valid_mask]
test_pred_csi = test_pred[test_mask]

# CSI MAE
train_csi_mae = mean_absolute_error(train_true_csi, train_pred_csi)
valid_csi_mae = mean_absolute_error(valid_true_csi, valid_pred_csi)
test_csi_mae = mean_absolute_error(test_true_csi, test_pred_csi)

# CSI sMAPE
train_csi_smape = smape(train_true_csi, train_pred_csi)
valid_csi_smape = smape(valid_true_csi, valid_pred_csi)
test_csi_smape = smape(test_true_csi, test_pred_csi)

# CSI R^2
train_csi_r2 = r2_score(train_true_csi, train_pred_csi)
valid_csi_r2 = r2_score(valid_true_csi, valid_pred_csi)
test_csi_r2 = r2_score(test_true_csi, test_pred_csi)

print("\nCSI-SPACE METRICS")
print("Training Error")
print("MAE: ", train_csi_mae)
print("sMAPE: ", train_csi_smape)
print("R^2:", train_csi_r2)

print("\nValidation Error")
print("MAE: ", valid_csi_mae)
print("sMAPE: ", valid_csi_smape)
print("R^2: ", valid_csi_r2)

print("\nTesting Error")
print("MAE: ", test_csi_mae)
print("sMAPE: ", test_csi_smape)
print("R^2: ", test_csi_r2)

# save results
with open("results/point_results/mlp_lag.txt", 'w') as file:
    file.write("GHI-SPACE METRICS\n")
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

    file.write("\nCSI-SPACE METRICS\n")
    file.write("Training Error\n")
    file.write("MAE: " + str(train_csi_mae) + "\n")
    file.write("sMAPE: " + str(train_csi_smape) + "\n")
    file.write("R^2: " + str(train_csi_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("MAE: " + str(valid_csi_mae) + "\n")
    file.write("sMAPE: " + str(valid_csi_smape) + "\n")
    file.write("R^2: " + str(valid_csi_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("MAE: " + str(test_csi_mae) + "\n")
    file.write("sMAPE: " + str(test_csi_smape) + "\n")
    file.write("R^2: " + str(test_csi_r2))

# plot the results
hours = np.arange(864) * (5/60) # 5 minutes to hours
plt.plot(hours, y_test_ghi[:864], label="Actual")
plt.plot(hours, test_pred_ghi[:864], label="Predicted")
plt.title(f"MLP ({max(lags)+1} Rows) GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/point_results/mlp_lag.pdf")
plt.show(block=False)