import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from solar_dataset import SolarDataset

num_epochs = 100
batch_size = 1024

early_stopping = True
patience = 8

# to only predict 12th timestamp, flip these
offset = 1 # number of rows ahead to predict
horizon = 12 # number of values to predict

seq_len = 12 # number of rows

target = 'CSI' # GHI or CSI

energy_metrics = True

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

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, horizon)
        self.bias = nn.Parameter(torch.zeros(horizon))

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        h = self.norm(h)
        return self.fc(h) + self.bias

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM(input_dim=x_train.shape[1], hidden_dim=256, num_layers=2, dropout=0.1).to(device)

# set other various parameters
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

train_ds = SolarDataset(data=x_train, targets=y_train, seq_len=seq_len, device=device, flatten=False)
valid_ds = SolarDataset(data=x_valid, targets=y_valid, seq_len=seq_len, device=device, flatten=False)
test_ds = SolarDataset(data=x_test, targets=y_test, seq_len=seq_len, device=device, flatten=False)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_epoch_loss += loss.item() * xb.size(0)

    train_epoch_loss /= len(train_ds)
    train_losses.append(train_epoch_loss)

    model.eval()
    valid_epoch_loss = 0.0

    with torch.no_grad():
        for xb, yb in valid_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            valid_epoch_loss += loss.item() * xb.size(0)

    valid_epoch_loss /= len(valid_ds)
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

train_loader = DataLoader(train_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

def predict(model, loader):
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)

model.eval()
train_pred = predict(model, train_loader)
valid_pred = predict(model, valid_loader)
test_pred = predict(model, test_loader)

train_idx = train_ds.valid
valid_idx = valid_ds.valid
test_idx = test_ds.valid

sza_train = train_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_valid = valid_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()
sza_test = test_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy()

train_true = train_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()[train_idx]
valid_true = valid_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()[valid_idx]
test_true = test_dataset[[f"Future_GHI_{h}" for h in range(horizon)]].to_numpy()[test_idx]

if target == 'CSI':
    train_pred = train_pred * train_csghi[train_idx]
    valid_pred = valid_pred * valid_csghi[valid_idx]
    test_pred = test_pred * test_csghi[test_idx]

valid_mask = (sza_valid[valid_idx] < 90).any(axis=1)
valid_true_day = valid_true[valid_mask]
valid_pred_day = valid_pred[valid_mask]

# bias correction
delta = np.mean(valid_pred_day - valid_true_day)
train_pred = train_pred - delta
valid_pred = valid_pred - delta
test_pred = test_pred - delta

# remove negative values
test_pred = np.maximum(test_pred, 0)
valid_pred = np.maximum(valid_pred, 0)
train_pred = np.maximum(train_pred, 0)

# zero out nighttime predictions
train_pred[sza_train[train_idx] >= 90] = 0
valid_pred[sza_valid[valid_idx] >= 90] = 0
test_pred[sza_test[test_idx] >= 90] = 0

def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

rmse_all = np.zeros((horizon, 3))
nrmse_all = np.zeros((horizon, 3))
mae_all = np.zeros((horizon, 3))
mbe_all = np.zeros((horizon, 3))
r2_all = np.zeros((horizon, 3))

# loop ends with target! that's how printing and saving metrics work!
for h_idx in range(horizon):
    train_true_h = train_true[:, h_idx]
    valid_true_h = valid_true[:, h_idx]
    test_true_h = test_true[:, h_idx]

    train_pred_h = train_pred[:, h_idx]
    valid_pred_h = valid_pred[:, h_idx]
    test_pred_h = test_pred[:, h_idx]

    sza_train_h = sza_train[:, h_idx]
    sza_valid_h = sza_valid[:, h_idx]
    sza_test_h = sza_test[:, h_idx]

    train_mask = sza_train_h[train_idx] < 90
    valid_mask = sza_valid_h[valid_idx] < 90
    test_mask = sza_test_h[test_idx] < 90

    # mask real values to daytime only
    train_true_day = train_true_h[train_mask]
    valid_true_day = valid_true_h[valid_mask]
    test_true_day = test_true_h[test_mask]

    # mask predictions to daytime only
    train_pred_day = train_pred_h[train_mask]
    valid_pred_day = valid_pred_h[valid_mask]
    test_pred_day = test_pred_h[test_mask]

    train_mse = mean_squared_error(train_true_day, train_pred_day)
    valid_mse = mean_squared_error(valid_true_day, valid_pred_day)
    test_mse = mean_squared_error(test_true_day, test_pred_day)

    train_rmse = np.sqrt(train_mse)
    valid_rmse = np.sqrt(valid_mse)
    test_rmse = np.sqrt(test_mse)
    rmse_all[h_idx, :] = [train_rmse, valid_rmse, test_rmse]

    train_average_GHI = np.mean(train_true_day)
    valid_average_GHI = np.mean(valid_true_day)
    test_average_GHI = np.mean(test_true_day)

    train_nrmse = train_rmse / train_average_GHI
    valid_nrmse = valid_rmse / valid_average_GHI
    test_nrmse = test_rmse / test_average_GHI
    nrmse_all[h_idx, :] = [train_nrmse, valid_nrmse, test_nrmse]

    train_mae = mean_absolute_error(train_true_day, train_pred_day)
    valid_mae = mean_absolute_error(valid_true_day, valid_pred_day)
    test_mae = mean_absolute_error(test_true_day, test_pred_day)
    mae_all[h_idx, :] = [train_mae, valid_mae, test_mae]

    train_mbe = mbe(train_true_day, train_pred_day)
    valid_mbe = mbe(valid_true_day, valid_pred_day)
    test_mbe = mbe(test_true_day, test_pred_day)
    mbe_all[h_idx, :] = [train_mbe, valid_mbe, test_mbe]

    train_r2 = r2_score(train_true_day, train_pred_day)
    valid_r2 = r2_score(valid_true_day, valid_pred_day)
    test_r2 = r2_score(test_true_day, test_pred_day)
    r2_all[h_idx, :] = [train_r2, valid_r2, test_r2]

if energy_metrics:
    train_actual_energy = train_true.sum(axis=1) * (5/60)
    train_pred_energy = train_pred.sum(axis=1) * (5/60)

    valid_actual_energy = valid_true.sum(axis=1) * (5/60)
    valid_pred_energy = valid_pred.sum(axis=1) * (5/60)

    test_actual_energy = test_true.sum(axis=1) * (5/60)
    test_pred_energy = test_pred.sum(axis=1) * (5/60)

    train_energy_mask = (sza_train[train_idx] < 90).any(axis=1)
    valid_energy_mask = (sza_valid[valid_idx] < 90).any(axis=1)
    test_energy_mask = (sza_test[test_idx] < 90).any(axis=1)

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
    train_energy_r2 = r2_score(train_actual_energy_day, train_pred_energy_day)

    valid_energy_mse = mean_squared_error(valid_actual_energy_day, valid_pred_energy_day)
    valid_energy_rmse = np.sqrt(valid_energy_mse)
    valid_energy_nrmse = valid_energy_rmse / np.mean(valid_actual_energy_day)
    valid_energy_mae = mean_absolute_error(valid_actual_energy_day, valid_pred_energy_day)
    valid_energy_mbe = np.mean(valid_pred_energy_day - valid_actual_energy_day)
    valid_energy_r2 = r2_score(valid_actual_energy_day, valid_pred_energy_day)

    test_energy_mse = mean_squared_error(test_actual_energy_day, test_pred_energy_day)
    test_energy_rmse = np.sqrt(test_energy_mse)
    test_energy_nrmse = test_energy_rmse / np.mean(test_actual_energy_day)
    test_energy_mae = mean_absolute_error(test_actual_energy_day, test_pred_energy_day)
    test_energy_mbe = np.mean(test_pred_energy_day - test_actual_energy_day)
    test_energy_r2 = r2_score(test_actual_energy_day, test_pred_energy_day)

# save results
with open(f"results/point_results/lstm_{target.lower()}_{seq_len}.txt", 'w') as file:
    file.write("GHI METRICS\n")
    file.write("Training Error\n")
    file.write("RMSE: " + str(train_rmse) + "\n")
    file.write("NRMSE: " + str(train_nrmse) + "\n")
    file.write("MAE: " + str(train_mae) + "\n")
    file.write("MBE: " + str(train_mbe) + "\n")
    file.write("R^2: " + str(train_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("RMSE: " + str(valid_rmse) + "\n")
    file.write("NRMSE: " + str(valid_nrmse) + "\n")
    file.write("MAE: " + str(valid_mae) + "\n")
    file.write("MBE: " + str(valid_mbe) + "\n")
    file.write("R^2: " + str(valid_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("RMSE: " + str(test_rmse) + "\n")
    file.write("NRMSE: " + str(test_nrmse) + "\n")
    file.write("MAE: " + str(test_mae) + "\n")
    file.write("MBE: " + str(test_mbe) + "\n")
    file.write("R^2: " + str(test_r2) + "\n")

    if energy_metrics:
        file.write("\nENERGY METRICS\n")
        file.write("Training Error\n")
        file.write("RMSE: " + str(train_energy_rmse) + "\n")
        file.write("NRMSE: " + str(train_energy_nrmse) + "\n")
        file.write("MAE: " + str(train_energy_mae) + "\n")
        file.write("MBE: " + str(train_energy_mbe) + "\n")
        file.write("R^2: " + str(train_energy_r2) + "\n")

        file.write("\nValidation Error\n")
        file.write("RMSE: " + str(valid_energy_rmse) + "\n")
        file.write("NRMSE: " + str(valid_energy_nrmse) + "\n")
        file.write("MAE: " + str(valid_energy_mae) + "\n")
        file.write("MBE: " + str(valid_energy_mbe) + "\n")
        file.write("R^2: " + str(valid_energy_r2) + "\n")

        file.write("\nTesting Error\n")
        file.write("RMSE: " + str(test_energy_rmse) + "\n")
        file.write("NRMSE: " + str(test_energy_nrmse) + "\n")
        file.write("MAE: " + str(test_energy_mae) + "\n")
        file.write("MBE: " + str(test_energy_mbe) + "\n")
        file.write("R^2: " + str(test_energy_r2) + "\n")

# print results
print("GHI METRICS")
print("Training Error")
print("RMSE:", train_rmse)
print("NRMSE:", train_nrmse)
print("MAE:", train_mae)
print("MBE:", train_mbe)
print("R^2:", train_r2)

print("\nValidation Error")
print("RMSE:", valid_rmse)
print("NRMSE:", valid_nrmse)
print("MAE:", valid_mae)
print("MBE:", valid_mbe)
print("R^2:", valid_r2)

print("\nTesting Error")
print("RMSE:", test_rmse)
print("NRMSE:", test_nrmse)
print("MAE:", test_mae)
print("MBE:", test_mbe)
print("R^2:", test_r2)

# plot the results
hours = np.arange(864) * (5/60) # 5 minutes to hours
plt.plot(hours, test_true_h[:864], label="Actual")
plt.plot(hours, test_pred_h[:864], label="Predicted")
plt.title(f"LSTM ({seq_len} Rows) GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig(f"results/point_results/lstm_{target.lower()}_{seq_len}.pdf")
plt.show(block=False)
plt.close('all')

if energy_metrics:
    print("\nENERGY METRICS")
    print("Training Error")
    print("RMSE:", train_energy_rmse)
    print("NRMSE:", train_energy_nrmse)
    print("MAE:", train_energy_mae)
    print("MBE:", train_energy_mbe)
    print("R^2:", train_energy_r2)

    print("\nValidation Error")
    print("RMSE:", valid_energy_rmse)
    print("NRMSE:", valid_energy_nrmse)
    print("MAE:", valid_energy_mae)
    print("MBE:", valid_energy_mbe)
    print("R^2:", valid_energy_r2)

    print("\nTesting Error")
    print("RMSE:", test_energy_rmse)
    print("NRMSE:", test_energy_nrmse)
    print("MAE:", test_energy_mae)
    print("MBE:", test_energy_mbe)
    print("R^2:", test_energy_r2)

    hours = np.arange(864) * (5/60) # 5 minutes to hours
    plt.plot(hours, test_actual_energy[:864], label="Actual")
    plt.plot(hours, test_pred_energy[:864], label="Predicted")
    plt.title(f"LSTM ({seq_len} Rows) Energy Pred vs Actual")
    plt.legend()
    plt.ylabel("Energy (Wh/m\u00b2)")
    plt.xlabel("Hour")
    plt.savefig(f"results/point_results/lstm_{target.lower()}_{seq_len}_energy.pdf")
    plt.show(block=False)