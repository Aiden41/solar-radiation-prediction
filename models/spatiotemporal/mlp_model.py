import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from solar_dataset import SpatioTemporalDataset

lagged = True
num_epochs = 10
batch_size = 1024

seq_len = None
path = "saves/preprocessed/"
if lagged:
    seq_len = 12
else:
    seq_len = 1

x_train = np.load(path + "train_grid.npy")
x_valid = np.load(path + "valid_grid.npy")
x_test = np.load(path + "test_grid.npy")

y_train = np.load(path + "y_train.npy")
y_valid = np.load(path + "y_valid.npy")
y_test = np.load(path + "y_test.npy")

train_weights = np.load(path + "train_weights.npy")
valid_weights = np.load(path + "valid_weights.npy")
test_weights = np.load(path + "test_weights.npy")

avg = np.load(path + "averages.npy")
train_average_GHI = avg[0]
valid_average_GHI = avg[1]
test_average_GHI = avg[2]

future_sza_train = np.load(path + "future_sza_train.npy")
future_sza_valid = np.load(path + "future_sza_valid.npy")
future_sza_test = np.load(path + "future_sza_test.npy")

future_cs_ghi_train = np.load(path+ "future_cs_ghi_train.npy")
future_cs_ghi_valid = np.load(path + "future_cs_ghi_valid.npy")
future_cs_ghi_test = np.load(path + "future_cs_ghi_test.npy")

y_train_ghi = np.load(path + "y_train_ghi.npy")
y_valid_ghi = np.load(path + "y_valid_ghi.npy")
y_test_ghi = np.load(path + "y_test_ghi.npy")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

train_ds = SpatioTemporalDataset(x_train, y_train, future_sza_train, seq_len=seq_len, flatten=True)
valid_ds = SpatioTemporalDataset(x_valid, y_valid, future_sza_valid, seq_len=seq_len, flatten=True)
test_ds = SpatioTemporalDataset(x_test, y_test, future_sza_test, seq_len=seq_len, flatten=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_x, _ = train_ds[0]
model = MLP(input_dim=sample_x.numel()).to(device)

# set other various parameters
criterion = nn.SmoothL1Loss(beta=0.1)
learning_rate = 0.0015
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)

        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(train_ds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("\n-------------------\n")

train_loader = DataLoader(train_ds, batch_size=batch_size)
valid_loader = DataLoader(valid_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

def predict(model, loader):
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0).reshape(-1)

train_pred = predict(model, train_loader)
valid_pred = predict(model, valid_loader)
test_pred = predict(model, test_loader)

train_idx = train_ds.valid
valid_idx = valid_ds.valid
test_idx = test_ds.valid

train_pred[future_sza_train[train_idx] >= 90] = 0
valid_pred[future_sza_valid[valid_idx] >= 90] = 0
test_pred[future_sza_test[test_idx] >= 90] = 0

train_pred_ghi = train_pred * future_cs_ghi_train[train_idx]
valid_pred_ghi = valid_pred * future_cs_ghi_valid[valid_idx]
test_pred_ghi = test_pred * future_cs_ghi_test[test_idx]

train_day = future_sza_train[train_idx] < 90
valid_day = future_sza_valid[valid_idx] < 90
test_day = future_sza_test[test_idx] < 90

train_true = y_train_ghi[train_idx][train_day]
valid_true = y_valid_ghi[valid_idx][valid_day]
test_true = y_test_ghi[test_idx][test_day]

train_pred_day = train_pred_ghi[train_day]
valid_pred_day = valid_pred_ghi[valid_day]
test_pred_day = test_pred_ghi[test_day]

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

train_true_csi = y_train[train_idx][train_day].flatten()
valid_true_csi = y_valid[valid_idx][valid_day].flatten()
test_true_csi = y_test[test_idx][test_day].flatten()

train_pred_csi = train_pred[train_day]
valid_pred_csi = valid_pred[valid_day]
test_pred_csi = test_pred[test_day]

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

path = None
if lagged:
    path = "mlp_lag"
else:
    path = "mlp"

# save results
with open("results/spatiotemporal_results/" + path + ".txt", 'w') as file:
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
plt.title("MLP GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/spatiotemporal_results/" + path + ".pdf")
plt.show(block=False)