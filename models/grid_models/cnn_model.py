import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from solar_dataset import SolarDataset

num_epochs = 100
batch_size = 512

early_stopping = True
patience = 8

# unlike point models, offset is handled in preprocessing. 
# to change, edit preprocess.py and rerun preprocessing. 
# offset is the number of rows ahead to predict.
horizon = 12 # number of values to predict

path = "saves/preprocessed/7x7_5km/"

target = 'CSI' # GHI or CSI

energy_metrics = True

x_train = np.load(path + "train_grid.npy")
x_valid = np.load(path + "valid_grid.npy")
x_test = np.load(path + "test_grid.npy")

if target == 'CSI':
    y_train = np.load(path + "y_train.npy")
    y_valid = np.load(path + "y_valid.npy")
    y_test = np.load(path + "y_test.npy")
else:
    y_train = np.load(path + "y_train_ghi.npy") / 1000
    y_valid = np.load(path + "y_valid_ghi.npy") / 1000
    y_test = np.load(path + "y_test_ghi.npy") / 1000

y_train_ghi = np.load(path + "y_train_ghi.npy")
y_valid_ghi = np.load(path + "y_valid_ghi.npy")
y_test_ghi = np.load(path + "y_test_ghi.npy")

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

class CNN(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizon)
        )

    def forward(self, x):
        x = x.squeeze(1)
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)

        h = self.gap(h).squeeze(-1).squeeze(-1)
        return self.fc(h)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_ds = SolarDataset(x_train, y_train)
valid_ds = SolarDataset(x_valid, y_valid)
test_ds = SolarDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

input_dim = train_ds[0][0].shape[1]
model = CNN(in_channels=input_dim, dropout=0.1).to(device)

# set other various parameters
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

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
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item() * xb.size(0)

    train_epoch_loss /= len(train_ds)
    train_losses.append(train_epoch_loss)

    model.eval()
    valid_epoch_loss = 0.0

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
    
    scheduler.step(valid_epoch_loss)

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

train_true = y_train_ghi[train_idx]
valid_true = y_valid_ghi[valid_idx]
test_true = y_test_ghi[test_idx]

if target == 'CSI':
    train_pred = train_pred * future_cs_ghi_train[train_idx]
    valid_pred = valid_pred * future_cs_ghi_valid[valid_idx]
    test_pred = test_pred * future_cs_ghi_test[test_idx]
else:
    train_pred *= 1000
    valid_pred *= 1000
    test_pred *= 1000

# remove negative values
valid_pred = np.maximum(valid_pred, 0)
train_pred = np.maximum(train_pred, 0)
test_pred = np.maximum(test_pred, 0)

# zero out nighttime predictions
train_pred[future_sza_train[train_idx] >= 90] = 0
valid_pred[future_sza_valid[valid_idx] >= 90] = 0
test_pred[future_sza_test[test_idx] >= 90] = 0

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

    sza_train_h = future_sza_train[:, h_idx]
    sza_valid_h = future_sza_valid[:, h_idx]
    sza_test_h = future_sza_test[:, h_idx]

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

    train_energy_mask = (future_sza_train[train_idx] < 90).any(axis=1)
    valid_energy_mask = (future_sza_valid[valid_idx] < 90).any(axis=1)
    test_energy_mask = (future_sza_test[test_idx] < 90).any(axis=1)

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

path = f"results/grid_results/{target.lower()}/cnn"

# save results
with open(path + ".txt", 'w') as file:
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
plt.title("CNN GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig(path + ".pdf")
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
    plt.title("CNN Energy Pred vs Actual")
    plt.legend()
    plt.ylabel("Energy (Wh/m\u00b2)")
    plt.xlabel("Hour")
    plt.savefig(path + "_energy.pdf")
    plt.show(block=False)