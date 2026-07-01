import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from solar_dataset import SolarDataset

lagged = True
num_epochs = 100
batch_size = 1024

early_stopping = True
patience = 8

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
        )

        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        h = self.net(x)
        h = self.norm(h)
        return self.fc(h)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_ds = SolarDataset(x_train, y_train, seq_len, flatten=True)
valid_ds = SolarDataset(x_valid, y_valid, seq_len, flatten=True)
test_ds = SolarDataset(x_test, y_test, seq_len, flatten=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

model = MLP(input_dim=len(train_ds[0][0])).to(device)

# set other various parameters
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

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
    return np.concatenate(preds, axis=0).reshape(-1)

model.eval()
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

train_true = y_train_ghi[train_idx]
valid_true = y_valid_ghi[valid_idx]
test_true = y_test_ghi[test_idx]

train_true_day = train_true[train_day]
valid_true_day = valid_true[valid_day]
test_true_day = test_true[test_day]

train_pred_day = train_pred_ghi[train_day]
valid_pred_day = valid_pred_ghi[valid_day]
test_pred_day = test_pred_ghi[test_day]

# MSE
train_mse = mean_squared_error(train_true_day, train_pred_day)
valid_mse = mean_squared_error(valid_true_day, valid_pred_day)
test_mse = mean_squared_error(test_true_day, test_pred_day)

# RMSE
train_rmse = np.sqrt(train_mse)
valid_rmse = np.sqrt(valid_mse)
test_rmse = np.sqrt(test_mse)

# NRMSE (normalize by daytime mean GHI)
train_nrmse = train_rmse / train_average_GHI
valid_nrmse = valid_rmse / valid_average_GHI
test_nrmse = test_rmse / test_average_GHI

# MAE
train_mae = mean_absolute_error(train_true_day, train_pred_day)
valid_mae = mean_absolute_error(valid_true_day, valid_pred_day)
test_mae = mean_absolute_error(test_true_day, test_pred_day)

# MBE
def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

train_mbe = mbe(train_true_day, train_pred_day)
valid_mbe = mbe(valid_true_day, valid_pred_day)
test_mbe = mbe(test_true_day, test_pred_day)

# R^2
train_r2 = r2_score(train_true_day, train_pred_day)
valid_r2 = r2_score(valid_true_day, valid_pred_day)
test_r2 = r2_score(test_true_day, test_pred_day)

# print results
print("GHI METRICS")
print("Training Error")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("NRMSE:", train_nrmse)
print("MAE:", train_mae)
print("MBE:", train_mbe)
print("R^2:", train_r2)

print("\nValidation Error")
print("MSE:", valid_mse)
print("RMSE:", valid_rmse)
print("NRMSE:", valid_nrmse)
print("MAE:", valid_mae)
print("MBE:", valid_mbe)
print("R^2:", valid_r2)

print("\nTesting Error")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("NRMSE:", test_nrmse)
print("MAE:", test_mae)
print("MBE:", test_mbe)
print("R^2:", test_r2)

path = None
if lagged:
    path = f"mlp_lag_{seq_len}"
else:
    path = "mlp"

# save results
with open("results/grid_results/" + path + ".txt", 'w') as file:
    file.write("GHI METRICS\n")
    file.write("Training Error\n")
    file.write("MSE: " + str(train_mse) + "\n")
    file.write("RMSE: " + str(train_rmse) + "\n")
    file.write("NRMSE: " + str(train_nrmse) + "\n")
    file.write("MAE: " + str(train_mae) + "\n")
    file.write("MBE: " + str(train_mbe) + "\n")
    file.write("R^2: " + str(train_r2) + "\n")

    file.write("\nValidation Error\n")
    file.write("MSE: " + str(valid_mse) + "\n")
    file.write("RMSE: " + str(valid_rmse) + "\n")
    file.write("NRMSE: " + str(valid_nrmse) + "\n")
    file.write("MAE: " + str(valid_mae) + "\n")
    file.write("MBE: " + str(valid_mbe) + "\n")
    file.write("R^2: " + str(valid_r2) + "\n")

    file.write("\nTesting Error\n")
    file.write("MSE: " + str(test_mse) + "\n")
    file.write("RMSE: " + str(test_rmse) + "\n")
    file.write("NRMSE: " + str(test_nrmse) + "\n")
    file.write("MAE: " + str(test_mae) + "\n")
    file.write("MBE: " + str(test_mbe) + "\n")
    file.write("R^2: " + str(test_r2) + "\n")

# plot the results
hours = np.arange(864) * (5/60) # 5 minutes to hours
plt.plot(hours, test_true[:864], label="Actual")
plt.plot(hours, test_pred_ghi[:864], label="Predicted")
if lagged:
    plt.title(f"MLP ({seq_len} Rows) GHI Pred vs Actual")
else:
    plt.title("MLP GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/grid_results/" + path + ".pdf")
plt.show(block=False)