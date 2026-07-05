import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

lagged = False

# unlike point models, offset is handled in preprocessing. 
# to change, edit preprocess.py and rerun preprocessing. 
# offset is the number of rows ahead to predict.
horizon = 12 # number of values to predict

seq_len = None
path = "saves/preprocessed/"
if lagged:
    seq_len = 12
else:
    seq_len = 1

target = 'CSI' # GHI or CSI

energy_metrics = True
importance = True

x_train = np.load(path + "train_grid.npy")
x_valid = np.load(path + "valid_grid.npy")
x_test = np.load(path + "test_grid.npy")

x_train = x_train.reshape(x_train.shape[0], -1)
x_valid = x_valid.reshape(x_valid.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

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

train_weights = np.load(path + "train_weights.npy")
valid_weights = np.load(path + "valid_weights.npy")
test_weights = np.load(path + "test_weights.npy")

avg = np.load(path + "averages.npy")
train_average_GHI = avg[0]
valid_average_GHI = avg[1]
test_average_GHI = avg[2]

sza_train = np.load(path + "future_sza_train.npy")
sza_valid = np.load(path + "future_sza_valid.npy")
sza_test = np.load(path + "future_sza_test.npy")

future_cs_ghi_train = np.load(path+ "future_cs_ghi_train.npy")
future_cs_ghi_valid = np.load(path + "future_cs_ghi_valid.npy")
future_cs_ghi_test = np.load(path + "future_cs_ghi_test.npy")

def build_lagged(arr, seq_len):
    if seq_len == 1:
        return arr

    lagged = []
    for t in range(seq_len - 1, len(arr)):
        window = arr[t - seq_len + 1 : t + 1].reshape(-1)
        lagged.append(window)
    return np.array(lagged)

x_train = build_lagged(x_train, seq_len)
x_valid = build_lagged(x_valid, seq_len)
x_test = build_lagged(x_test, seq_len)

max_lag = seq_len - 1
y_train = y_train[max_lag:]
y_valid = y_valid[max_lag:]
y_test = y_test[max_lag:]

sza_train = sza_train[max_lag:]
sza_valid = sza_valid[max_lag:]
sza_test = sza_test[max_lag:]

train_pred = np.zeros_like(y_train)
valid_pred = np.zeros_like(y_valid)
test_pred = np.zeros_like(y_test)

for h in range(horizon):
    train_weights = (sza_train[:, h] < 90).astype(np.float32)
    valid_weights = (sza_valid[:, h] < 90).astype(np.float32)

    model = XGBRegressor(n_estimators=1000, eval_metric="rmse", objective="reg:pseudohubererror", early_stopping_rounds=100, eta=0.05)

    model.fit(x_train, y_train[:, h], sample_weight=train_weights, eval_set=[(x_valid, y_valid[:, h])], 
    sample_weight_eval_set=[valid_weights], verbose=False)

    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # point importance
    if h == 11 and importance:
        importances = model.feature_importances_

        height = 7
        width = 7
        features_per_cell = x_train.shape[1] // (height * width)

        cell_importance = np.zeros((height, width))

        for f in range(features_per_cell):
            start = f * (height * width)
            end = start + (height * width)
            feature_slice = importances[start:end]
            feature_grid = feature_slice.reshape(height, width)
            cell_importance += feature_grid

        cell_importance /= cell_importance.sum()

        plt.figure(figsize=(6, 5))
        plt.imshow(cell_importance, cmap="viridis", origin="upper")
        plt.colorbar(label="Relative Importance")
        plt.title("XGBoost Feature Importance by Point")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")

        for i in range(height):
            for j in range(width):
                plt.text(j, i, f"{cell_importance[i,j]:.3f}",
                        ha="center", va="center", color="white")

        plt.tight_layout()
        if lagged:
            plt.savefig(f"results/grid_results/{target.lower()}/xgboost_lag_{seq_len}_point_importance.pdf")
        else:
            plt.savefig(f"results/grid_results/{target.lower()}/xgboost_no_lag_point_importance.pdf")
        plt.show(block=False)
        plt.close('all')

    train_pred[:, h] = model.predict(x_train)
    valid_pred[:, h] = model.predict(x_valid)
    test_pred[:, h] = model.predict(x_test)

train_true = y_train_ghi[max_lag:]
valid_true = y_valid_ghi[max_lag:]
test_true = y_test_ghi[max_lag:]

if target == 'CSI':
    train_pred = train_pred * future_cs_ghi_train[max_lag:]
    valid_pred = valid_pred * future_cs_ghi_valid[max_lag:]
    test_pred = test_pred * future_cs_ghi_test[max_lag:]
else:
    train_pred *= 1000
    valid_pred *= 1000
    test_pred *= 1000

valid_mask = (sza_valid < 90)
valid_true_day = valid_true[valid_mask]
valid_pred_day = valid_pred[valid_mask]

# remove negative values
test_pred = np.maximum(test_pred, 0)
valid_pred = np.maximum(valid_pred, 0)
train_pred = np.maximum(train_pred, 0)

# zero out nighttime predictions
train_pred[sza_train >= 90] = 0
valid_pred[sza_valid >= 90] = 0
test_pred[sza_test >= 90] = 0

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

    train_mask = sza_train_h < 90
    valid_mask = sza_valid_h < 90
    test_mask = sza_test_h < 90

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

    train_energy_mask = (sza_train < 90).any(axis=1)
    valid_energy_mask = (sza_valid < 90).any(axis=1)
    test_energy_mask = (sza_test < 90).any(axis=1)

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

path = f"results/grid_results/{target.lower()}/xgboost"
if lagged:
    path += f"_lag_{seq_len}"
else:
    path += "_no_lag"

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
if lagged:
    plt.title(f"XGBoost ({seq_len} Rows) GHI Pred vs Actual")
else:
    plt.title(f"XGBoost GHI Pred vs Actual")
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
    if lagged:
        plt.title(f"XGBoost ({seq_len} Rows) Energy Pred vs Actual")
    else:
        plt.title(f"XGBoost Energy Pred vs Actual")
    plt.legend()
    plt.ylabel("Energy (Wh/m\u00b2)")
    plt.xlabel("Hour")
    plt.savefig(path + "_energy.pdf")
    plt.show(block=False)