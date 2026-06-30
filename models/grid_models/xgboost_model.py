import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

path = "saves/preprocessed/"

x_train = np.load(path + "train_grid.npy")
x_valid = np.load(path + "valid_grid.npy")
x_test = np.load(path + "test_grid.npy")

x_train = x_train.reshape(x_train.shape[0], -1)
x_valid = x_valid.reshape(x_valid.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

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

model = XGBRegressor(n_estimators=1000, eval_metric="rmse", objective="reg:pseudohubererror", early_stopping_rounds=100, eta=0.05)
model.fit(x_train, y_train, sample_weight=train_weights, eval_set=[(x_train, y_train), (x_valid, y_valid)], sample_weight_eval_set=[train_weights, valid_weights], verbose=False)
results = model.evals_result()

# point importance
# importances = model.feature_importances_

# height = 7
# width = 7
# features = 27

# cell_importance = np.zeros((height, width))

# for i in range(height):
#     for j in range(width):
#         idx = (i * width + j) * features
#         cell_importance[i, j] = importances[idx:idx+features].sum()

# cell_importance /= cell_importance.sum()

# plt.figure(figsize=(6, 5))
# plt.imshow(cell_importance, cmap="viridis", origin="upper")
# plt.colorbar(label="Relative Importance")
# plt.title("XGBoost Feature Importance by Point")
# plt.xlabel("Grid X")
# plt.ylabel("Grid Y")

# for i in range(height):
#     for j in range(width):
#         plt.text(j, i, f"{cell_importance[i,j]:.3f}",
#                  ha="center", va="center", color="white")

# plt.tight_layout()
# plt.savefig("results/grid_results/xgboost_point_importance.pdf")
# plt.show(block=False)
# plt.close('all')

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

train_pred = model.predict(x_train)
valid_pred = model.predict(x_valid)
test_pred = model.predict(x_test)

train_pred[future_sza_train >= 90] = 0
valid_pred[future_sza_valid >= 90] = 0
test_pred[future_sza_test >= 90] = 0

train_pred_ghi = train_pred * future_cs_ghi_train
valid_pred_ghi = valid_pred * future_cs_ghi_valid
test_pred_ghi = test_pred * future_cs_ghi_test

train_mask = future_sza_train < 90
valid_mask = future_sza_valid < 90
test_mask = future_sza_test < 90

# Apply mask to GHI predictions and targets
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

# R^2
train_r2 = r2_score(train_true, train_pred_day)
valid_r2 = r2_score(valid_true, valid_pred_day)
test_r2 = r2_score(test_true, test_pred_day)

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

path = "xgboost"

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
plt.plot(hours, y_test_ghi[:864], label="Actual")
plt.plot(hours, test_pred_ghi[:864], label="Predicted")
plt.title("XGBoost GHI Pred vs Actual")
plt.legend()
plt.ylabel("GHI")
plt.xlabel("Hour")
plt.savefig("results/grid_results/" + path + ".pdf")
plt.show(block=False)