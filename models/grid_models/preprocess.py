import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

offset = 1 # adjust this!
horizon = 12 # adjust this!

all_x_train_raw = []
all_x_valid_raw = []
all_x_test_raw = []

rows = 7 # change these!!
path = "9x9_10km" # change these!!
out_path = f"saves/preprocessed/{rows}x{rows}_10km" # change these!!
os.makedirs(out_path, exist_ok=True)

max_rows = len(os.listdir("data/" + path))
center = (max_rows + 1) // 2
half = (rows - 1) // 2

start_index = center - half
end_index = center + half + 1

for x in range(start_index, end_index):
    for y in range(start_index, end_index):
        curr_path = "data/" + path + f"/row{x}/{y}/data.csv"
        dataset = pd.read_csv(curr_path, dtype=np.float32)
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

        cut = offset + horizon - 1
        train_dataset = train_dataset.iloc[:-cut]
        valid_dataset = valid_dataset.iloc[:-cut]
        test_dataset = test_dataset.iloc[:-cut]

        # all_columns = list(train_dataset.columns)

        # ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Solar Zenith Angle', 'Relative Humidity', 'GHI', 'DHI', 'DNI', 'Cloud Type_0.0', 'Cloud Type_1.0', 'Cloud Type_2.0', 'Cloud Type_3.0', 'Cloud Type_4.0', 'Cloud Type_5.0', 'Cloud Type_6.0', 'Cloud Type_7.0', 'Cloud Type_8.0', 'Cloud Type_9.0', 'Cloud Type_11.0', 'DayOfYear_Sin', 'DayOfYear_Cos', 'Sin_Hour', 'Cos_Hour', 'Sin_Month', 'Cos_Month', 'CSI']

        future_columns = []
        for h in range(horizon):
            future_columns += [f"Future_SZA_{h}", f"Future_GHI_{h}", f"Future_CSI_{h}", f"Future_CS_GHI_{h}"]

        drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

        x_train = train_dataset.drop(columns = drop_columns)
        x_valid = valid_dataset.drop(columns = drop_columns)
        x_test = test_dataset.drop(columns = drop_columns)

        remaining_columns = list(x_train.columns)
        # print(remaining_columns)

        x_train = x_train.astype(np.float32)
        x_valid = x_valid.astype(np.float32)
        x_test = x_test.astype(np.float32)

        all_x_train_raw.append(x_train.copy())
        all_x_valid_raw.append(x_valid.copy())
        all_x_test_raw.append(x_test.copy())

        if x == 4 and y == 4:
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

            future_csi_cols = [f"Future_CSI_{h}" for h in range(horizon)]
            y_train = train_dataset[future_csi_cols].to_numpy().astype(np.float32).copy()
            y_valid = valid_dataset[future_csi_cols].to_numpy().astype(np.float32).copy()
            y_test = test_dataset[future_csi_cols].to_numpy().astype(np.float32).copy()

            future_csghi_cols = [f"Future_CS_GHI_{h}" for h in range(horizon)]
            train_csghi = train_dataset[future_csghi_cols].to_numpy().astype(np.float32).copy()
            valid_csghi = valid_dataset[future_csghi_cols].to_numpy().astype(np.float32).copy()
            test_csghi = test_dataset[future_csghi_cols].to_numpy().astype(np.float32).copy()

            future_ghi_cols = [f"Future_GHI_{h}" for h in range(horizon)]
            y_train_ghi = train_dataset[future_ghi_cols].to_numpy().astype(np.float32).copy()
            y_valid_ghi = valid_dataset[future_ghi_cols].to_numpy().astype(np.float32).copy()
            y_test_ghi = test_dataset[future_ghi_cols].to_numpy().astype(np.float32).copy()

            sza_train = train_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy().astype(np.float32).copy()
            sza_valid = valid_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy().astype(np.float32).copy()
            sza_test = test_dataset[[f"Future_SZA_{h}" for h in range(horizon)]].to_numpy().astype(np.float32).copy()

# reorder to alphabetical
all_columns = sorted(set().union(*[df.columns for df in all_x_train_raw]))

# ['CSI', 'Cloud Type_0.0', 'Cloud Type_1.0', 'Cloud Type_11.0', 'Cloud Type_12.0', 'Cloud Type_2.0', 'Cloud Type_3.0', 'Cloud Type_4.0', 'Cloud Type_5.0', 'Cloud Type_6.0', 'Cloud Type_7.0', 'Cloud Type_8.0', 'Cloud Type_9.0', 'Cos_Hour', 'Cos_Month', 'DHI', 'DNI', 'DayOfYear_Cos', 'DayOfYear_Sin', 'GHI', 'Precipitable Water', 'Relative Humidity', 'SSA', 'Sin_Hour', 'Sin_Month', 'Wind Direction', 'Wind Speed']

# sort in case of missing cloud type feature
for i in range(49):
    all_x_train_raw[i] = all_x_train_raw[i].reindex(columns=all_columns, fill_value=0)
    all_x_valid_raw[i] = all_x_valid_raw[i].reindex(columns=all_columns, fill_value=0)
    all_x_test_raw[i] = all_x_test_raw[i].reindex(columns=all_columns, fill_value=0)

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']

# stack all training data from all 49 pixels
stacked = pd.concat([df[x_columns] for df in all_x_train_raw], axis=0)

global_scaler = StandardScaler().fit(stacked)

x_scaler = ColumnTransformer(
    [("scaler", global_scaler, x_columns)],
    remainder='passthrough'
)

x_scaler.fit(pd.concat(all_x_train_raw, axis=0))

scaled_x_train = []
scaled_x_valid = []
scaled_x_test = []

for i in range(49):
    Xtr = x_scaler.transform(all_x_train_raw[i]).astype(np.float32)
    Xva = x_scaler.transform(all_x_valid_raw[i]).astype(np.float32)
    Xte = x_scaler.transform(all_x_test_raw[i]).astype(np.float32)

    scaled_x_train.append(Xtr)
    scaled_x_valid.append(Xva)
    scaled_x_test.append(Xte)

train_weights = train_mask.to_numpy().astype(np.float32)
valid_weights = valid_mask.to_numpy().astype(np.float32)
test_weights = test_mask.to_numpy().astype(np.float32)

scaled_x_train = np.stack(scaled_x_train, axis=0)
scaled_x_valid = np.stack(scaled_x_valid, axis=0)
scaled_x_test = np.stack(scaled_x_test, axis=0)

def to_grid(arr):
    H = W = rows
    arr = arr.reshape(H, W, arr.shape[1], arr.shape[2])
    arr = arr.transpose(2, 3, 0, 1)
    return arr

train_grid = to_grid(scaled_x_train)
valid_grid = to_grid(scaled_x_valid)
test_grid = to_grid(scaled_x_test)

np.save(out_path + "/train_grid.npy", train_grid)
np.save(out_path + "/valid_grid.npy", valid_grid)
np.save(out_path + "/test_grid.npy", test_grid)

np.save(out_path + "/y_train.npy", y_train)
np.save(out_path + "/y_valid.npy", y_valid)
np.save(out_path + "/y_test.npy", y_test)

np.save(out_path + "/train_weights.npy", train_weights)
np.save(out_path + "/valid_weights.npy", valid_weights)
np.save(out_path + "/test_weights.npy", test_weights)

np.save(out_path + "/averages.npy", np.array([train_average_GHI, valid_average_GHI, test_average_GHI], dtype=np.float32))

np.save(out_path + "/future_sza_train.npy", sza_train)
np.save(out_path + "/future_sza_valid.npy", sza_valid)
np.save(out_path + "/future_sza_test.npy", sza_test)

np.save(out_path + "/future_cs_ghi_train.npy", train_csghi)
np.save(out_path + "/future_cs_ghi_valid.npy", valid_csghi)
np.save(out_path + "/future_cs_ghi_test.npy", test_csghi)

np.save(out_path + "/y_train_ghi.npy", y_train_ghi)
np.save(out_path + "/y_valid_ghi.npy", y_valid_ghi)
np.save(out_path + "/y_test_ghi.npy", y_test_ghi)
