import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

offset = 1 # number of hours ahead to predict

all_x_train_raw = []
all_x_valid_raw = []
all_x_test_raw = []

all_y = []
averages = []
masks = []

for x in range(1,8):
    for y in range(1,8):
        path = f"data/row{x}/{y}/data.csv"

        dataset = pd.read_csv(path)
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
        train_dataset["Future_SZA"] = train_dataset["Solar Zenith Angle"].shift(-1)
        valid_dataset["Future_SZA"] = valid_dataset["Solar Zenith Angle"].shift(-1)
        test_dataset["Future_SZA"] = test_dataset["Solar Zenith Angle"].shift(-1)

        train_dataset["Future_CS_GHI"] = train_dataset["Clearsky GHI"].shift(-1)
        valid_dataset["Future_CS_GHI"] = valid_dataset["Clearsky GHI"].shift(-1)
        test_dataset["Future_CS_GHI"] = test_dataset["Clearsky GHI"].shift(-1)

        train_dataset["Future_CS_DNI"] = train_dataset["Clearsky DNI"].shift(-1)
        valid_dataset["Future_CS_DNI"] = valid_dataset["Clearsky DNI"].shift(-1)
        test_dataset["Future_CS_DNI"] = test_dataset["Clearsky DNI"].shift(-1)

        train_dataset["Future_CS_DHI"] = train_dataset["Clearsky DHI"].shift(-1)
        valid_dataset["Future_CS_DHI"] = valid_dataset["Clearsky DHI"].shift(-1)
        test_dataset["Future_CS_DHI"] = test_dataset["Clearsky DHI"].shift(-1)

        train_dataset["Future_GHI"] = train_dataset["GHI"].shift(-1)
        valid_dataset["Future_GHI"] = valid_dataset["GHI"].shift(-1)
        test_dataset["Future_GHI"] = test_dataset["GHI"].shift(-1)

        train_dataset["Future_CSI"] = train_dataset["CSI"].shift(-1)
        valid_dataset["Future_CSI"] = valid_dataset["CSI"].shift(-1)
        test_dataset["Future_CSI"] = test_dataset["CSI"].shift(-1)

        train_dataset = train_dataset.iloc[:-1]
        valid_dataset = valid_dataset.iloc[:-1]
        test_dataset = test_dataset.iloc[:-1]

        # all_columns = list(train_dataset.columns)

        future_columns = ['Future_SZA', 'Future_CS_GHI', 'Future_CS_DNI', 'Future_CS_DHI', 'Future_GHI', 'Future_CSI']
        drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Solar Zenith Angle', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

        x_train = train_dataset.drop(columns = drop_columns)
        x_valid = valid_dataset.drop(columns = drop_columns)
        x_test = test_dataset.drop(columns = drop_columns)

        remaining_columns = list(x_train.columns)
        # print(remaining_columns)

        all_x_train_raw.append(x_train)
        all_x_valid_raw.append(x_valid)
        all_x_test_raw.append(x_test)

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

            masks = [train_mask.copy(), valid_mask.copy(), test_mask.copy()]

            y_train = train_dataset[["Future_CSI"]].to_numpy().copy()
            y_valid = valid_dataset[["Future_CSI"]].to_numpy().copy()
            y_test = test_dataset[["Future_CSI"]].to_numpy().copy()
            all_y.append(y_train.copy())
            all_y.append(y_valid.copy())
            all_y.append(y_test.copy())
            averages.append(train_average_GHI.copy())
            averages.append(valid_average_GHI.copy())
            averages.append(test_average_GHI.copy())
            future_sza_train = train_dataset["Future_SZA"].to_numpy().copy()
            future_sza_valid = valid_dataset["Future_SZA"].to_numpy().copy()
            future_sza_test = test_dataset["Future_SZA"].to_numpy().copy()

            future_cs_ghi_train = train_dataset["Future_CS_GHI"].to_numpy().copy()
            future_cs_ghi_valid = valid_dataset["Future_CS_GHI"].to_numpy().copy()
            future_cs_ghi_test = test_dataset["Future_CS_GHI"].to_numpy().copy()

            y_train_ghi = train_dataset["Future_GHI"].to_numpy().copy()
            y_valid_ghi = valid_dataset["Future_GHI"].to_numpy().copy()
            y_test_ghi = test_dataset["Future_GHI"].to_numpy().copy()


# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']

# stack all training data from all 49 pixels
stacked = pd.concat([df[x_columns] for df in all_x_train_raw], axis=0)

global_scaler = StandardScaler().fit(stacked)

x_scaler = ColumnTransformer(
    [("scaler", global_scaler, x_columns)],
    remainder='passthrough'
)

x_scaler.fit(all_x_train_raw[0])

scaled_x_train = []
scaled_x_valid = []
scaled_x_test = []

for i in range(49):
    Xtr = x_scaler.transform(all_x_train_raw[i])
    Xva = x_scaler.transform(all_x_valid_raw[i])
    Xte = x_scaler.transform(all_x_test_raw[i])

    scaled_x_train.append(Xtr)
    scaled_x_valid.append(Xva)
    scaled_x_test.append(Xte)

y_train = all_y[0]
y_valid = all_y[1]
y_test = all_y[2]

train_weights = np.where(masks[0], 1.0, 0.0)
valid_weights = np.where(masks[1], 1.0, 0.0)
test_weights = np.where(masks[2], 1.0, 0.0)

scaled_x_train = np.stack(scaled_x_train, axis=0)
scaled_x_valid = np.stack(scaled_x_valid, axis=0)
scaled_x_test = np.stack(scaled_x_test, axis=0)

np.save("saves/preprocessed/x_train.npy", scaled_x_train)
np.save("saves/preprocessed/x_valid.npy", scaled_x_valid)
np.save("saves/preprocessed/x_test.npy", scaled_x_test)

np.save("saves/preprocessed/y_train.npy", y_train)
np.save("saves/preprocessed/y_valid.npy", y_valid)
np.save("saves/preprocessed/y_test.npy", y_test)

np.save("saves/preprocessed/train_weights.npy", train_weights)
np.save("saves/preprocessed/valid_weights.npy", valid_weights)
np.save("saves/preprocessed/test_weights.npy", test_weights)

np.save("saves/preprocessed/averages.npy", np.array(averages))

np.save("saves/preprocessed/future_sza_train.npy", future_sza_train)
np.save("saves/preprocessed/future_sza_valid.npy", future_sza_valid)
np.save("saves/preprocessed/future_sza_test.npy", future_sza_test)

np.save("saves/preprocessed/future_cs_ghi_train.npy", future_cs_ghi_train)
np.save("saves/preprocessed/future_cs_ghi_valid.npy", future_cs_ghi_valid)
np.save("saves/preprocessed/future_cs_ghi_test.npy", future_cs_ghi_test)

np.save("saves/preprocessed/y_train_ghi.npy", y_train_ghi)
np.save("saves/preprocessed/y_valid_ghi.npy", y_valid_ghi)
np.save("saves/preprocessed/y_test_ghi.npy", y_test_ghi)
