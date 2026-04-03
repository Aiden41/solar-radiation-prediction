import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

offset = 12 # adjust this!

all_x_train_raw = []
all_x_valid_raw = []
all_x_test_raw = []

for x in range(1,8):
    for y in range(1,8):
        path = f"data/5min/row{x}/{y}/data.csv" # adjust this!

        dataset = pd.read_csv(path, dtype=np.float32)
        dataset = pd.get_dummies(dataset, columns=['Cloud Type'], dtype=int)
        mask = dataset['Year'] <= 2022
        train= dataset[mask].copy()
        mask = dataset['Year'] == 2023
        valid = dataset[mask].copy()
        mask = dataset['Year'] == 2024
        test = dataset[mask].copy()

        train.reset_index(drop=True, inplace=True) # set first index to 0
        valid.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        # create day of year column
        train['DayOfYear'] = pd.to_datetime(train[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
        # fix leap year day count
        train['DayOfYear'] = train['DayOfYear'].mask((train['DayOfYear'] >= 60) & ((train['Year'] == 2020)), train['DayOfYear']-1)

        valid['DayOfYear'] = pd.to_datetime(valid[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)

        # same for test set
        test['DayOfYear'] = pd.to_datetime(test[['Year', 'Month', 'Day']]).apply(lambda x: x.timetuple().tm_yday)
        test['DayOfYear'] = test['DayOfYear'].mask((test['DayOfYear'] >= 60) & ((test['Year'] == 2024)), test['DayOfYear']-1)

        # create cyclical columns
        train['DayOfYear_Sin'] = np.sin(2 * np.pi * train['DayOfYear'] / max(train['DayOfYear'])) 
        train['DayOfYear_Cos'] = np.cos(2 * np.pi * train['DayOfYear'] / max(train['DayOfYear']))
        valid['DayOfYear_Sin'] = np.sin(2 * np.pi * valid['DayOfYear'] / max(valid['DayOfYear'])) 
        valid['DayOfYear_Cos'] = np.cos(2 * np.pi * valid['DayOfYear'] / max(valid['DayOfYear']))
        test['DayOfYear_Sin'] = np.sin(2 * np.pi * test['DayOfYear'] / max(test['DayOfYear'])) 
        test['DayOfYear_Cos'] = np.cos(2 * np.pi * test['DayOfYear'] / max(test['DayOfYear']))

        train['Sin_Hour'] = np.sin(2 * np.pi * train['Hour'] / (max(train['Hour'])+1)) 
        train['Cos_Hour'] = np.cos(2 * np.pi * train['Hour'] / (max(train['Hour'])+1))
        valid['Sin_Hour'] = np.sin(2 * np.pi * valid['Hour'] / (max(valid['Hour'])+1))
        valid['Cos_Hour'] = np.cos(2 * np.pi * valid['Hour'] / (max(valid['Hour'])+1))
        test['Sin_Hour'] = np.sin(2 * np.pi * test['Hour'] / (max(test['Hour'])+1))
        test['Cos_Hour'] = np.cos(2 * np.pi * test['Hour'] / (max(test['Hour'])+1))

        train['Sin_Month'] = np.sin(2 * np.pi * train['Month'] / max(train['Month'])) 
        train['Cos_Month'] = np.cos(2 * np.pi * train['Month'] / max(train['Month']))
        valid['Sin_Month'] = np.sin(2 * np.pi * valid['Month'] / max(valid['Month'])) 
        valid['Cos_Month'] = np.cos(2 * np.pi * valid['Month'] / max(valid['Month']))
        test['Sin_Month'] = np.sin(2 * np.pi * test['Month'] / max(test['Month']))
        test['Cos_Month'] = np.cos(2 * np.pi * test['Month'] / max(test['Month']))

        # Calculate clear sky index for each row
        train['CSI'] = train['GHI'] / train['Clearsky GHI']
        valid['CSI'] = valid['GHI'] / valid['Clearsky GHI']
        test['CSI'] = test['GHI'] / test['Clearsky GHI']
        train['CSI'] = train['CSI'].fillna(0.0)
        valid['CSI'] = valid['CSI'].fillna(0.0)
        test['CSI'] = test['CSI'].fillna(0.0)
        mask = train['Solar Zenith Angle'] >= 90
        train.loc[mask, 'CSI'] = 0.0
        mask = valid['Solar Zenith Angle'] >= 90
        valid.loc[mask, 'CSI'] = 0.0
        mask = test['Solar Zenith Angle'] >= 90
        test.loc[mask, 'CSI'] = 0.0

        # move up deterministic columns and place into new columns
        train["Future_SZA"] = train["Solar Zenith Angle"].shift(-offset)
        valid["Future_SZA"] = valid["Solar Zenith Angle"].shift(-offset)
        test["Future_SZA"] = test["Solar Zenith Angle"].shift(-offset)

        train["Future_CS_GHI"] = train["Clearsky GHI"].shift(-offset)
        valid["Future_CS_GHI"] = valid["Clearsky GHI"].shift(-offset)
        test["Future_CS_GHI"] = test["Clearsky GHI"].shift(-offset)

        train["Future_CS_DNI"] = train["Clearsky DNI"].shift(-offset)
        valid["Future_CS_DNI"] = valid["Clearsky DNI"].shift(-offset)
        test["Future_CS_DNI"] = test["Clearsky DNI"].shift(-offset)

        train["Future_CS_DHI"] = train["Clearsky DHI"].shift(-offset)
        valid["Future_CS_DHI"] = valid["Clearsky DHI"].shift(-offset)
        test["Future_CS_DHI"] = test["Clearsky DHI"].shift(-offset)

        train["Future_GHI"] = train["GHI"].shift(-offset)
        valid["Future_GHI"] = valid["GHI"].shift(-offset)
        test["Future_GHI"] = test["GHI"].shift(-offset)

        train["Future_CSI"] = train["CSI"].shift(-offset)
        valid["Future_CSI"] = valid["CSI"].shift(-offset)
        test["Future_CSI"] = test["CSI"].shift(-offset)

        train = train.iloc[:-offset]
        valid = valid.iloc[:-offset]
        test = test.iloc[:-offset]
        
        if x == 4 and y == 4:
            lag_vars = ["CSI", "GHI", "DNI", "DHI",
                        "Wind Speed", "Wind Direction",
                        "Relative Humidity", "Precipitable Water"]

            lags = range(1, 7)

            def lag_block(df):
                out = {}
                for var in lag_vars:
                    for L in lags:
                        out[f"{var}_t{L}"] = df[var].shift(L)
                return pd.DataFrame(out)

            train = pd.concat([train, lag_block(train)], axis=1)
            valid = pd.concat([valid, lag_block(valid)], axis=1)
            test = pd.concat([test, lag_block(test)], axis=1)

            trim = max(lags)

            # train = train.iloc[trim:].reset_index(drop=True)
            # valid = valid.iloc[trim:].reset_index(drop=True)
            # test = test.iloc[trim:].reset_index(drop=True)

        # all_columns = list(train.columns)

        future_columns = ['Future_SZA', 'Future_CS_GHI', 'Future_CS_DNI', 'Future_CS_DHI', 'Future_GHI', 'Future_CSI']
        drop_columns = ['Minute', 'Month', 'Hour', 'Year', 'Day', 'DayOfYear', 'Temperature', 'Alpha', 'Ozone', 'Dew Point', 'Surface Albedo', 'Pressure', 'Aerosol Optical Depth', 'Asymmetry', 'Solar Zenith Angle', 'Clearsky DNI', 'Clearsky DHI', 'Clearsky GHI'] + future_columns

        x_train = train.drop(columns = drop_columns)
        x_valid = valid.drop(columns = drop_columns)
        x_test = test.drop(columns = drop_columns)

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
            train_mask = (train['Solar Zenith Angle'] < 90)
            valid_mask = (valid['Solar Zenith Angle'] < 90)
            test_mask = (test['Solar Zenith Angle'] < 90)
            daytime_average = np.mean(train['CSI'][train_mask])
            train_average_GHI = np.mean(train['GHI'][train_mask])
            valid_average = np.mean(valid['CSI'][valid_mask])
            valid_average_GHI = np.mean(valid['GHI'][valid_mask])
            test_average = np.mean(test['CSI'][test_mask])
            test_average_GHI = np.mean(test['GHI'][test_mask])

            y_train = train[["Future_CSI"]].to_numpy().copy()
            y_valid = valid[["Future_CSI"]].to_numpy().copy()
            y_test = test[["Future_CSI"]].to_numpy().copy()

            future_sza_train = train["Future_SZA"].to_numpy().copy()
            future_sza_valid = valid["Future_SZA"].to_numpy().copy()
            future_sza_test = test["Future_SZA"].to_numpy().copy()

            future_cs_ghi_train = train["Future_CS_GHI"].to_numpy().copy()
            future_cs_ghi_valid = valid["Future_CS_GHI"].to_numpy().copy()
            future_cs_ghi_test = test["Future_CS_GHI"].to_numpy().copy()

            y_train_ghi = train["Future_GHI"].to_numpy().copy()
            y_valid_ghi = valid["Future_GHI"].to_numpy().copy()
            y_test_ghi = test["Future_GHI"].to_numpy().copy()

            y_train = y_train.astype(np.float32)
            y_valid = y_valid.astype(np.float32)
            y_test = y_test.astype(np.float32)

            y_train_ghi = y_train_ghi.astype(np.float32)
            y_valid_ghi = y_valid_ghi.astype(np.float32)
            y_test_ghi = y_test_ghi.astype(np.float32)

            future_sza_train = future_sza_train.astype(np.float32)
            future_sza_valid = future_sza_valid.astype(np.float32)
            future_sza_test = future_sza_test.astype(np.float32)

            future_cs_ghi_train = future_cs_ghi_train.astype(np.float32)
            future_cs_ghi_valid = future_cs_ghi_valid.astype(np.float32)
            future_cs_ghi_test = future_cs_ghi_test.astype(np.float32)

# Apply center-pixel lag trim to ALL pixels and ALL arrays
for i in range(49):
    all_x_train_raw[i] = all_x_train_raw[i].iloc[trim:].reset_index(drop=True)
    all_x_valid_raw[i] = all_x_valid_raw[i].iloc[trim:].reset_index(drop=True)
    all_x_test_raw[i] = all_x_test_raw[i].iloc[trim:].reset_index(drop=True)

# Trim labels and masks
y_train = y_train[trim:]
y_valid = y_valid[trim:]
y_test = y_test[trim:]

train_mask = train_mask[trim:]
valid_mask = valid_mask[trim:]
test_mask = test_mask[trim:]

future_sza_train = future_sza_train[trim:]
future_sza_valid = future_sza_valid[trim:]
future_sza_test = future_sza_test[trim:]

future_cs_ghi_train = future_cs_ghi_train[trim:]
future_cs_ghi_valid = future_cs_ghi_valid[trim:]
future_cs_ghi_test = future_cs_ghi_test[trim:]

y_train_ghi = y_train_ghi[trim:]
y_valid_ghi = y_valid_ghi[trim:]
y_test_ghi = y_test_ghi[trim:]

all_columns = sorted(set().union(*[df.columns for df in all_x_train_raw]))

for i in range(49):
    all_x_train_raw[i] = all_x_train_raw[i].reindex(columns=all_columns, fill_value=0)
    all_x_valid_raw[i] = all_x_valid_raw[i].reindex(columns=all_columns, fill_value=0)
    all_x_test_raw[i] = all_x_test_raw[i].reindex(columns=all_columns, fill_value=0)

# get column titles for ColumnTransformer, excluding cyclical features
x_columns = ['Wind Speed', 'Wind Direction', 'Precipitable Water', 'SSA', 'Relative Humidity']

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

scaled_x_train = scaled_x_train.astype(np.float32)
scaled_x_valid = scaled_x_valid.astype(np.float32)
scaled_x_test = scaled_x_test.astype(np.float32)

np.save("saves/lagged_preprocessed/x_train.npy", scaled_x_train)
np.save("saves/lagged_preprocessed/x_valid.npy", scaled_x_valid)
np.save("saves/lagged_preprocessed/x_test.npy", scaled_x_test)

np.save("saves/lagged_preprocessed/y_train.npy", y_train)
np.save("saves/lagged_preprocessed/y_valid.npy", y_valid)
np.save("saves/lagged_preprocessed/y_test.npy", y_test)

np.save("saves/lagged_preprocessed/train_weights.npy", train_weights)
np.save("saves/lagged_preprocessed/valid_weights.npy", valid_weights)
np.save("saves/lagged_preprocessed/test_weights.npy", test_weights)

np.save("saves/lagged_preprocessed/averages.npy", np.array([train_average_GHI, valid_average_GHI, test_average_GHI], dtype=np.float32))

np.save("saves/lagged_preprocessed/future_sza_train.npy", future_sza_train)
np.save("saves/lagged_preprocessed/future_sza_valid.npy", future_sza_valid)
np.save("saves/lagged_preprocessed/future_sza_test.npy", future_sza_test)

np.save("saves/lagged_preprocessed/future_cs_ghi_train.npy", future_cs_ghi_train)
np.save("saves/lagged_preprocessed/future_cs_ghi_valid.npy", future_cs_ghi_valid)
np.save("saves/lagged_preprocessed/future_cs_ghi_test.npy", future_cs_ghi_test)

np.save("saves/lagged_preprocessed/y_train_ghi.npy", y_train_ghi)
np.save("saves/lagged_preprocessed/y_valid_ghi.npy", y_valid_ghi)
np.save("saves/lagged_preprocessed/y_test_ghi.npy", y_test_ghi)
