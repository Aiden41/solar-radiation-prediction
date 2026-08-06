import os
import pandas as pd

columns = ["Year","Month","Day","Hour","Minute","Temperature","Aerosol Optical Depth","Wind Speed","Wind Direction","Precipitable Water","Pressure","Surface Albedo","SSA","Solar Zenith Angle","Relative Humidity","Cloud Type","Dew Point","GHI","Asymmetry","Alpha","DHI","DNI","Ozone","Clearsky GHI","Clearsky DNI","Clearsky DHI"]

path = "data/target/"
dataset = []
for file in os.listdir(path):
    if file == "data.csv":
        continue
    dataframe = pd.read_csv(path + file, skiprows=2)
    dataset.append(dataframe)
new_dataset = pd.concat(dataset, axis=0)
new_dataset = new_dataset[columns]
new_path = path + "data.csv"
new_dataset.to_csv(new_path, index=False)