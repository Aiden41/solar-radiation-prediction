import os
import pandas as pd

columns = ["Year","Month","Day","Hour","Minute","Temperature","Aerosol Optical Depth","Wind Speed","Wind Direction","Precipitable Water","Pressure","Surface Albedo","SSA","Solar Zenith Angle","Relative Humidity","Cloud Type","Dew Point","GHI","Asymmetry","Alpha","DHI","DNI","Ozone","Clearsky GHI","Clearsky DNI","Clearsky DHI"]

for x in range(1,8):
    path = f"data/row{x}/"
    for folder in os.listdir(path):
        dataset = []
        for file in os.listdir(path + folder):
            dataframe = pd.read_csv(path + folder + "/" + file, skiprows=2)
            dataset.append(dataframe)
        new_dataset = pd.concat(dataset, axis=0)
        new_dataset = new_dataset[columns]
        new_path = path + folder + "/data.csv"
        new_dataset.to_csv(new_path, index=False)