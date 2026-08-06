import os
import pandas as pd

columns = ["Year","Month","Day","Hour","Minute","Temperature","Aerosol Optical Depth","Wind Speed","Wind Direction","Precipitable Water","Pressure","Surface Albedo","SSA","Solar Zenith Angle","Relative Humidity","Cloud Type","Dew Point","GHI","Asymmetry","Alpha","DHI","DNI","Ozone","Clearsky GHI","Clearsky DNI","Clearsky DHI"]

path = "data/7x7_5km" # change this!!
rows = len(os.listdir("data/" + path))

for x in range(1,rows+1):
    curr_path = path + f"/row{x}/"
    for folder in os.listdir(curr_path):
        dataset = []
        for file in os.listdir(curr_path + folder):
            if file == "data.csv":
                continue
            dataframe = pd.read_csv(curr_path + folder + "/" + file, skiprows=2)
            dataset.append(dataframe)
        new_dataset = pd.concat(dataset, axis=0)
        new_dataset = new_dataset[columns]
        new_path = curr_path + folder + "/data.csv"
        new_dataset.to_csv(new_path, index=False)