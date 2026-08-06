import os
import time
import requests
from dotenv import load_dotenv
from urllib.parse import quote

grid_size = 9 # change these!!
km_spacing = 10 # change these!!
path = "data/9x9_10km" # change these!!

load_dotenv()
API_KEY = os.getenv('NLR_API_KEY')

years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
interval = 5 # 5 minute rows
attributes = "all"

center_lat = 47.61
center_lon = -122.34

degree_lat = km_spacing / 111.0
degree_lon = km_spacing / 75.0

def build_grid():
    half = grid_size // 2
    grid = []

    for r in range(grid_size):
        for c in range(grid_size):
            lat_offset = (r - half) * degree_lat
            lon_offset = (c - half) * degree_lon

            lat = center_lat + lat_offset
            lon = center_lon + lon_offset

            grid.append((lat, lon))
    return grid

if grid_size > 1:
    grid_points = build_grid()

def download_csv(lat, lon, year, row, col):
    wkt = quote(f"POINT({lon} {lat})")
    url = (
        "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv?"
        f"api_key={API_KEY}"
        f"&wkt={wkt}"
        f"&names={year}"
        f"&interval={interval}"
        f"&email={os.getenv('NLR_EMAIL')}"
    )

    if grid_size == 1:
        folder = path
    else:
        folder = f"{path}/row{row}/{col}"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{year}.csv"

    print(f"Downloading {year} for ({lat}, {lon}): {filename}")

    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
    else:
        print(f"FAILED {year} for ({lat}, {lon}): {response.status_code}")
        print(response.text)

if grid_size == 1:
    lat, lon = center_lat, center_lon
    for year in years:
        download_csv(lat, lon, year, 1, 1)
        time.sleep(2)
else:
    index = 0
    for r in range(1, grid_size+1):
        for c in range(1, grid_size+1):
            lat, lon = grid_points[index]
            index += 1

            for year in years:
                download_csv(lat, lon, year, r, c)
                time.sleep(2)