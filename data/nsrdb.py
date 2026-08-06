import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from urllib.parse import quote

from requests import session

grid_size = 9 # change these!!
km_spacing = 10 # change these!!
path = "data/9x9_10km" # change these!!

load_dotenv()
API_KEY = os.getenv('NLR_API_KEY')
EMAIL = os.getenv('NLR_EMAIL')

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

rate_lock = asyncio.Lock()

async def rate_limited():
    async with rate_lock:
        await asyncio.sleep(1)

async def download_csv(session, lat, lon, year, row, col):
    wkt = quote(f"POINT({lon} {lat})")
    url = (
        "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv?"
        f"api_key={API_KEY}"
        f"&wkt={wkt}"
        f"&names={year}"
        f"&interval={interval}"
        f"&email={EMAIL}"
    )

    if grid_size == 1:
        folder = path
    else:
        folder = f"{path}/row{row}/{col}"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{year}.csv"

    if os.path.exists(filename):
        print(f"Skipping {year} for ({lat}, {lon}): {filename} already exists")
        return

    print(f"Downloading {year} for ({lat}, {lon}): {filename}")

    await rate_limited()
    async with session.get(url) as response:
        if response.status == 200:
            with open(filename, "wb") as f:
                f.write(await response.read())
            print(f"Saved {filename}")
        else:
            print(f"FAILED {year} for ({lat}, {lon}): {response.status_code}")
            print(response.text)

async def main():
    tasks = []
    async with aiohttp.ClientSession() as session:
        if grid_size == 1:
            lat, lon = center_lat, center_lon
            for year in years:
                tasks.append(download_csv(session, lat, lon, year, 1, 1))
        else:
            index = 0
            for r in range(1, grid_size+1):
                for c in range(1, grid_size+1):
                    lat, lon = grid_points[index]
                    index += 1

                    for year in years:
                        tasks.append(download_csv(session, lat, lon, year, r, c))
        sem = asyncio.Semaphore(20)
        async def sem_task(task):
            async with sem:
                await task
        await asyncio.gather(*[sem_task(task) for task in tasks])

asyncio.run(main())