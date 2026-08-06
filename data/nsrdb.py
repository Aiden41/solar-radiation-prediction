import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from urllib.parse import quote

grid_size = 9 # change these!!
km_spacing = 5 # change these!!
path = f"data/{grid_size}x{grid_size}_{km_spacing}km"

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

async def worker(session, queue, retry_queue):
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return
        
        lat, lon, year, row, col = job
        wkt = quote(f"POINT({lon} {lat})")
        url = (
            "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv?"
            f"api_key={API_KEY}"
            f"&wkt={wkt}"
            f"&names={year}"
            f"&interval={interval}"
            f"&email={EMAIL}"
        )

        folder = path if grid_size == 1 else f"{path}/row{row}/{col}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{year}.csv"

        print(f"Downloading {year} for ({lat}, {lon}): {filename}")

        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, "wb") as f:
                    f.write(await response.read())
                print(f"Saved {filename}")
            else:
                print(f"FAILED {year} for ({lat}, {lon}): {response.status}")
                print(await response.text())
                await retry_queue.put(job)

        queue.task_done()

async def scheduler(queue, retry_queue, jobs):
    for job in jobs:
        lat, lon, year, row, col = job
        folder = path if grid_size == 1 else f"{path}/row{row}/{col}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{year}.csv"

        if os.path.exists(filename):
            print(f"Skipping {year} for ({lat}, {lon}): {filename} already exists")
            continue

        await asyncio.sleep(2)
        await queue.put(job)

        while not retry_queue.empty():
            retry_job = await retry_queue.get()
            await asyncio.sleep(2)
            await queue.put(retry_job)
            retry_queue.task_done()

    for _ in range(15):
        await queue.put(None)

async def main():
    queue = asyncio.Queue()
    retry_queue = asyncio.Queue()
    jobs = []

    index = 0
    for r in range(1, grid_size+1):
        for c in range(1, grid_size+1):
            lat, lon = grid_points[index]
            index += 1
            for year in years:
                jobs.append((lat, lon, year, r, c))

    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(session, queue, retry_queue)) for _ in range(15)]
        sched = asyncio.create_task(scheduler(queue, retry_queue, jobs))
        await sched
        await queue.join()
        await asyncio.gather(*workers)

asyncio.run(main())