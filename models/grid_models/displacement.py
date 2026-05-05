import numpy as np
from scipy.signal import correlate2d
from math import atan2, degrees, sqrt

# DHI
# Mean displacement per hour: dx=-6.0000, dy=-6.0000
# Speed: 29.76 km per hour
# Direction: 133.9 degrees
# Downwind direction: NW
# Speed percentiles (km per hour):
# 5th percentile: 0.00
# 25th percentile: 20.00
# 50th percentile: 42.43
# 75th percentile: 42.43
# 90th percentile: 42.43
# 95th percentile: 42.43
# 99th percentile: 42.43

# DNI
# Mean displacement per hour: dx=-6.0000, dy=-6.0000
# Speed: 30.28 km per hour
# Direction: 134.0 degrees
# Downwind direction: NW
# Speed percentiles (km per hour):
# 5th percentile: 0.00
# 25th percentile: 20.00
# 50th percentile: 42.43
# 75th percentile: 42.43
# 90th percentile: 42.43
# 95th percentile: 42.43
# 99th percentile: 42.43

# GHI
# Mean displacement per hour: dx=0.0000, dy=0.0000
# Speed: 0.01 km per hour
# Direction: 135.0 degrees
# Downwind direction: NW
# Speed percentiles (km per hour):
# 5th percentile: 0.00
# 25th percentile: 0.00
# 50th percentile: 0.00
# 75th percentile: 0.00
# 90th percentile: 0.00
# 95th percentile: 0.00
# 99th percentile: 0.00

spacing_km = 5.0
grid = np.load("saves/preprocessed/train_grid.npy") # (T, F, 7, 7)

index = 19 # DHI = 15, DNI = 16, GHI = 19
target = grid[:, index, :, :]  # (T, 7, 7)

offset = 12
vectors = []

for t in range(len(target) - offset):
    A = target[t]
    B = target[t + offset]

    corr = correlate2d(B, A, mode='full')
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    dy = peak[0] - (A.shape[0] - 1)
    dx = peak[1] - (A.shape[1] - 1)

    vectors.append((dx, dy))

vectors = np.array(vectors)

# mean displacement
dx_mean = vectors[:, 0].mean()
dy_mean = vectors[:, 1].mean()

dx_km = dx_mean * spacing_km
dy_km = dy_mean * spacing_km
speed_km = sqrt(dx_km**2 + dy_km**2)

angle_math = degrees(atan2(-dy_mean, dx_mean)) % 360

def direction_16(angle):
    dirs = [
        "E", "ENE", "NE", "NNE",
        "N", "NNW", "NW", "WNW",
        "W", "WSW", "SW", "SSW",
        "S", "SSE", "SE", "ESE"
    ]
    ix = int((angle + 11.25) // 22.5) % 16
    return dirs[ix]

cardinal = direction_16(angle_math)

print(f"Mean displacement per hour: dx={dx:.4f}, dy={dy:.4f}")
print(f"Speed: {speed_km:.2f} km per hour")
print(f"Direction: {angle_math:.1f} degrees")
print(f"Downwind direction: {cardinal}")

dx = vectors[:, 0] * spacing_km
dy = vectors[:, 1] * spacing_km

speeds = np.sqrt(dx**2 + dy**2)

percentiles = [5, 25, 50, 75, 90, 95, 99]
speed_percentiles = np.percentile(speeds, percentiles)

print("Speed percentiles (km per hour):")
for p, val in zip(percentiles, speed_percentiles):
    print(f"{p}th percentile: {val:.2f}")