import numpy as np

def downsample_dataset(dataset, lon_bin_size, lat_bin_size):
    # dataset: list of [lon, lat, data, ...]
    # lon_bin_size: size of the longitude bin (e.g., 1 degree)
    # lat_bin_size: size of the latitude bin (e.g., 1 degree)

    # Empty dictionary
    grid = {}

    for data_point in dataset:
        lon, lat = data_point[0], data_point[1]

        # Determine the grid cell
        lon_grid = int(lon // lon_bin_size)
        lat_grid = int(lat // lat_bin_size)

        grid_key = (lon_grid, lat_grid)

        # Select the representative point for each cell (here we choose the first one)
        if grid_key not in grid:
            grid[grid_key] = data_point

    # Extract the reduced dataset from the grid
    reduced_dataset = list(grid.values())

    return reduced_dataset

# Example dataset
dataset = [
    [12.34, 56.78, 'data0'],
    [12.35, 56.79, 'data1'],
    [13.34, 57.78, 'data2'],
    [13.35, 57.79, 'data3'],
]

lon_bin_size = 2.0  # example - 1 degree longitude bins
lat_bin_size = 2.0  # example - 1 degree latitude bins
reduced_dataset = downsample_dataset(dataset, lon_bin_size, lat_bin_size)

print("Original dataset size:", len(dataset))
print("Reduced dataset size:", len(reduced_dataset))
print("Reduced dataset:", reduced_dataset)