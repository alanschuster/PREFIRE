import numpy as np

def downsample_dataset(dataset, lon_bin_size, lat_bin_size):
    # dataset: list of [lon, lat, data, ...]
    # lon_bin_size: size of the longitude bin (e.g., 1 degree)
    # lat_bin_size: size of the latitude bin (e.g., 1 degree)

    # Empty dictionary to store bins
    bins = {}

    for data_point in dataset:
        lon, lat = data_point[0], data_point[1]

        # Determine the grid cell
        lon_grid = int(lon // lon_bin_size)
        lat_grid = int(lat // lat_bin_size)

        grid_key = (lon_grid, lat_grid)

        # Initialize the bin if not already present
        if grid_key not in bins:
            bins[grid_key] = []

        # Append the data point to the corresponding bin
        bins[grid_key].append(data_point)

    # Calculate the mean of the data points within each bin
    reduced_dataset = []
    for grid_key, data_points in bins.items():
        # Calculate the mean longitude and latitude
        mean_lon = np.mean([point[0] for point in data_points])
        mean_lat = np.mean([point[1] for point in data_points])
        
        # Take the first data value as the representative value
        mean_data = data_points[0][2:]

        # Create the reduced data point
        reduced_data_point = [mean_lon, mean_lat] + mean_data
        reduced_dataset.append(reduced_data_point)

    return reduced_dataset

# Example dataset
dataset = [
    [12.34, 56.78, 'data0'],
    [12.35, 56.79, 'data1'],
    [13.34, 57.78, 'data2'],
    [13.35, 57.79, 'data3'],
]

lon_bin_size = 2.0  # example - 2 degree longitude bins
lat_bin_size = 2.0  # example - 2 degree latitude bins
reduced_dataset = downsample_dataset(dataset, lon_bin_size, lat_bin_size)

print("Original dataset size:", len(dataset))
print("Reduced dataset size:", len(reduced_dataset))
print("Reduced dataset:", reduced_dataset)
