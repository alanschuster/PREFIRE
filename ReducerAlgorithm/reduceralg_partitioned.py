import numpy as np

def partition_data(dataset, lon_bin_size, lat_bin_size):
    # Create an empty dictionary to store bins
    bins = {}

    for data_point in dataset:
        lon, lat = data_point[0], data_point[1]

        # Determine the grid cell
        lon_grid = int(lon // lon_bin_size)
        lat_grid = int(lat // lat_bin_size)

        grid_key = (lon_grid, lat_grid)

        # Add data points to the corresponding bin
        if grid_key not in bins:
            bins[grid_key] = []
        bins[grid_key].append(data_point)

    return bins

def map_data_to_bin(bins):
    reduced_dataset = []

    for bin_key, bin_data in bins.items():
        if bin_data:
            # Calculate the mean longitude, latitude, and other data
            mean_lon = np.mean([point[0] for point in bin_data])
            mean_lat = np.mean([point[1] for point in bin_data])
            mean_data = [mean_lon, mean_lat]

            # If there are additional data fields, calculate their mean as well
            for i in range(2, len(bin_data[0])):
                mean_value = np.mean([point[i] for point in bin_data])
                mean_data.append(mean_value)

            reduced_dataset.append(mean_data)

    return reduced_dataset

def downsample_dataset(dataset, lon_bin_size, lat_bin_size):
    # Partition the data into bins
    bins = partition_data(dataset, lon_bin_size, lat_bin_size)

    # Map data to bins around the mean
    reduced_dataset = map_data_to_bin(bins)

    return reduced_dataset

# Example dataset
dataset = [
    [12.34, 56.78, 0.5],
    [12.35, 56.79, 0.6],
    [13.34, 57.78, 0.7],
    [13.35, 57.79, 0.8],
]

lon_bin_size = 1.0  # example - 1 degree longitude bins
lat_bin_size = 1.0  # example - 1 degree latitude bins

reduced_dataset = downsample_dataset(dataset, lon_bin_size, lat_bin_size)

print("Original dataset size:", len(dataset))
print("Reduced dataset size:", len(reduced_dataset))
print("Reduced dataset:", reduced_dataset)
