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

def compute_dist(point, mean_data):
    dist = np.sqrt((point[0] - mean_data[0])**2 + (point[1] - mean_data[1])**2)
    for i in range(2, len(mean_data)):
        dist += (point[i] - mean_data[i])**2
    return dist

def reduce_bin_dataset(bin_dataset, topk=3, reduce_alg='mean'):
    new_bin_dataset = {}

    for bin_key, bin_data in bin_dataset.items():
        if bin_data:
            # Calculate the mean of the bin data
            mean_lon = np.mean([point[0] for point in bin_data])
            mean_lat = np.mean([point[1] for point in bin_data])
            mean_data = [mean_lon, mean_lat]

            for i in range(2, len(bin_data[0])):
                mean_value = np.mean([point[i] for point in bin_data])
                mean_data.append(mean_value)

            # Find the topk data points closest to the mean
            distances = [(point, compute_dist(point, mean_data)) for point in bin_data]
            distances.sort(key=lambda x: x[1])
            topk_data = [point for point, _ in distances[:topk]]

            new_bin_dataset[bin_key] = topk_data

    return new_bin_dataset

def downsample_dataset(dataset, lon_bin_size, lat_bin_size, topk=3, reduce_alg='mean'):
    # Partition the data into bins
    bins = partition_data(dataset, lon_bin_size, lat_bin_size)

    # Reduce the bin dataset
    reduced_dataset = reduce_bin_dataset(bins, topk, reduce_alg)

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
print("Reduced dataset size:", sum(len(v) for v in reduced_dataset.values()))
print("Reduced dataset:", reduced_dataset)
