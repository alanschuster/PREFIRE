import numpy as np

# Generate random input data
input_data = np.random.randn(100, 10)

# Save to a text file
np.savetxt('inputs.txt', input_data, delimiter=' ')