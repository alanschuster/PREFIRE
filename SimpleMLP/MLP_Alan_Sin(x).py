import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Function to generate inputs and targets
def generate_data(num_samples, input_size):
    x = np.random.rand(num_samples, input_size) * 2 * np.pi  # Random values in the range [0, 2*pi)
    y = np.sin(x)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

input_size = 10   # Example input size
hidden_size = 5   # Example hidden layer size 
output_size = 1   # Example output size

# Create the model
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Example loss function (Mean Squared Error)

# Test different learning rates
learning_rates = [0.2, 0.4, 0.6]

num_samples = 100  # Number of samples
num_epochs = 50

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Optimizer with current learning rate
    
    # Generate data
    inputs, targets = generate_data(num_samples, input_size)
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Print the loss every epoch
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
