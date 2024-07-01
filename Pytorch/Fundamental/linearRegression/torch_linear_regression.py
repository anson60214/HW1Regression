import numpy as np
import torch

# Set a manual seed for reproducibility
torch.manual_seed(0)

# Define the input data X and target values y
# X is a (4x2) matrix and y is a (4x1) matrix
X = torch.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=torch.float32)
y = torch.tensor([[8], [13], [26], [9]], dtype=torch.float32)

# Initialize the weights w randomly, requires gradient for optimization
w = torch.rand(2, 1, requires_grad=True, dtype=torch.float32)

# Set the number of iterations and learning rate
iter_count = 500
lr = 0.02

# Define the forward pass function to calculate the predicted values y_pred
def forward(X):
    return torch.matmul(X, w)  # Matrix multiplication of X and w

# Define the loss function (Mean Squared Error divided by 2)
def loss(y, y_pred):
    return ((y - y_pred) ** 2 / 2).sum()  # Sum of squared errors divided by 2

# Training loop
for i in range(iter_count):
    # Forward pass: compute predicted y
    y_pred = forward(X)
    
    # Compute the loss
    l = loss(y, y_pred)
    print(f'iter {i}, loss {l}')
    
    # Backward pass: compute gradients of the loss with respect to parameters (w)
    l.backward()
    
    # Update parameters using gradient descent
    with torch.no_grad():
        w -= lr * w.grad  # Update weights
        w.grad.zero_()  # Clear the gradients for the next iteration

# Print the final parameters
print(f'final parameters: {w}')

# Test the model with new input values
x1 = 4
x2 = 5
print(f'linear regression result: {forward(torch.tensor([[x1, x2]], dtype=torch.float32))}')
