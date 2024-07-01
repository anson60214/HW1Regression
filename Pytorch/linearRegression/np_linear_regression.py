import numpy as np

# Number of features: n
# Number of datasets: m
# X is of shape (m * n)
# w is of shape (n * 1)
# y is of shape (m * 1)

# Set a seed for reproducibility
np.random.seed(0)

# Define the input data X and target values y
# X is a (4x2) matrix and y is a (4x1) matrix
X = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
y = np.array([[8], [13], [26], [9]], dtype=np.float32)

# y = 2*x1 + 3*x2

# Initialize weights randomly
w = np.random.rand(2, 1)

# Set the number of iterations and learning rate
iter_count = 500
lr = 0.02

# Define the forward pass function to calculate the predicted values y_pred
def forward(X):
    return np.matmul(X, w)  # Matrix multiplication of X and w

# Define the loss function (Mean Squared Error divided by 2)
def loss(y, y_pred):
    return ((y - y_pred) ** 2 / 2).sum()  # Sum of squared errors divided by 2

# Define the gradient computation function
def gradient(X, y, y_pred):
    return np.matmul(X.T, y_pred - y)  # Gradient of the loss with respect to w

# Training loop
for i in range(iter_count):
    # Forward pass: compute predicted y
    y_pred = forward(X)
    
    # Compute the loss
    l = loss(y, y_pred)
    print(f'iter {i}, loss {l}')
    
    # Compute the gradient
    grad = gradient(X, y, y_pred)
    
    # Update weights using gradient descent
    w -= lr * grad

# Print the final parameters
print(f'final parameters: {w}')

# Test the model with new input values
x1 = 4
x2 = 5
print(f'linear regression result: {forward(np.array([[x1, x2]], dtype=np.float32))}')
