import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader

# python dataset.py

# Define a custom Dataset class
class MyDataset(Dataset):

    # Initialize the dataset
    def __init__(self):
        # Load the data from a text file
        txt_data = np.loadtxt('./sample_data.txt', delimiter=',')
        # Split the data into features (_x) and labels (_y)
        self._x = torch.from_numpy(txt_data[:, :2])  # Features are the first two columns
        self._y = torch.from_numpy(txt_data[:, 2])  # Labels are the third column
        # Store the length of the dataset
        self._len = len(txt_data)

    # Get a single sample from the dataset
    def __getitem__(self, item):
        # Return the features and label at the given index
        return self._x[item], self._y[item]

    # Return the length of the dataset
    def __len__(self):
        return self._len

# Create an instance of the MyDataset class
data = MyDataset()

#######################################################
# Print the length of the dataset
# print(len(data))

# Get the first sample from the dataset using an iterator
# first = next(iter(data))

# Print the first sample
# print(first)

# Print the type of the features of the first sample
# print(type(first[0]))
#######################################################

# Create a DataLoader to load the data in batches
dataloader = DataLoader(data, batch_size=3, shuffle=True, drop_last=True, num_workers=0)

# Initialize a counter for the number of batches
n = 0

# Iterate over the DataLoader
for data_val, label_val in dataloader:
    # Print the features and labels for each batch
    print('x', data_val, 'y', label_val)
    # Increment the counter
    n += 1

# Print the total number of iterations (batches)
print('iteration:', n)
