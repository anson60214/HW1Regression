import torch 
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# Create TensorDataset with correct parameters
torch_dataset = Data.TensorDataset(x, y)

# Create DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# Training loop
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # Training process...
        print('Epoch: ', epoch, '|Step: ', step, 'batch x: ',
              batch_x.numpy(), '|batch y: ', batch_y.numpy())