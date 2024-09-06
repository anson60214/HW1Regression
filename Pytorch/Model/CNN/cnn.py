import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE; HAS_SK=True

# hyper parameter
EPOCH = 1
BATCH_SIZE = 500
LR = 0.001
DOWNLOAD_MNIST = False

# Define a transform to convert the data to a tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(
    root = '.\Pytorch\Model\CNN\mnist',
    train = True,
    transform = transform, 
    download = DOWNLOAD_MNIST
)

# Plot one example
#print(train_data.data.size())  # (60000, 28, 28)
#print(train_data.targets.size())  # (60000)

# Assuming 'data' is already defined as a numpy array
#plt.imshow(train_data.data[0].numpy(), cmap='gray')
#plt.title('%i' % train_data.targets[0])
#plt.show()

# dataloader
train_loader = Data.DataLoader(dataset=train_data, 
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0)

test_data = torchvision.datasets.MNIST(
    root='.\Pytorch\Model\mnist',
    train = False
)

test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000]
#print(test_x.shape)
#print(test_data.data.shape[:2000])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(    # input (1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 # if stride=1, padding=(kernel_size-1)/2 = (5-1)/2=2
            ), # -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # ->(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2), # ->(32,14,14)  
            nn.ReLU(),
            nn.MaxPool2d(2) # ->(32,7,7)
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # (batch, 32,7,7)
        x = x.view(x.size(0), -1) # (batch, 32*7*7)
        output = self.out(x)
        return output
    

cnn = CNN()
#print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
criterion = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x) # batch x
        b_y = Variable(y) # batch y

        output = cnn(b_x)                   # cnn output
        loss = criterion(output, b_y)       # cross entropy loss
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y)/ test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)



# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')