import torch
from torch import nn
from torch.autograd import Variable
import torch.utils
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 1 
BATCH_SIZE = 64
TIME_STEP = 28 # run time step / image height
INPUT_SIZE = 28 # run input size / image width
LR = 10**-2 # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data

# Mnist digital dataset
train_data = dsets.MNIST(
    root='.\Pytorch\Model\mnist',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

train_loader = Data.DataLoader(dataset=train_data, 
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0)

test_data = dsets.MNIST(
    root='.\Pytorch\Model\mnist',
    train=False,                         # this is testing data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

test_x = Variable(test_data.test_data, volatile = True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]
# print(test_x.shape) # torch.Size([2000, 28, 28])
# print(test_y.shape)   # (2000,)

# plot one example
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True, # default: (time_step, batch, input) -> (batch, time_step, input)
        )

        self.out = nn.Linear(64, 10) # because we have 10 class of digit

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        # h_n == r_out[:, -1, :] which is the output of the last time_step
        # c_n is the condition of last time_step cell
        
        r_out, (h_n, h_c) = self.rnn(x, None) # None first hidden_step is None 
        out = self.out(r_out[:, -1, :]) # (batch, time_step, input) because we want the last time_step so we set -1
        return out
    
rnn = RNN()
# print(rnn)
# RNN(
#  (rnn): LSTM(28, 64, batch_first=True)
#  (out): Linear(in_features=64, out_features=10, bias=True)
# )

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # give batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size), -1 refer to all other dim will squeeze into 1 (from 4 to 3)
        b_y = Variable(y)
        
        output = rnn(b_x)                               # rnn output
        loss = criterion(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze() # torch.max(input, dim, keepdim=False, *, out=None) find the max across dim=1
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')