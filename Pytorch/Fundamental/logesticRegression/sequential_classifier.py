import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method 1:
class Model(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)
        return x

################################################
model = Model(n_feature=2, n_hidden=10, n_output=2) 
print(model)

# method 2:
model2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2), 
)

print(model2)


# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated

# contrl / -> 同時在多行前面添加 # 以將這些行注釋掉
lr = 0.02
num_epochs = 100

criterion = torch.nn.CrossEntropyLoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# train model
for epoch in range(num_epochs):
    # forward compute loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0 or epoch in [3, 6]:
        # plot and show learning process
        plt.cla()
        _, prediction = torch.max(F.softmax(y_pred), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.show()
        plt.pause(0.1)

plt.ioff()