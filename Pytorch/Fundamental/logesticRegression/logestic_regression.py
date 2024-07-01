import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

# load dataset
data = datasets.load_breast_cancer()
print(f'data shape: {data.data.shape}')
print(f'first 50 target: {data.target[:50]}')
X, y = data.data.astype(np.float32), data.target.astype(np.float32)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X,y, test_size=0.3)

sc = StandardScaler()
X_train_np = sc.fit_transform(X_train_np)
X_test_np = sc.transform(X_test_np)

X_train = torch.from_numpy(X_train_np)
X_test = torch.from_numpy(X_test_np)
y_train = torch.from_numpy(y_train_np)
y_test = torch.from_numpy(y_test_np)

# build model
class MyLogisticRegression(torch.nn.Module):
    
    def __init__(self, input_features):
        super().__init__()  # Initialize the parent class
        self.linear = torch.nn.Linear(input_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return torch.sigmoid(y)
    
input_features = X_train_np.shape[1]
model = MyLogisticRegression(input_features)

# loss and optimizer
lr = 0.2
num_epochs = 10

criterion = torch.nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# train model
for epoch in range(num_epochs):
    # forward compute loss
    y_pred = model(X_train.view(-1, input_features))
    loss = criterion(y_pred.view(-1, 1), y_train.view(-1, 1))

    # backward update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # eval
    with torch.no_grad():
        y_pred_test = model(X_test.view(-1, input_features))
        y_pred_test = y_pred_test.round().squeeze()
        total_correct = y_pred_test.eq(y_test).sum()
        prec = total_correct.item() / len(y_test)
        print(f'epoch: {epoch}, loss: {loss.item()}, prec: {prec}')


