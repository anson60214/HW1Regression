import numpy as np
import torch 

torch.manual_seed(0)

x = torch.rand(20)
print(f'x:{x}, x data ptr(內存地址):{x.data_ptr()}')
y = x.view(4, 5)
print(f'x.view: {y}, y data ptr(內存地址):{y.data_ptr()}')

y = x.reshape(4, 5)
print(f'x.reshape: {y}, y data ptr(內存地址):{x.data_ptr()}')

xt = y.T
# z = xt.view(1, 20) 
# view size is not compatible with input tensor's size and stride 
# (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

z = xt.contiguous().view(1, 20)
print(f'after view: {z}, data ptr(內存地址):{z.data_ptr()}') 

z = xt.reshape(1, 20)
print(f'after reshape:{z}, data ptr(內存地址):{z.data_ptr()}') 


#################################################################
x = torch.rand(20)
print(f'x:{x}, x data ptr(內存地址):{x.data_ptr()}')
print(f'before unsqueeze shape: {x.shape}')

y = x.unsqueeze(0)
print(f'after unsqueeze: {y}')
print(f'after unsqueeze shape: {y.shape}')

y = x.unsqueeze(1)
print(f'after unsqueeze: {y}')
print(f'after unsqueeze shape: {y.shape}')

z = y.squeeze(1)
print(f'before squeeze: {y}')
print(f'before unsqueeze shape: {y.shape}')
print(f'after squeeze: {z}')
print(f'after unsqueeze shape: {z.shape}')