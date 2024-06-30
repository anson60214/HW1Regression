import numpy as np
import torch 

torch.manual_seed(0)

x = torch.rand(4,5)
print(f'oringinal set: {x}')
print(f'x[0, 0]: {x[0,0]}')
print(f'x[0, :]: {x[0, :]}')
print(f'x[:, 1]: {x[:, 1]}')

print(f'x[1:3, 1:3]: {x[1:3, 1:3]}')
