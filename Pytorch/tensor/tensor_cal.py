import numpy as np
import torch 

# python tensor_cal.py

x = torch.tensor([[1, 2],[3, 4], [5, 6]], dtype=torch.float32)
y = torch.tensor([[1, 1],[2, 2], [3, 3]], dtype=torch.float32)

##################################################
# addition
# method 1 
print(f'method 1: {x+y}')

# method 2 
print(f'method 2: {torch.add(x,y)}')

# method 3
x.add_(y)
print(f'method 3: {x}')

##################################################
# substraction
# method 1 
print(f'method 1: {x-y}')

# method 2 
print(f'method 2: {torch.sub(x,y)}')

# method 3
x.sub_(y)
print(f'method 3: {x}')

##################################################
# muliplication
# method 1 
print(f'method 1: {x*y}')

# method 2 
print(f'method 2: {torch.mul(x,y)}')

# method 3
x.mul_(y)
print(f'method 3: {x}')

##################################################
# division
# method 1 
print(f'method 1: {x/y}')

# method 2 
print(f'method 2: {torch.div(x,y)}')

# method 3
x.div_(y)
print(f'method 3: {x}')

##################################################

x = torch.tensor([[1, 2],[3, 4], [5, 6]], dtype=torch.float32)
y = torch.tensor([[1, 1],[2, 2], [3, 3]], dtype=torch.float32)

print(f'all element sum: {x.sum()}')
print(f'all row element sum: {x.sum(axis=1)}')
print(f'all col element sum: {x.sum(axis=0)}')

print(f'all element mean: {x.mean()}')
print(f'all row element mean: {x.mean(axis=1)}')
print(f'all col element mean: {x.mean(axis=0)}')

print(f'x.Transpose: {x.T}')
print(f'x %*% y: {torch.matmul(x.T, y)}' )