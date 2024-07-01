import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad= True)

print(f'tensor: {tensor}')
print(f'variable: {variable}')

t_out = torch.mean(tensor*tensor) # x^2
v_out = torch.mean(variable*variable) 

print(f't_out: {t_out}')
print(f'v_out: {v_out}')

v_out.backward()
# v_out = 1/4* sum(var*var)
# d(v_out)/ d(var) = 1/4* 2* var = 1/2 * var
print(f'variable.grad: {variable.grad}')

print(f'To tensor: {variable.data}')
print(f'To numpy: {variable.data.numpy()}')

