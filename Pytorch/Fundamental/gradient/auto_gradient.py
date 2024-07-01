import numpy as np
import torch 

# Create tensor `a` with value 3.0 and tensor `b` with value 4.0, setting `b` to require gradient computation
a = torch.tensor(3.)
b = torch.tensor(4., requires_grad=True)

# Enable gradient computation for tensor `a`
a.requires_grad_(True)
print(a.requires_grad)  # Output should be True since gradient computation is enabled for `a`

###################################################################
# Calculate intermediate values f1 and f2
f1 = 2 * a  # f1 = 2*a
f2 = a * b  # f2 = a*b

# Calculate the final value z
z = f1 + f2  # z = 2*a + a*b

# Print the gradient functions for f1 and f2
print(f1.grad_fn, f2.grad_fn)
print(f'z function equal to: {z}')

# Perform backpropagation to compute the gradients of z with respect to `a` and `b`
z.backward()

# Print the gradients of z with respect to `a` and `b`
print(f'a.grad= {a.grad}')  # Should print the gradient of z with respect to a
print(f'b.grad= {b.grad}')  # Should print the gradient of z with respect to b

###################################################################
# Disable gradient computation temporarily
with torch.no_grad():
    f3 = a * b  # f3 = a*b
    print(f'f2.requires_grad= {f2.requires_grad}')  # Output should be True
    print(f'f3.requires_grad= {f3.requires_grad}')  # Output should be False since gradient computation is disabled

# Detach tensor `a` from the computation graph
a1 = a.detach()
print(f'a.requires_grad: {a.requires_grad}')  # Output should be True
print(f'a1.requires_grad: {a1.requires_grad}')  # Output should be False since a1 is detached from the graph
