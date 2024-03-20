import torch
import numpy as np
device = torch.device('cuda:0')
a = torch.Tensor(np.random.normal(size=10)).to(device)
print(torch.Tensor(a))