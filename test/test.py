import torch
import numpy as np

t = torch.tensor((1, 2, 3), device='cuda:1')
print(t)

# # 使用 torch.searchsorted() 在排序后的 t2 中查找 t1 的元素
# indices = torch.searchsorted(sorted_t2, t1)

# print(indices)