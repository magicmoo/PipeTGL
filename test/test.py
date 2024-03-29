import torch

t1 = torch.tensor([1, 2, 3, 3, 5, 5, 4])
t2 = torch.tensor([1, 2, 4, 5, 3])

# 使用 torch.argsort() 对 t2 进行排序并获取索引
sorted_indices = torch.argsort(t2)
sorted_t2 = t2[sorted_indices]

sorted_indices = torch.argsort(sorted_indices)
t2 = sorted_t2[sorted_indices]

print(sorted_indices)
print(t2)

# 使用 torch.searchsorted() 在排序后的 t2 中查找 t1 的元素
indices = torch.searchsorted(sorted_t2, t1)

print(indices)