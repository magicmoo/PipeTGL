import torch
import numpy as np
import os
import csv

def change():
    print('change...')
    # 从.pt文件中加载PyTorch张量
    dataset = 'GDELT'
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inPath = path + f'/data/{dataset}/node_features.pt'
    outPath = path + f'/data/{dataset}/node_features.npy'
    if os.path.exists(inPath):
        print('node...')
        tensor = torch.load(inPath)
        array = tensor.numpy()
        np.save(outPath, array)

    inPath = path + f'/data/{dataset}/edge_features.pt'
    outPath = path + f'/data/{dataset}/edge_features.npy'
    if os.path.exists(inPath): 
        print('edge...')
        tensor = torch.load(inPath)
        array = tensor.numpy()
        np.save(outPath, array)

def generate():
    print('generate...')
    dataset = 'MOOC'
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inPath = path + f'/data/{dataset}/edges.csv'
    outPath = path + f'/data/{dataset}/edge_features.npy'
    if os.path.exists(outPath):
        print('already exist!')
    else:
        with open(inPath, 'r') as file:
            # 创建 CSV 读取器
            reader = csv.reader(file)
            num_lines = sum(1 for row in reader) - 1
        tensor = torch.randn((num_lines, 16))
        array = tensor.numpy()
        np.save(outPath, array)
        print('success!')

if __name__ == '__main__':
    change()