import unittest
import torch
import random
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from modules.memory import Memory

class TestMemory(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.memory = Memory(num_nodes=10, dim_edge=4, dim_memory=4)


    def test_findOverlapMemory(self):
        data_loader = []
        mem = self.memory
        for _ in range(6):
            all_nodes = torch.sort(torch.tensor([random.randint(0, 9) for _ in range(10)])).values
            print(all_nodes)
            data_loader.append((all_nodes, None, None))
        mem.findOverlapMem(data_loader, 10, 0, 2)
        print(mem.send_msg)
        print(mem.push_msg)

        mem.findOverlapMem(data_loader, 10, 1, 2)
        print(mem.recv_msg)
        print(mem.pull_msg)

if __name__ == '__main__':
    unittest.main()