import torch
import threading
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from modules.util import recv

class recvClient:
    def __init__(self, rank, world_size, shapes_list, device):
        self.training = True
        self.shapes_list = shapes_list
        self.device = device
        self.params = []
        self.recv_thread = False
        self.rank = rank
        self.world_size = world_size

    def start(self):
        for shapes in self.shape_list:
            self.params = []
            for shape in shapes:
                self.params.append(torch.empty(shape, device=self.device))
            src = (self.rank-1+self.world_size) % self.world_size
            self.recv_thread = threading.Thread(target=recv, args=(self.params, self.rank, src))
            
    def get_data(self):
        pass