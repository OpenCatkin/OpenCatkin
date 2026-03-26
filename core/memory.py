import torch
import torch.nn as nn

class EpisodicMemory(nn.Module):
    def __init__(self, dim=16, memory_size=32):
        super().__init__()
        self.memory = []
        self.max_size = memory_size
        self.dim = dim

    def store(self, geo_state):
        self.memory.append(geo_state.detach())
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_context(self):
        if not self.memory:
            return torch.zeros(self.dim)
        return torch.stack(self.memory).mean(dim=0)

    def clear(self):
        self.memory = []