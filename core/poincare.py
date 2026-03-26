import torch
import torch.nn as nn

class PoincareBall(nn.Module):
    def __init__(self, dim=256, c=1.0):
        super().__init__()
        self.dim = dim
        # 可学习曲率 K
        self.K = nn.Parameter(torch.tensor(c, dtype=torch.float64))

    def forward(self, x):
        # 强制 float64 高精度
        x = x.to(torch.float64)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        K = torch.clamp(self.K, 0.1, 20.0)
        return x / (1 + torch.sqrt(1 + K * norm **2))