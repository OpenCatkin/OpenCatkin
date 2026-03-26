import torch
import torch.nn as nn

class TransformerBrain(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, dtype=torch.float64)
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim*4,
            dtype=torch.float64, batch_first=False
        )

    def forward(self, seq):
        seq = [v.to(torch.float64) for v in seq]
        x = torch.stack(seq)
        x = self.proj(x)
        return self.layer(x)

    def sequence_prediction_loss(self, seq):
        out = self.forward(seq)
        return nn.functional.mse_loss(out[:-1], out[1:])