import torch
import torch.nn as nn

class SequenceTransformer(nn.Module):
    def __init__(self, dim=16, heads=2, layers=2):
        super().__init__()
        self.dim = dim
        # ✅ 扩大到 100，支持超长句子！
        self.pos_emb = nn.Parameter(torch.randn(100, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2,
            activation='relu', batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, seq):
        seq_len = seq.size(0)
        # ✅ 自动取对应长度，不会报错
        pos = self.pos_emb[:seq_len]
        x = seq + pos
        x = x.unsqueeze(0)
        out = self.transformer(x).squeeze(0)
        return out