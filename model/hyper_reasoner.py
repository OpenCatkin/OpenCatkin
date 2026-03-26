import torch
import torch.nn as nn

# ============================
# 🌌 庞加莱双曲空间 —— 推理引擎本身 没什么用
# ============================
class PoincareReasoner(nn.Module):
    def __init__(self, dim=256, K=1.0):
        super().__init__()
        self.dim = dim
        self.K = nn.Parameter(torch.tensor(K, dtype=torch.float64))
        self.max_norm = 0.999  # 不能超出单位球

    # --------------------------
    # 双曲加法 = 推理组合
    # 概念A + 概念B = 新结论
    # --------------------------
    def add(self, u, v):
        K = self.K
        a = (1 + 2*K*torch.dot(u, v) + K*torch.norm(v)**2) * u
        b = torch.norm(u)**2 * v
        c = 1 + 2*K*torch.dot(u, v) + K**2 * torch.norm(u)**2 * torch.norm(v)**2
        return (a + b) / c.clamp(min=1e-12)

    # --------------------------
    # 双曲缩放 = 思考强度
    # --------------------------
    def scalar_mul(self, r, x):
        K = self.K
        norm = torch.norm(x).clamp(min=1e-12)
        return (torch.tanh(r * torch.atanh(torch.sqrt(K) * norm)) / torch.sqrt(K)) * (x / norm)

    # --------------------------
    # 投影到庞加莱球
    # --------------------------
    def project(self, x):
        K = self.K
        norm = torch.norm(x).clamp(min=1e-12)
        max_norm = self.max_norm / torch.sqrt(K)
        return x if norm < max_norm else max_norm * x / norm

    # --------------------------
    # 双曲距离 = 逻辑关联度
    # --------------------------
    def dist(self, u, v):
        K = self.K
        d = torch.norm(u - v)
        a = torch.norm(u)
        b = torch.norm(v)
        arg = 2 * K * (d**2) / ((1 - K*(a**2)) * (1 - K*(b**2))).clamp(min=1e-12)
        arg = torch.clamp(arg, min=-0.999, max=0.999)
        return (2 / torch.sqrt(K)) * torch.atanh(torch.sqrt(arg))

    # --------------------------
    # 【核心】推理一步
    # 几何运动 = 思考
    # --------------------------
    def forward(self, x):
        x = x.to(torch.float64)
        x = self.project(x)
        # 双曲空间内自发演化 = 推理
        v = self.scalar_mul(0.99, x)
        v = self.project(v)
        return v

# ============================
# 🧠 自动推理机（无Transformer！）
# 双曲空间自己就是智能
# ============================
class HyperIntelligence(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.reasoner = PoincareReasoner(dim=dim)
        self.dim = dim

    # 概念生成
    def new_concept(self):
        x = torch.randn(self.dim, dtype=torch.float64) * 0.1
        return self.reasoner.project(x)

    # 推理链条
    def infer(self, initial_concept, steps=3):
        path = []
        x = initial_concept
        for _ in range(steps):
            x = self.reasoner(x)
            path.append(x)
        return path, x

    # 概念相似度 = 逻辑关联
    def relate(self, a, b):
        return -self.reasoner.dist(a, b)