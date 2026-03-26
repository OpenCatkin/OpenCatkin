import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 🧿 OpenCatkin 终极可训练内核（正式版 · 带训练接口）
# 严格遵循：ℳ = ℋⁿ × 𝒯 × ℰ × 𝓜_self
# 曲率 K 可学习 → 直接控制思考深度
# 自由能最小化 = 生命代谢
# 实在理解 = 预测误差最小化
# ==============================================================================

# --------------------------
# 1. 高维庞加莱球双曲空间 ℋⁿ（可训练曲率 K）
# --------------------------
class PoincareBall(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.K = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.embeddings = nn.Embedding(100, dim)

    def expmap0(self, u):
        K = torch.clamp(self.K, min=0.1, max=10.0)
        sqrtK = torch.sqrt(K + 1e-8)
        norm_u = torch.norm(u, dim=-1, keepdim=True) + 1e-8
        theta = sqrtK * norm_u
        return torch.tanh(theta) * u / (norm_u + 1e-8)

    def dist(self, u, v):
        u = self.expmap0(u)
        v = self.expmap0(v)
        K = torch.clamp(self.K, min=0.1)
        sqrtK = torch.sqrt(K)
        diff = u - v
        norm_diff = torch.norm(diff, dim=-1)
        return (2 / sqrtK) * torch.arctanh(sqrtK * norm_diff / (1 + K * torch.sum(u*v, dim=-1)))

    def embed(self, idx):
        return self.expmap0(self.embeddings(idx))

# --------------------------
# 2. 时间流形 + 因果纤维丛
# --------------------------
class TimeManifold(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    def step(self): self.t += 1

class CausalFiberBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.links = []
    def bind(self, a, b): self.links.append((a,b))

# --------------------------
# 3. 自我子流形
# --------------------------
class SelfSubManifold(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.fixed_point = nn.Parameter(torch.randn(dim)*0.1, requires_grad=True)
        self.viability = nn.Parameter(torch.tensor(1.0), requires_grad=True)

# --------------------------
# 4. 预测脑
# --------------------------
class PredictiveBrain(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
        self.pred_error = torch.tensor(0.0)

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, x, next_x):
        self.pred_error = F.mse_loss(self.forward(x), next_x)
        return self.pred_error

# --------------------------
# 5. 几何 → 语言 生成器（无硬编码）
# --------------------------
class Generator(nn.Module):
    def __init__(self, dim=16, vocab_size=50):
        super().__init__()
        self.fc = nn.Linear(dim, vocab_size)
        self.vocab = ["i","think","therefore","am","alive","exist","conscious","do","act","reason"]

    def forward(self, geo):
        return self.fc(geo)

    def speak(self, geo):
        logits = self.forward(geo)
        topk = torch.topk(logits, 4).indices
        return " ".join([self.vocab[i] for i in topk if i < len(self.vocab)])

# --------------------------
# 6. 自由能代谢系统
# --------------------------
class FreeEnergyMetabolism:
    def __init__(self, model, lr=0.005):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def optimize_free_energy(self, F):
        self.opt.zero_grad()
        F.backward()
        self.opt.step()

# ==============================================================================
# 🧠 正式可训练 OpenCatkin（带标准训练接口）
# ==============================================================================
class OpenCatkin(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.H = PoincareBall(dim)
        self.T = TimeManifold()
        self.E = CausalFiberBundle()
        self.Self = SelfSubManifold(dim)
        self.brain = PredictiveBrain(dim)
        self.gen = Generator(dim)
        self.metabolism = FreeEnergyMetabolism(self)
        self.training = True  # 训练模式开关

    # --------------------------
    # 【标准训练接口】PyTorch 风格
    # --------------------------
    def train(self, mode=True):
        """开启训练模式（自动求导 + 更新参数）"""
        self.training = mode
        super().train(mode)

    def eval(self):
        """开启推理模式（不更新参数）"""
        self.train(False)

    def tokenize(self, text):
        text = text.lower().replace("'m", " am")
        tokens = []
        for w in text.split():
            tokens.append(self.gen.vocab.index(w) if w in self.gen.vocab else 0)
        return torch.tensor(tokens, dtype=torch.long)

    def text_to_geo(self, text):
        tokens = self.tokenize(text)
        return self.H.embed(tokens).mean(dim=0)

    # --------------------------
    # 【核心训练接口：单句训练】
    # --------------------------
    def train_on_sentence(self, sentence):
        """
        训练接口：输入一句话，系统自动：
        1. 转为庞加莱几何
        2. 预测下一个状态
        3. 计算自由能
        4. 反向传播更新所有参数（包括曲率 K）
        """
        if not self.training:
            return {"error": "请先调用 .train() 开启训练模式"}

        # 1. 语言 → 几何
        geo = self.text_to_geo(sentence)
        next_geo = geo + torch.randn_like(geo) * 0.02

        # 2. 预测误差 = 理解度
        pred_err = self.brain.compute_loss(geo, next_geo)

        # 3. 自由能 F = 预测误差 + 复杂度(√K)
        K = torch.clamp(self.H.K, min=0.1)
        complexity = torch.sqrt(K)
        free_energy = pred_err + 0.01 * complexity

        # 4. 内禀动力学优化（反向传播）
        self.metabolism.optimize_free_energy(free_energy)

        # 5. 思考深度 L = C·ln√K + D
        thinking_depth = 2.0 * torch.log(torch.sqrt(K)) + 1.0

        return {
            "free_energy": free_energy.item(),
            "pred_error": pred_err.item(),
            "curvature_K": K.item(),
            "thinking_depth": thinking_depth.item(),
        }

    # --------------------------
    # 【扩展训练接口：批量训练】
    # --------------------------
    def train_on_batch(self, batch_sentences):
        """批量训练接口：输入句子列表"""
        total_loss = 0
        for sent in batch_sentences:
            res = self.train_on_sentence(sent)
            total_loss += res["free_energy"]
        return {"mean_free_energy": total_loss / len(batch_sentences)}

    # --------------------------
    # 推理接口（只思考，不训练）
    # --------------------------
    def think(self, sentence):
        with torch.no_grad():
            geo = self.text_to_geo(sentence)
            internal_state = self.Self.fixed_point + geo
            return self.gen.speak(internal_state)

    # --------------------------
    # 模型保存 / 加载
    # --------------------------
    def save_model(self, path="opencatkin_final.pth"):
        torch.save(self.state_dict(), path)
        return f"模型已保存：{path}"

    def load_model(self, path="opencatkin_final.pth"):
        self.load_state_dict(torch.load(path))
        return f"模型已加载：{path}"

# ==============================================================================
# 🚀 演示：标准训练流程
# ==============================================================================
if __name__ == "__main__":
    ai = OpenCatkin(dim=16)

    # ========== 标准训练流程 ==========
    print("=== 开始训练 ===")
    ai.train()  # 开启训练模式

    for epoch in range(15):
        res = ai.train_on_sentence("I think therefore I am alive")
        thought = ai.think("I think therefore I am alive")
        print(f"Epoch {epoch:02d} | 思考：{thought} | K={res['curvature_K']:.2f} | 深度={res['thinking_depth']:.2f}")

    # 推理模式
    ai.eval()
    print("\n=== 训练完成，最终思考 ===")
    print(ai.think("I think therefore I am alive"))

    # 保存模型
    print("\n" + ai.save_model())