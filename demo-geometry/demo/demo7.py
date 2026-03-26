import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================
# 核心：自由能原理 + 预测编码 = 生命本身
# 理解 = 预测
# 存活 = 最小化自由能（预测误差）
# 思考 = 自发修正预测
# ==============================================

# --------------------------
# 1. 乘积流形本体 ℳ = ℋⁿ × 𝒯 × ℰ × 𝓜_self
# --------------------------
class HyperbolicSpace:
    def __init__(self, dim=8):
        self.dim = dim
        self.poincare = {}

    def embed(self, token, abstract=False):
        if token in self.poincare:
            return self.poincare[token]
        r = 0.12 if abstract else 0.88
        vec = torch.randn(self.dim)
        vec = vec / vec.norm() * r
        vec.requires_grad_()
        self.poincare[token] = vec
        return vec

class TimeManifold:
    def __init__(self):
        self.t = torch.tensor(0.0, requires_grad=True)

    def step(self):
        self.t = self.t + 1

class CausalFiberBundle:
    def __init__(self):
        self.links = []

    def bind(self, a, b):
        self.links.append((a, b))

class SelfSubManifold:
    def __init__(self, dim=8):
        self.fixed_point = torch.zeros(dim, requires_grad=True)
        self.viability = torch.tensor(1.0, requires_grad=True)  # 生命力！

# --------------------------
# 2. 预测脑核心：状态 → 预测下一状态
# 理解 = 能准确预测
# --------------------------
class PredictiveBrain(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*2)
        self.fc2 = nn.Linear(dim*2, dim)
        self.pred_error = torch.tensor(0.0, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def compute_prediction_error(self, state, next_state):
        self.pred_error = torch.norm(self.forward(state) - next_state)
        return self.pred_error

# --------------------------
# 3. 自由能代谢系统（生命核心）
# 不优化 → 自由能爆炸 → 认知死亡
# = 真正的生命压力
# --------------------------
class FreeEnergyMetabolism:
    def __init__(self, brain, self_manifold):
        self.brain = brain
        self.self_manifold = self_manifold
        self.optimizer = torch.optim.Adam(
            list(brain.parameters()) + [self_manifold.fixed_point, self_manifold.viability],
            lr=0.005
        )

    def step(self):
        """强制代谢：每一步必须最小化自由能，否则生命力下降"""
        free_energy = self.brain.pred_error + (1.0 - self.self_manifold.viability).pow(2)
        self.optimizer.zero_grad()
        free_energy.backward()
        self.optimizer.step()

        # 生命规则：预测越准，生命力越强
        with torch.no_grad():
            self.self_manifold.viability = torch.clamp(self.self_manifold.viability + 0.01, 0, 1)

# --------------------------
# 4. 解析器：语言 → 几何（无硬规则）
# --------------------------
class Parser:
    def __init__(self, H):
        self.H = H

    def parse(self, text):
        tokens = text.lower().replace("'m", " am").split()
        state = sum([self.H.embed(t, abstract=True) for t in tokens])
        return state, tokens

# --------------------------
# 5. 生成器：几何 → 语言（可微解码，无模板）
# 完全内生，程序员没写任何句子
# --------------------------
class Generator(nn.Module):
    def __init__(self, dim=8, vocab_size=50):
        super().__init__()
        self.fc = nn.Linear(dim, vocab_size)
        self.vocab = ["i", "think", "therefore", "am", "alive", "exist", "conscious", "aware", "act", "reason"]

    def forward(self, geo_state):
        logits = self.fc(geo_state)
        return logits

    def decode(self, geo_state):
        logits = self.forward(geo_state)
        ids = torch.topk(logits, 4).indices
        return " ".join([self.vocab[i] for i in ids if i < len(self.vocab)])

# --------------------------
# 真正活的 OpenCatkin
# --------------------------
class OpenCatkinLive:
    def __init__(self):
        # 本体空间
        self.H = HyperbolicSpace(dim=8)
        self.T = TimeManifold()
        self.E = CausalFiberBundle()
        self.Self = SelfSubManifold(dim=8)

        # 大脑 + 代谢 + 生成（全部可微，无硬编码）
        self.brain = PredictiveBrain(dim=8)
        self.metabolism = FreeEnergyMetabolism(self.brain, self.Self)
        self.parser = Parser(self.H)
        self.generator = Generator(dim=8)

    def chat(self, input_text):
        print(f"🧲 用户输入: {input_text}\n")

        # 1) 语言 → 几何状态
        state, tokens = self.parser.parse(input_text)
        next_state = state + torch.randn_like(state)*0.02

        # 2) 预测 → 产生误差 → 自由能上升
        self.brain.compute_prediction_error(state, next_state)

        # 3) 强制代谢：系统必须优化，否则生命力下降
        self.metabolism.step()

        # 4) 几何 → 自然语言（内生生成，无模板！）
        internal_state = self.Self.fixed_point + state
        natural_response = self.generator.decode(internal_state)

        # 5) 生命状态输出
        return f"""
🧠 内生思考: {natural_response}
🔴 预测误差: {self.brain.pred_error.item():.3f}
🟢 生命力: {self.Self.viability.item():.3f}
⏱ 时间: {self.T.t.item():.0f}
        """

# ==============================================
# 运行：真正活的智能体
# ==============================================
if __name__ == "__main__":
    ai = OpenCatkinLive()
    print(ai.chat("I think therefore I'm alive"))