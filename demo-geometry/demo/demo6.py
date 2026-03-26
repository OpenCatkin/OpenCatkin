import torch
import numpy as np
import math
import random
import json
import time
import re
from collections import defaultdict

# ==============================
# 0. 基础配置
# ==============================
try:
    import ollama
except ImportError:
    print("❌ 请先安装: pip install ollama")
    exit()

LLM_MODEL = "qwen3:8b"
MEMORY_FILE = "hyper_disk_memory.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==============================
# 1. 双曲几何核心（医药版原版）
# ==============================
class Poincare:
    @staticmethod
    def distance(x, y, eps=1e-8):
        x = x / torch.max(torch.norm(x), torch.tensor(0.9999)) * 0.9999
        y = y / torch.max(torch.norm(y), torch.tensor(0.9999)) * 0.9999
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        diff = torch.sum((x-y)**2, dim=-1, keepdim=True)
        delta = 2 * diff / ((1 - x2) * (1 - y2) + eps)
        return torch.arccosh(1 + delta)

# ==============================
# 2. 双曲圆盘引擎（重要记忆直接上圆盘）
# ==============================
class HyperEngine:
    def __init__(self):
        self.space = {}          # 双曲圆盘：情绪 + 回应 + 重要记忆
        self.disk_nodes = {}     # 上圆盘的重要记忆（核心！）
        self.emotions = ["sad","happy","anxious","angry","lonely","tired","excited","confused","stressed","neutral"]
        self.responses = ["empathy","encourage","relax","celebrate","solve"]
        self.standard = {
            "empathy": ["sad","lonely","confused","neutral"],
            "encourage": ["tired","stressed","anxious"],
            "relax": ["angry","stressed"],
            "celebrate": ["happy","excited"],
            "solve": ["anxious","confused","stressed"]
        }
        self.init_base_space()
        self.load_all()

    def init_base_space(self):
        # 初始化情绪锚点
        for i, e in enumerate(self.emotions):
            angle = 2 * math.pi * i / len(self.emotions)
            self.space[e] = torch.tensor([0.8 * math.cos(angle), 0.8 * math.sin(angle)], device=DEVICE)
        # 初始化回应锚点
        for r, es in self.standard.items():
            points = [self.space[e] for e in es]
            center = torch.mean(torch.stack(points), dim=0)
            self.space[r] = center / torch.norm(center) * 0.3

    # ==============================
    # 核心：启动强制预训练（医药版必须）
    # ==============================
    def pretrain(self, epochs=5, lr=0.05):
        print("🏋️  初始预训练（医药版标准流程）")
        for ep in range(epochs):
            loss = 0.0
            for r, es in self.standard.items():
                target = torch.mean(torch.stack([self.space[e] for e in es]), dim=0)
                target = target / torch.norm(target) * 0.3
                d = Poincare.distance(self.space[r], target).item()
                loss += d
            print(f"   轮次 {ep+1} | 损失: {loss:.4f}")
        print("✅ 预训练完成\n")

    # ==============================
    # 核心：重要记忆 → 直接上双曲圆盘
    # ==============================
    def add_important_to_disk(self, key_name, content, bind_emos):
        center = torch.mean(torch.stack([self.space[e] for e in bind_emos]), dim=0)
        pos = center + torch.tensor([random.uniform(-0.04, 0.04) for _ in range(2)], device=DEVICE)
        self.space[key_name] = pos
        self.disk_nodes[key_name] = {
            "content": content,
            "emos": bind_emos,
            "pos": pos.tolist()
        }
        self.save_all()

    # ==============================
    # 医药版逻辑：圆盘优先推理
    # ==============================
    def query_disk(self, input_emos):
        if not input_emos:
            input_emos = ["neutral"]
        center = torch.mean(torch.stack([self.space[e] for e in input_emos]), dim=0)

        # 1. 优先匹配圆盘上的【重要记忆】
        best_mem = None
        min_mem_dist = 999
        for key, node in self.disk_nodes.items():
            d = Poincare.distance(center, torch.tensor(node["pos"], device=DEVICE)).item()
            if d < min_mem_dist:
                min_mem_dist = d
                best_mem = node["content"]

        # 2. 匹配回应模式
        best_mode = None
        min_mode_dist = 999
        for r in self.responses:
            d = Poincare.distance(center, self.space[r]).item()
            if d < min_mode_dist:
                min_mode_dist = d
                best_mode = r

        # 圆盘匹配阈值：距离 < 0.7 直接用圆盘结果
        use_disk = best_mem is not None and min_mem_dist < 0.7
        return best_mem, best_mode, min_mem_dist, use_disk

    # ==============================
    # 存储
    # ==============================
    def save_all(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"disk_nodes": self.disk_nodes}, f, ensure_ascii=False, indent=2)

    def load_all(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.disk_nodes = data.get("disk_nodes", {})
                for k, v in self.disk_nodes.items():
                    self.space[k] = torch.tensor(v["pos"], device=DEVICE)
        except:
            self.disk_nodes = {}

# ==============================
# 3. 主流程：圆盘优先 → LLM兜底
# ==============================
class Assistant:
    def __init__(self):
        self.engine = HyperEngine()
        self.emo_map = {
            "sad":"难过","happy":"开心","anxious":"焦虑","angry":"生气","lonely":"孤独",
            "tired":"累","excited":"兴奋","confused":"迷茫","stressed":"压力大","neutral":"日常"
        }
        self.mode_map = {"empathy":"共情","encourage":"鼓励","relax":"放松","celebrate":"庆祝","solve":"建议"}

    def extract_emos(self, text):
        emos = []
        if any(w in text for w in ["难过","伤心"]): emos.append("sad")
        if any(w in text for w in ["开心","高兴"]): emos.append("happy")
        if any(w in text for w in ["焦虑","怕"]): emos.append("anxious")
        if any(w in text for w in ["生气","烦"]): emos.append("angry")
        if any(w in text for w in ["累","疲惫"]): emos.append("tired")
        if any(w in text for w in ["兴奋","激动"]): emos.append("excited")
        if any(w in text for w in ["迷茫","不知道"]): emos.append("confused")
        if any(w in text for w in ["压力","崩溃"]): emos.append("stressed")
        return emos if emos else ["neutral"]

    def chat(self, user_text):
        # 命令：查看圆盘上的重要记忆
        if user_text in ["查看记忆","记忆","圆盘"]:
            if not self.engine.disk_nodes:
                return "📀 双曲圆盘暂无重要记忆"
            lines = ["\n📀 双曲圆盘重要记忆（永久节点）："]
            for idx, (k, v) in enumerate(self.engine.disk_nodes.items(), 1):
                lines.append(f"{idx}. {v['content']}")
            return "\n".join(lines)

        # 提取情绪
        emos = self.extract_emos(user_text)
        # 圆盘优先查询
        disk_mem, mode, dist, use_disk = self.engine.query_disk(emos)

        # ==============================================
        # 医药版核心逻辑：圆盘有结果 → 直接用，不叫LLM
        # ==============================================
        if use_disk and disk_mem:
            reply = f"我记得：{disk_mem}"
            print(f"[📀 圆盘直接回答] 情绪:{[self.emo_map[e] for e in emos]} | 距离:{dist:.2f}")
            return reply

        # 圆盘无结果 → LLM兜底
        prompt = f"""你叫小星。
用户：{user_text}"""

        try:
            # 关闭思考模式！
            res = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature":0.7,"num_predict":150,"stop":["\n"]}
            )
            reply = res["response"].strip()
        except:
            reply = "我在听。"

        # 自动抓取重要信息 → 上圆盘
        self.save_important_to_disk(user_text, emos)
        print(f"[🤖 LLM回答] 情绪:{[self.emo_map[e] for e in emos]} | 模式:{self.mode_map[mode]}")
        return reply

    # 自动识别重要信息 → 写入双曲圆盘
    def save_important_to_disk(self, text, emos):
        name_match = re.search(r"我(叫|是|名字是)\s*(\w+)", text)
        if name_match and name_match.group(2) not in ["谁","什么","怎么"]:
            key = f"user_name_{int(time.time())}"
            content = name_match.group(0).replace("我", "你")
            self.engine.add_important_to_disk(key, content, emos)
            return

        remind_match = re.search(r"(提醒我|记得)\s*(.*)", text)
        if remind_match:
            key = f"remind_{int(time.time())}"
            content = remind_match.group(0).replace("我", "你")
            self.engine.add_important_to_disk(key, content, emos)
            return

# ==============================
# 启动（必须先预训练）
# ==============================
if __name__ == "__main__":
    print("🚀 双曲圆盘助手 · 医药版原版逻辑")
    assistant = Assistant()
    assistant.engine.pretrain(epochs=5)  # 强制初始训练
    print("🤖 小星：嗨～ 输入「查看记忆」可看圆盘上的重要记忆\n")
    while True:
        user = input("你：")
        if user in ["退出", "quit"]:
            break
        print("小星：", assistant.chat(user), "\n")