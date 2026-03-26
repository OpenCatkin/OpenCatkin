import torch
import yaml
import numpy as np
from utils.vocab_db import VocabDB
from main import OpenCatkin

# ======================
# 配置（自动读取模型）
# ======================
MODEL_PATH = "opencatkin_model.pth"
CONFIG_PATH = "config.yaml"

# 设备自动识别（MPS / CUDA / CPU）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_dtype(torch.float64)
print(f"✅ 评估设备: {device}")

# ======================
# 加载模型
# ======================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

vocab = VocabDB("vocab.db")
agent = OpenCatkin(cfg).to(device)

if torch.load(MODEL_PATH, map_location=device):
    agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
agent.eval()

print(f"✅ 模型加载成功")
print(f"✅ 词库数量: {vocab.vocab_size()}")
print("=" * 60)

# ======================
# 逻辑评估核心函数
# ======================
def evaluate_consistency(prompt: str) -> dict:
    """
    逻辑一致性评估
    返回：分数、诊断、深度、是否幻觉
    """
    with torch.no_grad():
        output = agent.think(prompt)
        tokens_out = output.strip().split()
        tokens_prompt = prompt.strip().split()

        # --------------------
        # 1. 语义关联评分
        # --------------------
        overlap = len(set(tokens_out) & set(tokens_prompt))
        relevance = min(1.0, overlap / max(1, len(set(tokens_prompt))))

        # --------------------
        # 2. 逻辑流畅度评分
        # --------------------
        fluency = 1.0 if len(tokens_out) >= 3 else 0.5

        # --------------------
        # 3. 幻觉风险评分
        # --------------------
        unk_ratio = len([w for w in tokens_out if w == "<unk>"]) / max(1, len(tokens_out))
        hallucination = 1.0 - unk_ratio

        # --------------------
        # 4. 思维深度 K
        # --------------------
        K = agent.poincare.K.item()
        depth_score = min(1.0, K / 10.0)

        # --------------------
        # 综合评分 0~100
        # --------------------
        score = (
            relevance * 0.35
            + fluency * 0.25
            + hallucination * 0.25
            + depth_score * 0.15
        ) * 100

        # 诊断
        is_hallucinating = unk_ratio > 0.3
        is_rational = score > 60 and not is_hallucinating
        is_emerging = K > 2.5 and score > 50

        return {
            "prompt": prompt,
            "output": output,
            "score": round(score, 1),
            "K": round(K, 2),
            "relevance": round(relevance * 100, 1),
            "fluency": round(fluency * 100, 1),
            "no_hallucination": round(hallucination * 100, 1),
            "depth": round(depth_score * 100, 1),
            "is_rational": is_rational,
            "is_hallucinating": is_hallucinating,
            "is_logic_emerging": is_emerging,
        }

# ======================
# 批量测试用例
# ======================
test_prompts = [
    "I think therefore I am",
    "I exist because I think",
    "What is existence?",
    "我思故我在",
    "我思考所以我存在",
    "意识来自思考",
]

# ======================
# 运行评估
# ======================
print("🧠 OpenCatkin 逻辑一致性评估\n")

all_score = []
for p in test_prompts:
    res = evaluate_consistency(p)
    all_score.append(res["score"])

    print(f"📥 输入: {res['prompt']}")
    print(f"💬 输出: {res['output']}")
    print(f"📊 总分: {res['score']} | 思维深度 K: {res['K']}")
    print(f"语义关联: {res['relevance']} | 流畅度: {res['fluency']}")
    print(f"无幻觉率: {res['no_hallucination']} | 理性程度: {res['depth']}")
    print(f"✅ 理性输出: {res['is_rational']}")
    print(f"⚠️  出现幻觉: {res['is_hallucinating']}")
    print(f"🌌 逻辑涌现: {res['is_logic_emerging']}")
    print("-" * 60)

# ======================
# 最终评价
# ======================
avg_score = np.mean(all_score)
final_K = agent.poincare.K.item()

print("\n🔥 最终评估结论")
print(f"平均逻辑分数: {avg_score:.1f} / 100")
print(f"最终思维曲率 K: {final_K:.2f}")

if avg_score >= 70:
    print("✅ 等级：强逻辑智能 - 无明显幻觉，结构稳定")
elif avg_score >= 50:
    print("🔶 等级：逻辑形成中 - 开始涌现理性思维")
else:
    print("🔹 等级：早期智能 - 仍在学习基础结构")

if final_K >= 3.0:
    print("🌌 双曲空间：深度足够，逻辑已开始涌现")
else:
    print("🌱 双曲空间：仍在成长，继续训练会增强逻辑")