import torch
import openai
import yaml
import time
import os
from utils.vocab_db import VocabDB
from main import OpenCatkin
from api.train_api import TrainAPI

# ======================
# 配置
# ======================
OPENAI_API_KEY = "你的OPENAI_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"
MODEL_SAVE_PATH = "opencatkin_model.pth"
TRAIN_EPOCHS = 8

# ======================
# ✅ 自动设备：MPS(CUDA)/CPU
# ======================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"✅ 使用设备: {device}")



openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_BASE_URL

# ======================
# 加载模型
# ======================
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

vocab = VocabDB("vocab.db")
# 强制全局 float64
torch.set_default_dtype(torch.float64)
# 模型加载
agent = OpenCatkin(cfg).to(device).to(torch.float64)
train_api = TrainAPI(agent, cfg, device)

# ======================
# 断点续训
# ======================
if os.path.exists(MODEL_SAVE_PATH):
    print(f"✅ 加载模型断点续训: {MODEL_SAVE_PATH}")
    agent.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
else:
    print("🆕 初始化新模型")

print("✅ 词库大小:", vocab.vocab_size())

# ======================
# 获取语料
# ======================
def get_high_quality_corpus():
    print("\n📥 从 OpenAI 获取高质量语料...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "生成5句有逻辑、哲学、深度的句子，用于AI训练。一句一行。"}],
            temperature=0.7,
            max_tokens=1024
        )
        lines = [l.strip() for l in response.choices[0].message.content.strip().split("\n") if len(l.strip()) > 2]
        return lines[:5]
    except Exception as e:
        print("❌ 获取失败:", e)
        return []

# ======================
# 保存模型
# ======================
def save_model():
    torch.save(agent.state_dict(), MODEL_SAVE_PATH)
    print("💾 模型已保存")

# ======================
# 训练
# ======================
def auto_train():
    print("\n🚀 开始自动训练")
    agent.train()
    sentences = get_high_quality_corpus()
    if not sentences:
        return

    for epoch in range(TRAIN_EPOCHS):
        total_err = 0.0
        for sent in sentences:
            res = train_api.train_on_sentence(sent)
            total_err += res["pred_error"]

        thought = agent.think(sentences[-1])
        print(f"Epoch {epoch+1:02d} | thought={thought} | err={total_err:.3f}")

    save_model()

# ======================
# 测试
# ======================
def test():
    agent.eval()
    print("\n💭 测试: 我思考所以我存在")
    print("输出:", agent.think("我思考所以我存在"))

if __name__ == "__main__":
    auto_train()
    test()