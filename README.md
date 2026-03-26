# 🚀 工业级 OpenCatkin 启动教程（1分钟跑起来）
我给你**最简单、最清晰、一步不坑**的启动方式，直接复制命令就能运行！

## 一、先确认你的项目文件结构（必须完全一样）
把所有文件建成这样：
```
opencatkin/
├── main.py                # 主入口
├── config.yaml            # 配置
├── requirements.txt       # 依赖
├── core/
│   ├── __init__.py
│   ├── poincare.py
│   ├── manifold.py
│   ├── free_energy.py
├── model/
│   ├── __init__.py
│   ├── brain.py
│   ├── generator.py
├── api/
│   ├── __init__.py
│   ├── train_api.py
│   ├── infer_api.py
├── utils/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── logger.py
│   ├── checkpoint.py
└── assets/
    └── checkpoints/  # 空文件夹即可
```

---

# 二、3步启动（直接复制命令）

## 1. 创建虚拟环境（推荐）
```bash
# 创建
python -m venv .venv

# 激活
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 3. 直接启动运行
```bash
python main.py
```

---

# ✅ 运行成功你会看到：
```
2026-03-26 10:00:00 | INFO | OpenCatkin AGI System Started
2026-03-26 10:00:00 | INFO | Start training...
2026-03-26 10:00:01 | INFO | Epoch 0 | thought=i think therefore alive | K=1.00 | depth=1.00
2026-03-26 10:00:02 | INFO | Epoch 1 | thought=i think alive exist | K=1.01 | depth=1.01
...
2026-03-26 10:00:20 | INFO | Final thought: i think therefore alive
```

