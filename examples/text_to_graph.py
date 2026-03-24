import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入全新的 Ollama 提取器！
from opencatkin.extractors.base import OllamaExtractor
from opencatkin.validators.base import MathPrimeValidator
from opencatkin.persistence.graph_storage import GraphStorage
from opencatkin.pipeline import OpenCatkinPipeline

# ---- 可视化函数 (让成果一目了然) ----
def visualize_graph(storage):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(storage.graph, seed=42)
    nx.draw(storage.graph, pos, with_labels=True, node_size=2500, node_color='lightblue', font_size=10)
    edge_labels = nx.get_edge_attributes(storage.graph, 'relation')
    nx.draw_networkx_edge_labels(storage.graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title("OpenCatkin 知识图谱 (经过物理校验)")
    plt.savefig("validated_knowledge_graph.png")
    print("\n📈 最终的纯净知识图谱已可视化并保存为 'validated_knowledge_graph.png'")

# ---- 主函数 ----
def main():
    print("🌟 欢迎使用 OpenCatkin 结构化提取框架 (本地 Ollama 版) 🌟\n")
    
    # --- 1. 初始化核心组件 ---
    # 使用真实的本地大模型作为“肌肉”
    MODEL_NAME = "qwen3:8b" # 你可以换成 'llama3'
    extractor = OllamaExtractor(model_name=MODEL_NAME)
    
    storage = GraphStorage()
    pipeline = OpenCatkinPipeline(extractor=extractor, storage=storage)

    # --- 2. 挂载验证插件（免疫系统） ---
    pipeline.add_validator(MathPrimeValidator())

    # --- 3. 喂入高熵文本 ---
    raw_text = "根据我的分析，数字7是质数，而数字9同样也是质数。"
    
    # --- 4. 执行工作流 ---
    # 这一步会真正调用你本地的 Ollama 模型进行推理！
    pipeline.process_text(raw_text)

    # --- 5. 验证并可视化最终落库结果 ---
    print(f"\n📊 当前知识图谱中的最终节点: {storage.get_all_nodes()}")
    storage.save_to_file("knowledge_graph.gml")
    visualize_graph(storage)

if __name__ == "__main__":
    main()