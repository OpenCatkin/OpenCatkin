import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencatkin.extractors.base import OllamaExtractor
from opencatkin.validators.base import MathPrimeValidator
from opencatkin.validators.drug_validator import DrugInteractionValidator
from opencatkin.persistence.graph_storage import GraphStorage
from opencatkin.pipeline import OpenCatkinPipeline
from opencatkin.pipeline import visualize_graph  # 如果你之前有这个函数
from text_to_graph import visualize_graph

def main():
    extractor = OllamaExtractor(model_name="qwen3:8b")
    storage = GraphStorage()
    pipeline = OpenCatkinPipeline(extractor=extractor, storage=storage)

    pipeline.add_validator(MathPrimeValidator())          # 已有免疫
    pipeline.add_validator(DrugInteractionValidator())    # 新增医药规则

    text = "根据文献，阿司匹林与华法林联合使用会增加出血风险，而布洛芬可用于治疗头痛。"
    pipeline.process_text(text)

    print(f"\n📊 医药知识图谱节点: {storage.get_all_nodes()}")
    storage.save_to_file("medical_graph.gml")
    visualize_graph(storage)   # 生成 PNG

if __name__ == "__main__":
    main()