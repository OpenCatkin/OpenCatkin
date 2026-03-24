import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opencatkin.extractors.base import OllamaExtractor
from opencatkin.validators.base import MathPrimeValidator
from opencatkin.validators.finance_validator import FinanceComplianceValidator
from opencatkin.persistence.graph_storage import GraphStorage
from opencatkin.pipeline import OpenCatkinPipeline
from text_to_graph import visualize_graph

def main():
    extractor = OllamaExtractor(model_name="qwen3:8b")
    storage = GraphStorage()
    pipeline = OpenCatkinPipeline(extractor=extractor, storage=storage)

    pipeline.add_validator(FinanceComplianceValidator())

    text = "根据反洗钱条款，金融机构必须对超过100万的交易进行报告，同时禁止匿名账户。"
    pipeline.process_text(text)

    print(f"\n📊 金融合规图谱节点: {storage.get_all_nodes()}")
    storage.save_to_file("finance_graph.gml")
    visualize_graph(storage)

if __name__ == "__main__":
    main()