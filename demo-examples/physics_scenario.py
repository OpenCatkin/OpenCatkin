import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencatkin.extractors.base import OllamaExtractor
from opencatkin.validators.base import MathPrimeValidator
from opencatkin.validators.physics_validator import PhysicsConstantValidator
from opencatkin.persistence.graph_storage import GraphStorage
from opencatkin.pipeline import OpenCatkinPipeline
from text_to_graph import visualize_graph

def main():
    
    extractor = OllamaExtractor(model_name="qwen3:8b")
    storage = GraphStorage()
    pipeline = OpenCatkinPipeline(extractor=extractor, storage=storage)

    pipeline.add_validator(PhysicsConstantValidator())

    text = "光速在真空中是 3e8 m/s，而普朗克常数约为 6.626e-34 J·s。"
    pipeline.process_text(text)

    print(f"\n📊 物理知识图谱节点: {storage.get_all_nodes()}")
    storage.save_to_file("physics_graph.gml")
    visualize_graph(storage)

if __name__ == "__main__":
    main()