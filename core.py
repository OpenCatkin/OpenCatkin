# opencatkin/pipeline.py

class OpenCatkin:
    def __init__(self, extractor, storage):
        self.extractor = extractor  # 负责调 LLM 提取 JSON
        self.storage = storage      # 负责存图 (NetworkX/Neo4j)
        self.validators = []        # 验证插件列表

    def add_validator(self, validator_func):
        """添加验证插件，例如化学价键检查、数学计算检查"""
        self.validators.append(validator_func)

    def process_text(self, text):
        # 1. 提取 (提案)
        proposals = self.extractor.extract(text) # 返回 [{"e1": "H2O", "rel": "is", "e2": "water"}]
        
        valid_facts = []
        for p in proposals:
            # 2. 校验 (三层防线)
            is_legal = True
            for validate in self.validators:
                if not validate(p):
                    is_legal = False
                    print(f"检测到非法数据，已拦截: {p}")
                    break
            
            if is_legal:
                valid_facts.append(p)

        # 3. 固化 (写入图谱)
        for fact in valid_facts:
            self.storage.upsert_edge(fact['e1'], fact['e2'], fact['rel'])
            
    def query(self, start_node, end_node):
        """寻找确定性的路径，不靠 LLM 瞎猜"""
        return self.storage.find_shortest_path(start_node, end_node)
