class OpenCatkinPipeline:
    """
    OpenCatkin 核心工作流：读取文本 -> 提纯提案 -> 插件校验 -> 写入图谱
    """
    def __init__(self, extractor, storage):
        self.extractor = extractor
        self.storage = storage
        self.validators = []

    def add_validator(self, validator):
        """挂载外部业务逻辑/物理校验插件"""
        self.validators.append(validator)

    def process_text(self, text: str):
        print("\n--- 🚀 开始处理新文本流 ---")
        # 1. 提案提取
        proposals = self.extractor.extract(text)
        
        valid_facts = []
        for p in proposals:
            # 2. 三层防线校验
            is_legal = True
            for v in self.validators:
                if not v.validate(p):
                    print(f"❌ [Pipeline 拦截] 校验失败，抛弃幻觉数据: {p}")
                    is_legal = False
                    break
            
            if is_legal:
                print(f"✅ [Pipeline 通过] 校验成功: {p}")
                valid_facts.append(p)

        # 3. 持久化固化
        for fact in valid_facts:
            self.storage.upsert_edge(fact["subject"], fact["object"], fact["relation"])
            
        print("--- 处理完成 ---\n")