import json
import ollama

class BaseExtractor:
    def extract(self, text: str) -> list[dict]:
        raise NotImplementedError("子类必须实现提取逻辑")

# ----------------- 这是旧的模拟器，可以删掉或注释掉 -----------------
# class LLMMockExtractor(BaseExtractor):
#     def extract(self, text: str) -> list[dict]:
#         # ...

# ----------------- 这是全新的、真实的 Ollama 提取器 -----------------
class OllamaExtractor(BaseExtractor):
    """
    一个真实的提取器，调用在本地运行的 Ollama 模型。
    """
    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name
        # 确认模型已在本地存在
        try:
            ollama.show(model_name)
        except ollama.ResponseError:
            print(f"❌ 错误: Ollama 模型 '{model_name}' 未找到。")
            print(f"请先运行 'ollama pull {model_name}' 进行拉取。")
            raise

    def extract(self, text: str) -> list[dict]:
        print(f"[Extractor] 正在调用本地 Ollama 模型 '{self.model_name}' 进行解析...")
        
        # 关键！这是驯服本地模型的“紧箍咒” Prompt
        prompt = f"""
        你是一个只会输出JSON的机器人。你的任务是从用户提供的文本中提取事实。
        你的响应必须是一个JSON对象，其中包含一个名为 "facts" 的键，其值是一个列表。
        列表中的每个元素都是一个字典，包含 "subject", "relation", "object" 三个键。
        不要包含任何解释、前言、或 markdown 代码块。只返回纯粹的JSON。

        例如:
        用户文本: "水由氢和氧组成。"
        你的响应: {{"facts": [{{"subject": "水", "relation": "由...组成", "object": "氢和氧"}}]}}
        
        现在，从以下文本中提取事实:
        "{text}"
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json', # 关键参数：强制 Ollama 输出 JSON 格式
            )
            
            content = response['message']['content']
            # 有时模型仍然会返回被字符串包裹的JSON，需要二次解析
            data = json.loads(content)
            
            # 确保返回的数据格式是我们期望的列表
            return data.get("facts", [])

        except json.JSONDecodeError:
            print(f"❌ [Extractor 警告] 模型返回了非JSON格式的响应，本次提取失败。")
            return []
        except Exception as e:
            print(f"❌ [Extractor 错误] 调用 Ollama 时发生错误: {e}")
            return []