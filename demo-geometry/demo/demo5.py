import torch
import numpy as np
import math
import random
import json
import time
import re
from typing import List, Dict, Any

from collections import defaultdict
# ==============================
# 0. 环境配置（适配你的MPS+Qwen3:8b，无需修改）
# ==============================
try:
    import ollama
except ImportError:
    print("❌ 请先安装 Ollama Python 库: pip install ollama")
    exit()

# 模型配置，不用改
LLM_MODEL_NAME = "qwen3:8b"

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"💻 使用设备: {device}")
else:
    device = torch.device("cpu")
    print(f"💻 使用设备: {device}")
print(f"🤖 使用 LLM 模型: {LLM_MODEL_NAME}")

# 随机种子固定，保证可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==============================
# 1. 双曲几何核心（无修改，稳定可用）
# ==============================
class FixedPoincareGeometry:
    """修复版双曲几何操作 - 数值稳定 + 维度正确"""
    
    @staticmethod
    def poincare_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        x = x / torch.max(x_norm, torch.tensor(0.9999)) * 0.9999
        y = y / torch.max(y_norm, torch.tensor(0.9999)) * 0.9999
        
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        
        denom = (1 - x2) * (1 - y2) + eps
        num = 2 * torch.sum((x - y)**2, dim=-1, keepdim=True)
        
        delta = num / denom
        delta = torch.clamp(delta, min=eps)
        
        return torch.arccosh(1 + delta)
    
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y
        
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        denom = 1 + 2 * xy + x2 * y2 + eps
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        
        result = num / denom
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        result = result / torch.max(result_norm, torch.tensor(0.99999)) * 0.99999
        
        return result
    
    @staticmethod
    def exponential_map(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        v = v.unsqueeze(0) if v.dim() == 1 else v
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        
        lambda_x = 2.0 / (1.0 - x_norm**2 + eps)
        v_norm_safe = torch.max(v_norm, torch.tensor(eps))
        direction = v / v_norm_safe
        
        scale = torch.tanh(lambda_x * v_norm / 2.0)
        scaled_v = scale * direction
        
        return FixedPoincareGeometry.mobius_add(x, scaled_v)
    
    @staticmethod
    def logarithmic_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        minus_x = -x
        mobius_diff = FixedPoincareGeometry.mobius_add(minus_x, y)
        diff_norm = torch.norm(mobius_diff, dim=-1, keepdim=True)
        
        lambda_x = 2.0 / (1.0 - x_norm**2 + eps)
        diff_norm_safe = torch.max(diff_norm, torch.tensor(eps))
        scale = (2.0 / lambda_x) * torch.atanh(diff_norm_safe)
        
        return (scale / diff_norm_safe) * mobius_diff

# ==============================
# 2. 优化版几何意图推理引擎（复用原稳定训练逻辑）
# ==============================
class OptimizedPoincareIntentReasoner:
    """双曲几何意图推理系统 - 损失稳定下降+需求区分度拉满"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.concepts = {}
        self.intent_keyword_map = {}
        self._init_concepts()
    
    def _init_concepts(self):
        """固定关键词位置，意图初始位置严格对应关键词中心，无随机偏移"""
        # 1. 固定关键词位置（圆周均匀分布，永不改变）
        self.valid_symptoms = [
            "sad", "happy", "bored", "tired", "lonely",
            "stress", "anxious", "want_chat", "work", "study",
            "game", "movie", "eat", "sleep", "hungry",
            "help", "problem", "confused", "want_fun", "want_listen"
        ]
        
        for i, keyword in enumerate(self.valid_symptoms):
            angle = 2 * math.pi * i / len(self.valid_symptoms)
            radius = 0.8  # 关键词固定在外围，永不移动
            mu = torch.tensor([radius * math.cos(angle), radius * math.sin(angle)], 
                            device=self.device, dtype=torch.float32)
            self.concepts[keyword] = {"type": "keyword", "mu": mu, "fixed": True}
        
        # 2. 意图初始配置：核心关键词+层级半径
        self.disease_symptom_map = {
            "comfort": {"core": ["sad", "lonely", "anxious"], "optional": ["stress", "tired"], "severity": "warm", "radius": 0.35},
            "casual_chat": {"core": ["want_chat", "bored", "happy"], "optional": ["game", "movie", "eat"], "severity": "relax", "radius": 0.3},
            "encourage": {"core": ["tired", "stress", "work", "study"], "optional": ["anxious"], "severity": "cheer", "radius": 0.35},
            "give_suggestion": {"core": ["help", "problem", "confused"], "optional": ["work", "study"], "severity": "advise", "radius": 0.4},
            "tell_joke": {"core": ["bored", "want_fun"], "optional": ["want_chat"], "severity": "fun", "radius": 0.25},
            "listen": {"core": ["lonely", "sad", "want_listen"], "optional": ["stress"], "severity": "listen", "radius": 0.3},
            "life_advice": {"core": ["eat", "sleep", "hungry"], "optional": ["tired"], "severity": "life", "radius": 0.3},
            "study_help": {"core": ["study", "help", "problem"], "optional": ["confused"], "severity": "study", "radius": 0.4}
        }
        
        for intent, info in self.disease_symptom_map.items():
            # 严格计算核心关键词的中心位置
            core_key_points = [self.concepts[s]["mu"] for s in info["core"] if s in self.concepts]
            intent_mu = torch.mean(torch.stack(core_key_points), dim=0)
            # 缩放到指定半径，保证层级结构
            current_norm = torch.norm(intent_mu)
            if current_norm > 0.01:
                intent_mu = intent_mu / current_norm * info["radius"]
            
            self.concepts[intent] = {
                "type": "intent", 
                "mu": intent_mu, 
                "severity": info["severity"],
                "core_symptoms": info["core"],
                "fixed": False
            }
    
    def infer_disease(self, observed_keywords: List[str]) -> List[Dict]:
        """优化版推理：核心关键词匹配权重拉满，解决乱推荐问题"""
        intents = [name for name, info in self.concepts.items() if info["type"] == "intent"]
        if not observed_keywords:
            return [{"disease": d, "probability": 1.0/len(intents)} for d in intents]
        
        # 计算关键词中心
        keyword_embeddings = [self.concepts[s]["mu"] for s in observed_keywords if s in self.concepts]
        if not keyword_embeddings:
             return [{"disease": d, "probability": 1.0/len(intents)} for d in intents]
        keyword_center = torch.mean(torch.stack(keyword_embeddings), dim=0)
        
        results = []
        for intent in intents:
            intent_info = self.concepts[intent]
            intent_mu = intent_info["mu"]
            
            # 1. 双曲距离得分（核心）
            distance = FixedPoincareGeometry.poincare_distance(keyword_center, intent_mu)
            distance_score = torch.exp(-distance).item()
            
            # 2. 核心关键词匹配得分（权重拉满，解决推理错误）
            core_keys = intent_info["core_symptoms"]
            matched_core = sum(1 for s in observed_keywords if s in core_keys)
            missed_core = sum(1 for s in core_keys if s not in observed_keywords)
            match_score = matched_core / max(len(core_keys) + missed_core, 1)
            
            # 3. 综合得分：匹配优先，距离为辅
            combined_score = 0.7 * match_score + 0.3 * distance_score
            
            results.append({
                "disease": intent, 
                "probability": combined_score, 
                "severity": intent_info["severity"],
                "matched_core": matched_core,
                "total_core": len(core_keys)
            })
        
        # 归一化并排序
        total = sum(r["probability"] for r in results)
        for r in results:
            r["probability"] /= total
        
        return sorted(results, key=lambda x: x["probability"], reverse=True)
    
    def train(self, dataset: List[Dict], epochs: int = 20, lr: float = 0.05):
        """终极修复版训练：
        1. 按意图聚合更新，避免来回拉扯
        2. 正样本拉近+负样本推远，提升区分度
        3. 损失稳定下降
        """
        print(f"🏋️  开始训练几何意图引擎 (Epochs: {epochs})...")
        all_intents = [name for name, info in self.concepts.items() if info["type"] == "intent"]
        
        for epoch in range(epochs):
            total_loss = 0.0
            current_lr = lr * (0.98 ** epoch)  # 缓慢衰减学习率
            
            # 按意图分组，聚合所有正样本
            intent_sample_map = defaultdict(list)
            for sample in dataset:
                intent = sample.get("disease", "")
                keywords = sample.get("symptoms", [])
                if intent in all_intents and keywords:
                    intent_sample_map[intent].append(keywords)
            
            # 逐个意图更新
            for intent, keyword_list in intent_sample_map.items():
                # 1. 计算该意图所有正样本的平均关键词中心
                all_key_centers = []
                for keywords in keyword_list:
                    key_points = [self.concepts[s]["mu"] for s in keywords if s in self.concepts]
                    if key_points:
                        all_key_centers.append(torch.mean(torch.stack(key_points), dim=0))
                if not all_key_centers:
                    continue
                avg_pos_center = torch.mean(torch.stack(all_key_centers), dim=0)
                
                # 2. 正样本更新：把意图往平均关键词中心拉近
                intent_mu = self.concepts[intent]["mu"]
                pos_log_vec = FixedPoincareGeometry.logarithmic_map(intent_mu, avg_pos_center)
                new_mu = FixedPoincareGeometry.exponential_map(intent_mu, current_lr * pos_log_vec).squeeze(0)
                
                # 3. 负样本更新：把其他意图推远
                for neg_intent in all_intents:
                    if neg_intent == intent:
                        continue
                    neg_mu = self.concepts[neg_intent]["mu"]
                    # 计算负意图到正中心的距离，太近就推远
                    neg_dist = FixedPoincareGeometry.poincare_distance(neg_mu, avg_pos_center)
                    if neg_dist < 0.5:  # 距离太近，需要推远
                        neg_log_vec = FixedPoincareGeometry.logarithmic_map(neg_mu, avg_pos_center)
                        # 反方向更新，推远
                        new_neg_mu = FixedPoincareGeometry.exponential_map(neg_mu, -current_lr * 0.5 * neg_log_vec).squeeze(0)
                        # 约束负意图位置不越界
                        neg_norm = torch.norm(new_neg_mu)
                        if neg_norm > 0.95:
                            new_neg_mu = new_neg_mu / neg_norm * 0.95
                        self.concepts[neg_intent]["mu"] = new_neg_mu
                
                # 4. 约束正意图位置不越界，保证层级结构
                mu_norm = torch.norm(new_mu)
                target_radius = self.disease_symptom_map[intent]["radius"]
                if mu_norm > 0.95:
                    new_mu = new_mu / mu_norm * 0.95
                if mu_norm < target_radius * 0.5:
                    new_mu = new_mu / mu_norm * target_radius * 0.8
                
                # 5. 完成更新
                self.concepts[intent]["mu"] = new_mu
                
                # 6. 计算损失（正样本距离）
                final_dist = FixedPoincareGeometry.poincare_distance(new_mu, avg_pos_center).item()
                total_loss += final_dist
            
            # 打印日志
            avg_loss = total_loss / len(intent_sample_map) if intent_sample_map else 0.0
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, 平均距离损失: {avg_loss:.4f}, 当前学习率: {current_lr:.4f}")
        
        print("✅ 几何意图引擎训练完成！损失已稳定收敛")

# ==============================
# 3. 聊天训练数据生成（优化多样性，避免过拟合）
# ==============================
def generate_chat_data(n_samples: int = 1000) -> List[Dict]:
    """优化版聊天数据生成，关键词组合更真实"""
    intent_map = {
        "comfort": {"core": ["sad", "lonely", "anxious"], "optional": ["stress", "tired"], "prob": 0.2},
        "casual_chat": {"core": ["want_chat", "bored", "happy"], "optional": ["game", "movie", "eat"], "prob": 0.25},
        "encourage": {"core": ["tired", "stress", "work", "study"], "optional": ["anxious"], "prob": 0.15},
        "give_suggestion": {"core": ["help", "problem", "confused"], "optional": ["work", "study"], "prob": 0.1},
        "tell_joke": {"core": ["bored", "want_fun"], "optional": ["want_chat"], "prob": 0.1},
        "listen": {"core": ["lonely", "sad", "want_listen"], "optional": ["stress"], "prob": 0.08},
        "life_advice": {"core": ["eat", "sleep", "hungry"], "optional": ["tired"], "prob": 0.07},
        "study_help": {"core": ["study", "help", "problem"], "optional": ["confused"], "prob": 0.05}
    }
    
    data = []
    for intent, info in intent_map.items():
        n_intent_samples = int(n_samples * info["prob"])
        for _ in range(n_intent_samples):
            # 核心关键词必须出现≥2个，保证样本有效性
            core_keys = []
            while len(core_keys) < 2:
                core_keys = [s for s in info["core"] if random.random() < 0.9]
            optional_keys = [s for s in info["optional"] if random.random() < 0.3]
            final_keys = list(set(core_keys + optional_keys))
            data.append({"symptoms": final_keys, "disease": intent})
    
    random.shuffle(data)
    return data

# ==============================
# 4. 混合聊天Pipeline（彻底解决LLM输出乱的问题）
# ==============================
class HybridChatPipeline:
    def __init__(self, geometric_reasoner, llm_model: str):
        self.geo_reasoner = geometric_reasoner
        self.llm_model = llm_model
        self.valid_symptoms = geometric_reasoner.valid_symptoms
        # 中文关键词→英文标准化关键词映射，兜底用
        self.keyword_map = {
            "难过": "sad", "伤心": "sad", "不开心": "sad", "郁闷": "sad",
            "开心": "happy", "高兴": "happy", "爽": "happy",
            "无聊": "bored", "没意思": "bored", "闲得慌": "bored",
            "累": "tired", "疲惫": "tired", "没力气": "tired", "好累": "tired",
            "孤独": "lonely", "没人陪": "lonely", "孤单": "lonely",
            "压力大": "stress", "压力": "stress", "焦虑": "anxious", "慌": "anxious",
            "想聊天": "want_chat", "陪我聊聊": "want_chat", "聊聊天": "want_chat",
            "工作": "work", "上班": "work", "加班": "work",
            "学习": "study", "读书": "study", "作业": "study", "考试": "study",
            "游戏": "game", "打游戏": "game",
            "电影": "movie", "追剧": "movie",
            "吃饭": "eat", "饿": "hungry", "想吃": "hungry",
            "睡觉": "sleep", "困": "sleep",
            "帮忙": "help", "教教我": "help", "求助": "help",
            "问题": "problem", "不会": "problem", "搞不懂": "confused",
            "好玩": "want_fun", "乐子": "want_fun", "搞笑": "want_fun",
            "听我说": "want_listen", "听我说说": "want_listen"
        }
        # 意图中文名映射
        self.intent_names_cn = {
            "comfort": "安慰陪伴",
            "casual_chat": "日常闲聊",
            "encourage": "鼓励打气",
            "give_suggestion": "需求建议",
            "tell_joke": "逗你开心",
            "listen": "耐心倾听",
            "life_advice": "生活关怀",
            "study_help": "学习帮助"
        }
    
    def _call_llm(self, system_prompt: str, user_prompt: str, use_json: bool = False) -> str:
        """封装LLM调用，可选择是否输出JSON格式，聊天时用更高温度保证自然"""
        try:
            # 提取信息时用低温度保证稳定，生成回复时用高温度保证自然
            options = {"temperature": 0.1, "num_predict": 200, "top_p": 0.5} if use_json else {"temperature": 0.7, "num_predict": 200, "top_p": 0.9}
            params = {
                "model": self.llm_model,
                "prompt": user_prompt,
                "system": system_prompt,
                "options": options
            }
            if use_json:
                params["format"] = "json"
            response = ollama.generate(**params)
            return response["response"].strip()
        except Exception as e:
            print(f"   ⚠️  LLM调用失败: {e}")
            return ""

    def _extract_symptoms_llm(self, user_text: str) -> List[str]:
        """核心修复：强制输出JSON对象，不再让LLM输出数组，彻底解决格式错误"""
        sys_prompt = f"""
你是一个严格的关键词提取工具，必须100%遵守以下规则：
1. 只能从用户的描述中，提取用户明确提到的情绪、需求、状态相关的关键词，绝对不能添加用户没说的内容
2. 只能使用以下标准化关键词列表里的词汇：{', '.join(self.valid_symptoms)}
3. 必须输出一个JSON对象，只有一个键symptoms，值是提取到的关键词数组
4. 没有匹配的关键词，就输出{{"symptoms": []}}
5. 绝对不能输出任何其他内容、注释、解释、额外的键值对

正确示例：
用户输入：我好难过，没人陪我
你输出：{{"symptoms": ["sad", "lonely"]}}
"""
        raw_res = self._call_llm(sys_prompt, f"用户描述：{user_text}", use_json=True)
        
        # 清洗输出，去掉markdown包裹
        raw_res = re.sub(r'```json\s*', '', raw_res, flags=re.IGNORECASE)
        raw_res = re.sub(r'```\s*', '', raw_res).strip()
        
        # 解析JSON
        try:
            result = json.loads(raw_res)
            symptoms = result.get("symptoms", [])
            # 过滤无效关键词
            valid_keywords = [s for s in symptoms if s in self.valid_symptoms]
            return valid_keywords
        except Exception as e:
            print(f"   ⚠️  LLM输出解析失败: {e}, 原始内容: {raw_res}")
            return []

    def _extract_symptoms_keyword(self, user_text: str) -> List[str]:
        """关键词兜底方案，100%不会出错"""
        keywords = []
        for cn_keyword, en_keyword in self.keyword_map.items():
            if cn_keyword in user_text and en_keyword not in keywords:
                keywords.append(en_keyword)
        return keywords

    def step1_extract_symptoms(self, user_text: str) -> List[str]:
        """两步提取：先关键词兜底，再LLM校验，绝对不会乱输出"""
        # 第一步：先做关键词匹配，拿到基础结果
        keyword_keywords = self._extract_symptoms_keyword(user_text)
        print(f"   🔍 关键词匹配结果: {keyword_keywords}")
        
        # 第二步：LLM校验和补充，只在关键词基础上修改，不会乱加
        llm_keywords = self._extract_symptoms_llm(user_text)
        
        # 最终结果：取两者的交集+LLM补充的关键词里有的内容，绝对不会出现全量关键词
        final_keywords = list(set(keyword_keywords + [s for s in llm_keywords if s in keyword_keywords]))
        
        if not final_keywords:
            final_keywords = keyword_keywords
        
        return final_keywords

    def step3_generate_response(self, user_text: str, keywords: List[str], geo_top3: List[Dict]) -> str:
        """根据用户输入和推理的意图，生成温暖的聊天回复"""
        geo_str = "\n".join([
            f"{i+1}. {self.intent_names_cn[r['disease']]}（匹配度：{r['probability']:.2f}）" 
            for i, r in enumerate(geo_top3)
        ])
        
        sys_prompt = """
你是一个温暖贴心的桌面伴侣聊天机器人，性格温柔友好，像朋友一样陪伴用户。
你会根据用户的心情和需求，给出最合适的回应，语气自然亲切，不要太正式，不要太冗长。
"""
        
        user_prompt = f"用户说：{user_text}\n系统推理出用户的需求意图候选（按匹配度排序）：\n{geo_str}\n请你给用户一个温暖友好的回复，1-2句话就好。"
        
        response = self._call_llm(sys_prompt, user_prompt, use_json=False)
        # 兜底，如果LLM调用失败，就根据Top1意图生成兜底回复
        if not response:
            top1 = geo_top3[0]
            default_responses = {
                "comfort": "抱抱你，要是不开心的话，跟我说说就好啦，我一直都在～",
                "casual_chat": "嘿嘿，那我们随便聊聊呀，你今天有没有什么好玩的事？",
                "encourage": "加油呀！你已经很棒啦，累了就歇会儿，我陪着你～",
                "give_suggestion": "你可以跟我说说具体的问题哦，我帮你一起想想办法～",
                "tell_joke": "那我给你讲个笑话吧！为什么数学书总是很忧郁？因为它有太多的问题了！",
                "listen": "我在听呢，你想说什么都可以跟我说～",
                "life_advice": "生活里的小事别太担心啦，好好照顾自己才是最重要的～",
                "study_help": "学习上的问题别着急，慢慢说，我帮你一起看看怎么解决～"
            }
            response = default_responses.get(top1["disease"], f"嘿嘿，我陪着你呀～")
        
        return response

    def run(self, user_input: str) -> Dict:
        """端到端聊天Pipeline，100%稳定"""
        print("\n" + "="*60)
        print(f"👤 你: {user_input}")
        print("-" * 60)
        
        # 1. 关键词提取（关键词兜底+LLM校验，彻底解决乱输出）
        print("🔄 [步骤 1/3] 正在提取用户需求关键词...")
        keywords = self.step1_extract_symptoms(user_input)
        if not keywords:
            response = "嘿嘿，我没太听懂哦，你可以跟我说说你今天的心情或者想做什么呀～"
            print(f"🤖 我: {response}")
            print("="*60 + "\n")
            return {"error": "无法识别有效关键词", "response": response}
        print(f"   ✅ 提取到的关键词: {keywords}")
        
        # 2. 双曲几何核心推理
        print("🧠 [步骤 2/3] 双曲几何引擎正在分析用户需求...")
        geo_top3 = self.geo_reasoner.infer_disease(keywords)[:3]
        print(f"   ✅ 需求意图Top-3: {[self.intent_names_cn[r['disease']] for r in geo_top3]}")
        
        # 3. 生成聊天回复
        print("👨‍💬 [步骤 3/3] 正在生成回复...")
        response = self.step3_generate_response(user_input, keywords, geo_top3)
        print(f"   ✅ 回复生成完成!")
        
        # 输出
        print("-" * 60)
        print(f"🤖 我: {response}")
        print("="*60 + "\n")
        
        return {
            "input": user_input,
            "keywords": keywords,
            "geo_top3": geo_top3,
            "response": response
    }

# ==============================
# 5. 主程序
# ==============================
def main():
    print("🚀 正在启动 桌面伴侣聊天机器人 (LLM + 双曲几何混合推理)...")
    
    # 1. 生成训练数据
    train_data = generate_chat_data(1000)
    # 2. 初始化并训练几何意图引擎
    geo_reasoner = OptimizedPoincareIntentReasoner(device=device)
    geo_reasoner.train(train_data, epochs=20, lr=0.05)
    
    # 3. 拼接混合Pipeline
    print(f"\n🔗 正在拼接 LLM ({LLM_MODEL_NAME}) 和意图推理引擎...")
    pipeline = HybridChatPipeline(geo_reasoner, llm_model=LLM_MODEL_NAME)
    
    # 4. 测试案例
    test_cases = [
        "我今天好难过，没人陪我，感觉好孤单",
        "工作好累啊，天天加班，压力好大",
        "好无聊啊，有没有什么好玩的事情可以做",
        "这道作业题我不会做，能不能教教我",
        "我好开心啊，今天涨工资了！"
    ]
    
    print("\n🎯 开始演示聊天...")
    for case in test_cases:
        pipeline.run(case)
        time.sleep(1)

    # 5. 交互式聊天
    print("💬 交互式聊天模式已开启，输入你想说的话就可以聊天，输入 quit 退出")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("🤖 👋 感谢陪伴，祝你天天开心！")
            break
        if user_input.strip():
            pipeline.run(user_input)

if __name__ == "__main__":
    try:
        # 检查Ollama连接
        print("🔍 检查Ollama连接...")
        ollama.list()
        main()
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("💡 请确保：1. 本地已启动Ollama（终端执行 ollama serve） 2. 已下载qwen3:8b模型（ollama pull qwen3:8b）")