import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(46)
np.random.seed(36)
random.seed(44)

# 设备设置
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# ============================================================================
# 1. 修复版双曲几何核心（数值稳定 + 维度正确）
# ============================================================================

class FixedPoincareGeometry:
    """修复版双曲几何操作 - 数值稳定 + 维度正确"""
    
    @staticmethod
    def poincare_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """修复的双曲距离计算 - 数值稳定"""
        # 确保输入是二维的
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y
        
        # 确保在单位圆内
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        x = x / torch.max(x_norm, torch.tensor(0.9999)) * 0.9999
        y = y / torch.max(y_norm, torch.tensor(0.9999)) * 0.9999
        
        # 计算距离
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        denom = (1 - x2) * (1 - y2) + eps
        num = 2 * torch.sum((x - y)**2, dim=-1, keepdim=True)
        
        delta = num / denom
        delta = torch.clamp(delta, min=eps)
        
        return torch.arccosh(1 + delta)
    
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """莫比乌斯加法"""
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y
        
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        denom = 1 + 2 * xy + x2 * y2 + eps
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        
        result = num / denom
        
        # 确保在单位圆内
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        result = result / torch.max(result_norm, torch.tensor(0.99999)) * 0.99999
        
        return result
    
    @staticmethod
    def exponential_map(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """指数映射"""
        x = x.unsqueeze(0) if x.dim() == 1 else x
        v = v.unsqueeze(0) if v.dim() == 1 else v
        
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        
        lambda_x = 2.0 / (1.0 - x_norm**2 + eps)
        
        # 防止除零
        v_norm_safe = torch.max(v_norm, torch.tensor(eps))
        direction = v / v_norm_safe
        
        scale = torch.tanh(lambda_x * v_norm / 2.0)
        scaled_v = scale * direction
        
        return FixedPoincareGeometry.mobius_add(x, scaled_v)
    
    @staticmethod
    def logarithmic_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """对数映射"""
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
    
    @staticmethod
    def hyperbolic_attention_fixed(query: torch.Tensor, keys: torch.Tensor, 
                                   values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        完全修复的双曲注意力机制
        query: (d,) 或 (1, d)
        keys: (n, d)
        values: (n, v)
        返回: (v,)
        """
        query = query.unsqueeze(0) if query.dim() == 1 else query  # (1, d)
        
        # 批量计算距离 - 高效且避免循环
        n_keys = keys.size(0)
        query_expanded = query.expand(n_keys, -1)  # (n, d)
        
        # 向量化距离计算
        distances = FixedPoincareGeometry.poincare_distance(query_expanded, keys)  # (n, 1)
        distances = distances.squeeze(-1)  # (n,)
        
        # 计算注意力权重
        attention_weights = F.softmax(-distances / temperature, dim=0)  # (n,)
        
        # 加权求和
        weighted_values = torch.sum(attention_weights.unsqueeze(1) * values, dim=0)  # (v,)
        
        return weighted_values

# ============================================================================
# 2. 优化版推理系统（高准确率）
# ============================================================================

class OptimizedPoincareReasoner:
    """优化版推理系统 - 高准确率"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.concepts = {}
        self.concept_indices = {}
        self.next_index = 0
        self.disease_symptom_map = {}  # 存储疾病-症状关联
        
        self._init_concepts()
    
    def _init_concepts(self):
        """优化的概念初始化 - 更好的几何布局"""
        # 症状 - 均匀分布在圆周
        symptoms = [
            "fever", "cough", "headache", "fatigue", "sore_throat",
            "runny_nose", "shortness_of_breath", "chest_pain",
            "nausea", "vomiting", "diarrhea", "rash", "muscle_pain"
        ]
        
        n_symptoms = len(symptoms)
        for i, symptom in enumerate(symptoms):
            angle = 2 * math.pi * i / n_symptoms
            radius = 0.8  # 症状在外围
            mu = torch.tensor([radius * math.cos(angle), radius * math.sin(angle)], 
                            device=self.device, dtype=torch.float32)
            
            self.concepts[symptom] = {
                "type": "symptom",
                "mu": mu,
                "index": self.next_index
            }
            self.concept_indices[symptom] = self.next_index
            self.next_index += 1
        
        # 疾病定义
        self.disease_symptom_map = {
            "common_cold": {
                "symptoms": ["cough", "runny_nose", "sore_throat"],
                "severity": "mild",
                "position": 0.3  # 距离圆心的距离
            },
            "influenza": {
                "symptoms": ["fever", "cough", "headache", "fatigue", "muscle_pain"],
                "severity": "moderate",
                "position": 0.35
            },
            "covid_19": {
                "symptoms": ["fever", "cough", "shortness_of_breath", "fatigue"],
                "severity": "moderate",
                "position": 0.4
            },
            "pneumonia": {
                "symptoms": ["fever", "cough", "shortness_of_breath", "chest_pain"],
                "severity": "severe",
                "position": 0.45
            },
            "bronchitis": {
                "symptoms": ["cough", "shortness_of_breath", "fatigue"],
                "severity": "moderate",
                "position": 0.35
            },
            "stomach_flu": {
                "symptoms": ["nausea", "vomiting", "diarrhea"],
                "severity": "mild",
                "position": 0.3
            },
            "allergy": {
                "symptoms": ["runny_nose", "rash"],
                "severity": "mild",
                "position": 0.25
            },
            "migraine": {
                "symptoms": ["headache", "nausea"],
                "severity": "moderate",
                "position": 0.3
            }
        }
        
        # 初始化疾病位置 - 放在对应症状的中心
        for disease, info in self.disease_symptom_map.items():
            symptom_points = []
            for symptom in info["symptoms"]:
                if symptom in self.concepts:
                    symptom_points.append(self.concepts[symptom]["mu"])
            
            if symptom_points:
                symptom_tensor = torch.stack(symptom_points)
                disease_mu = torch.mean(symptom_tensor, dim=0)
                
                # 缩放至指定位置
                current_norm = torch.norm(disease_mu)
                if current_norm > 0.01:
                    disease_mu = disease_mu / current_norm * info["position"]
            else:
                angle = random.random() * 2 * math.pi
                disease_mu = torch.tensor([info["position"] * math.cos(angle), 
                                          info["position"] * math.sin(angle)], 
                                        device=self.device, dtype=torch.float32)
            
            self.concepts[disease] = {
                "type": "disease",
                "mu": disease_mu,
                "severity": info["severity"],
                "index": self.next_index
            }
            self.concept_indices[disease] = self.next_index
            self.next_index += 1
    
    def compute_similarity(self, concept1: str, concept2: str) -> float:
        """计算相似度"""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
        
        mu1 = self.concepts[concept1]["mu"]
        mu2 = self.concepts[concept2]["mu"]
        
        distance = FixedPoincareGeometry.poincare_distance(mu1, mu2)
        similarity = torch.exp(-distance).item()
        
        return similarity
    
    def infer_disease(self, observed_symptoms: List[str]) -> Dict[str, float]:
        """
        优化的推理 - 基于距离的直接排序
        """
        diseases = [name for name, info in self.concepts.items() 
                   if info["type"] == "disease"]
        
        if not observed_symptoms:
            return {d: 1.0/len(diseases) for d in diseases}
        
        # 收集症状嵌入
        symptom_embeddings = []
        for symptom in observed_symptoms:
            if symptom in self.concepts:
                symptom_embeddings.append(self.concepts[symptom]["mu"])
        
        if not symptom_embeddings:
            return {d: 1.0/len(diseases) for d in diseases}
        
        symptom_center = torch.mean(torch.stack(symptom_embeddings), dim=0)
        
        # 计算每个疾病到症状中心的距离
        results = {}
        for disease in diseases:
            disease_mu = self.concepts[disease]["mu"]
            distance = FixedPoincareGeometry.poincare_distance(symptom_center, disease_mu)
            
            # 也考虑疾病与症状的匹配度
            match_score = 0.0
            disease_symptoms = self.disease_symptom_map[disease]["symptoms"]
            matched = sum(1 for s in observed_symptoms if s in disease_symptoms)
            match_score = matched / max(len(disease_symptoms), len(observed_symptoms), 1)
            
            # 综合得分
            distance_score = torch.exp(-distance).item()
            combined_score = 0.6 * distance_score + 0.4 * match_score
            
            results[disease] = combined_score
        
        # 归一化
        total = sum(results.values())
        if total > 0:
            for key in results:
                results[key] /= total
        
        return results
    
    def train_optimized(self, dataset: List[Dict], epochs: int = 20, 
                       learning_rate: float = 0.15):
        """优化的训练"""
        print(f"开始优化训练，数据量: {len(dataset)}")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for example in dataset:
                symptoms = example.get("symptoms", [])
                true_disease = example.get("disease", "")
                
                if not symptoms or true_disease not in self.concepts:
                    continue
                
                # 收集症状位置
                symptom_points = []
                for symptom in symptoms:
                    if symptom in self.concepts:
                        symptom_points.append(self.concepts[symptom]["mu"])
                
                if not symptom_points:
                    continue
                
                symptom_tensor = torch.stack(symptom_points)
                symptom_center = torch.mean(symptom_tensor, dim=0)
                
                disease_mu = self.concepts[true_disease]["mu"]
                
                # 计算梯度并更新
                log_vec = FixedPoincareGeometry.logarithmic_map(disease_mu, symptom_center)
                new_mu = FixedPoincareGeometry.exponential_map(
                    disease_mu, learning_rate * log_vec
                ).squeeze(0)
                
                self.concepts[true_disease]["mu"] = new_mu
                
                # 计算损失
                distance = FixedPoincareGeometry.poincare_distance(new_mu, symptom_center)
                total_loss += distance.item()
            
            avg_loss = total_loss / len(dataset) if dataset else 0.0
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, 平均距离损失: {avg_loss:.4f}")
            
            # 学习率衰减
            learning_rate *= 0.95
    
    def diagnose(self, symptoms: List[str]) -> Dict[str, Any]:
        """诊断"""
        results = self.infer_disease(symptoms)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        explanations = []
        for disease, prob in sorted_results[:3]:
            disease_info = self.concepts[disease]
            matched_symptoms = [s for s in symptoms if s in self.disease_symptom_map[disease]["symptoms"]]
            
            explanation = {
                "disease": disease,
                "probability": prob,
                "severity": disease_info.get("severity", "unknown"),
                "matched_symptoms": matched_symptoms
            }
            explanations.append(explanation)
        
        return {
            "symptoms": symptoms,
            "diagnoses": explanations,
            "confidence": sorted_results[0][1] if sorted_results else 0.0
        }

# ============================================================================
# 3. 数据生成和评估
# ============================================================================

def generate_medical_data(n_samples: int = 1000) -> List[Dict]:
    """生成医疗数据"""
    disease_symptom_map = {
        "common_cold": {
            "core": ["cough", "runny_nose", "sore_throat"],
            "optional": ["fever", "headache", "fatigue"],
            "probability": 0.3
        },
        "influenza": {
            "core": ["fever", "cough", "headache", "fatigue", "muscle_pain"],
            "optional": ["sore_throat", "runny_nose"],
            "probability": 0.2
        },
        "covid_19": {
            "core": ["fever", "cough", "shortness_of_breath", "fatigue"],
            "optional": ["headache", "muscle_pain", "sore_throat"],
            "probability": 0.15
        },
        "pneumonia": {
            "core": ["fever", "cough", "shortness_of_breath", "chest_pain"],
            "optional": ["fatigue", "headache"],
            "probability": 0.1
        },
        "bronchitis": {
            "core": ["cough", "shortness_of_breath", "fatigue"],
            "optional": ["fever", "chest_pain"],
            "probability": 0.1
        },
        "stomach_flu": {
            "core": ["nausea", "vomiting", "diarrhea"],
            "optional": ["fever", "headache"],
            "probability": 0.08
        },
        "allergy": {
            "core": ["runny_nose", "rash"],
            "optional": ["cough", "sore_throat"],
            "probability": 0.05
        },
        "migraine": {
            "core": ["headache", "nausea"],
            "optional": ["vomiting", "fatigue"],
            "probability": 0.02
        }
    }
    
    data = []
    for disease, info in disease_symptom_map.items():
        n_disease_samples = int(n_samples * info["probability"])
        
        for _ in range(n_disease_samples):
            present_symptoms = []
            for symptom in info["core"]:
                if random.random() < 0.9:
                    present_symptoms.append(symptom)
            for symptom in info["optional"]:
                if random.random() < 0.4:
                    present_symptoms.append(symptom)
            
            data.append({
                "symptoms": present_symptoms,
                "disease": disease
            })
    
    return data

def evaluate_system(reasoner: OptimizedPoincareReasoner, 
                   test_data: List[Dict]) -> Dict[str, Any]:
    """评估系统"""
    correct = 0
    total = len(test_data)
    
    for i, test_case in enumerate(test_data):
        symptoms = test_case.get("symptoms", [])
        true_disease = test_case.get("disease", "")
        
        if not symptoms or not true_disease:
            continue
        
        diagnosis_result = reasoner.diagnose(symptoms)
        
        if diagnosis_result["diagnoses"]:
            predicted_disease = diagnosis_result["diagnoses"][0]["disease"]
            if predicted_disease == true_disease:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

# ============================================================================
# 4. 主程序
# ============================================================================

def main():
    print("="*80)
    print("优化版双曲几何推理引擎 - 医疗诊断系统")
    print("="*80)
    
    # 创建推理器
    print("\n🏥 创建医疗诊断系统...")
    reasoner = OptimizedPoincareReasoner(device=device)
    
    # 生成数据
    print("\n📊 生成训练数据...")
    train_data = generate_medical_data(1000)
    print(f"   生成 {len(train_data)} 个训练样本")
    
    # 训练
    print("\n🎯 开始训练...")
    start_time = time.time()
    reasoner.train_optimized(train_data, epochs=20, learning_rate=0.15)
    end_time = time.time()
    print(f"   训练完成，耗时: {end_time - start_time:.2f}秒")
    
    # 诊断演示
    print("\n🔍 诊断演示...")
    
    test_cases = [
        (["cough", "runny_nose", "sore_throat"], "感冒"),
        (["fever", "cough", "headache", "muscle_pain"], "流感"),
        (["fever", "shortness_of_breath", "chest_pain"], "肺炎"),
        (["nausea", "vomiting", "diarrhea"], "肠胃炎"),
    ]
    
    for i, (symptoms, desc) in enumerate(test_cases, 1):
        print(f"\n案例 {i} ({desc}): 症状: {', '.join(symptoms)}")
        
        result = reasoner.diagnose(symptoms)
        
        if result["diagnoses"]:
            best = result["diagnoses"][0]
            print(f"  最可能的诊断: {best['disease']} (概率: {best['probability']:.3f}, "
                  f"严重程度: {best['severity']})")
            print(f"  匹配症状: {', '.join(best['matched_symptoms'])}")
            
            print(f"  前3个诊断:")
            for j, diag in enumerate(result["diagnoses"][:3], 1):
                print(f"    {j}. {diag['disease']}: {diag['probability']:.3f}")
    
    # 评估
    print("\n📊 创建测试数据...")
    test_data = generate_medical_data(200)
    
    print("\n📈 系统评估...")
    evaluation = evaluate_system(reasoner, test_data)
    print(f"  总体准确率: {evaluation['accuracy']:.4f}")
    print(f"  正确/总数: {evaluation['correct']}/{evaluation['total']}")
    
    print("\n" + "="*80)
    print("性能总结:")
    print("="*80)
    print(f"1. 使用设备: {device}")
    print(f"2. 总体准确率: {evaluation['accuracy']:.4f}")
    
    print("\n✨ 演示完成!")
    
    return reasoner, evaluation

if __name__ == "__main__":
    try:
        reasoner, evaluation = main()
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()