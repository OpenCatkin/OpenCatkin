# ==============================================
# 庞加莱圆盘元信息认知框架
# 扩展版：支持括号和多个运算符
# 严格实现论文：双曲嵌入 + 规则推理 + 路径搜索
# 支持任意整数和复杂表达式
# ==============================================
import math
from typing import List, Tuple, Dict, Optional, Any
import re

# ==============================================
# 一、庞加莱圆盘核心数学（论文公式直接实现）
# 定义：D^2 = { (x,y) | x²+y² < 1 }
# 距离公式、模长、几何约束
# ==============================================
def poincare_distance(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    """
    论文公式：庞加莱圆盘两点间双曲距离
    d_D(u,v) = arcosh( 1 + 2||u-v||² / [(1-||u||²)(1-||v||²)] )
    """
    norm_u_sq = u[0]**2 + u[1]**2
    norm_v_sq = v[0]**2 + v[1]**2
    diff_norm_sq = (u[0]-v[0])**2 + (u[1]-v[1])**2
    
    inside_arg = 1 + 2 * diff_norm_sq / ((1 - norm_u_sq) * (1 - norm_v_sq))
    inside_arg = max(inside_arg, 1.0)  # 数值稳定
    
    return math.acosh(inside_arg)

def poincare_norm(z: Tuple[float, float]) -> float:
    """计算庞加莱圆盘内点的模长（到圆心距离）"""
    return math.sqrt(z[0]**2 + z[1]**2)

def is_in_disk(z: Tuple[float, float]) -> bool:
    """检查点是否在单位圆盘内（几何合法性约束）"""
    return poincare_norm(z) < 1.0 - 1e-6

def mobius_addition(u: Tuple[float, float], v: Tuple[float, float]) -> Tuple[float, float]:
    """
    莫比乌斯加法（双曲平移）
    论文公式：u ⊕ v = (1 + 2⟨u,v⟩ + ||v||²)u + (1 - ||u||²)v / (1 + 2⟨u,v⟩ + ||u||²||v||²)
    简化实现：在切空间中计算
    """
    # 计算点积
    dot = u[0]*v[0] + u[1]*v[1]
    norm_u_sq = u[0]**2 + u[1]**2
    norm_v_sq = v[0]**2 + v[1]**2
    
    denominator = 1 + 2*dot + norm_u_sq*norm_v_sq
    
    x = ((1 + 2*dot + norm_v_sq)*u[0] + (1 - norm_u_sq)*v[0]) / denominator
    y = ((1 + 2*dot + norm_v_sq)*u[1] + (1 - norm_u_sq)*v[1]) / denominator
    
    # 确保仍在单位圆内
    norm = math.sqrt(x**2 + y**2)
    if norm >= 1.0:
        x, y = x/norm*0.999, y/norm*0.999
    
    return (x, y)

# ==============================================
# 二、原子概念空间（元信息知识库）
# 规则：
# 1. 具体概念（数字）→ 靠近边界（模长大）
# 2. 抽象概念（运算符）→ 靠近圆心（模长小）
# 3. 括号作为结构概念
# ==============================================
ATOM_CONCEPTS = {
    # 数字：具体实例，靠近圆盘边界
    "0": (0.95, 0.00),
    "1": (0.94, 0.05),
    "2": (0.93, 0.08),
    "3": (0.92, 0.10),
    "4": (0.91, 0.12),
    "5": (0.90, 0.14),
    "6": (0.89, 0.16),
    "7": (0.88, 0.18),
    "8": (0.87, 0.20),
    "9": (0.86, 0.22),
    
    # 运算符：抽象操作，靠近圆心
    # 乘法比加法更靠近圆心 → 更高优先级
    "+": (0.40, 0.00),  # 距离圆心约0.40
    "-": (0.41, 0.10),
    "*": (0.35, 0.00),  # 乘法比加法更靠近圆心
    "/": (0.36, 0.10),
    
    # 括号：结构运算符，用于分组
    "(": (0.60, 0.30),  # 左括号
    ")": (0.60, -0.30), # 右括号
}

# ==============================================
# 三、推理规则库（论文：规则 = 模式 + 变换 + 约束）
# 每条规则 r = (pattern, condition, transform, weight)
# ==============================================
RULE_SET = {
    "add": {
        "operator": "+",
        "priority": 1,  # 优先级
        "description": "加法规则",
        "compute": lambda a, b: a + b
    },
    "sub": {
        "operator": "-",
        "priority": 1,
        "description": "减法规则",
        "compute": lambda a, b: a - b
    },
    "mul": {
        "operator": "*",
        "priority": 2,  # 乘法优先级高于加法
        "description": "乘法规则",
        "compute": lambda a, b: a * b
    },
    "div": {
        "operator": "/",
        "priority": 2,
        "description": "除法规则",
        "compute": lambda a, b: a / b if b != 0 else float("inf")
    }
}

# ==============================================
# 四、括号处理机制（几何收缩变换）
# 核心思想：括号内的点向括号中心收缩，提高优先级
# ==============================================
def apply_parenthesis_contraction(tokens: List[Dict], start: int, end: int) -> List[Dict]:
    """
    对括号内的点进行几何收缩变换
    收缩因子：括号内的点向括号中心移动，提高优先级
    """
    if end - start <= 1:
        return tokens
    
    # 计算括号中心（左右括号的平均位置）
    left_pos = tokens[start]["embedding"]
    right_pos = tokens[end]["embedding"]
    
    center_x = (left_pos[0] + right_pos[0]) / 2
    center_y = (left_pos[1] + right_pos[1]) / 2
    center = (center_x, center_y)
    
    # 收缩因子：向中心移动50%
    contraction_factor = 0.5
    
    # 对括号内的每个点进行收缩
    for i in range(start + 1, end):
        token = tokens[i]
        if token["type"] in ["number", "operator"]:
            # 当前点位置
            pos = token["embedding"]
            
            # 计算从中心到点的向量
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            
            # 收缩：向中心移动
            new_x = center[0] + dx * contraction_factor
            new_y = center[1] + dy * contraction_factor
            
            # 确保仍在单位圆内
            norm = math.sqrt(new_x**2 + new_y**2)
            if norm >= 1.0:
                new_x, new_y = new_x/norm*0.999, new_y/norm*0.999
            
            # 更新点的位置
            token["embedding"] = (new_x, new_y)
            
            # 标记为括号内点，提高优先级
            token["in_parentheses"] = True
            if "priority_boost" not in token:
                token["priority_boost"] = 0
            token["priority_boost"] += 1
    
    return tokens

# ==============================================
# 五、双曲空间推理引擎（扩展版）
# 支持括号和多个运算符
# ==============================================
class ExtendedPoincareReasoningEngine:
    def __init__(self):
        self.concepts = ATOM_CONCEPTS
        self.rules = RULE_SET
        self.epsilon = 0.1
        
    def get_numeric_embedding(self, num_str: str) -> Tuple[float, float]:
        """
        扩展功能：自动生成任意整数的双曲嵌入点
        保持数字越大越靠近边界的几何规律
        """
        num = int(num_str)
        
        # 数字越大，模长越接近1（靠近圆盘边界）
        base_norm = 0.85 + min(0.14, num * 0.01)
        base_norm = min(base_norm, 0.99)
        
        # 固定角度分布，保持有序
        angle = 0.1 + (num % 100) * 0.01
        
        x = base_norm * math.cos(angle)
        y = base_norm * math.sin(angle)
        
        return (x, y)
    
    def get_concept_embedding(self, symbol: str) -> Tuple[float, float]:
        """
        获取概念的双曲嵌入
        """
        if symbol.isdigit():
            emb = self.get_numeric_embedding(symbol)
        elif symbol in self.concepts:
            emb = self.concepts[symbol]
        else:
            raise ValueError(f"未知概念：{symbol}")
        
        assert is_in_disk(emb), f"概念点必须在庞加莱圆盘内：{symbol}"
        return emb
    
    def tokenize_expression(self, expr: str) -> List[Dict]:
        """
        分词和标记化
        返回token列表，每个token包含类型、值、嵌入等信息
        """
        # 移除空格
        expr = expr.replace(" ", "")
        
        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]
            
            if char.isdigit():
                # 读取整个数字
                j = i
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                num_str = expr[i:j]
                
                tokens.append({
                    "type": "number",
                    "value": num_str,
                    "embedding": self.get_concept_embedding(num_str),
                    "in_parentheses": False
                })
                i = j
                
            elif char in "+-*/()":
                tokens.append({
                    "type": "operator" if char in "+-*/" else "parenthesis",
                    "value": char,
                    "embedding": self.get_concept_embedding(char),
                    "in_parentheses": False
                })
                i += 1
            else:
                raise ValueError(f"非法字符：{char}")
        
        return tokens
    
    def process_parentheses(self, tokens: List[Dict]) -> List[Dict]:
        """
        处理括号：找到匹配的括号对，应用几何收缩
        """
        # 查找括号对
        stack = []
        paren_pairs = []
        
        for i, token in enumerate(tokens):
            if token["value"] == "(":
                stack.append(i)
            elif token["value"] == ")":
                if stack:
                    start = stack.pop()
                    paren_pairs.append((start, i))
        
        # 从最内层括号开始处理
        paren_pairs.sort(key=lambda p: p[1] - p[0], reverse=True)
        
        for start, end in paren_pairs:
            tokens = apply_parenthesis_contraction(tokens, start, end)
        
        return tokens
    
    def compute_operator_priority(self, token: Dict) -> float:
        """
        计算运算符的几何优先级
        优先级基于：是否在括号内、到圆心的距离
        """
        priority_score = 0.0
        
        # 基础优先级：到圆心的距离越小，优先级越高
        # 乘法运算符比加法更靠近圆心
        pos = token["embedding"]
        distance_to_origin = poincare_norm(pos)
        priority_score += 100.0 * (1.0 - distance_to_origin)
        
        # 括号内提升优先级
        if token.get("in_parentheses", False):
            priority_score += 1000.0
        
        # 运算符本身的优先级
        if token["type"] == "operator":
            for rule in self.rules.values():
                if rule["operator"] == token["value"]:
                    priority_score += rule["priority"] * 10.0
        
        return priority_score
    
    def find_next_operation(self, tokens: List[Dict]) -> Optional[Dict]:
        """
        找到下一个要执行的操作
        基于几何优先级
        """
        operators = []
        
        for i, token in enumerate(tokens):
            if token["type"] == "operator":
                # 找到左右操作数
                left_val = None
                right_val = None
                left_idx = None
                right_idx = None
                
                # 向左找左操作数
                for j in range(i-1, -1, -1):
                    if tokens[j]["type"] == "number" and tokens[j]["value"] is not None:
                        left_val = float(tokens[j]["value"])
                        left_idx = j
                        break
                
                # 向右找右操作数
                for j in range(i+1, len(tokens)):
                    if tokens[j]["type"] == "number" and tokens[j]["value"] is not None:
                        right_val = float(tokens[j]["value"])
                        right_idx = j
                        break
                
                if left_val is not None and right_val is not None:
                    # 计算优先级
                    priority = self.compute_operator_priority(token)
                    
                    operators.append({
                        "index": i,
                        "operator": token["value"],
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "left_val": left_val,
                        "right_val": right_val,
                        "priority": priority
                    })
        
        if not operators:
            return None
        
        # 选择优先级最高的运算符
        operators.sort(key=lambda op: op["priority"], reverse=True)
        return operators[0]
    
    def execute_operation(self, tokens: List[Dict], op_info: Dict) -> List[Dict]:
        """
        执行一个操作，更新token列表
        """
        i = op_info["index"]
        left_idx = op_info["left_idx"]
        right_idx = op_info["right_idx"]
        
        # 获取计算规则
        operator = op_info["operator"]
        rule = None
        for r in self.rules.values():
            if r["operator"] == operator:
                rule = r
                break
        
        if rule is None:
            raise ValueError(f"无可用规则：{operator}")
        
        # 执行计算
        result = rule["compute"](op_info["left_val"], op_info["right_val"])
        
        print(f"执行计算: {op_info['left_val']} {operator} {op_info['right_val']} = {result}")
        
        # 更新token列表
        # 将结果放在运算符位置
        tokens[i] = {
            "type": "number",
            "value": str(result),
            "embedding": mobius_addition(
                tokens[left_idx]["embedding"],
                tokens[right_idx]["embedding"]
            ),
            "in_parentheses": tokens[i].get("in_parentheses", False)
        }
        
        # 标记操作数已使用
        tokens[left_idx]["value"] = None
        tokens[right_idx]["value"] = None
        
        return tokens
    
    def reason(self, expr: str) -> float:
        """
        推理主函数
        支持括号和多个运算符
        """
        print("=" * 60)
        print(f"【推理引擎启动】输入查询：{expr}")
        print("-" * 60)
        
        # 1. 分词和标记化
        tokens = self.tokenize_expression(expr)
        
        print("[状态 Z0] 初始概念嵌入：")
        for i, token in enumerate(tokens):
            pos = token["embedding"]
            print(f"  {token['value']:3s} → ({pos[0]:.3f}, {pos[1]:.3f}) | 模长={poincare_norm(pos):.3f}")
        
        # 2. 处理括号（几何收缩）
        tokens = self.process_parentheses(tokens)
        
        print("\n[括号处理] 收缩变换后：")
        for i, token in enumerate(tokens):
            if token["type"] in ["number", "operator"]:
                pos = token["embedding"]
                in_paren = token.get("in_parentheses", False)
                print(f"  {token['value']:3s} → ({pos[0]:.3f}, {pos[1]:.3f}) | "
                      f"模长={poincare_norm(pos):.3f} | 括号内={in_paren}")
        
        # 3. 循环执行计算，直到没有运算符
        step = 1
        while True:
            # 找到下一个要执行的操作
            op_info = self.find_next_operation(tokens)
            if op_info is None:
                break
            
            print(f"\n[步骤 {step}] 选择操作：{tokens[op_info['left_idx']]['value']} "
                  f"{op_info['operator']} {tokens[op_info['right_idx']]['value']}")
            print(f"  优先级得分：{op_info['priority']:.2f}")
            
            # 执行操作
            tokens = self.execute_operation(tokens, op_info)
            step += 1
        
        # 4. 提取最终结果
        for token in tokens:
            if token["type"] == "number" and token["value"] is not None:
                result = float(token["value"])
                break
        else:
            result = 0.0
        
        print("-" * 60)
        print(f"【推理完成】步骤数：{step-1}")
        print(f"✅ 最终结果 = {result}")
        print("=" * 60 + "\n")
        
        return result

# ==============================================
# 主程序：运行推理引擎
# ==============================================
if __name__ == "__main__":
    # 初始化引擎
    engine = ExtendedPoincareReasoningEngine()
    
    # 测试表达式
    test_cases = [
        "3+(5 * 2)",     # 应该先计算5 * 2，然后3+10=13
        "(3+5)*2",     # 应该先计算3+5，然后8 * 2=16
        "10-2 * 3",      # 应该先计算2 * 3，然后10-6=4
        "12/4+3",      # 应该先计算12/4，然后3+3=6
        "2+3 * 4-5",     # 应该先计算3 * 4，然后2+12-5=9
    ]
    
    for expr in test_cases:
        result = engine.reason(expr)
        
        # 验证结果
        expected = eval(expr.replace("*", "*"))  # 使用Python的eval计算期望值
        
        if abs(result - expected) < 0.0001:
            print(f"✓ 验证通过：{expr} = {result} (期望: {expected})")
        else:
            print(f"✗ 验证失败：{expr} = {result} (期望: {expected})")
        print()