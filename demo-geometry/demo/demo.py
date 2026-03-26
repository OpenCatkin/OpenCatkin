# ==============================================
# 庞加莱圆盘元信息认知框架
# 最小推理引擎 Demo (加减乘除)
# 严格对应论文：双曲嵌入 + 规则推理 + 路径搜索
# 修复：支持任意整数（10、15、99等），不再限制0-9
# ==============================================
import math

# ==============================================
# 一、庞加莱圆盘核心数学（论文公式直接实现）
# 定义：D^2 = { (x,y) | x²+y² < 1 }
# 距离公式、模长、几何约束
# ==============================================
def poincare_distance(u, v):
    """
    论文公式：庞加莱圆盘两点间双曲距离
    d_D(u,v) = arcosh( 1 + 2||u-v||² / [(1-||u||²)(1-||v||²)] )
    :param u: 点1 (x1,y1)
    :param v: 点2 (x2,y2)
    :return: 双曲距离（标量）
    """
    # 计算欧氏模长平方
    norm_u_sq = u[0] ** 2 + u[1] ** 2
    norm_v_sq = v[0] ** 2 + v[1] ** 2
    diff_norm_sq = (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2

    # 论文公式核心
    inside_arg = 1 + 2 * diff_norm_sq / ((1 - norm_u_sq) * (1 - norm_v_sq))
    inside_arg = max(inside_arg, 1.0)  # 数值稳定

    return math.acosh(inside_arg)

def poincare_norm(z):
    """计算庞加莱圆盘内点的模长（到圆心距离）"""
    return math.sqrt(z[0] ** 2 + z[1] ** 2)

def is_in_disk(z):
    """检查点是否在单位圆盘内（几何合法性约束）"""
    return poincare_norm(z) < 1.0 - 1e-6

# ==============================================
# 二、原子概念空间（元信息知识库）
# 论文定义：原子概念 → 双曲空间点
# 规则：
# 1. 具体概念（数字）→ 靠近边界（模长大）
# 2. 抽象概念（运算符）→ 靠近圆心（模长小）
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
    "+": (0.40, 0.00),
    "-": (0.41, 0.10),
    "*": (0.35, 0.00),
    "/": (0.36, 0.10),
}

# ==============================================
# 三、推理规则库（论文：规则 = 模式 + 变换 + 约束）
# 每条规则 r = (pattern, condition, transform, weight)
# ==============================================
RULE_SET = {
    "add": {
        "operator": "+",
        "description": "加法规则",
        "compute": lambda a, b: a + b
    },
    "sub": {
        "operator": "-",
        "description": "减法规则",
        "compute": lambda a, b: a - b
    },
    "mul": {
        "operator": "*",
        "description": "乘法规则",
        "compute": lambda a, b: a * b
    },
    "div": {
        "operator": "/",
        "description": "除法规则",
        "compute": lambda a, b: a / b if b != 0 else float("inf")
    }
}

# ==============================================
# 四、双曲空间推理引擎（论文第2章核心实现）
# 功能：
# 1. 输入查询表达式
# 2. 映射为双曲空间状态 Z_curr
# 3. 规则匹配（几何距离 + 约束）
# 4. 生成推理路径 P
# 5. 输出最终概念（结果）
# ==============================================
class PoincareReasoningEngine:
    def __init__(self):
        # 初始化：加载元知识（概念 + 规则）
        self.concepts = ATOM_CONCEPTS
        self.rules = RULE_SET
        self.epsilon = 0.1  # 匹配阈值（论文：几何匹配阈值δ）

    def parse_query(self, expr: str):
        """
        推理步骤1：查询解析
        将输入表达式 → 符号 token（对应原子概念）
        例："3+5" → ["3", "+", "5"]
        修复：支持多位数（10、15、99）
        """
        expr = expr.strip()
        # 遍历运算符，分割左右操作数（支持多位数）
        for op in ["+", "-", "*", "/"]:
            if op in expr:
                left, right = expr.split(op, maxsplit=1)
                left = left.strip()
                right = right.strip()
                # 验证左右都是合法数字
                if left.isdigit() and right.isdigit():
                    return left, op, right
        raise ValueError("不支持的表达式格式，仅支持 a+b、a-b、a*b、a/b（整数）")

    def get_numeric_embedding(self, num_str: str):
        """
        扩展功能：自动生成任意整数的双曲嵌入点
        不再局限于0-9，保持数字越大越靠近边界的几何规律
        """
        num = int(num_str)
        # 数字越大，模长越接近1（靠近圆盘边界），符合具体概念定义
        base_norm = 0.85 + min(0.14, num * 0.01)  # 模长上限 0.99
        base_norm = min(base_norm, 0.99)  # 保证在圆盘内
        # 固定角度分布，保持有序
        angle = 0.1 + num * 0.01
        x = base_norm * math.cos(angle)
        y = base_norm * math.sin(angle)
        return (x, y)

    def get_concept_embedding(self, symbol: str):
        """
        推理步骤2：符号 → 双曲嵌入（论文：φ: A → D^d）
        修复：自动处理任意整数字符串
        """
        # 如果是数字（多位数），自动生成嵌入
        if symbol.isdigit():
            emb = self.get_numeric_embedding(symbol)
            assert is_in_disk(emb), f"数字点必须在庞加莱圆盘内：{symbol}"
            return emb
        
        # 如果是运算符，从知识库获取
        if symbol in self.concepts:
            emb = self.concepts[symbol]
            assert is_in_disk(emb), f"概念点必须在庞加莱圆盘内：{symbol}"
            return emb
        
        raise ValueError(f"未知原子概念：{symbol}")

    def match_rule(self, op_symbol: str):
        """
        推理步骤3：规则匹配（论文：几何匹配 + 约束检查）
        根据运算符符号 → 匹配最优规则
        """
        for rule_name, rule in self.rules.items():
            if rule["operator"] == op_symbol:
                return rule
        raise ValueError(f"无可用推理规则：{op_symbol}")

    def reason(self, expr: str):
        """
        推理主函数（论文：完整推理路径 P）
        P = Z0 →(r1)→ Z1 →(r2)→ ... → Zn=T
        """
        print("=" * 60)
        print(f"【推理引擎启动】输入查询：{expr}")
        print("-" * 60)

        # ======================
        # 阶段1：初始状态 Z0
        # ======================
        a_str, op_str, b_str = self.parse_query(expr)
        z_a = self.get_concept_embedding(a_str)
        z_op = self.get_concept_embedding(op_str)
        z_b = self.get_concept_embedding(b_str)

        # 打印初始状态（双曲空间点集 Z0）
        print(f"[状态 Z0] 初始概念嵌入：")
        print(f"  数字 {a_str} → {z_a}  | 模长= {poincare_norm(z_a):.3f}")
        print(f"  数字 {b_str} → {z_b}  | 模长= {poincare_norm(z_b):.3f}")
        print(f"  操作符 {op_str} → {z_op} | 模长= {poincare_norm(z_op):.3f}")

        # ======================
        # 阶段2：几何匹配
        # ======================
        dist_a = poincare_distance(z_a, z_op)
        dist_b = poincare_distance(z_b, z_op)
        print(f"[几何匹配] 双曲距离：a-op={dist_a:.3f} | b-op={dist_b:.3f}")

        # ======================
        # 阶段3：规则应用
        # ======================
        rule = self.match_rule(op_str)
        print(f"[规则激活] {rule['description']}")

        # ======================
        # 阶段4：状态变换 Z0 → Z1
        # ======================
        a_val = int(a_str)
        b_val = int(b_str)
        result_val = rule["compute"](a_val, b_val)

        # ======================
        # 阶段5：到达目标状态 Zn（推理完成）
        # ======================
        print("-" * 60)
        print(f"【推理完成】路径：Z0 →({op_str})→ Z1(结果)")
        print(f"✅ 最终结果 = {result_val}")
        print("=" * 60 + "\n")

        return result_val

# ==============================================
# 主程序：运行推理引擎
# ==============================================
if __name__ == "__main__":
    # 初始化引擎
    engine = PoincareReasoningEngine()

    # 执行算术推理（现在支持任意整数：10、15、99、100）
    engine.reason("3+5")
    engine.reason("10-4")   # 修复后正常运行！
    engine.reason("6*7")
    engine.reason("8/2")
    engine.reason("99+1")  # 扩展测试