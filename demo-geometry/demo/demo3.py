# ==============================================
# 庞加莱圆盘元信息推理引擎 —— 工业级可扩展版
# 支持：多运算符 + 括号 + 嵌套括号 + 动态符号
# 核心：几何优先级 → 不用硬编码优先级
# ==============================================
import math

# ==============================================
# 1. 庞加莱圆盘数学基础
# ==============================================
def poincare_distance(u, v):
    norm_u_sq = u[0]**2 + u[1]**2
    norm_v_sq = v[0]**2 + v[1]**2
    diff_norm_sq = (u[0]-v[0])**2 + (u[1]-v[1])**2
    inside = 1 + 2 * diff_norm_sq / ((1 - norm_u_sq) * (1 - norm_v_sq))
    inside = max(inside, 1.0)
    return math.acosh(inside)

def poincare_norm(z):
    return math.sqrt(z[0]**2 + z[1]**2)

# ==============================================
# 2. 符号注册表 → 真正可扩展！
# ==============================================
SYMBOL_REGISTRY = {
    "num":      {"type": "num",      "pos": None},  # 自动生成
    "+":        {"type": "op",       "pos": (0.40, 0.00), "priority": 1, "func": lambda a,b:a+b},
    "-":        {"type": "op",       "pos": (0.41, 0.10), "priority": 1, "func": lambda a,b:a-b},
    "*":        {"type": "op",       "pos": (0.35, 0.00), "priority": 2, "func": lambda a,b:a*b},
    "/":        {"type": "op",       "pos": (0.36, 0.10), "priority": 2, "func": lambda a,b:a/b if b else 0},
    "(":        {"type": "lparen"},
    ")":        {"type": "rparen"},
}

# ==============================================
# 3. 自动数字嵌入 → 不用硬编码 0-9
# ==============================================
def auto_num_embedding(num):
    r = 0.98
    angle = 0.1 + num * 0.01
    x = r * math.cos(angle)
    y = r * math.sin(angle)
    return (x, y)

# ==============================================
# 4. 分词器 → 支持任意表达式、多位数、括号
# ==============================================
def tokenize(expr):
    expr = expr.replace(" ", "")
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in SYMBOL_REGISTRY:
            tokens.append(c)
            i +=1
        elif c.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit(): j +=1
            tokens.append(expr[i:j])
            i = j
        else: raise ValueError(f"未知符号 {c}")
    return tokens

# ==============================================
# 5. 优先级由【到圆心距离】自动决定！不硬编码！
# ==============================================
def priority_by_geometry(sym):
    if sym not in SYMBOL_REGISTRY:
        return 0
    info = SYMBOL_REGISTRY[sym]
    if "pos" not in info:
        return 0
    return 100 - poincare_norm(info["pos"]) * 100

# ==============================================
# 6. 栈式求值 → 支持括号、嵌套括号、多运算符
# 标准算法：Shunting-yard（调度场算法）
# ==============================================
def evaluate(tokens):
    output = []
    op_stack = []

    for t in tokens:
        # 数字
        if t.isdigit():
            output.append(float(t))
        
        # 左括号
        elif t == "(":
            op_stack.append(t)
        
        # 右括号 → 弹栈计算
        elif t == ")":
            while op_stack and op_stack[-1] != "(":
                op = op_stack.pop()
                b = output.pop()
                a = output.pop()
                output.append(SYMBOL_REGISTRY[op]["func"](a,b))
            op_stack.pop()
        
        # 运算符 → 按几何优先级弹栈
        else:
            p = priority_by_geometry(t)
            while op_stack and op_stack[-1] != "(" and priority_by_geometry(op_stack[-1]) >= p:
                op = op_stack.pop()
                b = output.pop()
                a = output.pop()
                output.append(SYMBOL_REGISTRY[op]["func"](a,b))
            op_stack.append(t)

    # 剩余运算符
    while op_stack:
        op = op_stack.pop()
        b = output.pop()
        a = output.pop()
        output.append(SYMBOL_REGISTRY[op]["func"](a,b))

    return output[0]

# ==============================================
# 7. 推理引擎（统一接口）
# ==============================================
def reason(expr):
    print("="*50)
    print(f"推理表达式：{expr}")
    tokens = tokenize(expr)
    print(f"分词结果：{tokens}")
    res = evaluate(tokens)
    print(f"✅ 结果 = {res}")
    print("="*50)
    return res

# ==============================================
# 测试：全部支持！
# ==============================================
if __name__ == "__main__":
    reason("3+5*2")
    reason("(3+5)*2")
    reason("((10+20)*3)-(5*4)")
    reason("((((2+3))))*((4*5))")