import networkx as nx

class GraphStorage:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        print("🧬 [存储骨架] GraphStorage 初始化完成（MultiDiGraph）")

    def upsert_edge(self, s, r, o):
        """固化一条三元组为图谱边"""
        self.graph.add_edge(s, o, relation=r)
        print(f"💎 [固化] 写入真理: ({s}) --[{r}]--> ({o})")

    def get_all_nodes(self):
        """返回当前图谱中所有节点（用于 Demo 打印）"""
        nodes = list(self.graph.nodes())
        print(f"📍 [查询] 当前图谱共有 {len(nodes)} 个节点")
        return nodes

    def save_to_file(self, filename: str = "knowledge_graph.gml"):
        """将图谱保存为 GML 格式（可被 Gephi / NetworkX 直接打开）"""
        nx.write_gml(self.graph, filename)
        print(f"💾 [固化] 图谱已永久保存 → {filename}")

    # 可选扩展方法（未来更方便调试）
    def get_all_edges(self):
        """返回所有边（带关系标签）"""
        return [(u, v, d['relation']) for u, v, d in self.graph.edges(data=True)]

    def print_summary(self):
        """打印图谱概要"""
        print(f"📊 图谱摘要 → 节点: {self.graph.number_of_nodes()} | 边: {self.graph.number_of_edges()}")