from .base import BaseValidator

class DrugInteractionValidator(BaseValidator):
    def validate(self, triplet: dict) -> bool:
        # 简单硬规则示例：subject 和 object 必须是常见药物名（实际可扩展成数据库查）
        common_drugs = {"阿司匹林", "华法林", "布洛芬", "对乙酰氨基酚"}
        if triplet.get("relation") in ["interacts_with", "treats"]:
            s = triplet.get("subject", "")
            o = triplet.get("object", "")
            return s in common_drugs or o in common_drugs
        return True  # 其他关系放行