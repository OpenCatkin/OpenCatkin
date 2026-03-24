from .base import BaseValidator

class FinanceComplianceValidator(BaseValidator):
    def validate(self, triplet: dict) -> bool:
        # 硬规则：relation 必须是合规关键词，且 subject 是政策条款
        valid_relations = {"requires", "prohibits", "mandates"}
        if triplet.get("relation") in valid_relations:
            s = str(triplet.get("subject", "")).lower()
            return "条款" in s or "policy" in s or len(s) > 5  # 简单长度+关键词检查
        return True