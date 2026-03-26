from .base import BaseValidator
import re

class PhysicsConstantValidator(BaseValidator):
    def validate(self, triplet: dict) -> bool:
        # 硬规则：object 必须是科学数字格式（支持科学计数法）
        obj = str(triplet.get("object", ""))
        if triplet.get("relation") in ["is", "equals", "value"]:
            return bool(re.match(r'^\d+(\.\d+)?(e[+-]?\d+)?$', obj.strip()))
        return True