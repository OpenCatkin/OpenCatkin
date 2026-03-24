class BaseValidator:
    def validate(self, triplet: dict) -> bool:
        """返回 True 表示通过校验，False 表示拦截"""
        raise NotImplementedError

class MathPrimeValidator(BaseValidator):
    """一个简单的具体验证器：专门拦截瞎编素数的幻觉"""
    def validate(self, triplet: dict) -> bool:
        if triplet["relation"] == "IS_PRIME":
            try:
                num = int(triplet["subject"])
                if num < 2: return False
                for i in range(2, int(num**0.5) + 1):
                    if num % i == 0: 
                        return False # 发现合数，物理拦截！
                return True
            except ValueError:
                return False
        # 如果不是自己负责检查的关系，默认放行交给其他验证器
        return True