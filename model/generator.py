import torch
import torch.nn as nn

class LanguageGenerator(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Linear(dim, 100000)

    def forward(self, x):
        return self.proj(x)

    def speak(self, geo_state, vocab, top_k=6):
        logits = self.forward(geo_state)
        
        # 只取真实存在的词
        valid_size = vocab.vocab_size()
        logits = logits[:valid_size]
        
        top_ids = torch.topk(logits, top_k).indices.tolist()

        # 只保留有效词，不补充任何符号！
        words = []
        for idx in top_ids:
            word = vocab.get_word(idx)
            if word and word != "<unk>":
                words.append(word)

        # 直接返回真实结果，不补pad！
        return " ".join(words)