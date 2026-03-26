class Tokenizer:
    def __init__(self, vocab):
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_idx[w] for w in words if w in self.word_to_idx]