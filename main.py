import torch
import yaml
from core.poincare import PoincareBall
from core.memory import EpisodicMemory
from model.brain import PredictiveBrain
from model.generator import LanguageGenerator
from utils.logger import LOG
from utils.vocab_db import VocabDB
from api.train_api import TrainAPI

class OpenCatkin(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ✅ 数据库词库（永不爆内存）
        self.vocab = VocabDB("vocab.db")

        self.poincare = PoincareBall(cfg['model']['dim'])
        self.brain = PredictiveBrain(cfg['model']['dim'])
        self.generator = LanguageGenerator(cfg['model']['dim'])
        self.memory = EpisodicMemory(cfg['model']['dim'])

        self.training = True

    # ✅ 句子 → 几何序列（用 VocabDB）
    def text_to_geo_sequence(self, text):
        ids = self.vocab.encode(text)
        seq = []
        for idx in ids:
            x = torch.tensor([idx], dtype=torch.float64)
            x = x.repeat(self.cfg['model']['dim'])
            geo = self.poincare(x)
            seq.append(geo)
        return seq

    def think(self, sentence):
        with torch.no_grad():
            geo_seq = self.text_to_geo_sequence(sentence)
            seq = torch.stack(geo_seq)
            context = self.brain.transformer(seq)
            mem = self.memory.get_context()
            final = context[-1] * 2 + mem
            return self.generator.speak(final, self.vocab)

    def train(self, mode=True):
        self.training = mode
        super().train(mode)

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    agent = OpenCatkin(cfg)
    train_api = TrainAPI(agent, cfg)

    LOG.info("OpenCatkin 数据库词库版 AGI 启动")

    sentences = [
        "I think therefore I am alive",
        "I am conscious because I think",
        "If I think I exist"
    ]

    agent.train()
    train_api.train_on_multiple_sentences(sentences, epochs=30)

    agent.eval()
    LOG.info("Final thought: " + agent.think("I think therefore I am alive"))