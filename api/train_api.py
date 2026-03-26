import torch
from utils.logger import LOG

class TrainAPI:
    def __init__(self, agent, cfg, device):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=cfg['train']['lr'])

    def train_on_sentence(self, sentence):
        if not self.agent.training:
            return None

        self.optimizer.zero_grad()
        geo_seq = self.agent.text_to_geo_sequence(sentence)
        geo_seq = [x.to(self.device) for x in geo_seq]

        pred_err = self.agent.brain.sequence_prediction_loss(geo_seq)
        K = torch.clamp(self.agent.poincare.K, 0.1, 20.0)
        penalty = 1.0 / torch.sqrt(K)
        loss = pred_err + 0.01 * penalty

        loss.backward()
        self.optimizer.step()

        depth = 2.0 * torch.log(torch.sqrt(K) + 1e-8)
        return {
            "pred_error": pred_err.item(),
            "K": K.item(),
            "depth": depth.item()
        }

    def train_on_multiple_sentences(self, sentences, epochs=30):
        LOG.info("🧠 OpenCatkin 多句子因果学习启动")
        for e in range(epochs):
            for sent in sentences:
                res = self.train_on_sentence(sent)
            thought = self.agent.think(sentences[-1])
            LOG.info(f"Epoch {e:02d} | thought={thought} | K={res['K']:.2f} | depth={res['depth']:.2f} | err={res['pred_error']:.3f}")