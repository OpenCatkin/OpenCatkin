import torch

class InferAPI:
    def __init__(self, agent):
        self.agent = agent

    def think(self, sentence):
        with torch.no_grad():
            geo = self.agent.text_to_geo(sentence)
            state = self.agent.self_manifold.fixed_point + geo
            return self.agent.generator.speak(state)