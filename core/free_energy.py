import torch

class FreeEnergyMetabolism:
    def __init__(self, model, lr=0.005):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def minimize(self, free_energy):
        self.opt.zero_grad()
        free_energy.backward()
        self.opt.step()