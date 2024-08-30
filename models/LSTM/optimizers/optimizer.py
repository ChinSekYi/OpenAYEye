import torch

class Optimizer():
    def __init__(self, config, model):
        self.config = config
        self.model = model
    
    def get_Optim(self):
        if self.config.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.LR, momentum=self.config.MOMENTUM, dampening=self.config.DAMPENING)
        elif self.config.OPTIMIZER == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.config.LR)
        return optimizer