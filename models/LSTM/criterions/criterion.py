import torch.nn as nn

class Criterion():
    def __init__(self, config):
        self.config = config

    def get_Criterion(self):
        if self.config.LOSS == "MSELoss":
            criterion = nn.MSELoss()
        elif self.config.LOSS == "L1Loss":
            criterion = nn.L1Loss()
        return criterion
