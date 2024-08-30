# import pandas as pd
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, config,
            act_layer = nn.Identity, norm_layer = nn.LayerNorm, device=torch.device('cpu')):
        super(LSTM, self).__init__()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        self.config = config
        self.device = device
        
        # self.norm_layer = norm_layer((7, 1))
        self.lstm = nn.LSTM(self.config.INPUT_SIZE, self.config.HIDDEN_SIZE, self.config.NUM_LAYERS, batch_first=True)
        self.drop = nn.Dropout(self.config.DROP)
        self.fc1 = nn.Linear(self.config.HIDDEN_SIZE, 1)
        self.act = act_layer()
        

    def forward(self, x):
        # x = self.norm_layer(x)
        h_start = torch.zeros(self.config.NUM_LAYERS, x.size(0), self.config.HIDDEN_SIZE, device=self.device, dtype = torch.float64)
        c_start = torch.zeros(self.config.NUM_LAYERS, x.size(0), self.config.HIDDEN_SIZE, device=self.device, dtype = torch.float64)


        out, (h_state, c_state)  = self.lstm(x, (h_start, c_start))
        x = self.drop(out[:, -1, :])
        x = self.fc1(x)
        x = self.act(x)
    
        return x