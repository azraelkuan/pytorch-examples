import torch.nn as nn
import numpy as np


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform(self.linear.weight, gain=np.sqrt(2))

    def forward(self, x):
        out = self.linear(x)
        return out
