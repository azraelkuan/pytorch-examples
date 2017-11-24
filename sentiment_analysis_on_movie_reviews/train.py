import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from load_data import CommentDataSet


class DnnModel(nn.ModuleList):

    def __init__(self, input_dim, output_dim, hidden_size):
        super(DnnModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)
        self.nonlinear = nn.Sigmoid()

    def forward(self, x):
        out = self.nonlinear(self.linear1(x))
        out = self.nonlinear(self.linear2(out))
        out = self.linear3(out)
        return out


def main():



if __name__ == '__main__':
    main()
