# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_feeder import MAX_LENGTH
from torch.nn import functional as F


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, use_cuda=True):
        super(EncoderRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, use_cuda=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden


class AttentionDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, drop_out_p=0.1, max_length=MAX_LENGTH, use_cuda=True):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_out_p = drop_out_p
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.dropout = nn.Dropout(self.drop_out_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layers)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden





