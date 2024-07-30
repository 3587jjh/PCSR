import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, residual=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_list = hidden_list
        self.residual = residual
        if residual:
            self.convert = nn.Linear(in_dim, out_dim)

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        if self.residual:
            y = y + self.convert(x)
        return y