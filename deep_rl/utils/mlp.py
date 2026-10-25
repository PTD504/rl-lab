import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64), type='policy'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        for layer in hidden_dims:
            self.layers.append(nn.Linear(input_dim, layer))
            self.layers.append(nn.ReLU())
            input_dim = layer

        self.layers.append(nn.Linear(input_dim, output_dim))

        if type == 'policy':
            self.layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)