# The neural network used for both the actor and the critic:
# 3 fully connected layers (input -> hidden_in -> hidden_out -> output) with ReLU in between.
#   actor  (actor=True):  observation -> 2 actions, squashed to [-1, 1] by tanh
#   critic (actor=False): full state + all actions -> 1 Q-value (no squashing)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# range for the starting weights of a hidden layer: +-1/sqrt(number of inputs),
# so the layer's outputs start at a reasonable size whatever its input count
def hidden_init(layer):
    fan_in = layer.weight.data.size()[1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        # (unused: input batch normalization, kept disabled from the lab code)
        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = F.relu #leaky_relu
        self.actor = actor
        self.reset_parameters()

    # DDPG-style starting weights. The last layer starts very small (+-0.001), so at first
    # the actor's actions are near 0 (not stuck at +-1 where tanh is flat and learns slowly)
    # and the critic's Q-values are near 0.
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # x: own observations, shape (batch, 24)
            h1 = self.nonlin(self.fc1(x))

            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            # 2 actions per agent (move towards/away from the net, jump), each in [-1, 1]:
            # exactly the range Tennis expects
            return torch.tanh(h3)

        else:
            # x: full state + all actions, shape (batch, 52)
            # critic network simply outputs a number (the Q-value, can be any size)
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3