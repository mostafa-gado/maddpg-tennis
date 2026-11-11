"""Actor and Critic networks for MADDPG on Tennis.

Same structure as the p2 (Reacher) networks. The one difference is what the critic sees:
    Actor:  ONE agent's own observation (24 values)          -> its 2 actions
    Critic: the FULL state (both observations, 2 x 24 = 48)
            + BOTH agents' actions (2 x 2 = 4)               -> 1 Q-value
The critic sees everything, because it is only used during training. When playing,
each agent only needs its actor, which only uses what that agent can see.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    # weight shape is (outputs, inputs), so size()[1] is the number of inputs (fan-in)
    fan_in = layer.weight.data.size()[1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of one agent's observation (24)
            action_size (int): Dimension of one agent's action (2)
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # tiny last layer: actions start near 0, where tanh is not flat
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh keeps both actions in [-1, 1], the range Tennis expects
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, full_state_size, full_action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            full_state_size (int): Both agents' observations joined (48)
            full_action_size (int): Both agents' actions joined (4)
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+full_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_state, full_action):
        """Build a critic (value) network that maps (full state, all actions) -> Q-value."""
        # same as p2: the state goes through one layer first, then the actions are added
        xs = F.relu(self.fcs1(full_state))
        x = torch.cat((xs, full_action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
