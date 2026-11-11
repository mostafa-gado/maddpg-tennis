# individual network settings for each actor + critic pair
# see networkforall for details
#
# One DDPG agent = one Tennis racket. It has 4 networks:
#   actor, critic                 -> the ones being trained
#   target_actor, target_critic   -> slowly-following copies, used to compute stable
#                                    learning targets (updated with soft_update in maddpg.py)
# plus its own OU noise for exploration.

from networkforall import Network
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    # in_actor: observation size (24), out_actor: number of actions (2)
    # in_critic: full state + all actions (52)
    # hidden_*: neurons in the 2 hidden layers of each network
    # lr_actor < lr_critic: the critic learns faster, so the actor follows estimates that are
    # already reasonable instead of chasing a critic that is still wrong
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        # exploration noise, one value per action; its size is scaled by `noise` in act()
        self.noise = OUNoise(out_actor, scale=1.0 )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # weight_decay: a small penalty on large critic weights (L2 regularization)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)


    # action = actor's output + noise. The noise is added after tanh, so actions can go past
    # +-1; train.py clips them before sending them to Unity.
    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action

    # same with the target actor (maddpg.py calls it with noise=0)
    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
