import torch
import numpy as np

# swap the first two levels of a nested list: [[a1, a2], [b1, b2]] -> [[a1, b1], [a2, b2]]
def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

# same swap, turning each new group into one float tensor.
# e.g. a batch of observations obs[sample][agent] -> one (batch, 24) tensor per agent
def transpose_to_tensor(input_list):
    # np.array first: building a tensor from a list of numpy arrays is very slow in new PyTorch
    make_tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
