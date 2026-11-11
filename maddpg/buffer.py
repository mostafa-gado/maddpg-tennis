from collections import deque
import random
from utilities import transpose_list


# Replay buffer: stores past transitions; learning samples random batches from it.
# Random sampling breaks the correlation between consecutive steps, which makes training stabler.
# When full, the oldest transitions are dropped (deque with maxlen).
class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    # transition = (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones),
    # where every part is a list with one entry per parallel environment.
    # (The lab ran several environments; train.py runs one, so each list has length 1.)
    def push(self,transition):
        """push into the buffer"""

        # regroup per environment: one stored item = one environment's full transition
        input_to_buffer = transpose_list(transition)
    
        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        # from [transition1, transition2, ...] to [all obs, all obs_full, all actions, ...],
        # the format MADDPG.update() expects
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)



