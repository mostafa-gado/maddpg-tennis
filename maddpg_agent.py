"""MADDPG agent for the Tennis environment.

Each actor acts on its own agent's observation. The critic is centralized: it receives
both agents' observations and actions, ordered from the learning agent's point of view
(own observation and action first).

With shared=True (default), both agents use the same actor and critic. The environment
is symmetric, so the agents can share one policy and learn from each other's experience.
With shared=False, each agent has its own actor and critic.
"""

import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay of the critic
LEARN_EVERY = 1        # learn every LEARN_EVERY environment steps
LEARN_TIMES = 1         # number of updates per agent each time

# The networks are small, so CPU is faster than GPU here.
device = torch.device("cpu")


class Agent():
    """Actor and critic of one agent, each with a target network and an optimizer."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        # Actor: own observation -> own action
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic: all observations and all actions -> Q-value
        full_state_size = state_size * num_agents
        full_action_size = action_size * num_agents
        self.critic_local = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initialize the target networks as copies of the local networks
        for local, target in [(self.actor_local, self.actor_target), (self.critic_local, self.critic_target)]:
            for target_param, local_param in zip(target.parameters(), local.parameters()):
                target_param.data.copy_(local_param.data)


class MADDPG():
    """Multi-agent DDPG with a shared replay buffer."""

    def __init__(self, state_size, action_size, num_agents, random_seed, noise_sigma=0.2, shared=True):
        """Initialize a MADDPG object.

        Params
        ======
            state_size (int): dimension of one agent's observation
            action_size (int): dimension of one agent's action
            num_agents (int): number of agents
            random_seed (int): random seed
            noise_sigma (float): sigma of the OU exploration noise
            shared (bool): if True, all agents share one actor and one critic
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.shared = shared
        self.seed = random.seed(random_seed)

        if shared:
            self.agents = [Agent(state_size, action_size, num_agents, random_seed)] * num_agents
        else:
            self.agents = [Agent(state_size, action_size, num_agents, random_seed + i) for i in range(num_agents)]
        # Distinct networks (one if shared), so each is updated and saved only once
        self.unique_agents = list({id(a): a for a in self.agents}.values())

        # One OU process per agent
        self.noise = OUNoise((num_agents, action_size), random_seed, sigma=noise_sigma)
        # Scale applied to the noise; decayed by the training loop
        self.noise_scale = 1.0

        # Each row stores one environment step for all agents
        self.memory = ReplayBuffer(num_agents, state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save one step of experience and learn every LEARN_EVERY steps.

        Params
        ======
            states (array): shape (num_agents, state_size)
            actions (array): shape (num_agents, action_size)
            rewards (list): one reward per agent
            next_states (array): shape (num_agents, state_size)
            dones (list): one done flag per agent
        """
        self.memory.add(states, actions, rewards, next_states, dones)

        self.t_step = (self.t_step + 1) % LEARN_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(LEARN_TIMES):
                # One update from each agent's point of view
                for agent_i in range(self.num_agents):
                    self.learn(agent_i, self.memory.sample(), GAMMA)
                self.update_targets()

    def act(self, states, add_noise=True):
        """Return one action per agent, each from its own observation.

        Params
        ======
            states (array): shape (num_agents, state_size)
            add_noise (bool): add exploration noise (False for evaluation)
        """
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            agent.actor_local.eval()
            with torch.no_grad():
                actions[i] = agent.actor_local(states[i].unsqueeze(0)).cpu().data.numpy()[0]
            agent.actor_local.train()
        if add_noise:
            actions += self.noise_scale * self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        """Reset the noise process (at the start of each episode)."""
        self.noise.reset()

    def learn(self, agent_i, experiences, gamma):
        """Update the actor and critic used by agent_i.

        Q_targets = r_i + γ * critic_target(next_states, target_actions)

        Batch shapes (B = BATCH_SIZE):
            states, next_states: (B, num_agents, state_size)
            actions:             (B, num_agents, action_size)
            rewards, dones:      (B, num_agents)
        """
        states, actions, rewards, next_states, dones = experiences
        agent = self.agents[agent_i]

        # Critic inputs are ordered with agent_i first, then the other agents
        order = [agent_i] + [i for i in range(self.num_agents) if i != agent_i]
        full_states = torch.cat([states[:, i] for i in order], dim=1)
        full_next_states = torch.cat([next_states[:, i] for i in order], dim=1)
        full_actions = torch.cat([actions[:, i] for i in order], dim=1)
        r = rewards[:, agent_i].unsqueeze(1)
        d = dones[:, agent_i].unsqueeze(1)

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            actions_next = torch.cat([self.agents[i].actor_target(next_states[:, i]) for i in order], dim=1)
            Q_targets_next = agent.critic_target(full_next_states, actions_next)
            Q_targets = r + (gamma * Q_targets_next * (1 - d))
        Q_expected = agent.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Only agent_i's action carries gradients; the other actions are detached
        actions_pred = torch.cat([
            self.agents[i].actor_local(states[:, i]) if i == agent_i
            else self.agents[i].actor_local(states[:, i]).detach()
            for i in order], dim=1)
        actor_loss = -agent.critic_local(full_states, actions_pred).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self):
        """Soft-update the target networks of every distinct agent."""
        for agent in self.unique_agents:
            self.soft_update(agent.critic_local, agent.critic_target, TAU)
            self.soft_update(agent.actor_local, agent.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path, **extra):
        """Save a training checkpoint: networks, optimizers and noise scale.

        Params
        ======
            path (str): output file
            extra: additional entries to store, e.g. scores=[...]
        """
        # Store plain floats so the file can be loaded with torch.load's safe mode
        if 'scores' in extra:
            extra['scores'] = [float(s) for s in extra['scores']]
        checkpoint = {
            'agents': [{
                'actor_local': a.actor_local.state_dict(),
                'actor_target': a.actor_target.state_dict(),
                'critic_local': a.critic_local.state_dict(),
                'critic_target': a.critic_target.state_dict(),
                'actor_optimizer': a.actor_optimizer.state_dict(),
                'critic_optimizer': a.critic_optimizer.state_dict(),
            } for a in self.unique_agents],
            'noise_scale': float(self.noise_scale),
            'shared': self.shared,
        }
        checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load(self, path):
        """Load a checkpoint written by save() and return it."""
        checkpoint = torch.load(path, map_location=device)
        saved_shared = checkpoint.get('shared', False)
        if saved_shared != self.shared:
            raise ValueError('{} was trained with shared={}, but this agent has shared={}. '
                             'Use the same setting (train.py: --separate for shared=False).'
                             .format(path, saved_shared, self.shared))
        for a, saved in zip(self.unique_agents, checkpoint['agents']):
            a.actor_local.load_state_dict(saved['actor_local'])
            a.actor_target.load_state_dict(saved['actor_target'])
            a.critic_local.load_state_dict(saved['critic_local'])
            a.critic_target.load_state_dict(saved['critic_target'])
            a.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            a.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.noise_scale = checkpoint['noise_scale']
        return checkpoint

    def weight_files(self, prefix='checkpoint', kind='actor'):
        """File names of the weight files: one if shared, one per agent otherwise."""
        if self.shared:
            return ['{}_{}.pth'.format(prefix, kind)]
        return ['{}_{}_{}.pth'.format(prefix, kind, i) for i in range(len(self.unique_agents))]

    def save_weights(self, prefix='checkpoint'):
        """Save the actor and critic weights."""
        for a, actor_file, critic_file in zip(self.unique_agents,
                                              self.weight_files(prefix, 'actor'),
                                              self.weight_files(prefix, 'critic')):
            torch.save(a.actor_local.state_dict(), actor_file)
            torch.save(a.critic_local.state_dict(), critic_file)

    def load_actor_weights(self, paths):
        """Load actor weights, one file per distinct network."""
        if len(paths) != len(self.unique_agents):
            raise ValueError('Expected {} actor file(s) for shared={}, got {}: {}'
                             .format(len(self.unique_agents), self.shared, len(paths), paths))
        for a, path in zip(self.unique_agents, paths):
            a.actor_local.load_state_dict(torch.load(path, map_location=device))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        Params
        ======
            size (tuple): shape of the noise, e.g. (num_agents, action_size)
            seed (int): random seed
            mu (float): long-run mean
            theta (float): mean reversion rate
            sigma (float): volatility
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size replay buffer stored in pre-allocated float32 arrays.

    Each row holds one environment step for all agents.
    """

    def __init__(self, num_agents, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of one agent's observation
            action_size (int): dimension of one agent's action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((buffer_size, num_agents, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, num_agents, state_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.pos = 0     # next row to write
        self.size = 0    # number of stored rows

    def add(self, states, actions, rewards, next_states, dones):
        """Add one step; the oldest row is overwritten when the buffer is full."""
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.next_states[self.pos] = next_states
        self.dones[self.pos] = dones
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idx = self.rng.integers(0, self.size, size=self.batch_size)

        states = torch.from_numpy(self.states[idx]).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)
        next_states = torch.from_numpy(self.next_states[idx]).to(device)
        dones = torch.from_numpy(self.dones[idx]).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
