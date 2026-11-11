"""MADDPG agent for Tennis, written in the same style as the p2 DDPG code (ddpg_multi_agent.py).

How it differs from the p2 MultiAgent:
    p2 (Reacher):  20 arms SHARE one actor and one critic. Each arm's critic only
                   looks at that arm (its state, its action). The arms don't affect
                   each other, so that's enough.
    here (Tennis): the rackets DO affect each other (where the ball goes depends on both),
                   so the critic looks at BOTH observations and BOTH actions. That way
                   the other racket's behaviour isn't a mysterious, changing part of the
                   environment: the critic sees it directly. This is MADDPG.

Shared or separate networks (the `shared` option):
    shared=True (default): both rackets use the SAME actor and critic, like the 20 arms in p2.
        Tennis is symmetric (each agent sees the court from its own side), so a skill one
        racket learns is immediately the other's too, and every update learns from both
        sides' experience. Without this, one agent can learn to hit while the other never
        does (seen in training: 16 vs 3 hits in 30 test episodes), and rallies never start.
    shared=False: each racket has its own actor and critic (classic MADDPG).

The critic's input is always ordered "me first": [my observation, other's observation] and
[my action, other's action]. So the same critic can judge the game from either side.

Everything else is the same as p2: one shared replay buffer, OU noise, soft updates,
critic gradient clipping, and the same method names (act, step, reset, learn, save, load).
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
ACTOR_WEIGHT_DECAY = 0  # starting L2 weight decay of the actor (train.py lowers it once agents improve)
LEARN_EVERY = 1        # learn once every LEARN_EVERY environment steps...
LEARN_TIMES = 1         # ...and run LEARN_TIMES updates (per agent) each time

# CPU on purpose: the networks are small and every step does many tiny operations, so copying
# data to and from the GPU costs more than it saves (measured: ~12 ms vs ~20 ms per step).
device = torch.device("cpu")


class Agent():
    """The networks of ONE racket: actor + critic, each with a target copy and an optimizer."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        # Actor: sees only this agent's observation
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        # Weight decay keeps the actor's weights (and so its raw, pre-tanh outputs) small.
        # Without it, the actor follows the critic's tiny, meaningless early hints all the way
        # until tanh saturates (actions stuck at +-1), and then it can't learn any more.
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR,
                                          weight_decay=ACTOR_WEIGHT_DECAY)

        # Critic: sees the full state and all actions
        full_state_size = state_size * num_agents
        full_action_size = action_size * num_agents
        self.critic_local = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Start the target networks as exact copies of the local networks
        for local, target in [(self.actor_local, self.actor_target), (self.critic_local, self.critic_target)]:
            for target_param, local_param in zip(target.parameters(), local.parameters()):
                target_param.data.copy_(local_param.data)


class MADDPG():
    """Two Agents (one per racket) trained together, with one shared replay buffer."""

    def __init__(self, state_size, action_size, num_agents, random_seed, noise_sigma=0.2, shared=True):
        """Initialize a MADDPG object.

        Params
        ======
            state_size (int): dimension of ONE agent's observation (24)
            action_size (int): dimension of ONE agent's action (2)
            num_agents (int): number of rackets (2)
            random_seed (int): random seed
            noise_sigma (float): size of the exploration noise (smaller = less random actions)
            shared (bool): True = both rackets use the same actor and critic (see the top of the file)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.shared = shared
        self.seed = random.seed(random_seed)

        if shared:
            # ONE Agent, used by every racket: self.agents[0] and self.agents[1] are the same object
            self.agents = [Agent(state_size, action_size, num_agents, random_seed)] * num_agents
        else:
            # One Agent per racket. Different seeds, so the two don't start identical.
            self.agents = [Agent(state_size, action_size, num_agents, random_seed + i) for i in range(num_agents)]
        # the distinct Agents (1 if shared, 2 if not): used where each network must be handled once
        self.unique_agents = list({id(a): a for a in self.agents}.values())

        # Noise process: shape (num_agents, action_size) -> one independent process per racket
        self.noise = OUNoise((num_agents, action_size), random_seed, sigma=noise_sigma)
        # Multiplies the noise. train.py lowers it over time: explore a lot early, play precisely later.
        self.noise_scale = 1.0

        # Replay memory - shared by both rackets (every row holds BOTH agents' data for one step)
        self.memory = ReplayBuffer(num_agents, state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Counts environment steps, to know when to learn
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save one step of experience (both agents), then learn every LEARN_EVERY steps.

        Params
        ======
            states (array): shape (num_agents, state_size)
            actions (array): shape (num_agents, action_size)
            rewards (list): one reward per agent
            next_states (array): shape (num_agents, state_size)
            dones (list): one done flag per agent
        """
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn only every LEARN_EVERY steps, and only once the buffer has enough samples
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(LEARN_TIMES):
                # One update per racket's point of view, each from its own random batch.
                # When shared, both updates train the same networks: once from each side.
                for agent_i in range(self.num_agents):
                    self.learn(agent_i, self.memory.sample(), GAMMA)
                self.update_targets()

    def act(self, states, add_noise=True):
        """Return one action per agent. Each actor only sees its OWN observation.

        Params
        ======
            states (array): shape (num_agents, state_size)
            add_noise (bool): add exploration noise (set False when evaluating)
        """
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            agent.actor_local.eval()
            with torch.no_grad():
                # row i of states -> agent i's actor -> row i of actions
                actions[i] = agent.actor_local(states[i].unsqueeze(0)).cpu().data.numpy()[0]
            agent.actor_local.train()
        if add_noise:
            # Each row of the noise belongs to one racket
            actions += self.noise_scale * self.noise.sample()
        # The environment only accepts actions in [-1, 1]
        return np.clip(actions, -1, 1)

    def reset(self):
        """Reset the noise of every agent (call at the start of each episode)."""
        self.noise.reset()

    @property
    def actor_weight_decay(self):
        """The actors' current weight decay (the same for every agent)."""
        return self.agents[0].actor_optimizer.param_groups[0]['weight_decay']

    def set_actor_weight_decay(self, value):
        """Change the actors' weight decay during training.

        The optimizers' state_dicts include it, so save()/load() keep the current value.
        """
        for agent in self.unique_agents:
            for group in agent.actor_optimizer.param_groups:
                group['weight_decay'] = value

    def learn(self, agent_i, experiences, gamma):
        """Update the actor and critic of ONE agent (agent_i) from a batch of experience.

        Q_targets = r_i + γ * critic_target_i(full_next_state, all target actions)

        Shapes in the batch (B = BATCH_SIZE):
            states, next_states: (B, num_agents, state_size)
            actions:             (B, num_agents, action_size)
            rewards, dones:      (B, num_agents)
        """
        states, actions, rewards, next_states, dones = experiences
        agent = self.agents[agent_i]

        # "Me first" order: this agent, then the other(s). The critic always sees
        # [my obs, other's obs] and [my action, other's action], so it judges the game from
        # agent_i's side. That's what lets ONE shared critic work for both rackets.
        order = [agent_i] + [i for i in range(self.num_agents) if i != agent_i]
        full_states = torch.cat([states[:, i] for i in order], dim=1)            # (B, 48)
        full_next_states = torch.cat([next_states[:, i] for i in order], dim=1)  # (B, 48)
        full_actions = torch.cat([actions[:, i] for i in order], dim=1)          # (B, 4)
        # this agent's own reward and done flag, as columns
        r = rewards[:, agent_i].unsqueeze(1)
        d = dones[:, agent_i].unsqueeze(1)

        # ---------------------------- update critic ---------------------------- #
        # Next actions of every agent (me first), each from its own target actor and own observation
        with torch.no_grad():
            actions_next = torch.cat([self.agents[i].actor_target(next_states[:, i]) for i in order], dim=1)
            Q_targets_next = agent.critic_target(full_next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = r + (gamma * Q_targets_next * (1 - d))
        # Compute critic loss
        Q_expected = agent.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the critic's gradients so one bad batch can't make a huge update
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # This agent's action comes from its current actor (so gradients flow into it).
        # The other agent's action also comes from its current actor, but detached:
        # only this agent's side is trained here. (Same "me first" order as above.)
        actions_pred = torch.cat([
            self.agents[i].actor_local(states[:, i]) if i == agent_i
            else self.agents[i].actor_local(states[:, i]).detach()
            for i in order], dim=1)
        # Compute actor loss: the actor wants actions the critic scores highly
        actor_loss = -agent.critic_local(full_states, actions_pred).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self):
        """Soft-update the target networks of every agent (after all agents learned)."""
        # unique_agents: a shared network must be soft-updated only once per step, not once per racket
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
        """Save everything needed to continue training later.

        Saves every agent's local AND target networks plus both optimizers, and the
        current noise scale. The replay buffer is not saved (it would be huge).

        Params
        ======
            path (str): file to write, e.g. 'checkpoint.pth'
            extra: anything else to store, e.g. scores=[...]
        """
        # numpy numbers can't be read back by torch.load's safe mode -> store plain floats
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
            } for a in self.agents],      # when shared, both entries hold the same networks
            'noise_scale': float(self.noise_scale),
            'shared': self.shared,
        }
        checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load(self, path):
        """Load a checkpoint written by save() and return it.

        The returned dict also holds the extras passed to save(), e.g. checkpoint['scores'].
        """
        checkpoint = torch.load(path, map_location=device)
        # checkpoints from before the `shared` option had separate networks
        saved_shared = checkpoint.get('shared', False)
        if saved_shared != self.shared:
            raise ValueError('{} was trained with shared={}, but this agent has shared={}. '
                             'Use the same setting (train.py: --separate for shared=False).'
                             .format(path, saved_shared, self.shared))
        for a, saved in zip(self.agents, checkpoint['agents']):
            a.actor_local.load_state_dict(saved['actor_local'])
            a.actor_target.load_state_dict(saved['actor_target'])
            a.critic_local.load_state_dict(saved['critic_local'])
            a.critic_target.load_state_dict(saved['critic_target'])
            a.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            a.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.noise_scale = checkpoint['noise_scale']
        return checkpoint

    def save_weights(self, prefix='checkpoint'):
        """Save plain actor/critic weights per agent, e.g. checkpoint_actor_0.pth (for the submission)."""
        for i, a in enumerate(self.agents):
            torch.save(a.actor_local.state_dict(), '{}_actor_{}.pth'.format(prefix, i))
            torch.save(a.critic_local.state_dict(), '{}_critic_{}.pth'.format(prefix, i))

    def load_actor_weights(self, paths):
        """Load plain actor weights, one file per agent (e.g. for test.py)."""
        for a, path in zip(self.agents, paths):
            a.actor_local.load_state_dict(torch.load(path, map_location=device))


class OUNoise:
    """Ornstein-Uhlenbeck process, with one independent process per agent."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        Params
        ======
            size (tuple): shape of the noise, e.g. (num_agents, action_size)
            seed (int): random seed
            mu (float): long-run mean the noise is pulled towards
            theta (float): how strongly the noise is pulled back to mu
            sigma (float): size of the random kicks
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
        # Gaussian kicks (mean 0), drawn independently for every agent and every action
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer of experiences, stored in pre-allocated float32 arrays (like p2).

    One row = one environment step, holding the data of BOTH agents, because each
    critic needs to see both agents' observations and actions at the same moment.
    """

    def __init__(self, num_agents, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            num_agents (int): number of agents (2)
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
        self.size = 0    # how many rows hold real data

    def add(self, states, actions, rewards, next_states, dones):
        """Add one step (both agents). When full, the oldest row is overwritten."""
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
