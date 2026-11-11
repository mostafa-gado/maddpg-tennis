# MADDPG for Tennis: one DDPG agent (actor + critic) per racket, trained together
#
#   - each ACTOR only sees its own agent's observation (24 values) -> 2 actions.
#     That's all an agent has when playing.
#   - each CRITIC sees everything: both observations (the "full state", 48 values)
#     plus both agents' actions (2 x 2). Knowing what the other racket did makes the
#     other agent's learning look less like random changes in the environment.
#
# This file: creating the agents, acting, and the learning update (critic, then actor).
# See ddpg.py for one agent's networks and optimizers, networkforall.py for the layers.

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    # defaults = Tennis: 2 agents x 24 observations (8 values x 3 stacked frames),
    # full state = both observations joined (48), 2 actions per agent
    def __init__(self, obs_sizes=(24, 24), full_state_size=48, action_size=2,
                 discount_factor=0.99, tau=0.005):
        super(MADDPG, self).__init__()

        # networks for each agent:
        #   actor:  own observation (24) -> 256 -> 128 -> 2 actions (tanh, in [-1, 1])
        #   critic: full state + all actions (48 + 2 + 2 = 52) -> 256 -> 256 -> 1 Q-value
        critic_in = full_state_size + action_size * len(obs_sizes)
        self.maddpg_agent = [DDPGAgent(obs_size, 256, 128, action_size, critic_in, 256, 256)
                             for obs_size in obs_sizes]

        # discount_factor (gamma): how much future rewards count
        # tau: how fast the target networks follow the trained ones
        self.discount_factor = discount_factor
        self.tau = tau
        # number of learning updates so far (x-axis of the TensorBoard loss plots)
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    # obs_all_agents: one tensor per agent, shape (batch, 24)
    # returns one action tensor per agent, shape (batch, 2)
    # noise > 0 adds OU exploration noise (used in training; 0 for testing)
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    # same as act(), but with the slowly-changing target actors (used to compute the critic's target)
    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    # one learning step for ONE agent (agent_number 0 or 1): first its critic, then its actor
    # samples: a batch from the replay buffer; logger: TensorBoard writer for the losses
    def update(self, samples, agent_number, logger):
        """update the critic and actor of one agent"""

        # The buffer gives the batch as samples[i] = i-th transition, grouped as
        # obs[sample][agent]. Transposing flips that to obs[agent][sample]:
        # one tensor per agent, shape (batch, 24). Same for actions, rewards and dones.
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        # full state: transposing gives 48 tensors of shape (batch,); stack -> (48, batch).
        # The .t() calls below turn it into (batch, 48).
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # ---- critic update ----
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # (the loss used below is Huber, a less outlier-sensitive version of the squared error)
        # next actions of BOTH agents come from their target actors
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        # (1 - done): no future value after the ball drops / the episode ends
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        # actions both agents really took (from the buffer), joined: (batch, 4)
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ---- actor update ----
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # This agent's action comes from its CURRENT actor (so gradients flow into it).
        # The other agent's action also comes from its actor, but is detached:
        # only this agent's actor is trained in this update, which also saves computation.
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # (full state 48 + both actions 4 = 52, the same input the critic was trained on)
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)

        # get the policy gradient
        # maximize the critic's Q-value of the actor's actions = minimize -Q
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # log both losses to TensorBoard (agent0/losses, agent1/losses)
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    # move every target network a small step (tau) towards its trained network.
    # Called once after both agents were updated.
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




