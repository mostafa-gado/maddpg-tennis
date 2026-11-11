"""Test the trained MADDPG agents on the Tennis environment.

Loads the trained actors (one per agent) and plays WITHOUT exploration noise,
so the scores show what the learned policies can really do.

Examples
--------
    # watch the trained agents for 1 episode
    python test.py --graphics --episodes 1

    # check the score over 100 episodes, headless (fast)
    python test.py --episodes 100

    # test the latest training checkpoint instead of the solved weights
    python test.py --checkpoint checkpoint.pth
"""

import argparse

import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg_agent import MADDPG


def parse_args():
    """Read the command-line options."""
    p = argparse.ArgumentParser(description='Test trained MADDPG agents on the Tennis environment.')
    p.add_argument('--env', default='Tennis_Linux/Tennis.x86_64', help='path to the Unity environment')
    p.add_argument('--actors', nargs='+', default=['checkpoint_actor_0.pth', 'checkpoint_actor_1.pth'],
                   help='trained actor weights, one file per agent (saved when the env is solved)')
    p.add_argument('--checkpoint', help='load a full training checkpoint instead (e.g. checkpoint.pth)')
    p.add_argument('--separate', action='store_true',
                   help='the actors were trained with train.py --separate (only needed with --actors)')
    p.add_argument('--episodes', type=int, default=5, help='how many test episodes to run')
    p.add_argument('--graphics', action='store_true', help='show the Unity window, at normal speed')
    p.add_argument('--worker-id', type=int, default=0, help='change if another Unity env is already running')
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------- environment ---------------------------- #
    env = UnityEnvironment(file_name=args.env, no_graphics=not args.graphics, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)                    # 2 rackets
    action_size = brain.vector_action_space_size         # 2 per agent
    state_size = env_info.vector_observations.shape[1]   # 24 per agent

    # ------------------------------ agents ------------------------------- #
    # Only the actors are needed to act; the critics are only used during training.
    # Shared or separate networks: a full checkpoint records it; for plain actor files use --separate.
    shared = not args.separate
    if args.checkpoint:
        shared = torch.load(args.checkpoint, map_location='cpu').get('shared', False)
    agent = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents,
                   random_seed=0, shared=shared)
    if args.checkpoint:
        agent.load(args.checkpoint)
        print('Loaded {}'.format(args.checkpoint))
    else:
        agent.load_actor_weights(args.actors)
        print('Loaded {}'.format(', '.join(args.actors)))

    # ------------------------------- test -------------------------------- #
    # train_mode=False runs the simulation at normal speed, so it can be watched
    train_mode = not args.graphics
    scores = []
    try:
        for i_episode in range(1, args.episodes + 1):
            env_info = env.reset(train_mode=train_mode)[brain_name]
            states = env_info.vector_observations          # shape (2, 24): one row per agent
            episode_scores = np.zeros(num_agents)          # one running score per agent
            while True:
                actions = agent.act(states, add_noise=False)   # no noise: the actors' own actions
                env_info = env.step(actions)[brain_name]
                states = env_info.vector_observations
                episode_scores += env_info.rewards
                if np.any(env_info.local_done):            # the episode ends for both agents together
                    break
            # same score as in training: the higher of the 2 agents' scores
            score = np.max(episode_scores)
            scores.append(score)
            print('Episode {}\tScore: {:.3f}\t(agent 0: {:.3f}, agent 1: {:.3f})'.format(
                i_episode, score, episode_scores[0], episode_scores[1]))
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        env.close()

    if scores:
        print('\nAverage score over {} episodes: {:.3f}  (target: 0.5)'.format(len(scores), np.mean(scores)))


if __name__ == '__main__':
    main()
