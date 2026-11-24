"""Train MADDPG agents on the Tennis environment.

The score of an episode is the maximum over the two agents' scores. The environment is
solved when the average score over 100 consecutive episodes reaches 0.5.

Examples
--------
    # start (or resume) training
    python train.py

    # start over, ignoring an existing checkpoint
    python train.py --no-resume

    # run in the background and follow the log
    nohup python train.py > /dev/null 2>&1 &
    tail -f train.log

Ctrl+C (or SIGTERM) saves a checkpoint before exiting.
"""

import argparse
import logging
import os
import signal
import time
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from maddpg_agent import MADDPG


def parse_args():
    """Read the command-line options."""
    p = argparse.ArgumentParser(description='Train MADDPG on the Tennis environment.')
    p.add_argument('--env', default='Tennis_Linux/Tennis.x86_64', help='path to the Unity environment')
    p.add_argument('--episodes', type=int, default=10000, help='how many MORE episodes to run')
    p.add_argument('--max-t', type=int, default=1000, help='max steps per episode')
    p.add_argument('--sigma', type=float, default=0.2, help='OU noise: size of the random kicks')
    p.add_argument('--noise-start', type=float, default=1.0, help='noise scale at the start of training')
    p.add_argument('--noise-decay', type=float, default=0.999, help='noise scale is multiplied by this every episode')
    p.add_argument('--noise-min', type=float, default=0, help='the noise scale never goes below this')
    p.add_argument('--separate', action='store_true',
                   help='give each racket its own actor and critic (default: both share one)')
    p.add_argument('--warmup-episodes', type=int, default=500,
                   help='random-action episodes that only fill the replay buffer, before learning starts')
    p.add_argument('--seed', type=int, default=2, help='random seed')
    p.add_argument('--checkpoint', default='checkpoint.pth', help='full checkpoint file (save + resume)')
    p.add_argument('--no-resume', action='store_true', help='ignore an existing checkpoint and start fresh')
    p.add_argument('--save-every', type=int, default=50, help='save a checkpoint every N episodes')
    p.add_argument('--log', default='train.log', help='log file')
    p.add_argument('--plot', default='scores.png', help='where to save the score plot')
    p.add_argument('--graphics', action='store_true', help='show the Unity window (slower)')
    p.add_argument('--worker-id', type=int, default=0, help='change if another Unity env is already running')
    return p.parse_args()


def setup_logging(log_file):
    """Log to the terminal and to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,                    # replace the logging setup done by unityagents
    )
    return logging.getLogger('train')


def save_plot(scores, path):
    """Plot the episode scores and their 100-episode moving average."""
    if not scores:
        return
    rolling = [np.mean(scores[max(0, i - 99):i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores, alpha=0.4, label='Episode score')
    ax.plot(np.arange(1, len(scores) + 1), rolling, label='Average of last 100')
    ax.axhline(0.5, color='gray', linestyle='--', label='Target (0.5)')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score (max over the 2 agents)')
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def warm_up(env, brain_name, agent, num_agents, action_size, episodes, max_t, log):
    """Fill the replay buffer with random-action episodes before learning starts.

    Rewards are sparse at the start, so this gives the buffer some rewarded transitions.
    These episodes are not included in the score history.
    """
    hits = 0
    for _ in range(episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        for t in range(max_t):
            actions = np.random.uniform(-1, 1, (num_agents, action_size))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            agent.memory.add(states, actions, env_info.rewards, next_states, env_info.local_done)
            hits += sum(r > 0 for r in env_info.rewards)
            states = next_states
            if np.any(env_info.local_done):
                break
    log.info('Warm-up: %d random episodes, %d steps in the buffer, %d hits over the net',
             episodes, len(agent.memory), hits)


def stop_on_sigterm(signum, frame):
    """Handle SIGTERM like Ctrl+C, so a checkpoint is saved."""
    raise KeyboardInterrupt


def main():
    args = parse_args()
    log = setup_logging(args.log)
    signal.signal(signal.SIGTERM, stop_on_sigterm)

    # ---------------------------- environment ---------------------------- #
    env = UnityEnvironment(file_name=args.env, no_graphics=not args.graphics, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    log.info('Agents: %d, state size: %d, action size: %d', num_agents, state_size, action_size)

    # ------------------------------- agent ------------------------------- #
    agent = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents,
                   random_seed=args.seed, noise_sigma=args.sigma, shared=not args.separate)
    log.info('Networks: %s', 'separate per racket' if args.separate else 'one actor + critic shared by both rackets')
    agent.noise_scale = args.noise_start

    scores = []
    if not args.no_resume and os.path.exists(args.checkpoint):
        checkpoint = agent.load(args.checkpoint)
        scores = checkpoint['scores']
        log.info('Resumed from %s after %d episodes (last 100 avg: %.3f, noise scale: %.3f)',
                 args.checkpoint, len(scores), np.mean(scores[-100:]), agent.noise_scale)
    else:
        log.info('Starting fresh')

    scores_deque = deque(scores[-100:], maxlen=100)
    start_episode = len(scores) + 1

    # ------------------------------ training ----------------------------- #
    try:
        # The replay buffer is not part of the checkpoint, so the warm-up also runs on resume
        if args.warmup_episodes > 0:
            warm_up(env, brain_name, agent, num_agents, action_size, args.warmup_episodes, args.max_t, log)

        for i_episode in range(start_episode, start_episode + args.episodes):
            t0 = time.time()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            agent.reset()
            episode_scores = np.zeros(num_agents)
            for t in range(args.max_t):
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                episode_scores += rewards
                if np.any(dones):
                    break

            agent.noise_scale = max(args.noise_min, agent.noise_scale * args.noise_decay)

            # Episode score: maximum over the agents
            score = float(np.max(episode_scores))
            scores.append(score)
            scores_deque.append(score)
            log.info('Episode %d\tScore: %.3f\tAverage (last 100): %.3f\tSteps: %d\tNoise: %.3f\tTime: %.1fs',
                     i_episode, score, np.mean(scores_deque), t + 1, agent.noise_scale, time.time() - t0)

            if i_episode % args.save_every == 0:
                agent.save(args.checkpoint, scores=scores)
                save_plot(scores, args.plot)

            if len(scores_deque) == 100 and np.mean(scores_deque) >= 0.5:
                log.info('Environment solved at episode %d! Average of episodes %d-%d: %.3f',
                         i_episode, i_episode - 99, i_episode, np.mean(scores_deque))
                agent.save_weights('checkpoint')
                break
    except KeyboardInterrupt:
        log.info('Interrupted after %d episodes', len(scores))
    finally:
        agent.save(args.checkpoint, scores=scores)
        save_plot(scores, args.plot)
        log.info('Saved %s and %s', args.checkpoint, args.plot)
        env.close()


if __name__ == '__main__':
    main()
