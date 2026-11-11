"""Train the MADDPG agents (maddpg_agent.py) on the Tennis environment from the terminal.

Same layout as the p2 train.py. Tennis-specific parts:
    - the score of an episode is the HIGHER of the two agents' scores, target 0.5
    - a random warm-up fills the replay buffer before learning (rewards are rare at first)
    - the exploration noise shrinks every episode (precise play is needed for long rallies)

Examples
--------
    # start (or resume) training, headless, with the default settings
    python train.py

    # 1000 more episodes
    python train.py --episodes 1000

    # start over (ignore the saved checkpoint)
    python train.py --no-resume

    # keep training after the terminal / VS Code closes, and follow the log
    nohup python train.py > /dev/null 2>&1 &
    tail -f train.log

Stopping with Ctrl+C (or `kill <pid>`) saves a checkpoint first, so no work is lost.
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
matplotlib.use('Agg')                  # draw plots to a file, no window needed
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
    p.add_argument('--wd-start-score', type=float, default=0.05,
                   help="start lowering the actors' weight decay once the 100-episode average reaches this")
    p.add_argument('--wd-decay', type=float, default=0,
                   help="actors' weight decay is multiplied by this every episode (while above --wd-start-score)")
    p.add_argument('--wd-min', type=float, default=1e-5, help="the actors' weight decay never goes below this")
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
    """Log to the terminal AND to a file, so progress can be checked with `tail -f`."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,                    # unityagents sets up logging on import; replace its setup
    )
    return logging.getLogger('train')


def save_plot(scores, path):
    """Save the score of every episode plus the 100-episode average to an image."""
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
    """
    Play random actions to fill the replay buffer before any learning.

    Rewards in Tennis are rare: early on, almost every episode is "ball drops, nothing else".
    If learning starts on such a buffer, the critic learns "everything is worth 0" and the
    actors can get stuck at a fixed action. Random play hits the ball now and then, so this
    gives the buffer some useful (rewarded) examples first.
    These episodes don't count towards the score history.
    """
    hits = 0
    for _ in range(episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        for t in range(max_t):
            actions = np.random.uniform(-1, 1, (num_agents, action_size))   # random, in Tennis's range
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            # straight into the buffer: no learning yet
            agent.memory.add(states, actions, env_info.rewards, next_states, env_info.local_done)
            hits += sum(r > 0 for r in env_info.rewards)
            states = next_states
            if np.any(env_info.local_done):
                break
    log.info('Warm-up: %d random episodes, %d steps in the buffer, %d hits over the net',
             episodes, len(agent.memory), hits)


def stop_on_sigterm(signum, frame):
    """Turn `kill <pid>` into a KeyboardInterrupt, so it saves a checkpoint like Ctrl+C."""
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
    num_agents = len(env_info.agents)                    # 2 rackets
    action_size = brain.vector_action_space_size         # 2: move towards/away from the net, jump
    state_size = env_info.vector_observations.shape[1]   # 24 per agent (8 values x 3 stacked frames)
    log.info('Agents: %d, state size: %d, action size: %d', num_agents, state_size, action_size)

    # ------------------------------- agent ------------------------------- #
    agent = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents,
                   random_seed=args.seed, noise_sigma=args.sigma, shared=not args.separate)
    log.info('Networks: %s', 'separate per racket' if args.separate else 'one actor + critic shared by both rackets')
    agent.noise_scale = args.noise_start

    scores = []                                          # score of every episode
    if not args.no_resume and os.path.exists(args.checkpoint):
        # resume: networks, optimizers, score history and noise scale
        checkpoint = agent.load(args.checkpoint)
        scores = checkpoint['scores']
        log.info('Resumed from %s after %d episodes (last 100 avg: %.3f, noise scale: %.3f, actor WD: %.1e)',
                 args.checkpoint, len(scores), np.mean(scores[-100:]), agent.noise_scale,
                 agent.actor_weight_decay)
    else:
        log.info('Starting fresh')

    # the rolling window continues from the resumed history
    scores_deque = deque(scores[-100:], maxlen=100)
    start_episode = len(scores) + 1

    # ------------------------------ training ----------------------------- #
    try:
        # The buffer is not saved in the checkpoint, so this also runs when resuming.
        # (Use --warmup-episodes 0 to skip it when resuming a run that already plays well.)
        if args.warmup_episodes > 0:
            warm_up(env, brain_name, agent, num_agents, action_size, args.warmup_episodes, args.max_t, log)

        for i_episode in range(start_episode, start_episode + args.episodes):
            t0 = time.time()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations          # shape (2, 24): one row per agent
            agent.reset()                                  # reset the exploration noise
            episode_scores = np.zeros(num_agents)          # one running score per agent
            for t in range(args.max_t):
                actions = agent.act(states)                # shape (2, 2): one action per agent
                env_info = env.step(actions)[brain_name]   # send both actions to the environment
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                agent.step(states, actions, rewards, next_states, dones)  # store, and learn
                states = next_states
                episode_scores += rewards
                if np.any(dones):                          # the episode ends for both agents together
                    break

            # explore a bit less every episode, down to --noise-min
            agent.noise_scale = max(args.noise_min, agent.noise_scale * args.noise_decay)

            # the project's score for an episode = the higher of the 2 agents' scores
            score = float(np.max(episode_scores))
            scores.append(score)
            scores_deque.append(score)

            # Lower the actors' weight decay, but ONLY while the agents play well enough
            # (100-episode average >= --wd-start-score). Strong decay protects the actors early,
            # while the critic still knows nothing; once they really hit the ball, the critic gives
            # real hints and the actors can be freed to make sharper moves. If the average drops
            # below the threshold again, the decay pauses at its current value.
            if np.mean(scores_deque) >= args.wd_start_score:
                agent.set_actor_weight_decay(max(args.wd_min, agent.actor_weight_decay * args.wd_decay))

            log.info('Episode %d\tScore: %.3f\tAverage (last 100): %.3f\tSteps: %d\tNoise: %.3f\tActor WD: %.1e\tTime: %.1fs',
                     i_episode, score, np.mean(scores_deque), t + 1, agent.noise_scale,
                     agent.actor_weight_decay, time.time() - t0)

            # save regularly, so a crash loses at most --save-every episodes
            if i_episode % args.save_every == 0:
                agent.save(args.checkpoint, scores=scores)
                save_plot(scores, args.plot)

            # solved = average of +0.5 over the last 100 episodes
            if len(scores_deque) == 100 and np.mean(scores_deque) >= 0.5:
                log.info('Environment solved at episode %d! Average of episodes %d-%d: %.3f',
                         i_episode, i_episode - 99, i_episode, np.mean(scores_deque))
                # plain weights per agent (checkpoint_actor_0.pth, ...), for test.py and the submission
                agent.save_weights('checkpoint')
                break
    except KeyboardInterrupt:
        log.info('Interrupted after %d episodes', len(scores))
    finally:
        # always save and clean up, however training ended
        agent.save(args.checkpoint, scores=scores)
        save_plot(scores, args.plot)
        log.info('Saved %s and %s', args.checkpoint, args.plot)
        env.close()


if __name__ == '__main__':
    main()
