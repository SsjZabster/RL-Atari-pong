import random
import numpy as np
import gym
import matplotlib.pyplot as plt

import time
import os
import itertools

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

def graph_rewards(episode_rewards):
    x = range(len(episode_rewards))
    plt.plot(x,episode_rewards)
    #plt.title("Plot of Rewards for Episode #")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("Plot.png")
    plt.show()

def generate_policy_animation(env, agent):
    """
    Follows (deterministic) greedy policy
    with respect to the given q-value estimator
    and saves animation using openAI gym's Monitor
    wrapper. Monitor will throw an error if monitor
    files already exist in save_dir so use unique
    save_dir for each call.
    """
    path = "playthrough"
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        env = gym.wrappers.Monitor(
            env, path, video_callable=lambda episode_id: True, force=True)
    except gym.error.Error as e:
        print(e)

    # Reset the environment
    state = env.reset()
    for t in itertools.count():
        time.sleep(0.01)  # Slow down animation
        env.render()  # Animate
        action = agent.act(np.array(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            print('Solved in {} steps'.format(t))
            break

if __name__ == '__main__':

    params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10
    }

    np.random.seed(params["seed"])
    random.seed(params["seed"])

    assert "NoFrameskip" in params["env"], "Require environment with no frameskip"
    env = gym.make(params["env"])
    env.seed(params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    #Warp the iamge frame into the required right w, h & channels
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=params["use-double-dqn"],
        lr=params["learning-rate"],
        batch_size=params["batch-size"],
        gamma=params["discount-factor"]
    )

    eps_timesteps = params["eps-fraction"] * float(params["num-steps"])
    episode_rewards = [0.0]

    state = env.reset()
    for t in range(params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = params["eps-start"] + fraction * (params["eps-end"] - params["eps-start"])
        sample = random.random()
        # TODO
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if t > params["learning-starts"] and t % params["learning-freq"] == 0:
            agent.optimise_td_loss()

        if t > params["learning-starts"] and t % params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and params["print-freq"] is not None and len(episode_rewards) % params["print-freq"] == 0:
            mean_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    generate_policy_animation(env,agent)
    graph_rewards(episode_rewards)
