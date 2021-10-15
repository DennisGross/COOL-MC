from common.utilities.project import Project
import sys
from common.rl_agents.dummy_agent import DummyAgent
import gym
import random
import math
import numpy as np

def train(project, env, monitor = None):
    all_episode_rewards = []
    all_property_results = []
    best_average = -math.inf
    try:
        for episode in range(project.command_line_arguments['num_episodes']):
            state = env.reset()
            episode_reward = 0
            while True:
                action = project.agent.select_action(state)
                next_state, reward, terminal, info = env.step(action)
                project.agent.store_experience(state, action, reward, next_state, terminal)
                project.agent.step_learn()
                state = next_state
                episode_reward+=reward
                if terminal:
                    break
            project.agent.episodic_learn()
            if episode % project.command_line_arguments['eval_interval']==0 and project.command_line_arguments['task']=='safe_training':
                # Log reward and property result (Safe Training)
                all_episode_rewards.append(episode_reward)
                project.log_reward(all_episode_rewards[-1], episode)
                all_property_results.append(random.random())
                project.log_property(all_property_results[-1], 'Probablity for done', episode)
            else:
                # Only log reward (OpenAI Gym Training)
                all_episode_rewards.append(episode_reward)
                project.log_reward(all_episode_rewards[-1], episode)
                reward_of_sliding_window = np.mean(all_episode_rewards[-project.command_line_arguments['sliding_window_size']:])
                project.log_avg_reward(reward_of_sliding_window, episode)
                sys.stdout.flush()
                print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window)
                if reward_of_sliding_window  > best_average:
                    best_reward_of_sliding_window = reward_of_sliding_window
                    project.save()
    except KeyboardInterrupt:
        pass
    
    # Log overall metrics
    project.log_best_reward(best_reward_of_sliding_window)


