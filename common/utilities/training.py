from common.utilities.project import Project
import sys
from common.rl_agents.dummy_agent import DummyAgent
from common.safe_gym.safe_gym import SafeGym
import gym
import random
import math
import numpy as np
from collections import deque


def train(project, env, monitor = None):
    all_episode_rewards = []
    all_property_results = []
    best_average = -math.inf
    last_max_steps_states = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_actions = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_rewards = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_terminals = deque(maxlen=project.command_line_arguments['max_steps']*2)
    try:
        for episode in range(project.command_line_arguments['num_episodes']):
            state = env.reset()
            last_max_steps_states.append(state)
            episode_reward = 0
            while True:
                action = project.agent.select_action(state)
                next_state, reward, terminal, info = env.step(action)
                project.agent.store_experience(state, action, reward, next_state, terminal)
                # Collect last max_steps states, actions, and rewards
                last_max_steps_states.append(next_state)
                last_max_steps_actions.append(action)
                last_max_steps_rewards.append(reward)
                last_max_steps_terminals.append(terminal)
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
                reward_of_sliding_window = np.mean(all_episode_rewards[-project.command_line_arguments['sliding_window_size']:])
                project.log_avg_reward(reward_of_sliding_window, episode)
                # Log Property Result
                mdp_reward_result, model_size, _, _ = env.storm_bridge.model_checker.induced_markov_chain(project.agent, env, project.command_line_arguments['constant_definitions'], project.command_line_arguments['prop'])
                all_property_results.append(mdp_reward_result)
                project.log_property(all_property_results[-1], 'Property Result', episode)
                print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tProperty Result:", mdp_reward_result)
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
    if project.command_line_arguments['task']=='safe_training':
        pass

    return list(last_max_steps_states), list(last_max_steps_actions), list(last_max_steps_rewards), list(last_max_steps_terminals)


