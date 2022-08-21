from matplotlib.pyplot import fill
from common.utilities.project import Project
import sys
from common.rl_agents.dummy_agent import DummyAgent
from common.safe_gym.safe_gym import SafeGym
from common.utilities.front_end_printer import *
from common.adversarial_attacks.adversarial_attack_builder import *
from common.utilities.state_observer_reward import StateObserverReward
import gym
import random
import math
import numpy as np
import torch
from collections import deque
import gc


def train(project, env, prop_type=''):
    try:
        m_state_observer_reward = StateObserverReward(env.storm_bridge.state_mapper.mapper[project.command_line_arguments['feature_observer'].split('=')[0]], project.command_line_arguments['feature_observer'].split('=')[1])
    except:
        pass
    all_episode_rewards = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    all_property_results = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    best_reward_of_sliding_window = -math.inf
    last_max_steps_states = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_actions = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_rewards = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_terminals = deque(maxlen=project.command_line_arguments['max_steps']*2)
    if project.command_line_arguments['task'] == 'safe_training':
        adv_builder = AdversarialAttackBuilder()
        adversarial_attack = adv_builder.build_adversarial_attack(env.storm_bridge.state_mapper, project.command_line_arguments['attack_config'])
    try:
        for episode in range(project.command_line_arguments['num_episodes']):
            state = env.reset()
            try:
                m_state_observer_reward.reset()
            except:
                pass
            last_max_steps_states.append(state)
            episode_reward = 0
            step_counter = 0
            while True:
                if state.__class__.__name__ == 'int':
                    state = [state]
                #print(project.command_line_arguments['deploy'])
                # Adversarial Attack here
                if project.command_line_arguments['deploy'] == True and project.command_line_arguments['attack_config'] != '':
                    state = adversarial_attack.attack(project.agent, state)
                action = project.agent.select_action(state, project.command_line_arguments['deploy'])
                step_counter +=1
                next_state, reward, terminal, info = env.step(action)
                try:
                    m_state_observer_reward.observe(next_state, reward)
                except:
                    pass
                if next_state.__class__.__name__ == 'int':
                    next_state = [next_state]
                if project.command_line_arguments['deploy']==False:
                    project.agent.store_experience(state, action, reward, next_state, terminal)
                    project.agent.step_learn()
                # Collect last max_steps states, actions, and rewards
                last_max_steps_states.append(next_state)
                last_max_steps_actions.append(action)
                last_max_steps_rewards.append(reward)
                last_max_steps_terminals.append(terminal)
                state = next_state
                episode_reward+=reward
                if terminal:
                    break
            if project.command_line_arguments['deploy']==False:
                project.agent.episodic_learn()
            if episode % project.command_line_arguments['eval_interval']==0 and project.command_line_arguments['task']=='safe_training':
                # Log reward and property result (Safe Training)
                all_episode_rewards.append(episode_reward)
                project.mlflow_bridge.log_reward(all_episode_rewards[-1], episode)
                reward_of_sliding_window = np.mean(list(all_episode_rewards))
                project.mlflow_bridge.log_avg_reward(reward_of_sliding_window, episode)
                # Log Property Result
                if prop_type != 'reward':
                    mdp_reward_result, model_size = env.storm_bridge.model_checker.induced_markov_chain(project.agent, env, project.command_line_arguments['constant_definitions'], project.command_line_arguments['prop'])
                    all_property_results.append(mdp_reward_result)

                    if (all_property_results[-1] == min(all_property_results) and prop_type == "min_prop") or (all_property_results[-1] == max(all_property_results) and prop_type == "max_prop"):
                        if project.command_line_arguments['deploy']==False:
                            project.save()
    
                    project.mlflow_bridge.log_property(all_property_results[-1], 'Property Result', episode)
                else:
                    mdp_reward_result = None
                print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", mdp_reward_result)
            else:
                # Only log reward
                all_episode_rewards.append(episode_reward)
                project.mlflow_bridge.log_reward(all_episode_rewards[-1], episode)
                reward_of_sliding_window = np.mean(list(all_episode_rewards))
                project.mlflow_bridge.log_avg_reward(reward_of_sliding_window, episode)
                if len(all_property_results) > 0:
                    print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", all_property_results[-1])
                else:
                    print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", None)
                

            if reward_of_sliding_window  > best_reward_of_sliding_window and len(all_episode_rewards)>=project.command_line_arguments['sliding_window_size']:
                best_reward_of_sliding_window = reward_of_sliding_window
                if prop_type=='reward' and project.command_line_arguments['deploy']==False:
                    project.save()

            gc.collect()
            #print(state)
            try:
                m_state_observer_reward.after_episode()
            except:
                pass
            

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        torch.cuda.empty_cache()
        # Log overall metrics
        if project.command_line_arguments['deploy']==0:
            project.mlflow_bridge.log_best_reward(best_reward_of_sliding_window)
            if project.command_line_arguments['task']=='safe_training':
                #TODO: Save Metrics
                pass
    try:
        return list(last_max_steps_states), list(last_max_steps_actions), list(last_max_steps_rewards), list(last_max_steps_terminals), all_episode_rewards[-1], m_state_observer_reward.rewards[-1]
    except:
        return list(last_max_steps_states), list(last_max_steps_actions), list(last_max_steps_rewards), list(last_max_steps_terminals), all_episode_rewards[-1], 0


