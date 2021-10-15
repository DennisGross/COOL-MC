from project import Project
import sys
sys.path.insert(0, '..')
from rl_agents.dummy_agent import DummyAgent
import gym
import random
import math
import numpy as np
def train(project, env, monitor = None):
    all_episode_rewards = []
    all_property_results = []
    best_average = -math.inf
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
        if episode % project.command_line_arguments['eval_episodes']==0 and project.command_line_arguments['task']=='safe_training':
            # Log reward and property result
            all_episode_rewards.append(episode_reward)
            project.log_reward(all_episode_rewards[-1], episode)
            all_property_results.append(random.random())
            project.log_property(all_property_results[-1], 'Probablity for done', episode)
        else:
            # Only log reward
            all_episode_rewards.append(episode_reward)
            project.log_reward(all_episode_rewards[-1], episode)
            reward_of_sliding_window = np.mean(all_episode_rewards[-project.command_line_arguments['sliding_window_size']:])
            if reward_of_sliding_window  > best_average:
                best_reward_of_sliding_window = reward_of_sliding_window
                project.save()
    
    # Log overall metrics
    project.log_best_reward(best_reward_of_sliding_window)


'''
if __name__ == '__main__':
    command_line_arguments = {'project_dir':'projects', 'sliding_window_size':3, 'always_action':0, 'num_episodes':40, 'eval_episodes':5 ,'project_name':'CartPole-v0', 'env':'CartPole-v0', 'task':'training', 'architecture':'dummy_agent', 'parent_run_id':'1dad1b8801a34c608bd0d2bed740e98a'}
    m_project = Project(command_line_arguments, (2,3), 1)
    env = gym.make(command_line_arguments['env'])
    #print(env.reset())

    train(m_project, env)
'''