import math
import gym
from gym import spaces
from gym.spaces import Discrete
import numpy as np
from common.safe_gym.storm_bridge import StormBridge
from common.safe_gym.action_mapper import ActionMapper

class SafeGym(gym.Env):

    def __init__(self, prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag, permissive_input, disabled_features=''):
        self.storm_bridge = StormBridge(prism_file_path, constant_definitions, wrong_action_penalty, reward_flag, disabled_features, permissive_input)
        self.action_mapper = ActionMapper.collect_actions(self.storm_bridge)
        self.steps = 0
        self.max_steps = max_steps
        self.state = self.reset()
        # Observation Space
        self.observation_space = spaces.Box(np.array([-math.inf]*self.state.shape[0]), np.array([math.inf]*self.state.shape[0]))
        # Action Space
        self.action_space = Discrete(len(self.action_mapper.actions))

    def step(self, action_index):
        action_name = self.action_mapper.action_index_to_action_name(action_index)
        n_state, reward, self.done = self.storm_bridge.step(action_name)
        self.steps+=1
        if self.steps >= self.max_steps:
            #print("DONE")
            self.done = True
        self.state = n_state
        return self.state, reward, self.done, {}
    
    def reset(self):
        self.steps = 0
        self.state = self.storm_bridge.reset()
        self.done = False
        return self.state
