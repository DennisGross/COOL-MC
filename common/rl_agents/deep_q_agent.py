import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import Agent
import torch
import random
import gym
import numpy as np
import mlflow
import os
import shutil
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return torch.tensor(states).to(device), torch.tensor(actions).to(device), torch.tensor(rewards).to(device), torch.tensor(states_).to(device), torch.tensor(terminal).to(device)

class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, number_of_actions, lr, name='bla', chkpt_dir='asdg'):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc3 = nn.Linear(1024, number_of_actions)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        state = torch.tensor(state).float().to(device)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(state))
        actions = self.fc3(flat1)
        return actions

    def save_checkpoint(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name):
        self.load_state_dict(torch.load(file_name))


class DQNAgent(Agent):

    def __init__(self, state_dim, number_of_actions, epsilon=1, epsilon_dec=0.99999, epsilon_min=0.05, gamma=0.99, learning_rate=0.0001, replace=100, batch_size=32):
        self.number_of_actions = number_of_actions
        self.replay_buffer = ReplayBuffer(10000, state_dim)
        self.q_eval = DeepQNetwork(state_dim, number_of_actions, learning_rate)
        self.q_next = DeepQNetwork(state_dim, number_of_actions, learning_rate)
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = torch.tensor(gamma).to(device)
        self.replace = 100
        self.batch_size = batch_size
        self.exp_counter = 0
        self.learn_step_counter = 0

    def save(self):
        try:
            os.mkdir('tmp_model')
        except:
            pass
        self.q_eval.save_checkpoint('tmp_model/q_eval.chkpt')
        self.q_next.save_checkpoint('tmp_model/q_next.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path="model")
        shutil.rmtree('tmp_model')
        #print(mlflow.get_artifact_uri(artifact_path="model"))

    def load(self, model_root_folder_path):
        
        try:
            self.q_eval.load_checkpoint(os.path.join(model_root_folder_path,'q_eval.chkpt'))
            self.q_next.load_checkpoint(os.path.join(model_root_folder_path,'q_next.chkpt'))
        except:
            pass
    

    def store_experience(self, state, action, reward, n_state, done):
        self.replay_buffer.store_transition(state, action, reward, n_state, done)
        self.exp_counter+=1


    def select_action(self, state, deploy=False):
        if deploy:
            return int(torch.argmax(self.q_eval.forward(state)).item())
        if torch.rand(1).item() < self.epsilon:
            self.epsilon *= self.epsilon_dec
            self.epsilon = max(self.epsilon, self.epsilon_min)
            return int(torch.randint(0,self.number_of_actions,(1,)).item())
        else:
            return int(torch.argmax(self.q_eval.forward(state)).item())

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_learn(self):
        if self.exp_counter<self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        indices = torch.arange(0,self.batch_size).long()
        action_batch = action_batch.long()
        q_pred = self.q_eval.forward(state_batch)[indices, action_batch]
        q_next = self.q_next.forward(n_state_batch).max(dim=1).values.to(device)
        q_next[done_batch] = 0
        q_target = reward_batch.to(device) + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
   



