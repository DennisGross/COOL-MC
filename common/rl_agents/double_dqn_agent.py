import torch
import numpy as np
import mlflow
import shutil
import torch.nn.functional as F
from common.rl_agents.agent import Agent
from collections import OrderedDict
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:

  def __init__(self, state_dimension, mem_size=2):
    self.states = torch.zeros(mem_size, state_dimension)
    self.actions = torch.zeros(mem_size).long()
    self.rewards = torch.zeros(mem_size)
    self.next_states = torch.zeros(mem_size, state_dimension)
    self.dones = torch.zeros(mem_size, dtype=torch.long)
    self.mem_counter = 0
    self.mem_size = mem_size

  def to_torch(self, state, action, reward, next_state, done):
    state = torch.tensor(state)
    action = torch.tensor(action)
    reward = torch.tensor(reward)
    next_state = torch.tensor(next_state)
    if done:
      done = 1
    else:
      done = 0
    done = torch.tensor(done)
    return state, action, reward, next_state, done

  def store_transition(self, state, action, reward, next_state, done):
    mem_idx = self.mem_counter % self.mem_size
    state, action, reward, next_state, done = self.to_torch(state, action, reward, next_state, done)
    self.states[mem_idx] = state
    self.actions[mem_idx] = action
    self.rewards[mem_idx] = reward
    self.next_states[mem_idx] = next_state
    self.dones[mem_idx] = done
    self.mem_counter += 1

  def sample_batch(self, size):
    if self.mem_counter < size:
      return None
    rnd_idizes = torch.randperm(self.states.size()[0], dtype=torch.long)[0:size]
    return (self.states[rnd_idizes],self.actions[rnd_idizes].to(device),self.rewards[rnd_idizes].to(device),self.next_states[rnd_idizes].to(device), self.dones[rnd_idizes].to(device))


class DQNetwork(torch.nn.Module):

  def __init__(self, state_dimension, number_of_actions, number_of_neurons, learning_rate):
    super(DQNetwork, self).__init__()
    layers = OrderedDict()
    previous_neurons = state_dimension
    for i in range(len(number_of_neurons)+1):
      if i == len(number_of_neurons):
        layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_actions)
      else:
        layers[str(i)]  = torch.nn.Linear(previous_neurons, number_of_neurons[i])
        previous_neurons = number_of_neurons[i]
    self.layers = torch.nn.Sequential(layers)
    self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
    self.to(device)


  def to_torch(self, x):
    return torch.tensor(x).float().to(device)

  def forward(self, x):
    x = self.to_torch(x)
    for i in range(len(self.layers)):
      x = F.relu(self.layers[i](x))
    return x

  def save_checkpoint(self, file_name):
    torch.save(self.state_dict(), file_name)

  def load_checkpoint(self, file_name):
    self.load_state_dict(torch.load(file_name))


class DoubleDQNAgent(Agent):

  def __init__(self, state_dimension, number_of_actions, number_of_neurons, mem_size, epsilon, epsilon_dec, epsilon_min, gamma, replace, learning_rate, batch_size):
    self.state_dimension = state_dimension
    self.number_of_actions = number_of_actions
    self.number_of_neurons = number_of_neurons
    self.mem_size = mem_size
    self.epsilon = epsilon
    self.epsilon_dec = epsilon_dec
    self.epsilon_min = epsilon_min
    self.gamma = gamma
    self.replace = replace
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.exp_counter = 0
    self.learn_step_counter = 0
    # Neural Networks
    self.q_predict = DQNetwork(self.state_dimension, self.number_of_actions, self.number_of_neurons, learning_rate)
    self.q_target = DQNetwork(self.state_dimension, self.number_of_actions, self.number_of_neurons, learning_rate)
    # Replay Buffer
    self.replay_buffer = ReplayBuffer(self.state_dimension, mem_size=self.mem_size)

  def save(self):
    try:
      os.mkdir('tmp_model')
    except:
      pass
    self.q_predict.save_checkpoint('tmp_model/q_predict.chkpt')
    self.q_target.save_checkpoint('tmp_model/q_target.chkpt')
    mlflow.log_artifacts("tmp_model", artifact_path="model")
    shutil.rmtree('tmp_model')
    #print(mlflow.get_artifact_uri(artifact_path="model"))


  def load(self):
    self.q_predict.load_checkpoint(os.path.join(mlflow.get_artifact_uri(artifact_path="model"),'q_predict.chkpt'))
    self.q_target.load_checkpoint(os.path.join(mlflow.get_artifact_uri(artifact_path="model"),'q_target.chkpt'))
  

  def store_experience(self, state, action, reward, next_state, terminal):
    self.replay_buffer.store_transition(state, action, reward, next_state, terminal)


  def select_action(self, state, deploy=False):
    if torch.rand(1).item() < self.epsilon:
      self.epsilon *= self.epsilon_dec
      self.epsilon = max(self.epsilon, self.epsilon_min)
      return int(torch.randint(0,self.number_of_actions,(1,)).item())
    else:
      return int(torch.argmax(self.q_predict.forward(state)).item())

  def replace_target_network(self):
    if self.learn_step_counter % self.replace == 0:
      self.q_target.load_state_dict(self.q_predict.state_dict())
    self.learn_step_counter += 1

  def step_learn(self):
    if self.replay_buffer.mem_counter<self.batch_size:
      return
    # Replace target network
    self.replace_target_network()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)

    # Q Target Values
    max_actions = self.q_predict.forward(next_state_batch).detach().argmax(dim=1).long()
    next_q_values = self.q_target.forward(next_state_batch).detach().gather(1, max_actions.unsqueeze(1))
    target_q_values = reward_batch + (self.gamma * next_q_values.squeeze(1) * (1-done_batch.float()))
    target_q_values = target_q_values.unsqueeze(1)

    # Q Predicted
    expected_q_values = self.q_predict(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute Loss
    loss = F.mse_loss(expected_q_values, target_q_values)
    # Minimize Loss
    self.q_predict.optimizer.zero_grad()
    loss.backward()
    self.q_predict.optimizer.step()
   