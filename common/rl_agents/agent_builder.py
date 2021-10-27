import os
from common.rl_agents.dummy_agent import DummyAgent
from common.rl_agents.dummy_frozen_lake_agent import DummyFrozenLakeAgent
from common.rl_agents.double_dqn_agent import DoubleDQNAgent
from common.rl_agents.sarsa_max_agent import SarsaMaxAgent
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class AgentBuilder():

    @staticmethod
    def layers_neurons_to_number_of_neurons(layers, neurons):
        number_of_neurons = []
        for i in range(layers):
            number_of_neurons.append(neurons)
        return number_of_neurons

    @staticmethod
    def build_agent(model_root_folder_path, command_line_arguments, observation_space, action_space):
        print('Build model with', model_root_folder_path, command_line_arguments)
        print('Environment', observation_space.shape[0], action_space.n)
        agent = None
        if command_line_arguments['rl_algorithm'] == 'dummy_agent':
            print("Build Dummy Agent.")
            agent = DummyAgent(observation_space.shape[0], action_space.n, command_line_arguments['always_action'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'dummy_frozen_lake_agent':
            print("Build Dummy Frozen Lake Agent.")
            agent = DummyFrozenLakeAgent(observation_space.shape[0], action_space.n, command_line_arguments['always_action'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'sarsamax':
            print("Build Dummy SARSAMAX Agent.")
            agent = SarsaMaxAgent(action_space.n, epsilon=1, epsilon_dec=0.9999, epsilon_min=0.1, alpha=0.6, gamma=0.99)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'double_dqn_agent':
            print("Build Dummy Double DQN Agent.")
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = DoubleDQNAgent(observation_space.shape[0], action_space.n, number_of_neurons, command_line_arguments['replay_buffer_size'], command_line_arguments['epsilon'], command_line_arguments['epsilon_dec'], command_line_arguments['epsilon_min'], command_line_arguments['gamma'], command_line_arguments['replace'], command_line_arguments['lr'], command_line_arguments['batch_size'])
            agent.load(model_root_folder_path)
        return agent