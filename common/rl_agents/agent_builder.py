import os
from rl_agents.dummy_agent import DummyAgent

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
    def build_agent(model_root_folder_path, command_line_arguments, observation_space, number_of_actions):
        print('Build model with', model_root_folder_path, command_line_arguments)
        agent = None
        if command_line_arguments['architecture'] == 'dummy_agent':
            print("Build Dummy Agent.")
            agent = DummyAgent(observation_space, number_of_actions, command_line_arguments['always_action'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        return agent