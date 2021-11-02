import argparse
import sys
import os
import gym
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym


def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='CartPole-v0')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=100)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='double_dqn_agent')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='slippery=0.04')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=0)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)        
    # Permissive Options
    arg_parser.add_argument('--permissive_input', help='Define the state variables which we want to ignore for the permissive policy', type=str,
                            default='')
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Sarsamax
    arg_parser.add_argument('--alpha', help='Gamma', type=float,
                            default=0.99)
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9994)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.004)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)

    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)

def parse_prop_type(prop):
    if prop.find("min") < prop.find("=") and prop.find("min") != -1:
        return "min_prop"
    elif prop.find("max") < prop.find("=") and prop.find("max") != -1:
        return "max_prop"
    else:
        return "reward"



if __name__ == '__main__':
    command_line_arguments = get_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['prop_type'] = parse_prop_type(command_line_arguments['prop'])
    command_line_arguments['reward_flag'] = command_line_arguments['reward_flag']  == 1
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    # Environment
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'])
    # Project
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'],command_line_arguments['task'],command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    train(m_project, env, prop_type=command_line_arguments['prop_type'])