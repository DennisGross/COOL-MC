import pytest
from helper import *
import os
import argparse
import sys
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.utilities.front_end_printer import *
from common.tasks.safe_gym_training import *
from common.tasks.verify_rl_agent import *

def test_dummy_abc_reward():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='dummy_abc_unittesting')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=101)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=100)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='dummy_abc.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F FALL_OFF=true]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=20)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)        
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
                            default=0.001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)

    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    parent_run_id = run_safe_gym_training(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    result = run_verify_rl_agent(command_line_arguments)
    assert 0 == result[0]
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['permissive_input'] = 'a=[0;1]'
    command_line_arguments['prop'] = "Pmin=? [F FALL_OFF=true]"
    result_min = run_verify_rl_agent(command_line_arguments)
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['permissive_input'] = 'a=[0;1],b=[0;1],c=[0;1]'
    command_line_arguments['prop'] = "Pmax=? [F FALL_OFF=true]"
    result_max = run_verify_rl_agent(command_line_arguments)
    assert result_min[0] != result_max[0]


