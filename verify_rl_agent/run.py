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
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--disabled_features', help='Disabled features', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='slippery=0.04')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)

def manage_disable_features(command_line_arguments):
    if 'disabled_features' not in command_line_arguments.keys():
        command_line_arguments['disabled_features'] = ''
    return command_line_arguments


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments = manage_disable_features(command_line_arguments)
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'], command_line_arguments['task'], command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    prism_file_path = os.path.join(m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,m_project.command_line_arguments['constant_definitions'], 1, 1, False, command_line_arguments['permissive_input'],  m_project.command_line_arguments['disabled_features'])
    m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    print(m_project.command_line_arguments)
    mdp_reward_result, model_size, _, _ = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, env, command_line_arguments['constant_definitions'], command_line_arguments['prop'])
    print(command_line_arguments['prop'], ':', mdp_reward_result)
    m_project.mlflow_bridge.log_best_property_result(mdp_reward_result)
    m_project.save()