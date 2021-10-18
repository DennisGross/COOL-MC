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
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
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


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['num_episodes'] = 1
    command_line_arguments['eval_episodes'] = 1
    m_project = Project(command_line_arguments, None, None)
    prism_file_path = os.path.join(m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['max_steps'], m_project.command_line_arguments['wrong_action_penalty'], m_project.command_line_arguments['reward_flag'], m_project.command_line_arguments['disabled_features'])
    m_project.agent = m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    print(m_project.command_line_arguments)
    mdp_reward_result, model_size, _, _ = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, env, command_line_arguments['constant_definitions'], command_line_arguments['prop'])
    print(command_line_arguments['prop'], ':', mdp_reward_result)