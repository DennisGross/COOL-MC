import argparse
import sys
import os
import gym
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.safe_gym.constant_definition_parser import ConstantDefinitionParser
import numpy as np
import matplotlib.pyplot as plt

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (e.g. slippery=0.1 or range: slippery=[0.1;0.1;1.1])', type=str,
                            default='slippery=0.04')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    arg_parser.add_argument('--disabled_features', help='Features for disabling', type=str,
                            default='')

    #IGNORE THIS ARGUMENTS (BUT DO NOT DELETE THEM)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9994)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)



if __name__ == '__main__':
    command_line_arguments = get_arguments()
    command_line_arguments['task'] = 'rl_model_checking'
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'], command_line_arguments['task'], command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.command_line_arguments['disabled_features'] = command_line_arguments['disabled_features']
    prism_file_path = os.path.join(m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    print(m_project.command_line_arguments)
    if command_line_arguments['constant_definitions'].count('[') == 0 and command_line_arguments['constant_definitions'].count(']') == 0:
        env = SafeGym(prism_file_path,m_project.command_line_arguments['constant_definitions'], 1, 1, False, command_line_arguments['permissive_input'],  m_project.command_line_arguments['disabled_features'])
        m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
        mdp_reward_result, model_size, _, _ = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, env, command_line_arguments['constant_definitions'], command_line_arguments['prop'])
        print(command_line_arguments['prop'], ':', mdp_reward_result)
        print("Model Size", model_size)
        m_project.mlflow_bridge.log_best_property_result(mdp_reward_result)
        if command_line_arguments['permissive_input'] == '':
            command_line_arguments['prop'] = command_line_arguments['prop'].replace("max","").replace("min","")
        m_project.mlflow_bridge.set_property_query_as_run_name(command_line_arguments['prop'] + " for " + command_line_arguments['constant_definitions'])
        m_project.save()
    elif command_line_arguments['constant_definitions'].count('[') == 1 and command_line_arguments['constant_definitions'].count(']') == 1:
        # For each step make model checking and save results in list
        all_constant_definitions, range_tuple, range_state_variable = ConstantDefinitionParser.parse_constant_definition(
                command_line_arguments['constant_definitions'])
        all_prop_results = []
        for constant_definitions in all_constant_definitions:
            env = SafeGym(prism_file_path,constant_definitions, 1, 1, False, command_line_arguments['permissive_input'],  m_project.command_line_arguments['disabled_features'])
            m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
            mdp_reward_result, model_size, _, _ = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, env, constant_definitions, command_line_arguments['prop'])
            print(command_line_arguments['prop'], ':', mdp_reward_result)
            
            all_prop_results.append(mdp_reward_result)
            # Plot results
            # Data for plotting
        x = np.arange(range_tuple[0], range_tuple[2], range_tuple[1])
        y = np.array(all_prop_results)

        fig, ax = plt.subplots()
        ax.plot(x,y)

        ax.set(xlabel=range_state_variable, ylabel=command_line_arguments['prop'].replace("max","").replace("min",""), title='Property Results over a range of different constant assignments (' + str(range_state_variable) + ')')
        ax.grid()

        #fig.savefig(os.path.join(project_folder, "properties.png"))
        plt.show()
    else:
        raise ValueError("We only support plotting for one state variable...")
