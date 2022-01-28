import os
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.safe_gym.constant_definition_parser import ConstantDefinitionParser
import numpy as np
from common.utilities.front_end_printer import *
from common.tasks.verify_rl_agent import run_verify_rl_agent
import gc

def get_all_assignments_of_feature(feature_part):
    feature_name = feature_part.split("=")[0]
    start_value = int(feature_part[(feature_part.find("[")+1):feature_part.find(";")])
    end_value = int(feature_part[(feature_part.find(";")+1):feature_part.find("]")])+1
    return feature_name, list(range(start_value, end_value))

def run_sensitivity_analysis(command_line_arguments):
    '''    
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'], command_line_arguments['task'], command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.command_line_arguments['disabled_features'] = command_line_arguments['disabled_features']
    prism_file_path = os.path.join(m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,m_project.command_line_arguments['constant_definitions'], 1, 1, False, command_line_arguments['seed'], command_line_arguments['permissive_input'],  m_project.command_line_arguments['disabled_features'], abstraction_input=m_project.command_line_arguments['abstract_features'], noisy_feature_str=m_project.command_line_arguments['noisy_features'])
    all_features = env.storm_bridge.get_features()
    print(all_features)
    '''
    N = command_line_arguments['sensitive_iterations']
    features_results = {}
    if command_line_arguments['noisy_features'].find(":")==0:
        noisy_features = command_line_arguments['noisy_features'][1:]
        command_line_arguments['noisy_features'] = ""
        for feature_part in noisy_features.split(","):
            feature_name, assignment_list = get_all_assignments_of_feature(feature_part)
            features_results[feature_name] = 0

            for assignment in assignment_list:
                input = feature_name + "=["+str(assignment)+";"+str(assignment)+"]"
                command_line_arguments["permissive_input"] = input
                tmp_args = command_line_arguments.copy()
                prop = run_verify_rl_agent(tmp_args)
                features_results[feature_name] += prop[0]
                print(input, prop[0])
            print(features_results[feature_name], len(assignment_list))
            features_results[feature_name] /= len(assignment_list)
            print(features_results[feature_name])
    else:
        for feature_part in command_line_arguments['noisy_features'].split(','):
            tmp_args = command_line_arguments.copy()
            tmp_args['noisy_features'] = feature_part
            feature_name = feature_part.split("=")[0]
            features_results[feature_name] = 0
            for i in range(N):
                tmp_args = command_line_arguments.copy()
                tmp_args['noisy_features'] = feature_part
                prop = run_verify_rl_agent(tmp_args)
                features_results[feature_name] += prop[0]
            features_results[feature_name] /= N
        gc.collect()
    print("Average Property Results:")
    for feature_name in features_results.keys():
        print(feature_name, features_results[feature_name])