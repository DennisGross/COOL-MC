import argparse
import sys
import os
import gym
sys.path.insert(0, '../')
from common.safe_gym.safe_gym import SafeGym

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--open_ai_env', help='In which environment do you want to train your RL agent?', type=str,
                            default='FrozenLake-v1')
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which environment do you want to train your RL agent?', type=str,
                            default='frozen_lake-v1.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='slippery=0.0')
    arg_parser.add_argument('--disable_features', help='Disable features (seperate by commata)', type=str,
                            default='')
    
   
    
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    safe_env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], 10, 1, False, "", command_line_arguments['disable_features'])
    actions = {
        'Left': 0,
        'Down': 1,
        'Right': 2, 
        'Up': 3
    }
    
    
    env = gym.make(command_line_arguments['open_ai_env'], is_slippery=True)
    state = env.reset()
    print("Safe Gym Actions:\t", safe_env.action_mapper.actions)
    print("Safe Gym:\t", safe_env.storm_bridge.state_json_example)
    print("Safe Gym:\t",safe_env.reset())
    print("OpenAI Gym:\t",state)

    steps = [1,1,1,1]
    for step in steps:
        state, _, __, _ = env.step(step)
        state_safe, _, __, _ = safe_env.step(step)
        print(state)
        env.render()

# It seems that randomness for allowed actions

    