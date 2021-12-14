import argparse
import sys
import os
import mlflow




def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='avoid')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=300)
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
                            default='avoid.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='xMax=4,yMax=4,slickness=0')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Tmin=? [F COLLISION=true]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=0)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)        
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




if __name__ == '__main__':
    args = get_arguments()
    if args['task'] == 'safe_training':
        mlflow.run(
            "safe_gym_training",
            use_conda=False,
            parameters=dict(args)
        )
    elif args['task'] == 'openai_training':
        mlflow.run(
            "openai_gym_training",
            parameters=dict(args),
            use_conda=False
        )
    elif args['task'] == 'rl_model_checking':
        mlflow.run(
            "verify_rl_agent",
            parameters=dict(args),
            use_conda=False
        )
   
