import argparse
import sys
import gym
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train


def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='CartPole')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='CartPole-v0')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='double_dqn_agent')
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=0)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=0)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    command_line_arguments['task'] = 'openai_gym_training'
    command_line_arguments['eval_interval'] = 1
    command_line_arguments['max_steps'] = 10
    env = gym.make(command_line_arguments['env'])
    m_project = Project(command_line_arguments, env.observation_space, env.action_space)
    train(m_project, env, prop_type='reward')