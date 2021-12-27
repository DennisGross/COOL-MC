
import sys
sys.path.insert(0, '../')
from common.tasks.safe_gym_training import *



if __name__ == '__main__':
    command_line_arguments = get_arguments()
    experiment_id = run_safe_gym_training(command_line_arguments)
    print("Experiment ID:", experiment_id)