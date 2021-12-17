
import sys
sys.path.insert(0, '../')
from common.tasks.helper import *
from common.tasks.gym_matcher import *



if __name__ == '__main__':
    command_line_arguments = get_arguments()
    run_gym_matcher(command_line_arguments)