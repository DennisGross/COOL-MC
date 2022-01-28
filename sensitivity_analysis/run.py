import sys
sys.path.insert(0, '../')
from common.tasks.sensitivity_analysis import *
from common.tasks.helper import *




if __name__ == '__main__':
    command_line_arguments = get_arguments()
    run_sensitivity_analysis(command_line_arguments)