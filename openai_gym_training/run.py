sys.path.insert(0, '../')
from common.tasks.helper import *
from common.tasks.openai_gym_training import *


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    run_openai_gym_training(command_line_arguments)