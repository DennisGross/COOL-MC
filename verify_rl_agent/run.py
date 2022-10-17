import sys
sys.path.insert(0, '../')
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import time




if __name__ == '__main__':
    command_line_arguments = get_arguments()
    if command_line_arguments['permissive_input'].startswith("robustness,"):
        start_time = time.time()
        command_line_arguments['prop'] = command_line_arguments['prop'].replace("max", "min")
        min_result = run_verify_rl_agent(command_line_arguments)[0]
        command_line_arguments['prop'] = command_line_arguments['prop'].replace("min", "max")
        max_result = run_verify_rl_agent(command_line_arguments)[0]
        print("Robustness:", max_result - min_result)
        print("Time:", time.time() - start_time)
    else:
        run_verify_rl_agent(command_line_arguments)
    
