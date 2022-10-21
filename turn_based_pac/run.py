import sys
sys.path.insert(0, '../')
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import random
from helper import *
from math import *

# Experiments
# For tic-tac-toe
# Load player 2 from another training process and apply model checking of current player 1 against new player 2
# Do the same for multiple player 2s with different architecture parameters
# Calculate the average of the results of the model checking and compare with pac guarantees

if __name__ == '__main__':
    start_time = time.time()
    start_timestemp = get_current_timestemp()
    command_line_arguments = get_arguments()
    alpha = command_line_arguments['alpha']
    outputs = run_verify_rl_agent(command_line_arguments)
    result = outputs[0]
    N = command_line_arguments['range_plotting']
    print("Result", result)
    print("Half Model size", int(outputs[1]/2))
    all_results = []
    for i in range(N):
        rand_state_idx = random.randint(0, int(outputs[1]/2) -1)
        outputs = run_verify_rl_agent(command_line_arguments,random_state_idx=rand_state_idx, prop_extension="min")
        clean_folder("../mlruns",start_timestemp)
        print("Current Sample IDX:", i)
        all_results.append(outputs[0])
    print("Average Result:", sum(all_results)/N)
    g = exp(-N *(alpha**2)/2) + exp(-N *(alpha**2)/3)
    print(f"Probability that we are further away from the Expected Result than {alpha}:", g)
    print("Running time:", time.time() - start_time)
        
    
