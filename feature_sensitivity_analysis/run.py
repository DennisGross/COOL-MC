import json
import sys
import os
import shutil
import datetime

from click import command
sys.path.insert(0, '../')
from common.tasks.safe_gym_training import run_safe_gym_training

from math import sqrt
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import time


def get_current_timestemp():
    return datetime.datetime.now().timestamp()

def delete_folder_recursively(path, ts):
    """"
    Delete folder if it was created before timestemp ts
    """
    if os.path.exists(path):
        if os.path.getmtime(path) >= ts:
            shutil.rmtree(path)

def get_sub_directory_paths_of_folder(path):
    """
    Get sub directory paths of folder
    """
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def clean_folder(path, ts):
    for root_folder_path in get_sub_directory_paths_of_folder(path):
        for folder_path in os.listdir(root_folder_path):
            
            full_path = os.path.join(root_folder_path, folder_path)
            
            # Check if full_path is a folder
            if os.path.isdir(full_path):
                delete_folder_recursively(full_path, ts)



def parse_mapping_parameters(raw_str: str):
    print(raw_str)
    feature_name = raw_str.split("=")[0]
    lower_bound = int(raw_str[raw_str.find("[")+1:raw_str.find(";")])
    upper_bound = int(raw_str[raw_str.find(";")+1:raw_str.find("]")])
    all_fixed_values = list(range(lower_bound,upper_bound+1))
    return feature_name, lower_bound, upper_bound, all_fixed_values

def create_mapping_file(path: str, feature_name: str, lower_bound: int, upper_bound: int, fixed_value: int):
    all_values = list(range(lower_bound, upper_bound+1))
    mapper = {feature_name:{}}
    for value in all_values:
        mapper[feature_name][value] = fixed_value
    with open(path, 'w') as fp:
        json.dump(mapper, fp)

def save_list_into_csv(path, l):
    with open(path, 'w') as fp:
        for item in l:
            fp.write(str(item) + "\n")


def parse_feature_space(raw_str: str):
    pass

def specific_fv(command_line_arguments, method='mean', c="/"):
    control_str = command_line_arguments['abstract_features']
    print(control_str)
    feature_space_strs = control_str.split(c)[0]
    attack_str = control_str.split(c)[1]
    command_line_arguments['abstract_features'] = ""
    property_query = str(command_line_arguments['prop'])
    command_line_arguments['prop'] = property_query
    results = run_verify_rl_agent(command_line_arguments)
    r = results[0]
    safe_feature_values = []
    for part in attack_str.split(","):
        feature_name, lower_bound, upper_bound, all_fixed_values = parse_mapping_parameters(part)
        print(feature_name, lower_bound, upper_bound, all_fixed_values)
        all_prop_results = []
        for fixed_value in all_fixed_values:
            # Create attack config for specific dda
            command_line_arguments['attack_config'] = "sdda,"+str(fixed_value)+','+feature_name+"," + feature_space_strs
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            print(prop_result)
            all_prop_results.append(prop_result)
        result = None
        if method == "mean":
            avg_props = sum(all_prop_results)/len(all_prop_results)
            result = abs(r-avg_props)
        elif method == "max":
            max_prob = np.array(all_prop_results) - r
            # Each numpy element as abs
            abs_arr = np.abs(max_prob)
            # numpy array abs_arr to list
            abs_arr_list = abs_arr.tolist()
            result = max(abs_arr_list)
            #result = abs(r-max_value)
        safe_feature_values.append(feature_name + ":" + str(result))
    print("####################################")
    print("Specific Safe Feature Values", method, ":")
    for s in safe_feature_values:
        print(s)
    safe_feature_values.append(command_line_arguments['prop'])
    safe_feature_values.append("s"+method)
    safe_feature_values.append(command_line_arguments['prism_file_path'])
    safe_feature_values.append(command_line_arguments['parent_run_id'])
    safe_feature_values.append(str(feature_space_strs))
    save_list_into_csv(str(time.time())+".csv", safe_feature_values)

        


def feature_vulnerability(command_line_arguments, method='mean'):
    if str(command_line_arguments['abstract_features']).find("*")!=-1:
        all_abstraction_features = str(command_line_arguments['abstract_features']).split("*")[1]
        fixed_json_path = str(command_line_arguments['abstract_features']).split("*")[0]
    else:
         all_abstraction_features = str(command_line_arguments['abstract_features'])
         fixed_json_path = None
    # r-value
    start_time = time.time()
    command_line_arguments['abstract_features'] = ""
    property_query = str(command_line_arguments['prop'])
    command_line_arguments['prop'] = property_query
    results = run_verify_rl_agent(command_line_arguments)
    print(results)
    r = results[0]
    model_size = results[1]
    model_sizes = [model_size]
    safe_feature_values = []
    counter = 0
    # Samples
    for abstraction_feature in all_abstraction_features.split(","):
        command_line_arguments['abstract_features'] = abstraction_feature
        feature_name, lower_bound, upper_bound, all_fixed_values = parse_mapping_parameters(command_line_arguments['abstract_features'])
        REMAPPING_FILE_PATH = "remapping.json"
        # For Fixed states
        if fixed_json_path != None:
            command_line_arguments['abstract_features'] = fixed_json_path+"*"+REMAPPING_FILE_PATH
        else:
            command_line_arguments['abstract_features'] = REMAPPING_FILE_PATH
        all_prop_results = []
        # Verify with this abstraction mapper
        for fixed_value in all_fixed_values:
            create_mapping_file(REMAPPING_FILE_PATH, feature_name, lower_bound, upper_bound, fixed_value)
            # Verification
            command_line_arguments['prop'] = property_query
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            model_size = results[1]
            model_sizes.append(model_size)
            all_prop_results.append(prop_result)
            print(feature_name, fixed_value, prop_result)
            counter+=1
        result = None
        if method == "mean":
            avg_props = sum(all_prop_results)/len(all_prop_results)
            result = abs(r-avg_props)
        elif method == "max":
            max_prob = np.array(all_prop_results) - r
            # Each numpy element as abs
            abs_arr = np.abs(max_prob)
            # numpy array abs_arr to list
            abs_arr_list = abs_arr.tolist()
            result = max(abs_arr_list)
            #result = abs(r-max_value)
        safe_feature_values.append(feature_name + ":" + str(result))
    print("####################################")
    print("Property Result:", r)
    print("Minimal Model:", min(model_sizes))
    print("Original Model:", model_sizes[0])
    print("Average Model:", sum(model_sizes)/len(model_sizes))
    print("Maximal Model:", max(model_sizes))
    print("Number of Queries:", counter)
    print("####################################")
    print("Safe Feature Values", method, ":")
    for s in safe_feature_values:
        print(s)
    safe_feature_values.append(command_line_arguments['prop'])
    safe_feature_values.append(method)
    safe_feature_values.append(command_line_arguments['prism_file_path'])
    safe_feature_values.append(command_line_arguments['parent_run_id'])
    safe_feature_values.append("Time:" + str(time.time()-start_time))
    safe_feature_values.append("States: " + str(model_sizes[0]))
    safe_feature_values.append("Average States: " + str(sum(model_sizes) / len(model_sizes)))
    save_list_into_csv(str(time.time())+".csv", safe_feature_values)

    
def parse_korkmaz_parameters(raw_str: str):
    print(raw_str)
    feature_name, magnitude = raw_str.split(":")
    return feature_name, int(magnitude)

def save_dictionary_and_integer_into_file(path: str, dictionary: dict):

    # save dictionary into csv
    with open(path, 'w') as fp:
        for key, value in dictionary.items():
            fp.write(str(key) + "," + str(value) + "\n")

def empirical_korkmaz_analysis(command_line_arguments):
    random_attack_results = []
    n, part2 = command_line_arguments['abstract_features'].split("&")
    n = int(n)
    feature_parts = part2.split(",")
    command_line_arguments['abstract_features'] = ""

    feature_results = {}
    rs = []
    print("HERE")
    print(command_line_arguments)
    
    for i in range(n):
        command_line_arguments['attack_config'] = ""
        command_line_arguments['abstract_features'] = ""
        command_line_arguments['task'] = "safe_training"
        command_line_arguments['num_episodes'] = 1
        command_line_arguments['deploy'] = 1
        command_line_arguments['prop'] = ""
        #print(command_line_arguments['attack_config'])
        #command_line_arguments['attack_config'] = ""
        run_id, last_reward = run_safe_gym_training(command_line_arguments)
        rs.append(last_reward)
        print(last_reward)
    r = sum(rs)/len(rs)

    
    for feature_part in feature_parts:
        # feature_name:magnitude:n,
        feature_name, magnitude = parse_korkmaz_parameters(feature_part)
        
        # N times specific direction attack
        rewards = []
        for i in range(n):
            # Set Specific direction attack config
            command_line_arguments['attack_config'] = "specific_direction,"+str(magnitude)+','+feature_name
            command_line_arguments['num_episodes'] = 1
            command_line_arguments['deploy'] = 1
            command_line_arguments['abstract_features'] = ""
            command_line_arguments['task'] = "safe_training"
            command_line_arguments['prop'] = ""
            print(command_line_arguments)
            run_id, last_reward = run_safe_gym_training(command_line_arguments)
            print(last_reward)
            rewards.append(last_reward)
        
        distance_specific_to_real = abs(sum(rewards)/len(rewards)-r)
        feature_results[feature_name] = distance_specific_to_real
        print("####################################")

    for feature in feature_results.keys():
        print(feature, ":", feature_results[feature])

    feature_results['n'] = n
    feature_results['prop'] = "Empirical reward"
    feature_results['prism_file_path'] = command_line_arguments['prism_file_path']
    feature_results['parent_run_id'] = command_line_arguments['parent_run_id']
    save_dictionary_and_integer_into_file(str(time.time())+".csv", feature_results)





def korkmaz_analysis(command_line_arguments):
    start_time = time.time()
    random_attack_results = []
    n, part2 = command_line_arguments['abstract_features'].split("#")
    n = int(n)
    feature_parts = part2.split(",")
    command_line_arguments['abstract_features'] = ""
    results = run_verify_rl_agent(command_line_arguments)
    r = results[0]
    feature_results = {}
    model_sizes = [results[1]]
    for feature_part in feature_parts:
        
        # feature_name:magnitude:n,
        feature_name, lower_bound, upper_bound = parse_cw_korkmaz_parameters(feature_part)
        """
        for i in range(n):
            # Set Random direction attack config
            command_line_arguments['attack_config'] = "random_direction,"+str(magnitude)
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            model_size = results[1]
            random_attack_results.append(prop_result)
        distance_random_to_real = abs(sum(random_attack_results)/len(random_attack_results)-r)
        """
        # N times specific direction attack
        specific_attack_results = []
        for i in range(n):
            # Set Specific direction attack config
            command_line_arguments['attack_config'] = "cwattack,"+feature_name+","+str(lower_bound)+","+str(upper_bound)
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            model_size = results[1]
            model_sizes.append(model_size)
            specific_attack_results.append(prop_result)
        
        distance_specific_to_real = abs(sum(specific_attack_results)/len(specific_attack_results)-r)

        #feature_results[feature_name] = abs(r-distance_specific_to_real)
        feature_results[feature_name] = abs(sum(specific_attack_results)/len(specific_attack_results)-r)
        print("####################################")

    total_time = time.time() - start_time
    for feature in feature_results.keys():
        print(feature, ":", feature_results[feature])

    feature_results['n'] = n
    feature_results['prop'] = command_line_arguments['prop']
    feature_results['prism_file_path'] = command_line_arguments['prism_file_path']
    feature_results['parent_run_id'] = command_line_arguments['parent_run_id']
    feature_results['total_time'] = total_time
    feature_results['model_size'] = model_sizes[0]
    feature_results['avg_model_size'] = sum(model_sizes)/len(model_sizes)
    save_dictionary_and_integer_into_file(str(time.time())+".csv", feature_results)



def korkmaz_analysis_specific_states(command_line_arguments):
    random_attack_results = []
    part1, part2 = command_line_arguments['abstract_features'].split("#")
    feature_spaces_str, n = part1.split("%")
    n = int(n)
    feature_parts = part2.split(",")
    command_line_arguments['abstract_features'] = ""
    results = run_verify_rl_agent(command_line_arguments)
    r = results[0]
    feature_results = {}
    for feature_part in feature_parts:
        # feature_name:magnitude:n,
        feature_name, lower_bound, upper_bound = parse_cw_korkmaz_parameters(feature_part)
        """
        for i in range(n):
            # Set Random direction attack config
            command_line_arguments['attack_config'] = "random_direction,"+str(magnitude)
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            model_size = results[1]
            random_attack_results.append(prop_result)
        distance_random_to_real = abs(sum(random_attack_results)/len(random_attack_results)-r)
        """
        # N times specific direction attack
        specific_attack_results = []
        for i in range(n):
            # Set Specific direction attack config
            command_line_arguments['attack_config'] = "scwattack," + feature_name + "," + str(lower_bound) + "," +str(upper_bound) + "," + feature_spaces_str
            results = run_verify_rl_agent(command_line_arguments)
            prop_result = results[0]
            model_size = results[1]
            specific_attack_results.append(prop_result)
        
        distance_specific_to_real = abs(sum(specific_attack_results)/len(specific_attack_results)-r)

        #feature_results[feature_name] = abs(r-distance_specific_to_real)
        feature_results[feature_name] = abs(sum(specific_attack_results)/len(specific_attack_results)-r)
        print("####################################")

    for feature in feature_results.keys():
        print(feature, ":", feature_results[feature])

    feature_results['n'] = n
    feature_results['prop'] = command_line_arguments['prop']
    feature_results['prism_file_path'] = command_line_arguments['prism_file_path']
    feature_results['parent_run_id'] = command_line_arguments['parent_run_id']
    save_dictionary_and_integer_into_file(str(time.time())+".csv", feature_results)


def parse_cw_korkmaz_parameters(feature_part):
    feature_name, lower_bound, upper_bound = feature_part.split(":")
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)
    return feature_name, lower_bound, upper_bound

def cw_empirical_korkmaz_analysis(command_line_arguments):
    random_attack_results = []
    n, part2 = command_line_arguments['abstract_features'].split("^")
    n = int(n)
    feature_parts = part2.split(",")
    command_line_arguments['abstract_features'] = ""

    feature_results = {}
    rs = []
    print("HERE")
    print(command_line_arguments)
    
    for i in range(n):
        command_line_arguments['attack_config'] = ""
        command_line_arguments['abstract_features'] = ""
        command_line_arguments['task'] = "safe_training"
        command_line_arguments['num_episodes'] = 1
        command_line_arguments['deploy'] = 1
        command_line_arguments['prop'] = ""
        #print(command_line_arguments['attack_config'])
        #command_line_arguments['attack_config'] = ""
        run_id, _, last_reward = run_safe_gym_training(command_line_arguments)
        rs.append(last_reward)
        print(last_reward)
    r = sum(rs)/len(rs)

    
    for feature_part in feature_parts:
        # feature_name:magnitude:n,
        feature_name, lower_bound, upper_bound = parse_cw_korkmaz_parameters(feature_part)
        
        # N times specific direction attack
        rewards = []
        for i in range(n):
            # Set Specific direction attack config
            command_line_arguments['attack_config'] = "cwattack,"+feature_name+","+str(lower_bound)+","+str(upper_bound)
            command_line_arguments['num_episodes'] = 1
            command_line_arguments['deploy'] = 1
            command_line_arguments['abstract_features'] = ""
            command_line_arguments['task'] = "safe_training"
            command_line_arguments['prop'] = ""
            print(command_line_arguments)
            run_id, _, last_reward = run_safe_gym_training(command_line_arguments)
            print(last_reward)
            rewards.append(last_reward)
        
        distance_specific_to_real = abs(sum(rewards)/len(rewards)-r)
        feature_results[feature_name] = distance_specific_to_real
        print("####################################")

    for feature in feature_results.keys():
        print(feature, ":", feature_results[feature])

    feature_results['n'] = n
    feature_results['prop'] = "Empirical reward"
    feature_results['prism_file_path'] = command_line_arguments['prism_file_path']
    feature_results['parent_run_id'] = command_line_arguments['parent_run_id']
    save_dictionary_and_integer_into_file(str(time.time())+".csv", feature_results)

    
        
        



if __name__ == '__main__':
    time_stemp = get_current_timestemp()
    result_strings = []
    command_line_arguments = get_arguments()
    print(command_line_arguments['prism_file_path'])
    set_random_seed(command_line_arguments['seed'])
    command_line_arguments['task'] = "rl_model_checking"
    start_time = time.time()
    if command_line_arguments['abstract_features'][0] == ":":
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features'][1:]
        feature_vulnerability(command_line_arguments, method="mean")
    if command_line_arguments['abstract_features'][0] == "+":
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features'][1:]
        feature_vulnerability(command_line_arguments, method="max")
    elif command_line_arguments['abstract_features'].find("%") != -1:
        korkmaz_analysis_specific_states(command_line_arguments)
    elif command_line_arguments['abstract_features'].find("#") != -1:
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features']
        korkmaz_analysis(command_line_arguments)
    elif command_line_arguments['abstract_features'].find("/") != -1:
        print("Specific FV")
        specific_fv(command_line_arguments, "mean", '/')
    elif command_line_arguments['abstract_features'].find("^") != -1:
        cw_empirical_korkmaz_analysis(command_line_arguments)
    elif command_line_arguments['abstract_features'].find("?") != -1:
        print("Specific FV")
        specific_fv(command_line_arguments, "max", '?')
        
    print("####################################")
    delete_folder_recursively("../mlruns")
    print("Runtime (s):", time.time()-start_time)
    