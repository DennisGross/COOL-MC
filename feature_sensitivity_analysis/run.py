import json
import sys
import gc
sys.path.insert(0, '../')
from math import sqrt
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import time

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



def feature_vulnerability(command_line_arguments, method='mean'):
    if str(command_line_arguments['abstract_features']).find("*")!=-1:
        all_abstraction_features = str(command_line_arguments['abstract_features']).split("*")[1]
        fixed_json_path = str(command_line_arguments['abstract_features']).split("*")[0]
    else:
         all_abstraction_features = str(command_line_arguments['abstract_features'])
         fixed_json_path = None
    # r-value
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

def parse_korkmaz_parameters(raw_str: str):
    print(raw_str)
    feature_name, magnitude = raw_str.split(":")
    return feature_name, int(magnitude)

def korkmaz_analysis(command_line_arguments, method='mean'):
    random_attack_results = []
    n, part2 = command_line_arguments['abstract_features'].split("#")
    n = int(n)
    feature_parts = part2.split(",")
    command_line_arguments['abstract_features'] = ""
    results = run_verify_rl_agent(command_line_arguments)
    r = results[0]
    feature_results = {}
    for feature_part in feature_parts:
        
        # feature_name:magnitude:n,
        feature_name, magnitude = parse_korkmaz_parameters(feature_part)
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
            command_line_arguments['attack_config'] = "specific_direction,"+str(magnitude)+','+feature_name
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
        
        



if __name__ == '__main__':
    result_strings = []
    command_line_arguments = get_arguments()
    set_random_seed(command_line_arguments['seed'])
    start_time = time.time()
    if command_line_arguments['abstract_features'][0] == ":":
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features'][1:]
        feature_vulnerability(command_line_arguments, method="mean")
    if command_line_arguments['abstract_features'][0] == "+":
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features'][1:]
        feature_vulnerability(command_line_arguments, method="max")
    elif command_line_arguments['abstract_features'].find("#") != -1:
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features']
        korkmaz_analysis(command_line_arguments)
    print("####################################")
    print("Runtime (s):", time.time()-start_time)