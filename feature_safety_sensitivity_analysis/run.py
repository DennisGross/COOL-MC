import json
import sys

from numpy import fix
sys.path.insert(0, '../')
from math import sqrt
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *

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


def safe_pdp(command_line_arguments):
    all_abstraction_features = str(command_line_arguments['abstract_features'])
    property_query = str(command_line_arguments['prop'])
    for abstraction_feature in all_abstraction_features.split(","):
        command_line_arguments['abstract_features'] = abstraction_feature
        feature_name, lower_bound, upper_bound, all_fixed_values = parse_mapping_parameters(command_line_arguments['abstract_features'])
        REMAPPING_FILE_PATH = "remapping.json"
        # Verify with this abstraction mapper
        command_line_arguments['abstract_features'] = REMAPPING_FILE_PATH
        all_prop_results = []
        for fixed_value in all_fixed_values:
            create_mapping_file(REMAPPING_FILE_PATH, feature_name, lower_bound, upper_bound, fixed_value)
            # Verification
            command_line_arguments['prop'] = property_query
            prop_result = run_verify_rl_agent(command_line_arguments)[0]
            all_prop_results.append(prop_result)
        # Calculate I(s_i)
        inner_square_part = 0
        for i in range(len(all_prop_results)):
            inner_power_part = all_prop_results[i]
            # Second Average
            inner_part  = sum(all_prop_results)
            inner_part /= len(all_prop_results)
            # Prop minus second sum 
            inner_power_part-= inner_part
            # Power 2
            inner_power_part=inner_power_part**2
            inner_square_part+=inner_power_part
        # Average
        inner_square_part = inner_square_part / (len(all_prop_results)-1)
        # Square
        final_result = sqrt(inner_square_part)
        result_str = "I("+feature_name+")="+ str(final_result)
        result_strings.append(result_str)
    print("Safe-PDP values:")
    for s in result_strings:
        print(s)

def safe_feature_values(command_line_arguments):
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
    r = run_verify_rl_agent(command_line_arguments)[0]
    safe_feature_values = []
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
            prop_result = run_verify_rl_agent(command_line_arguments)[0]
            all_prop_results.append(prop_result)
        avg_props = sum(all_prop_results)/len(all_prop_results)
        result = abs(r-avg_props)
        safe_feature_values.append(feature_name + ":" + str(result))

    print("Property Result:", r)
    print("Safe Feature Values:")
    for s in safe_feature_values:
        print(s)





if __name__ == '__main__':
    result_strings = []
    command_line_arguments = get_arguments()
    if command_line_arguments['abstract_features'][0] == ":":
        command_line_arguments['abstract_features'] = command_line_arguments['abstract_features'][1:]
        safe_feature_values(command_line_arguments)
    else:
        safe_pdp(command_line_arguments)