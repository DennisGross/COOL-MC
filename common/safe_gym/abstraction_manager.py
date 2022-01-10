import numpy as np
import math
import os
import json
class AStateVariable:

    def __init__(self, name, lower_bound, upper_bound, step_size, mapping=None):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.all_assignments = range(self.lower_bound,self.upper_bound+self.step_size,self.step_size)
        self.mapping = mapping
        

    def round(self, value):
        if self.mapping is None:
            min_distance = math.inf
            min_idx = 0
            for idx in range(len(self.all_assignments)):
                distance = abs(value - self.all_assignments[idx])
                if distance < min_distance:
                    min_idx = idx
                    min_distance = distance
            return self.all_assignments[min_idx]
        else:
            for original_values in self.mapping.keys():
                for part_value in original_values.split(","):
                    if part_value == str(int(value)):
                        return self.mapping[original_values]
            raise ValueError("Mapping for " + str(value) + " does not exist.")


    def set_idx(self, idx):
        self.idx = idx

    def get_idx(self):
        return self.idx

    def __str__(self):
        return self.name + "=["+str(self.lower_bound)+','+str(self.upper_bound)+"]" + " Index: " + str(self.idx) + " Step Size: " + str(self.step_size) + " Rounding Method: " + self.method


    @staticmethod
    def parse_abstraction_variables(abstraction_input):
        # VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX]
        astate_variables = []
        for abstraction_variable_str in abstraction_input.split(","):
            name = abstraction_variable_str.split('=')[0]
            start_interval = abstraction_variable_str.find('[')
            end_interval = abstraction_variable_str.find(']')
            interval = abstraction_variable_str[(start_interval+1):end_interval]
            step_size = 1
            if interval.count(';')==2:
                parts = abstraction_variable_str[(start_interval+1):end_interval].split(';')
                lower_bound = int(parts[0])
                step_size = int(parts[1])
                upper_bound = int(parts[2])
                astate_variable = AStateVariable(name, lower_bound, upper_bound, step_size)
                astate_variables.append(astate_variable)
            else:
                raise ValueError("Abstraction Input is the wrong format (VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX;METHOD])")
            
        return astate_variables

    @staticmethod
    def parse_abstraction_from_dict(abstraction_mapping):
        astate_variables = []
        for state_variable_name in abstraction_mapping.keys():
            astate_variable = AStateVariable(state_variable_name, -1, -1, -1, mapping=abstraction_mapping[state_variable_name])
            astate_variables.append(astate_variable)
        return astate_variables


class AbstractionManager:


    def __init__(self, state_var_mapper, abstraction_input) -> None:
        self.is_active = (abstraction_input != '')
        self.state_var_mapper = state_var_mapper.mapper
        if self.is_active:
            if os.path.isfile(abstraction_input):
                with open(abstraction_input) as json_file:
                    abstraction_mapping = json.load(json_file)
                    self.astate_variables = AStateVariable.parse_abstraction_from_dict(abstraction_mapping)
            else:
                self.astate_variables = AStateVariable.parse_abstraction_variables(abstraction_input)
            
            for i in range(len(self.astate_variables)):
                self.astate_variables[i].set_idx(self.state_var_mapper[self.astate_variables[i].name])


    def preprocess_state(self, state):
        for astate_variable in self.astate_variables:
            idx = astate_variable.get_idx()
            state[idx] = astate_variable.round(state[idx])
        return state