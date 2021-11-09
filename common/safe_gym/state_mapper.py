import json
import numpy as np
import os
class StateMapper:

    def __init__(self, transformation_file_path, state_json_example):
        self.openai_state_variable_positions = self.load_mappings(transformation_file_path, state_json_example)
        

    def load_mappings(self, transformation_file_path, state_json_example):
        mapper = None
        print(transformation_file_path)
        if os.path.exists(transformation_file_path):
            with open(transformation_file_path) as json_file:
                mapper = json.load(json_file)
        else:
            json_example = str(state_json_example)
            mapper = {}
            i = 0
            for k in json.loads(json_example):
                mapper[k] = i
                i+=1

        return mapper

    def map(self, state, state_variables):
        size = len(state_variables)
        mapped_state = np.zeros(size, np.int32)
        idx = 0
        for name in state_variables:
            n_idx = self.openai_state_variable_positions[name]
            mapped_state[n_idx] = state[idx]
            idx+=1
        return mapped_state




        



    