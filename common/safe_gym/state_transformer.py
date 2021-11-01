import json
import numpy as np
import os
class StateTransformer:

    def __init__(self, transformation_file_path):
        self.openai_state_variable_positions = self.load_transformations(transformation_file_path)
        

    def load_transformations(self, transformation_file_path):
        if os.path.exists(transformation_file_path):
            with open(transformation_file_path) as json_file:
                data = json.load(json_file)
                return data
        else:
            return None

    def transform(self, state, state_variables):
        if self.openai_state_variable_positions == None:
            return state
        else:
            size = len(state_variables)
            transformed_state = np.zeros(size)
            idx = 0
            for name in state_variables:
                n_idx = self.openai_state_variable_positions[name]
                transformed_state[n_idx] = state[idx]
                idx+=1
            return transformed_state




        



    