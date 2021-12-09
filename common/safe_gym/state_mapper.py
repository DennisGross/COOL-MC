import json
import numpy as np
import os
class StateMapper:

    def __init__(self, transformation_file_path, state_json_example, disabled_features):
        self.mapper = self.load_mappings(transformation_file_path, state_json_example)
        self.original_format = list(self.mapper)
        if disabled_features == "":
            self.disabled_features = []
        else:
            self.disabled_features = disabled_features.split(",")
        self.new_variable_format = self.get_format( self.original_format, self.disabled_features)
        print(self.mapper)
        self.mapper = self.update_mapper(self.mapper, self.original_format, self.new_variable_format)
        print(self.mapper)

    def get_format(self,original_var_format, disabled_features):
        new_var_format = []
        for idx, name in enumerate(original_var_format):
            if name in self.disabled_features:
                continue
            else:
                new_var_format.append(name)
        return new_var_format

    def update_mapper(self, mapper, original_format, new_variable_format):
        print(original_format, new_variable_format)
        for idx, name in enumerate(original_format):
            try:
                if name in self.disabled_features:
                    del mapper[name]
                else:
                    mapper[name] += (new_variable_format.index(name)-idx)
                    if mapper[name] < 0:
                        mapper[name] = 0
                    elif mapper[name] >= len(new_variable_format):
                        mapper[name] = len(new_variable_format)-1
            except:
                pass
        return mapper

        

        

    def load_mappings(self, transformation_file_path, state_json_example):
        mapper = None
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

    def map(self, state):
        size = len(self.original_format)-len(self.disabled_features)
        mapped_state = np.zeros(size, np.int32)
        for idx, name in enumerate(self.original_format):
            if name not in self.disabled_features:
                n_idx = self.mapper[name]
                #print("put", name, 'to', n_idx, 'original', idx)
                mapped_state[n_idx] = state[idx]
        return mapped_state

    def inverse_mapping(self, idx):
        for name in self.mapper:
            try:
                if self.mapper[name] == idx:
                    return name
            except:
                pass
        return -1

    



        



    