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
        print(self.mapper)
        self.mapper = self.update_mapper(self.mapper, self.disabled_features)
        print(self.mapper)


    def update_mapper(self, mapper, disabled_features):
        if len(disabled_features) == 0:
            return mapper
        else:
            for disabled_key in disabled_features:
                disabled_index = mapper[disabled_key]
                for key in mapper.keys():
                    if key != disabled_key and mapper[key] > disabled_index:
                        mapper[key]-=1
                del mapper[disabled_key]
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
        size = len(self.mapper.keys())
        mapped_state = np.zeros(size, np.int32)
        print(state)
        for idx, name in enumerate(self.original_format):
            if name not in self.disabled_features:
                n_idx = self.mapper[name]
                print("put", name, 'to', n_idx, 'original', idx)
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

    



        



    