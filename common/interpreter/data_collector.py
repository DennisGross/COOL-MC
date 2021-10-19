import pandas as pd
import os
import json
import mlflow

class DataCollector():

    def __init__(self, root_folder, state_json_example):
        '''
        Initialize data collecot
        :root_folder, root folder for data file
        :state_json_example, state JSON example
        '''
        self.csv_file_path = os.path.join(root_folder, 'data.csv')
        self.features = self.__extract_features(state_json_example)
        if os.path.exists(self.csv_file_path):
            self.df = pd.read_csv(self.csv_file_path)
        else:
            self.df = pd.DataFrame(columns=self.features)

    def __extract_features(self, state_json_example):
        '''
        Initialize data collecot
        :state_json_example, state JSON example
        '''
        state_dic = json.loads(state_json_example)
        feature_list = []
        for feature in state_dic.keys():
            feature_list.append(feature)
        return feature_list

    def store_state(self, n_state):
        '''
        Store time step as data point
        :time_step, time step
        '''
        state = {}
        i = 0
        for i in range(len(self.features)):
            state[self.features[i]] = int(n_state[i])
        self.df = self.df.append(state, ignore_index=True)
        

    def save(self):
        '''
        Save data file without duplicates as CSV.
        '''
        self.df = self.df.drop_duplicates()
        self.df.to_csv(self.csv_file_path, index=False)


    


