import random
import io
import pandas as pd
import json


class DataGenerator:

    def __init__(self, csv_file_path, storm_bridge, agent, env):
        self.df = pd.read_csv(csv_file_path)
        self.features = []
        for feature in self.df.columns:
            self.features.append(feature)
        self.csv_file_path = csv_file_path
        self.storm_bridge = storm_bridge
        self.agent = agent
        self.env = env

    def __row_to_state(self, row, features):
        state_dict = {}
        # row to json
        for feature in features:
            state_dict[feature] = int(row[feature])
        # dict to json
        json_state = json.dumps(state_dict)
        # json to state
        state = self.storm_bridge.parse_state(json_state)
        # state to time_step
        return state


    def label_each_row_with_rl_agent_action(self):
        actions = []
        for index, row in self.df.iterrows():
            # Create TimeStep
            time_step = self.__row_to_state(row, self.features)
            # Get rl agent action
            action_index = self.agent.select_action(time_step, True)
            # Action to action label
            action_name = self.env.action_mapper.action_index_to_action_name(action_index)
            # Action to actions
            actions.append(action_name)

        # Add List to Data Frame as column
        self.df['action'] = actions


    def generate_dataset(self, destination_path):
        f = open(destination_path, 'w')
        for index, row in self.df.iterrows():
            tmp_str = ''
            for feature in self.features:
                tmp_str += feature + str(int(row[feature])) + ' '
            tmp_str = tmp_str.strip()
            tmp_str += '\t'
            tmp_str += row['action']
            tmp_str += '\n'
            f.write(tmp_str)
        f.close()




    
