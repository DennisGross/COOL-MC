import numpy as np

class PartialObservableManager:

    def __init__(self, prism_file_path, state_mapper) -> None:
        self.prism_file_path = prism_file_path
        # Read text file
        with open(self.prism_file_path, 'r') as file:
            self.prism_lines = file.readlines()
        self.state_mapper = state_mapper
        self.all_agent_variables = self.extract_partial_observable_state_variables(self.prism_lines)
    
    def extract_partial_observable_state_variables(self, path):
        """Extract the partial observable state variables from the PRISM file.
        
        Args:
            path (str): Path to the PRISM file
        
        Returns:
            list: List of partial observable state variables
        """
        assert isinstance(path, list)
        all_agent_variables = []
        agent_counter=0
        for i, line in enumerate(path):
            if line.startswith('//AGENT'):
                agent_variables = []
                line_parts = line.split(":")
                variable_parts = line_parts[1].split(" ")
                for j, variable_part in enumerate(variable_parts):
                    if variable_part.strip() != '':
                        agent_variables.append(variable_part)
                agent_counter+=1
                all_agent_variables.append(agent_variables)
        return all_agent_variables

    def get_observation_dimension_for_agent_idx(self, idx):
        print(len(self.all_agent_variables[idx]))
        print(self.all_agent_variables[idx])
        return len(self.all_agent_variables[idx])

    def get_observation(self, full_state, agent_index):
        observation = []
        for variable_name in self.all_agent_variables[agent_index]:
            if variable_name.strip() != '':
                variable_idx = self.state_mapper.mapper[variable_name]
                observation.append(full_state[variable_idx])
        return np.array(observation)