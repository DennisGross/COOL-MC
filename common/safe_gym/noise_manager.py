from typing import List
import numpy as np
import random
import math

def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

class NoiseVariable():

    def __init__(self, name, lower_bound, upper_bound, percentage):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.percentage = percentage

    def set_idx(self, idx: int):
        """Set index of state variable for abstraction.

        Args:
            idx (int): index
        """
        assert isinstance(idx, int)
        self.idx = idx

    def add_noise_to(self, value):
        domain_size = abs(self.upper_bound-self.lower_bound)
        noise = random.randint(0,normal_round(self.percentage*domain_size))
        if random.randint(0,1) == 0:
            noise *= (-1)
        value = value + noise
        if value<self.lower_bound:
            value = self.lower_bound
        elif value>self.upper_bound:
            value = self.upper_bound
        #print(value-noise,value,noise)
        return value


    def get_idx(self) -> int:
        """Get index of state variable for abstractoin

        Returns:
            int: Index
        """
        return self.idx

class NoiseManager:

    def __init__(self, state_var_mapper, noise_input) -> None:
        self.is_active = self.__is_active(noise_input)
        self.noise_variables = []
        self.state_var_mapper = state_var_mapper.mapper
        if self.is_active:
            noise_input = noise_input[1:]
            self.noise_variables = self.parse_random_variables(noise_input)

            for i in range(len(self.noise_variables)):
                self.noise_variables[i].set_idx(self.state_var_mapper[self.noise_variables[i].name])


    def __is_active(self, noise_input):
        if len(noise_input) > 0:
            if noise_input[0] == '#':
                return True
            else:
                return False
        return False

    @staticmethod
    def parse_random_variables(noise_input: str) -> List:
        """Parse random variables fro abstraction input string.

        Args:
            abstraction_input ([str]): Abstraction input: VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX]

        Raises:
            ValueError: Abstraction Input is the wrong format

        Returns:
            List: List of AStateVariables
        """
        assert isinstance(noise_input, str)
        # VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX]
        noise_variables = []
        for abstraction_variable_str in noise_input.split(","):
            name = abstraction_variable_str.split('=')[0]
            start_interval = abstraction_variable_str.find('[')
            end_interval = abstraction_variable_str.find(']')
            interval = abstraction_variable_str[(
                start_interval+1):end_interval]

            if interval.count(';') == 2:
                parts = abstraction_variable_str[(
                    start_interval+1):end_interval].split(';')
                lower_bound = int(parts[0])
                upper_bound = int(parts[1])
                percentage = float(parts[2])
                noise_variable = NoiseVariable(
                    name, lower_bound, upper_bound, percentage)
                noise_variables.append(noise_variable)
            else:
                raise ValueError(
                    "Abstraction Input is the wrong format (VAR=[LOW;MAX;percentage], ...)")
        assert isinstance(noise_variables, list)
        return noise_variables


    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Apply the abstraction on the state.

        Args:
            state (np.ndarray): raw state

        Returns:
            np.ndarray: abstracted state
        """
        assert isinstance(state, np.ndarray)
        for noise_variable in self.noise_variables:
            idx = noise_variable.get_idx()
            state[idx] = noise_variable.add_noise_to(state[idx])
        assert isinstance(state, np.ndarray)
        return state