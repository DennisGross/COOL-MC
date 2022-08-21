
# Import AdversarialAttack
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
# import numpy
import numpy as np

class FeatureSpace:

    def __init__(self, name, lower_bound, upper_bound, feature_index) -> None:
        self.name = name
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.feature_index = feature_index

    def check_state(self, state):
        return self.is_in_range(int(state[self.feature_index]))

    def is_in_range(self, value):
        return self.lower_bound <= value <= self.upper_bound

    @staticmethod
    def parse_feature_space(raw_str, state_mapper):
        # featurename=lower_bound:uppper_bound
        parts = raw_str.split('=')
        feature_name = parts[0]
        lower_bound, upper_bound = parts[1].split(':')
        feature_index = state_mapper.mapper[str(feature_name)]
        return FeatureSpace(feature_name, lower_bound, upper_bound, feature_index)

class SDDA(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,fixed_value,feature_name,feature_assignments" and store each component into a variable
        """
        print(attack_config_str)
        attack_name, fixed_value, feature_name, feature_spaces = attack_config_str.split(',')
        self.feature_spaces = []
        for feature_space in feature_spaces.split(';'):
            feature_space = FeatureSpace.parse_feature_space(feature_space, self.state_mapper)
            self.feature_spaces.append(feature_space)
        self.attack_name = attack_name
        self.fixed_value = int(fixed_value)
        self.feature_name = feature_name


    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        all_true = True
        for feature_space in self.feature_spaces:
            value = state[feature_space.feature_index]  # Get feature assignment of current state
            all_true &= feature_space.is_in_range(value)
        
        #print("all_true:", all_true)
        if all_true:
            print("HEEEERE!!!!!!!!!!!!!!!!!!!!!!")
            state[self.state_mapper.mapper[str(self.feature_name)]] = self.fixed_value
        return state





        
        
        


