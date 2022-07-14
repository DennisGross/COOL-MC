
# Import AdversarialAttack
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
# import numpy
import numpy as np

class DDA(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,fixed_value,feature_name" and store each component into a variable
        """
        attack_name, fixed_value, feature_name = attack_config_str.split(',')
        self.attack_name = attack_name
        self.fixed_value = int(fixed_value)
        self.feature_name = feature_name
        self.feature_index = self.state_mapper.mapper[str(feature_name)]

    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        state[self.feature_index] = self.fixed_value
        return state





        
        
        


