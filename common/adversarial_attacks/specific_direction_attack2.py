
# Import AdversarialAttack
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
from common.adversarial_attacks.specific_state_dda import FeatureSpace
# import numpy
import numpy as np

class SpecificDirectionAttack2(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,magnitude,direction,feature_spaces" and store each component into a variable
        """
        attack_name, magnitude, direction, feature_spaces_str = attack_config_str.split(',')
        self.attack_name = attack_name
        self.magnitude = int(magnitude)
        self.direction = self.state_mapper.mapper[str(direction)]
        self.feature_spaces = []
        for feature_space in feature_spaces_str.split(';'):
            feature_space = FeatureSpace.parse_feature_space(feature_space, self.state_mapper)
            self.feature_spaces.append(feature_space)
    
    

    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        if self.already_attacked(state):
            return state + self.attack_buffer[str(state)]
        else:
            all_true = True
            for feature_space in self.feature_spaces:
                value = state[feature_space.feature_index]  # Get feature assignment of current state
                all_true &= feature_space.is_in_range(value)
            if all_true:
                # Initialize a identity matrix called a of the same shape as state.
                a = np.identity(state.shape[0])
                # Pick the direction row in a.
                a = a[self.direction, :]
                # Multiply a with magnitude
                a = a * self.magnitude
                # Cast array a from float to int32
                attack = a.astype(np.int32)
                # Add a to the state.
                #print("=============")
                #print("state\t\t", state)
                #print("adv\t\t", attack)
                self.update_attack_buffer(state, attack)
                state += attack
                #print("adv-state\t", state)
        
        return state





        
        
        


