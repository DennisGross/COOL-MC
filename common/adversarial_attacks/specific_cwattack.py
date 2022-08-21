
# Import AdversarialAttack
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
from common.adversarial_attacks.specific_state_dda import FeatureSpace
# import numpy
import numpy as np

class SpecificCWAttack(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,direction,lower_bound,upper_bound,feature_assignment(featurename:value)" and store each component into a variable
        """
        print(attack_config_str)
        attack_name, direction, lower_bound, upper_bound, raw_feature_assignment = attack_config_str.split(',')
        self.attack_name = attack_name
        self.direction = self.state_mapper.mapper[str(direction)]
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        print(raw_feature_assignment)
        self.feature_space = FeatureSpace.parse_feature_space(raw_feature_assignment, self.state_mapper)

    

    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        if self.feature_space.check_state(state):
            if self.already_attacked(state):
                return state + self.attack_buffer[str(state)]
            else:
                # Get action index
                action_index = rl_agent.select_action(state, deploy=True)
                original_feature_assignment = state[self.direction]
                # Upper bound direction
                upper_distance = 0
                success1 = False
                for scale1 in range(original_feature_assignment, self.upper_bound):
                    a = np.identity(state.shape[0])
                    a = a[self.direction, :]
                    a = a * scale1
                    attack = a.astype(np.int32)
                    upper_distance +=1
                    if rl_agent.select_action(state+attack) != action_index:
                        success1 = True
                        break
                lower_distance = 0
                own_range = list(reversed(list(range(self.lower_bound, original_feature_assignment))))
                success2 = False
                for scale2 in own_range:
                    a2 = np.identity(state.shape[0])
                    a2 = a2[self.direction, :]
                    a2 = a2 * scale2
                    attack2 = a2.astype(np.int32)
                    lower_distance +=1
                    if rl_agent.select_action(state+attack2) != action_index:
                        success2 = True
                        break

                if upper_distance <= lower_distance and success1 and success2:
                    self.update_attack_buffer(state, attack)
                    state += attack
                elif lower_distance < upper_distance and success1 and success2:
                    self.update_attack_buffer(state, attack2)
                    state += attack2
                    
                elif success1:
                    self.update_attack_buffer(state, attack)
                    state += attack
                elif success2:
                    self.update_attack_buffer(state, attack2)
                    state += attack2
                else:
                    pass

                # Initialize a identity matrix called a of the same shape as state.
                #a = np.identity(state.shape[0])
                # Pick the direction row in a.
                #a = a[self.direction, :]
                # Multiply a with scale
                #a = a * self.magnitude
                # Cast array a from float to int32
                #attack = a.astype(np.int32)
                # Add a to the state.
                #print("=============")
                #print("state\t\t", state)
                #print("adv\t\t", attack)
                #self.update_attack_buffer(state, attack)
                #state += attack
                #print("adv-state\t", state)
        
        return state





        
        
        


