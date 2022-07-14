
# Import AdversarialAttack
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
# import numpy
import numpy as np

class SpecificDirectionAttack(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,magnitude,direction" and store each component into a variable
        """
        attack_name, magnitude, direction = attack_config_str.split(',')
        self.attack_name = attack_name
        self.magnitude = int(magnitude)
        self.direction = self.state_mapper.mapper[str(direction)]

    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        # Initialize a identity matrix called a of the same shape as state.
        a = np.identity(state.shape[0])
        # Pick randomly a row in a.
        a = a[self.direction, :]
        # Multiply a with magnitude and either a -1 or 1.
        a = a * self.magnitude * np.random.choice([-1, 1], size=state.shape)
        # Cast array a from float to int32
        a = a.astype(np.int32)
        # Add a to the state.
        state += a
        return state





        
        
        


