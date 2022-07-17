from common.adversarial_attacks.adversarial_attack import AdversarialAttack
import numpy as np

class RandomAttack(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        """
        Parse the string attack_config_str "attack_name,magnitude" and store each component into a variable
        """
        attack_name, magnitude = attack_config_str.split(',')
        self.attack_name = attack_name
        self.magnitude = int(magnitude)


    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        """
        Generate an integer numpy array called a of length state.
        Make sure that the l^1-norm of a is less equal than the magnitude.
        Randomly assign each element in a to be either positive or negative.
        Add a to the state.
        Return the adversarial state.
        """
        if self.already_attacked(state):
            return state + self.attack_buffer[str(state)]
        else:
            a = np.random.randint(0, self.magnitude, state.shape)
            while np.linalg.norm(a, ord=1) > self.magnitude:
                a = np.random.randint(0, self.magnitude, state.shape)
            # Randomly multiply each element in a with either -1 or 1.
            a = a * np.random.choice([-1, 1], size=state.shape)
            # Cast array a from float to int32
            attack = a.astype(np.int32)
            self.update_attack_buffer(state, attack)
            # Add a to the state.
            state += attack
            return state
        
    
        

        

