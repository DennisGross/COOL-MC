import numpy as np
class AdversarialAttack:

    def __init__(self, state_mapper, attack_config_str:str) -> None:
        """
        Initialize the attack.
        :param state_mapper: The state mapper.
        :param attack_config: The attack configuration.
        """
        self.state_mapper = state_mapper
        self.attack_config_str = attack_config_str
        self.attack_buffer = {}

    def parse_attack_config(self, attack_config_str:str) -> None:
        """
        Parse the attack configuration.
        :param attack_config_str: The attack configuration.
        """
        raise NotImplementedError()

    def attack(self, rl_agent, state:np.ndarray, current_action: str = "") -> np.ndarray:
        """
        Perform the attack.
        :param state: The state.
        :return: The adversarial state.
        """
        raise NotImplementedError()


