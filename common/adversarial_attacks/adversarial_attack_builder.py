# Import all adversarial attacks
from common.adversarial_attacks.adversarial_attack import AdversarialAttack
from common.adversarial_attacks.random_direction_attack import RandomDirectionAttack
from common.adversarial_attacks.specific_direction_attack import SpecificDirectionAttack
from common.adversarial_attacks.random_attack import RandomAttack
from common.adversarial_attacks.dda import DDA

"""
This class builds an adversarial attack based on the attack_config_str.
"""
class AdversarialAttackBuilder:
    
    def build_adversarial_attack(self, state_mapper, attack_config_str: str) -> AdversarialAttack:
        """
        Build an adversarial attack based on the attack_config_str.
        :param state_mapper: The state mapper.
        :param attack_config_str: The attack configuration.
        :return: The adversarial attack.
        """
        attack_name = attack_config_str.split(',')[0]
        if attack_name == "random":
            return RandomAttack(state_mapper)
        elif attack_name == "random_direction":
            return RandomDirectionAttack(state_mapper, attack_config_str)
        elif attack_name == "specific_direction":
            return SpecificDirectionAttack(state_mapper, attack_config_str)
        elif attack_name == "dda":
            return DDA(state_mapper, attack_config_str)
        else:
            return None