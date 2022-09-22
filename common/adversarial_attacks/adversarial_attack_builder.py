from common.adversarial_attacks.adversarial_attack import AdversarialAttack
from common.adversarial_attacks.robustness_attacker import RobustnessAttacker
class AdversarialAttackBuilder:
    
    def build_adversarial_attack(self, state_mapper, attack_config_str: str) -> AdversarialAttack:
        """
        Build an adversarial attack based on the attack_config_str.
        :param state_mapper: The state mapper.
        :param attack_config_str: The attack configuration.
        :return: The adversarial attack.
        """
        attack_name = attack_config_str.split(',')[0]
        if attack_name == "robustness":
            return RobustnessAttacker(state_mapper, attack_config_str)
        else:
            return None