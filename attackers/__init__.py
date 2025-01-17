import os
from global_utils import import_all_modules, Register

attacker_registry = Register()
# import all files in the directory, so that the registry decorator can be read and used
import_all_modules(os.path.dirname(__file__))

# pure data poisoning attacks
data_poisoning_attacks = [name for name in attacker_registry.keys(
) if "data_poisoning" in attacker_registry[name]._attributes]

# hybrid attackers with data poisoning and model poisoning capabilities simultaneously
hybrid_attacks = [name for name in attacker_registry.keys() if all(
    attr in attacker_registry[name]._attributes for attr in ["model_poisoning", "data_poisoning"])]

# get pure model poisoning attacks
model_poisoning_attacks = [name for name in attacker_registry.keys(
) if "data_poisoning" not in attacker_registry[name]._attributes]


def get_attacker_handler(name):
    assert name != "NoAttack", f"NoAttack should not specify num_adv argument"
    return attacker_registry[name]
