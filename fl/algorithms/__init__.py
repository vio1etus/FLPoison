from global_utils import import_all_modules, Register
import os

algorithm_registry = Register()
# import all files in the directory, so that the registry decorator can be read and used
# os.path.dirname(__file__) get the current directory path
import_all_modules(os.path.dirname(__file__), 1, "fl")
all_algorithms = list(algorithm_registry.keys())


def get_algorithm_handler(name):
    return algorithm_registry[name]
