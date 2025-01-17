from global_utils import import_all_modules, Register
import os

aggregator_registry = Register()

# import all files in the directory, so that the registry decorator can be read and used

# os.path.dirname(__file__) get the current directory path
import_all_modules(os.path.dirname(__file__))
all_aggregators = list(aggregator_registry.keys())


def get_aggregator(name):
    return aggregator_registry[name]
