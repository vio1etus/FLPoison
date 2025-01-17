import time
from functools import wraps
import importlib
import os
import logging
import random
import numpy as np
import torch


def setup_logger(logger_name, log_file, level=logging.INFO, stream=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fileHandler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger


def actor(category, *attributes):
    """class decorator for categorizing attackers, data poisoners (backdoor, others), and model poisoners (omniscient, non_omniscient, others)
    """
    def decorator(cls):
        # key is the actor, value is the attributes of the actor
        categories = {"benign": ['always', 'temporary'],
                      "attacker": ['data_poisoning', 'model_poisoning', "non_omniscient", "omniscient"]}

        if not set(attributes).issubset(set(categories[category])):
            raise ValueError(
                "Invalid sub-category. Please change or add the sub-category.")
        cls._category = category
        cls._attributes = attributes
        # change __init__ method to realize it in objects
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Check if the object has already been decorated to avoid redundant decoration due to inheritance.
            # When attacker init, one class, two different object (attacker, super init)
            if not hasattr(self, "_decorated"):
                self.category = cls._category
                self.attributes = cls._attributes
                # Mark self object as decorated to prevent re-decoration on inherited classes
                self._decorated = True
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator


def import_all_modules(current_dir, depth=0, depth_prefix=None):
    pkg_name = depth_prefix + "." + os.path.basename(
        current_dir) if depth else os.path.basename(current_dir)

    for filename in os.listdir(current_dir):
        # filter our __init__.py and non-python files
        if filename.endswith(".py") and (filename != "__init__.py"):
            module_name = filename[:-3]  # remove ".py"
            importlib.import_module(
                f".{module_name}", package=pkg_name)


class Register(dict):
    """Register class is a dict class with 2 functions: 1. serve as the registry 2. register the callable object, function, to this registry
    """

    def __init__(self, *args, **kwargs):
        # init the dict class, so that it can be used as a normal dict
        super().__init__(*args, **kwargs)

    def __call__(self, target):
        def register_item(name, func):
            self[name] = func
            return func

        # if target is a string, return a function to receive the callable object. @register('name')
        if isinstance(target, str):
            ret_func = (lambda x: register_item(target, x))
        # if target is a callable object, then register it, and return it, @register
        elif callable(target):
            ret_func = register_item(target.__name__, target)
        return ret_func


def print_filtered_args(args, logger):
    args_dict = vars(args)
    filtered_args = {k: v for k, v in args_dict.items() if k not in [
        'attacks', 'defenses', 'logger']}
    msg = ', '.join([f'{key}: {value}' for key,
                     value in filtered_args.items()]) + '\n'
    logger.info(msg)


def avg_value(x):
    return sum(x) / len(x)


def setup_seed(seed):
    """
    fix all possible randomness for reproduction
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def frac_or_int_to_int(frac_or_int, total_num):
    return int(frac_or_int) if frac_or_int >= 1 else int(frac_or_int * total_num)


class TimingRecorder:
    def __init__(self, id, output_file):
        self.id = id
        # record the duration and number of call of func
        self.global_timings = {}
        time_log_path = output_file.replace(
            "logs/", "logs/time_logs/", 1)[:-4]+'.log'
        self.logger = setup_logger(
            __name__, time_log_path, level=logging.INFO)
        self.client_log_flag = False
        epoch_level = False
        self.record_epochs = [2, 4, 6, 8, 10, 20,
                              50, 100, 150, 200] if epoch_level else []

    def timing_decorator(self, func):
        """decorator to record the running time of each function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # start timer
            result = func(*args, **kwargs)  # call the function
            end_time = time.time()  # end timer
            duration = end_time - start_time

            # update the global_timings with the duration
            method_name = func.__name__
            if method_name not in self.global_timings:
                self.global_timings[method_name] = {
                    "total_time": 0, "calls": 0}
            self.global_timings[method_name]["total_time"] += duration
            self.global_timings[method_name]["calls"] += 1

            if self.client_log_flag:
                # log data during training
                self.report(f"Worker ID {self.id}")

            # for client
            self.client_log_flag = True if method_name == "local_training" and self.global_timings[
                method_name]["calls"] in self.record_epochs else False

            if method_name == "aggregation" and self.global_timings[method_name]["calls"] in self.record_epochs:
                # log data during training
                self.report(f"Worker ID {self.id}")
            return result
        return wrapper

    def get_average_time(self, func_name):
        """get average running time of func_name for all epoch"""
        if func_name in self.global_timings:
            total_time = self.global_timings[func_name]["total_time"]
            calls = self.global_timings[func_name]["calls"]
            return total_time / calls if calls > 0 else 0
        return 0

    def report(self, id=None):
        for method_name, stats in self.global_timings.items():
            avg_time = stats["total_time"] / stats["calls"]
            self.logger.info(
                f"{id}, {method_name} averge time: {avg_time:.6f} s, call time: {stats['calls']}")
