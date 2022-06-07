from typing import Any, Optional, List, Tuple, Type, Dict, Union, cast
import os
import json
import inspect
import importlib

from setproctitle import setproctitle as ptitle

import torch
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.algorithms.onpolicy_sync.runner import CONFIG_KWARGS_STR
from allenact.utils.system import get_logger, init_logging, HUMAN_LOG_LEVELS

from test_scripts.test_class import TestEngine, TestRunner


NUM_PROCESSES = 2
MP_START_METHOD = "forkserver"
DEVICE = "cuda"
MODE = "train"
SEED = 0

torch.autograd.set_detect_anomaly(True)


def _config_source(config_type: Type) -> Dict[str, str]:
    if config_type is ExperimentConfig:
        return {}

    try:
        module_file_path = inspect.getfile(config_type)
        module_dot_path = config_type.__module__
        sources_dict = {module_file_path: module_dot_path}
        for super_type in config_type.__bases__:
            sources_dict.update(_config_source(super_type))

        return sources_dict
    except TypeError as _:
        return {}


def find_sub_modules(path: str, module_list: Optional[List] = None):
    if module_list is None:
        module_list = []

    path = os.path.abspath(path)
    if path[-3:] == ".py":
        module_list.append(path)
    elif os.path.isdir(path):
        contents = os.listdir(path)
        if any(key in contents for key in ["__init__.py", "setup.py"]):
            new_paths = [os.path.join(path, f) for f in os.listdir(path)]
            for new_path in new_paths:
                find_sub_modules(new_path, module_list)
    return module_list


def load_config(
    experiment_base: str,
    experiment: str,
):
    assert os.path.exists(
        experiment
    ), f"The path '{experiment_base}' does not seem to exist (your current working directory is '{os.getcwd()}')."

    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(experiment_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = experiment
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            f"Could not import experiment '{module_path}', are you sure this is the right path?"
            f" Possibly relevant files include {relevant_submodules}."
            f" Note that the experiment must be reachable along your `PYTHONPATH`, it might"
            f" be helpful for you to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your"
            f" project's top level directory."
        ) from e
    
    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config_kwargs = {}
    config = experiments[0](**config_kwargs)
    sources = _config_source(config_type=experiments[0])
    sources[CONFIG_KWARGS_STR] = json.dumps(config_kwargs)
    return config, sources


def main():
    init_logging("info")
    get_logger().info("Starting Main Process")

    ptitle(f"Testing...")
    cfg, src = load_config(
        experiment_base=".",
        experiment="experiments/sensor_test.py"
    )

    TestRunner(
        config=cfg,
        seed=SEED,
        mode=MODE,
        mp_ctx=None,
        multiprocessing_start_method=MP_START_METHOD,
    ).start_script()


if __name__ == "__main__":
    main()