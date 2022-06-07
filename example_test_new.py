#%%
from typing import List, Optional, Dict, Any, DefaultDict, Union, cast
import os
import numpy as np
import torch
import importlib
import inspect
import multiprocessing as mp

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
from custom.hlsm.voxel_grid import DefaultGridParameters
from experiments.sensor_test import SensorTestExperimentConfig
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from example_utils import advantage_stats, collect_step_across_all_task_samplers, find_sub_modules, get_sampler_fn_args, to_tensor, batch_observations, set_seed, worker_seed


NUM_PROCESSES = 2
MP_START_METHOD = "forkserver"
DEVICE = "cuda"
MODE = "valid"
SEED = 0


experimental_base = ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
rel_base_dir = os.path.relpath(experimental_base, os.getcwd())
rel_base_dot_path = rel_base_dir.replace("/", ".")
if rel_base_dot_path == ".":
    rel_base_dot_path = ""

exp_dot_path = "experiments/sensor_test.py"
if exp_dot_path[-3:] == ".py":
    exp_dot_path = exp_dot_path[:-3]
exp_dot_path = exp_dot_path.replace("/", ".")

module_path = (
    f"{rel_base_dot_path}.{exp_dot_path}"
    if len(rel_base_dot_path) != 0
    else exp_dot_path
)
import importlib
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
config = experiments[0]()
machine_params = config.machine_params(MODE)
create_model_kwargs = {}
sensor_preprocessor_graph = None
if machine_params.sensor_preprocessor_graph is not None:
    sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(DEVICE)
    create_model_kwargs["sensor_preprocessor_graph"] = sensor_preprocessor_graph

# import pdb; pdb.set_trace()     # python - 1997 MiB
create_model_kwargs["mode"] = MODE
create_model_kwargs["device"] = DEVICE

set_seed(SEED)
actor_critic = config.create_model(**create_model_kwargs).to(DEVICE)

training_pipeline = config.training_pipeline()
# import pdb; pdb.set_trace()     # python - 2339 MiB
cur_stage_training_settings = training_pipeline.current_stage.training_settings
rollout_storage_uuid = training_pipeline.rollout_storage_uuid
uuid_to_storage = training_pipeline.current_stage_storage
rollout_storage = uuid_to_storage[rollout_storage_uuid]

############################################################################################
mp_ctx = mp.get_context(MP_START_METHOD)
seeds = worker_seed(
    nprocs=NUM_PROCESSES,
    init_seed=SEED
)
sampler_fn_args = get_sampler_fn_args(
    config=config,
    machine_params=machine_params,
    mode=MODE, 
    device=DEVICE,
    num_procs=NUM_PROCESSES,
    seeds=seeds,
)
if __name__ == "__main__":
    vector_tasks = VectorSampledTasks(
        make_sampler_fn=config.make_sampler_fn,
        sampler_fn_args=sampler_fn_args,
        multiprocessing_start_method=None,
        mp_ctx=mp_ctx
    )

    import pdb; pdb.set_trace()     
    # python - 2339 MiB / VectorSampledTask: 0 - 2339 MiB / VectorSampledTask: 0 - 2339 MiB / 
    # ...bac8ad630462b5c0c4cea1f5f - 125 MiB x 4
    num_datapoints = vector_tasks.command('sampler_attr', ['total_unique'] * NUM_PROCESSES)
    obs_spaces = vector_tasks.attr(['observation_space'] * NUM_PROCESSES)
    act_spaces = vector_tasks.attr(['action_space'] * NUM_PROCESSES)
    num_steps_taken = vector_tasks.call(['num_steps_taken'] * NUM_PROCESSES)
    
    obs = vector_tasks.get_observations()
    batch_obs = batch_observations(obs)
    preproc_obs = sensor_preprocessor_graph.get_observations(batch_obs) if sensor_preprocessor_graph else batch_obs

    import pdb; pdb.set_trace()
    # python - 2687 MiB / VectorSampledTask: 0 - 2339 MiB / VectorSampledTask: 0 - 2339 MiB / 
    # ...bac8ad630462b5c0c4cea1f5f - 125 MiB x 4

    rollout_storage.to(DEVICE)
    rollout_storage.initialize(
        observations=preproc_obs,
        num_samplers=NUM_PROCESSES,
        recurrent_memory_specification=actor_critic.recurrent_memory_specification,
        action_space=actor_critic.action_space,
    )

    import pdb; pdb.set_trace()
    # python - 2687 MiB / VectorSampledTask: 0 - 2339 MiB / VectorSampledTask: 0 - 2339 MiB / 
    # ...bac8ad630462b5c0c4cea1f5f - 125 MiB x 4
    
    manual_input = True
    for step in range(cur_stage_training_settings.num_steps):
        npaused = collect_step_across_all_task_samplers(
            vector_tasks=vector_tasks,
            actor_critic=actor_critic,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
            rollout_storage_uuid=rollout_storage_uuid,
            uuid_to_storage=uuid_to_storage,
            device=DEVICE,
            manual_step_inputs=manual_input,
        )
    
# %%
# with torch.no_grad():
#     actor_critic_output, _ = actor_critic(
#         **rollout_storage.agent_input_for_next_step()
#     )

# before_update_info = dict(
#     next_value=actor_critic_output.values.detach(),
#     use_gae=cur_stage_training_settings.use_gae,
#     gamma=cur_stage_training_settings.gamma,
#     tau=cur_stage_training_settings.gae_lambda,
#     adv_stats_callback=advantage_stats,
# )

# rollout_storage.before_updates(**before_update_info)
# batch_iterator = rollout_storage.batched_experience_generator(
#     num_mini_batch=cur_stage_training_settings.num_mini_batch
# )

# batch = next(batch_iterator)
# # %%
# actor_critic_output_for_batch, _ = actor_critic(
#     observations=batch["observations"],
#     memory=batch["memory"],
#     prev_actions=batch["prev_actions"],
#     masks=batch["masks"],
# )
# # %%
