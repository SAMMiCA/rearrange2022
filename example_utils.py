from typing import List, Optional, Dict, Any, DefaultDict, Union, cast
import os
import torch
import numbers
import random
import pdb
import sys
from collections import defaultdict
import numpy as np
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.utils import spaces_utils as su
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import VectorSampledTasks, COMPLETE_TASK_METRICS_KEY, SingleProcessVectorSampledTasks
from allenact.algorithms.onpolicy_sync.storage import ExperienceStorage, RolloutStorage
from allenact.base_abstractions.misc import RLStepResult, Memory, ActorCriticOutput, GenericAbstractLoss


TRAIN_MODE_STR = "train"
VALID_MODE_STR = "valid"
TEST_MODE_STR = "test"


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            

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
    

def get_sampler_fn_args(
    config: ExperimentConfig, 
    machine_params: MachineParams,
    mode: str, 
    device: torch.device,
    num_procs: Optional[int]= None,
    is_distributed: bool = False,
    num_samplers_per_worker: Optional[List[int]] = None,
    worker_id: Optional[int] = None,
    seeds: Optional[List[int]] = None,
):
    sampler_devices = machine_params.sampler_devices

    if mode == TRAIN_MODE_STR:
        fn = config.train_task_sampler_args
    elif mode == VALID_MODE_STR:
        fn = config.valid_task_sampler_args
    elif mode == TEST_MODE_STR:
        fn = config.test_task_sampler_args
    else:
        raise NotImplementedError(
            f"mode must be one of ('train', 'valid', 'test')"
        )
    
    if is_distributed:
        assert (
            num_samplers_per_worker is not None
            and worker_id is not None
        )
        num_samplers = len(num_samplers_per_worker)
        total_processes = sum(num_samplers_per_worker)
        process_offset = sum(num_samplers_per_worker[:worker_id])
    else:
        assert num_procs is not None
        num_samplers = num_procs
        total_processes = num_procs
        process_offset = 0

    device = torch.device(device)
    
    sampler_devices_as_ints: Optional[List[int]] = None
    if mode == TEST_MODE_STR and device.index is not None:
        sampler_devices_as_ints = [device.index]
    elif sampler_devices is not None:
        sampler_devices_as_ints = [
            -1 if sd.index is None else sd.index
            for sd in sampler_devices
        ]
    
    return [
        fn(
            process_ind=process_offset + it,
            total_processes=total_processes,
            devices=sampler_devices_as_ints,
            seeds=seeds,
        )
        for it in range(num_samplers)
    ]


# initialize storage
def to_tensor(v) -> torch.Tensor:
    """Return a torch.Tensor version of the input.

    # Parameters

    v : Input values that can be coerced into being a tensor.

    # Returns

    A tensor version of the input.
    """
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(
            v, dtype=torch.int64 if isinstance(v, numbers.Integral) else torch.float
        )


def preprocess_observations(
    sensor_preprocessor_graph,
    batched_observations
):
    if sensor_preprocessor_graph is None:
        return batched_observations
    return sensor_preprocessor_graph.get_observations(batched_observations)


def batch_observations(
    observations: List[Dict],
    device: Optional[torch.device] = None,
):
    def dict_from_observation(
        observation: Dict[str, Any]
    ) -> Dict[str, Union[Dict, List]]:
        batch_dict: DefaultDict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                batch_dict[sensor] = dict_from_observation(observation[sensor])
            else:
                batch_dict[sensor].append(to_tensor(observation[sensor]))

        return batch_dict

    def fill_dict_from_observations(
        input_batch: Any, observation: Dict[str, Any]
    ) -> None:
        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                fill_dict_from_observations(input_batch[sensor], observation[sensor])
            else:
                input_batch[sensor].append(to_tensor(observation[sensor]))

    def dict_to_batch(input_batch: Any) -> None:
        for sensor in input_batch:
            if isinstance(input_batch[sensor], Dict):
                dict_to_batch(input_batch[sensor])
            else:
                input_batch[sensor] = torch.stack(
                    [batch.to(device=device) for batch in input_batch[sensor]], dim=0
                )

    if len(observations) == 0:
        return cast(Dict[str, Union[Dict, torch.Tensor]], observations)

    batch = dict_from_observation(observations[0])

    for obs in observations[1:]:
        fill_dict_from_observations(batch, obs)

    dict_to_batch(batch)

    return cast(Dict[str, Union[Dict, torch.Tensor]], batch)


def remove_paused(
    observations: List[Dict],
    vector_tasks: VectorSampledTasks,
    device: Optional[torch.device] = None,
):
    paused, keep, running = [], [], []
    for it, obs in enumerate(observations):
        if obs is None:
            paused.append(it)
        else:
            keep.append(it)
            running.append(obs)
    
    for p in reversed(paused):
        vector_tasks.pause_at(p)

    batch = batch_observations(running, device=device)

    return len(paused), keep, batch


def set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_seed(nprocs: int, init_seed: Optional[int]) -> List[int]:
    rstate = None
    if init_seed is not None:
        rstate = random.getstate()
        random.seed(init_seed)

    seeds = [random.randint(0, (2 ** 31) - 1) for _ in range(nprocs)]
    if init_seed is not None:
        random.setstate(rstate)

    return seeds


def collect_step_across_all_task_samplers(
    vector_tasks: VectorSampledTasks,
    actor_critic: ActorCriticModel,
    sensor_preprocessor_graph,
    rollout_storage_uuid: str,
    uuid_to_storage: Dict[str, ExperienceStorage],
    dist_wrapper_class: Optional[type] = None,
    device: torch.device = "cpu",
    manual_step_inputs: bool = False,
):
    rollout_storage = cast(RolloutStorage, uuid_to_storage[rollout_storage_uuid])
    actions, actor_critic_output, memory, _ = act(
        actor_critic=actor_critic,
        rollout_storage=rollout_storage,
        dist_wrapper_class=dist_wrapper_class
    )
    if manual_step_inputs: 
        action_idx = int(input(f'action index = '))
        actions = torch.ones_like(actions) * action_idx

    flat_actions = su.flatten(actor_critic.action_space, actions)

    assert len(flat_actions.shape) == 3, (
        "Distribution samples must include step and task sampler dimensions [step, sampler, ...]. The simplest way"
        "to accomplish this is to pass param tensors (like `logits` in a `CategoricalDistr`) with these dimensions"
        "to the Distribution."
    )

    outputs: List[RLStepResult] = vector_tasks.step(
        su.action_list(actor_critic.action_space, flat_actions)
    )

    for step_result in outputs:
        if (
            step_result.info is not None
            and COMPLETE_TASK_METRICS_KEY in step_result.info
        ):
            # TODO:?
            del step_result.info[COMPLETE_TASK_METRICS_KEY]
    
    rewards: Union[List, torch.Tensor]
    observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

    rewards = torch.tensor(
        rewards, dtype=torch.float, device=device,
    )

    if len(rewards.shape) == 1:
        # Rewards are of shape [sampler, ]
        rewards = rewards.unsqueeze(-1)
    elif len(rewards.shape) > 1:
        raise NotImplementedError()

    masks = (
        1.0 
        - torch.tensor(
            dones, dtype=torch.float32, device=device,
        )
    ).view(
        -1, 1 
    )   # [sampler, 1]

    npaused, keep, batch = remove_paused(
        observations=observations,
        vector_tasks=vector_tasks,
        device=device
    )

    if npaused > 0:
        for s in uuid_to_storage.values():
            if isinstance(s, RolloutStorage):
                s.sampler_select(keep)
            
    to_add_to_storage = dict(
        observations=preprocess_observations(
            sensor_preprocessor_graph=sensor_preprocessor_graph, 
            batched_observations=batch
        ) if len(keep) > 0 else batch,
        memory=active_memory(memory, keep),
        actions=flat_actions[0, keep],
        action_log_probs=actor_critic_output.distributions.log_prob(actions)[0, keep],
        value_preds=actor_critic_output.values[0, keep],
        rewards=rewards[keep],
        masks=masks[keep],
    )
    for storage in uuid_to_storage.values():
        storage.add(**to_add_to_storage)

    return npaused


def act(
    actor_critic: ActorCriticModel,
    rollout_storage: RolloutStorage,
    dist_wrapper_class: Optional[type] = None,
    deterministic_agents: bool = False,
):
    with torch.no_grad():
        agent_input = rollout_storage.agent_input_for_next_step()
        actor_critic_output, memory = actor_critic(**agent_input)

        distr = actor_critic_output.distributions
        if dist_wrapper_class is not None:
            distr = dist_wrapper_class(distr=distr, obs=agent_input["observations"])

        actions = distr.sample() if not deterministic_agents else distr.mode()

    return actions, actor_critic_output, memory, agent_input["observations"]


def active_memory(memory, keep):
    return memory.sampler_select(keep) if memory is not None else memory


def advantage_stats(advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mean": advantages.mean(), "std": advantages.std()}