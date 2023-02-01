import numbers
from typing import Dict, List, Optional, Union
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from torch.optim.lr_scheduler import _LRScheduler
except (ImportError, ModuleNotFoundError):
    raise ImportError("`_LRScheduler` was not found in `torch.optim.lr_scheduler`")

from allenact.utils.misc_utils import NumpyJSONEncoder
from allenact.utils.system import get_logger, init_logging, _new_logger
from allenact.utils.tensor_utils import batch_observations, detach_recursively
from allenact.utils import spaces_utils as su
from allenact.base_abstractions.misc import (
    ActorCriticOutput,
    GenericAbstractLoss,
    Memory,
    RLStepResult,
)
from allenact.algorithms.onpolicy_sync.misc import TrackingInfo, TrackingInfoType
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import (
    COMPLETE_TASK_CALLBACK_KEY,
    COMPLETE_TASK_METRICS_KEY,
    SingleProcessVectorSampledTasks,
    VectorSampledTasks,
)
from allenact.algorithms.onpolicy_sync.storage import (
    ExperienceStorage,
    MiniBatchStorageMixin,
    RolloutStorage,
    StreamingStorageMixin,
)
from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from experiments.test_exp import ExpertTestExpConfig
from task_aware_rearrange.subtasks import IDX_TO_SUBTASK


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', type=str, default="train")
    parser.add_argument('--visualize', action='store_true', )

    args = parser.parse_args()

    return args

def advantage_stats(advantage: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "mean": advantage.mean(),
        "std": advantage.std(),
    }


if __name__ == "__main__":
    # init_logging("info")
    args = parse_args()
    stage = args.stage
    visualize = args.visualize

    device = torch.device(0)
    tracking_info_list: List[TrackingInfo] = []
    
    def get_sampler_fn_args(stage: str):
        if stage == "train":
            fn = ExpertTestExpConfig.train_task_sampler_args
        elif stage == "val":
            fn = ExpertTestExpConfig.valid_task_sampler_args
        elif stage == "test":
            fn = ExpertTestExpConfig.test_task_sampler_args
        else:
            raise NotImplementedError
        
        machine_params = ExpertTestExpConfig.machine_params(stage)
        total_processes = sum(machine_params.nprocesses)

        sampler_devices_as_ints = [0]

        return [
            fn(
                process_ind=it,
                total_processes=total_processes,
                devices=[it],
                seeds=None,
            )
            for it in range(total_processes)
        ]

    vector_tasks = VectorSampledTasks(
        make_sampler_fn=ExpertTestExpConfig.make_sampler_fn,
        sampler_fn_args=get_sampler_fn_args(stage=stage),
        callback_sensors=None,
        multiprocessing_start_method="forkserver",
        mp_ctx=None,
        max_processes=20,
        read_timeout=60,
    )
    single_process_metrics: List = []
    single_process_task_callback_data: List = []

    sensor_preprocessor_graph = ExpertTestExpConfig.create_preprocessor_graph(mode="train")().to(device)
    agent_model = ExpertTestExpConfig.create_model(**{"sensor_preprocessor_graph": sensor_preprocessor_graph}).to(device)
    # RolloutStorages
    training_pipeline = ExpertTestExpConfig.training_pipeline()
    rollout_storage = training_pipeline.rollout_storage
    uuid_to_storage = training_pipeline.current_stage_storage
    recurrent_memory_specification = agent_model.recurrent_memory_specification

    # For Training
    optimizer: optim.Optimizer = (
        training_pipeline.optimizer_builder(
            params=[p for p in agent_model.parameters() if p.requires_grad]
        )
    )
    lr_scheduler: Optional[_LRScheduler] = None
    if training_pipeline.lr_scheduler_builder is not None:
        lr_scheduler = training_pipeline.lr_scheduler_builder(
            optimizer=optimizer
        )

    # how_many_unique_datapoints = sum(
    #     vector_tasks.command("sampler_attr", ["total_unique"] * ExpertTestExpConfig.NUM_PROCESSES)
    # )

    observations = vector_tasks.get_observations() # Lists
    batch = batch_observations(observations, device=device)   # DefaultDict
    preprocessed_obs = sensor_preprocessor_graph.get_observations(batch)    # Dict

    rollout_storage.to(device)
    rollout_storage.set_partition(index=0, num_parts=1)
    rollout_storage.initialize(
        observations=preprocessed_obs,
        num_samplers=ExpertTestExpConfig.NUM_PROCESSES,
        recurrent_memory_specification=recurrent_memory_specification,
        action_space=agent_model.action_space,
    )

    while True:
        former_steps = training_pipeline.current_stage.steps_taken_in_stage
        former_storage_experiences = {
            k: v.total_experiences
            for k, v in training_pipeline.current_stage_storage.items()
        }

        step = -1
        # Collect data and store in RolloutStorage
        action_history = [[] for _ in range(ExpertTestExpConfig.NUM_PROCESSES)]
        subtask_history = [[] for _ in range(ExpertTestExpConfig.NUM_PROCESSES)]
        while step < training_pipeline.training_settings.num_steps - 1:
            step += 1

            try:
                with torch.no_grad():
                    agent_input = rollout_storage.agent_input_for_next_step()
                    ac_out, memory = agent_model(**agent_input)

                    distr = ac_out.distributions
                    actions = agent_input["observations"]["expert_action"][..., 0]
                    subtasks = agent_input["observations"]["expert_subtask"][..., 0]
                    for it in range(ExpertTestExpConfig.NUM_PROCESSES):
                        action_history[it].append(actions[0, it].item())
                        subtask_history[it].append(subtasks[0, it].item())
                
                flat_actions = su.flatten(agent_model.action_space, actions)    # (nsteps, nsamplers, 1)
                outputs: List[RLStepResult] = vector_tasks.step(
                    su.action_list(agent_model.action_space, flat_actions)
                )

                for step_result in outputs:
                    if step_result is not None:
                        if COMPLETE_TASK_METRICS_KEY in step_result.info:
                            single_process_metrics.append(
                                step_result.info[COMPLETE_TASK_METRICS_KEY]
                            )
                            del step_result.info[COMPLETE_TASK_METRICS_KEY]
                        if COMPLETE_TASK_CALLBACK_KEY in step_result.info:
                            single_process_task_callback_data.append(
                                step_result.info[COMPLETE_TASK_CALLBACK_KEY]
                            )
                            del step_result.info[COMPLETE_TASK_CALLBACK_KEY]

                rewards: Union[List, torch.Tensor]
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

                rewards = torch.tensor(
                    rewards, dtype=torch.float, device=device
                )

                if len(rewards.shape) == 1:
                    rewards = rewards.unsqueeze(-1) # [nsamplers,]
                elif len(rewards.shape) > 1:
                    raise NotImplementedError()

                masks = (
                    1.0
                    - torch.tensor(
                        dones, dtype=torch.float32, device=device
                    )
                ).view(-1, 1)   # [nsamplers, 1]

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
                preprocessed_obs = sensor_preprocessor_graph.get_observations(batch) if len(keep) > 0 else batch
                npaused = len(paused)

                if npaused > 0:
                    for s in uuid_to_storage.values():
                        if isinstance(s, RolloutStorage):
                            s.sampler_select(keep)
                
                to_add_to_storage = dict(
                    observations=preprocessed_obs,
                    memory=memory.sampler_select(keep),
                    actions=flat_actions[0, keep],
                    action_log_probs=ac_out.distributions.log_prob(actions)[0, keep],
                    value_preds=ac_out.values[0, keep],
                    rewards=rewards[keep],
                    masks=masks[keep],
                )
                for s in uuid_to_storage.values():
                    s.add(**to_add_to_storage)

            except Exception as e:
                print(f"????: {e}")
                import pdb; pdb.set_trace()
                pass

        with torch.no_grad():
            ac_out, _ = agent_model(
                **rollout_storage.agent_input_for_next_step()
            )
        
        import pdb; pdb.set_trace()
        training_pipeline.rollout_count += 1
        before_update_info = dict(
            next_value=ac_out.values.detach(),
            use_gae=training_pipeline.training_settings.use_gae,
            gamma=training_pipeline.training_settings.gamma,
            tau=training_pipeline.training_settings.gae_lambda,
            adv_stats_callback=advantage_stats,
        )

        for storage in training_pipeline.current_stage_storage.values():
            storage.before_updates(**before_update_info)

        for sc in training_pipeline.current_stage.stage_components:
            component_storage = uuid_to_storage[sc.storage_uuid]
            stage = training_pipeline.current_stage

            # Update loss
            training_settings = sc.training_settings
            loss_names = sc.loss_names
            losses = [training_pipeline.get_loss(ln) for ln in loss_names]
            loss_weights = [stage.uuid_to_loss_weight[ln] for ln in loss_names]
            loss_update_repeats_list = training_settings.update_repeats
            if isinstance(loss_update_repeats_list, numbers.Integral):
                loss_update_repeats_list = [loss_update_repeats_list] * len(loss_names)

            enough_data_for_update = True
            for current_update_repeat_index in range(
                max(loss_update_repeats_list, default=0)
            ):
                if isinstance(component_storage, MiniBatchStorageMixin):
                    batch_iterator = component_storage.batched_experience_generator(
                        num_mini_batch=training_settings.num_mini_batch
                    )
                elif isinstance(component_storage, StreamingStorageMixin):
                    assert (
                        training_settings.num_mini_batch is None
                        or training_settings.num_mini_batch == 1
                    )
                    # pass
                    pass
                else:
                    raise NotImplementedError(
                        f"Storage {component_storage} must be a subclass of `MiniBatchStorageMixin` or `StreamingStorageMixin`."
                    )
                
                for batch in batch_iterator:
                    if batch is None:
                        assert isinstance(storage, StreamingStorageMixin)
                        pass

                    info: Dict[str, float] = {}

                    bsize: Optional[int] = None
                    total_loss: Optional[torch.Tensor] = None
                    b_ac_out: Optional[ActorCriticOutput] = None
                    b_memory = Memory()

                    for loss, loss_name, loss_weight, max_update_repeats_for_loss in zip(
                        losses, loss_names, loss_weights, loss_update_repeats_list
                    ):
                        if current_update_repeat_index >= max_update_repeats_for_loss:
                            continue

                        if isinstance(loss, AbstractActorCriticLoss):
                            bsize = batch["size"]

                            if b_ac_out is None:
                                try:
                                    b_ac_out, _ = agent_model(
                                        observations=batch["observations"],
                                        memory=batch["memory"],
                                        prev_actions=batch["prev_actions"],
                                        masks=batch["masks"],
                                    )
                                except ValueError:
                                    raise

                                import pdb; pdb.set_trace()
                            
                            loss_return = loss.loss(
                                step_count=training_pipeline.current_stage.steps_taken_in_stage,
                                batch=batch,
                                actor_critic_output=b_ac_out,
                            )

                            per_epoch_info = {}
                            if len(loss_return) == 2:
                                current_loss, current_info = loss_return
                            elif len(loss_return) == 3:
                                current_loss, current_info, per_epoch_info = loss_return
                            else:
                                raise NotImplementedError
                        
                        elif isinstance(loss, GenericAbstractLoss):
                            loss_output = loss.loss(
                                model=agent_model,
                                batch=batch,
                                batch_memory=b_memory,
                                stream_memory=stage.stage_component_uuid_to_stream_memory[
                                    sc.uuid
                                ],
                            )
                            current_loss = loss_output.value
                            current_info = loss_output.info
                            per_epoch_info = loss_output.per_epoch_info
                            b_memory = loss_output.batch_memory
                            stage.stage_component_uuid_to_stream_memory[
                                sc.uuid
                            ] = loss_output.stream_memory
                            bsize = loss_output.bsize
                        
                        else:
                            raise NotImplementedError()

                        if total_loss is None:
                            total_loss = loss_weight * current_loss
                        else:
                            total_loss += loss_weight * current_loss

                        for key, value in current_info.items():
                            info[f"{loss_name}/{key}"] = value

                        if per_epoch_info is not None:
                            for key, value in per_epoch_info.items():
                                if max(loss_update_repeats_list, default=0) > 1:
                                    info[
                                        f"{loss_name}/{key}_epoch{current_update_repeat_index:02d}"
                                    ] = value
                                    info[f"{loss_name}/{key}_combined"] = value
                                else:
                                    info[f"{loss_name}/{key}"] = value
                        
                    assert total_loss is not None

                    total_loss_scalar = total_loss.item()
                    info[f"total_loss"] = total_loss_scalar

                    tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.LOSS,
                            info=info,
                            n=bsize,
                            storage_uuid=sc.storage_uuid,
                            stage_component_uuid=sc.uuid,
                        )
                    )

                    aggregate_bsize = torch.tensor(bsize, 1).item()
                    to_track = {
                        "lr": optimizer.param_groups[0]["lr"],
                        "rollout_epochs": max(loss_update_repeats_list, default=0),
                        "global_batch_size": aggregate_bsize,
                        "worker_batch_size": bsize,
                    }
                    if training_settings.num_mini_batch is not None:
                        to_track[
                            "rollout_num_mini_batch"
                        ] = training_settings.num_mini_batch
                    
                    for k, v in to_track.items():
                        tracking_info_list.append(
                            TrackingInfo(
                                type=TrackingInfoType.UPDATE_INFO,
                                info={k: v},
                                n=bsize,
                                storage_uuid=sc.storage_uuid,
                                stage_component_uuid=sc.uuid,
                            )
                        )

                    # Backprop!!
                    optimizer.zero_grad()
                    if isinstance(total_loss, torch.Tensor):
                        total_loss.backward()

                    nn.utils.clip_grad_norm_(
                        agent_model.parameters(), max_norm=training_settings.max_grad_norm,
                    )

                    optimizer.step()

                    stage.stage_component_uuid_to_stream_memory[
                        sc.uuid
                    ] = detach_recursively(
                        input=stage.stage_component_uuid_to_stream_memory[
                            sc.uuid
                        ],
                        inplace=True,
                    )

        for storage in training_pipeline.current_stage_storage.values():
            storage.after_updates()

        change_in_storage_experiences = {}
        for k in sorted(training_pipeline.current_stage_storage.keys()):
            delta = (
                training_pipeline.current_stage_storage[k].total_experiences
                - former_storage_experiences[k]
            )
            assert delta >= 0
            change_in_storage_experiences[k] = torch.tensor(delta * 1).item()

        for storage_uuid, delta in change_in_storage_experiences.items():
            training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage[
                storage_uuid
            ] += delta

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=training_pipeline.total_steps)
