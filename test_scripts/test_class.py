from collections import defaultdict
import datetime
import itertools
import logging
import queue
import sys
from functools import partial
from typing import Any, Optional, List, Sequence, Tuple, Type, Dict, Union, cast
import os
import time
import json
import numbers
import inspect
import pathlib
import importlib
import atexit
import random
import signal
import traceback
import filelock

import numpy as np
import torch
import torch.distributed as dist  # type: ignore
import torch.distributions  # type: ignore
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from setproctitle import setproctitle as ptitle

from multiprocessing.process import BaseProcess
from multiprocessing.context import BaseContext

from allenact.utils import spaces_utils as su
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.distributions import TeacherForcingDistr
from allenact.base_abstractions.misc import RLStepResult, Memory, ActorCriticOutput, GenericAbstractLoss
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import VectorSampledTasks, SingleProcessVectorSampledTasks, COMPLETE_TASK_METRICS_KEY
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from allenact.algorithms.onpolicy_sync.storage import ExperienceStorage, MiniBatchStorageMixin, StreamingStorageMixin, RolloutStorage
from allenact.algorithms.onpolicy_sync.misc import TrackingInfoType, TrackingInfo
from allenact.utils.system import get_logger, init_logging, HUMAN_LOG_LEVELS
from allenact.utils.model_utils import md5_hash_of_state_dict
from allenact.utils.misc_utils import all_equal, NumpyJSONEncoder
from allenact.utils.experiment_utils import set_seed, TrainingPipeline, LoggingPackage, PipelineStage, ScalarMeanTracker, StageComponent
from allenact.utils.tensor_utils import SummaryWriter, batch_observations, detach_recursively
from allenact.utils.viz_utils import VizSuite

from example_utils import ForkedPdb


TRAIN_MODE_STR = "train"
VALID_MODE_STR = "valid"
TEST_MODE_STR = "test"


class TestEngine:
    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: Optional[
            mp.Queue
        ],  # to write/read (trainer/evaluator) ready checkpoints
        mode: str = "train",
        seed: Optional[int] = None,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_ip: str = "127.0.0.1",
        distributed_port: int = 0,
        max_sampler_processes_per_worker: Optional[int] = None,
        initial_model_state_dict: Optional[Union[Dict[str, Any], int]] = None,
        # kwargs for training
        distributed_preemption_threshold: float = 0.7,
        first_local_worker_id: int = 0,
        **kwargs,
    ):
        self.config = config
        self.results_queue = results_queue
        self.checkpoints_queue = checkpoints_queue
        self.mp_ctx = mp_ctx
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = torch.device("cpu") if device == -1 else torch.device(device)  # type: ignore
        self.distributed_ip = distributed_ip
        self.distributed_port = distributed_port

        self.mode = mode.lower().strip()
        assert self.mode in [
            TRAIN_MODE_STR,
            VALID_MODE_STR,
            TEST_MODE_STR,
        ], 'Only "train", "valid", "test" modes supported'

        self.seed = seed
        set_seed(self.seed)

        self.experiment_name = experiment_name

        assert (
            max_sampler_processes_per_worker is None
            or max_sampler_processes_per_worker >= 1
        ), "`max_sampler_processes_per_worker` must be either `None` or a positive integer."
        self.max_sampler_processes_per_worker = max_sampler_processes_per_worker

        machine_params = config.machine_params(self.mode)
        self.machine_params: MachineParams
        if isinstance(machine_params, MachineParams):
            self.machine_params = machine_params
        else:
            self.machine_params = MachineParams(**machine_params)

        self.num_samplers_per_worker = self.machine_params.nprocesses
        self.num_samplers = self.num_samplers_per_worker[self.worker_id]

        self._vector_tasks: Optional[
            Union[VectorSampledTasks, SingleProcessVectorSampledTasks]
        ] = None

        self.sensor_preprocessor_graph = None
        self.actor_critic: Optional[ActorCriticModel] = None
        if self.num_samplers > 0:
            create_model_kwargs = {}
            if self.machine_params.sensor_preprocessor_graph is not None:
                self.sensor_preprocessor_graph = self.machine_params.sensor_preprocessor_graph.to(
                    self.device
                )
                create_model_kwargs[
                    "sensor_preprocessor_graph"
                ] = self.sensor_preprocessor_graph

            set_seed(self.seed)
            self.actor_critic = cast(
                ActorCriticModel, self.config.create_model(**create_model_kwargs),
            ).to(self.device)

        if initial_model_state_dict is not None:
            if isinstance(initial_model_state_dict, int):
                assert (
                    md5_hash_of_state_dict(self.actor_critic.state_dict())
                    == initial_model_state_dict
                ), (
                    f"Could not reproduce the correct model state dict on worker {self.worker_id} despite seeding."
                    f" Please ensure that your model's initialization is reproducable when `set_seed(...)`"
                    f"] has been called with a fixed seed before initialization."
                )
            else:
                self.actor_critic.load_state_dict(
                    state_dict=cast(
                        "OrderedDict[str, Tensor]", initial_model_state_dict
                    )
                )
        else:
            assert mode != TRAIN_MODE_STR or self.num_workers == 1, (
                "When training with multiple workers you must pass a,"
                " non-`None` value for the `initial_model_state_dict` argument."
            )
        
        if get_logger().level == logging.DEBUG:
            model_hash = md5_hash_of_state_dict(self.actor_critic.state_dict())
            get_logger().debug(
                f"[{self.mode} worker {self.worker_id}] model weights hash: {model_hash}"
            )

        self.is_distributed = False
        self.store: Optional[torch.distributed.TCPStore] = None  # type:ignore
        if self.num_workers > 1:
            self.store = torch.distributed.TCPStore(  # type:ignore
                host_name=self.distributed_ip,
                port=self.distributed_port,
                world_size=self.num_workers,
                is_master=self.worker_id == 0,
            )
            cpu_device = self.device == torch.device("cpu")  # type:ignore

            # "gloo" required during testing to ensure that `barrier()` doesn't time out.
            backend = "gloo" if cpu_device or self.mode == TEST_MODE_STR else "nccl"
            get_logger().debug(
                f"Worker {self.worker_id}: initializing distributed {backend} backend with device {self.device}."
            )
            dist.init_process_group(  # type:ignore
                backend=backend,
                store=self.store,
                rank=self.worker_id,
                world_size=self.num_workers,
                # During testing, we sometimes found that default timeout was too short
                # resulting in the run terminating surprisingly, we increase it here.
                timeout=datetime.timedelta(minutes=3000)
                if self.mode == TEST_MODE_STR
                else dist.default_pg_timeout,
            )
            self.is_distributed = True

        self._is_closing: bool = (
            False  # Useful for letting the RL runner know if this is closing
        )
        self._is_closed: bool = False

        self.training_pipeline: Optional[TrainingPipeline] = None

        # Keeping track of metrics during training/inference
        self.single_process_metrics: List = []

        # Trainig Engine
        if self.mode == TRAIN_MODE_STR:
            self.actor_critic.train()
            self.training_pipeline = config.training_pipeline()

            if self.num_workers != 1:
                # Ensure that we're only using early stopping criterions in the non-distributed setting.
                if any(
                    stage.early_stopping_criterion is not None
                    for stage in self.training_pipeline.pipeline_stages
                ):
                    raise NotImplementedError(
                        "Early stopping criterions are currently only allowed when using a single training worker, i.e."
                        " no distributed (multi-GPU) training. If this is a feature you'd like please create an issue"
                        " at https://github.com/allenai/allenact/issues or (even better) create a pull request with this "
                        " feature and we'll be happy to review it."
                    )

            self.optimizer: optim.optimizer.Optimizer = (
                self.training_pipeline.optimizer_builder(
                    params=[p for p in self.actor_critic.parameters() if p.requires_grad]
                )
            )
            self.lr_scheduler: Optional[_LRScheduler] = None
            if self.training_pipeline.lr_scheduler_builder is not None:
                self.lr_scheduler = self.training_pipeline.lr_scheduler_builder(
                    optimizer=self.optimizer
                )
            
            if self.is_distributed:
                # Tracks how many workers have finished their rollout
                self.num_workers_done = torch.distributed.PrefixStore(  # type:ignore
                    "num_workers_done", self.store
                )
                # Tracks the number of steps taken by each worker in current rollout
                self.num_workers_steps = torch.distributed.PrefixStore(  # type:ignore
                    "num_workers_steps", self.store
                )
                self.distributed_preemption_threshold = distributed_preemption_threshold
                # Flag for finished worker in current epoch
                self.offpolicy_epoch_done = torch.distributed.PrefixStore(  # type:ignore
                    "offpolicy_epoch_done", self.store
                )
                # Flag for finished worker in current epoch with custom component
                self.insufficient_data_for_update = torch.distributed.PrefixStore(  # type:ignore
                    "insufficient_data_for_update", self.store
                )
            else:
                self.num_workers_done = None
                self.num_workers_steps = None
                self.distributed_preemption_threshold = 1.0
                self.offpolicy_epoch_done = None

            # Keeping track of training state
            self.tracking_info_list: List[TrackingInfo] = []
            self.former_steps: Optional[int] = None
            self.last_log: Optional[int] = None
            self.last_save: Optional[int] = None
            # The `self._last_aggregated_train_task_metrics` attribute defined
            # below is used for early stopping criterion computations
            self._last_aggregated_train_task_metrics: ScalarMeanTracker = (
                ScalarMeanTracker()
            )

            self.first_local_worker_id = first_local_worker_id

    @property
    def vector_tasks(
        self,
    ) -> Union[VectorSampledTasks, SingleProcessVectorSampledTasks]:
        if self._vector_tasks is None and self.num_samplers > 0:
            if self.is_distributed:
                total_processes = sum(
                    self.num_samplers_per_worker
                )  # TODO this will break the fixed seed for multi-device test
            else:
                total_processes = self.num_samplers

            seeds = self.worker_seeds(
                total_processes,
                initial_seed=self.seed,  # do not update the RNG state (creation might happen after seed resetting)
            )

            # TODO: The `self.max_sampler_processes_per_worker == 1` case below would be
            #   great to have but it does not play nicely with us wanting to kill things
            #   using SIGTERM/SIGINT signals. Would be nice to figure out a solution to
            #   this at some point.
            # if self.max_sampler_processes_per_worker == 1:
            #     # No need to instantiate a new task sampler processes if we're
            #     # restricted to one sampler process for this worker.
            #     self._vector_tasks = SingleProcessVectorSampledTasks(
            #         make_sampler_fn=self.config.make_sampler_fn,
            #         sampler_fn_args_list=self.get_sampler_fn_args(seeds),
            #     )
            # else:
            self._vector_tasks = VectorSampledTasks(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(seeds),
                multiprocessing_start_method="forkserver"
                if self.mp_ctx is None
                else None,
                mp_ctx=self.mp_ctx,
                max_processes=self.max_sampler_processes_per_worker,
            )
        return self._vector_tasks

    @staticmethod
    def worker_seeds(nprocesses: int, initial_seed: Optional[int]) -> List[int]:
        """Create a collection of seeds for workers without modifying the RNG
        state."""
        rstate = None  # type:ignore
        if initial_seed is not None:
            rstate = random.getstate()
            random.seed(initial_seed)
        seeds = [random.randint(0, (2 ** 31) - 1) for _ in range(nprocesses)]
        if initial_seed is not None:
            random.setstate(rstate)
        return seeds

    def get_sampler_fn_args(self, seeds: Optional[List[int]] = None):
        sampler_devices = self.machine_params.sampler_devices

        if self.mode == TRAIN_MODE_STR:
            fn = self.config.train_task_sampler_args
        elif self.mode == VALID_MODE_STR:
            fn = self.config.valid_task_sampler_args
        elif self.mode == TEST_MODE_STR:
            fn = self.config.test_task_sampler_args
        else:
            raise NotImplementedError(
                "self.mode must be one of `train`, `valid` or `test`."
            )

        if self.is_distributed:
            total_processes = sum(self.num_samplers_per_worker)
            process_offset = sum(self.num_samplers_per_worker[: self.worker_id])
        else:
            total_processes = self.num_samplers
            process_offset = 0

        sampler_devices_as_ints: Optional[List[int]] = None
        if (
            self.is_distributed or self.mode == TEST_MODE_STR
        ) and self.device.index is not None:
            sampler_devices_as_ints = [self.device.index]
        elif sampler_devices is not None:
            sampler_devices_as_ints = [
                -1 if sd.index is None else sd.index for sd in sampler_devices
            ]

        return [
            fn(
                process_ind=process_offset + it,
                total_processes=total_processes,
                devices=sampler_devices_as_ints,
                seeds=seeds,
            )
            for it in range(self.num_samplers)
        ]

    def _preprocess_observations(self, batched_observations):
        if self.sensor_preprocessor_graph is None:
            return batched_observations
        return self.sensor_preprocessor_graph.get_observations(batched_observations)

    def remove_paused(self, observations):
        paused, keep, running = [], [], []
        for it, obs in enumerate(observations):
            if obs is None:
                paused.append(it)
            else:
                keep.append(it)
                running.append(obs)

        for p in reversed(paused):
            self.vector_tasks.pause_at(p)

        # Group samplers along new dim:
        batch = batch_observations(running, device=self.device)

        return len(paused), keep, batch

    def initialize_storage_and_viz(
        self,
        storage_to_initialize: Optional[Sequence[ExperienceStorage]],
        visualizer: Optional[VizSuite] = None,
    ):
        observations = self.vector_tasks.get_observations()

        npaused, keep, batch = self.remove_paused(observations)
        observations = self._preprocess_observations(batch) if len(keep) > 0 else batch

        assert npaused == 0, f"{npaused} samplers are paused during initialization."

        num_samplers = len(keep)
        recurrent_memory_specification = (
            self.actor_critic.recurrent_memory_specification
        )

        if storage_to_initialize is not None:
            for s in storage_to_initialize:
                s.to(self.device)
                s.set_partition(index=self.worker_id, num_parts=self.num_workers)
                s.initialize(
                    observations=observations,
                    num_samplers=num_samplers,
                    recurrent_memory_specification=recurrent_memory_specification,
                    action_space=self.actor_critic.action_space,
                )

        if visualizer is not None and len(keep) > 0:
            visualizer.collect(vector_task=self.vector_tasks, alive=keep)

        return npaused

    @property
    def num_active_samplers(self):
        return self.vector_tasks.num_unpaused_tasks

    def act(
        self,
        rollout_storage: RolloutStorage,
        dist_wrapper_class: Optional[type] = None,
    ):
        if self.mode == TRAIN_MODE_STR:
            if self.training_pipeline.current_stage.teacher_forcing is not None:
                assert dist_wrapper_class is None
                dist_wrapper_class = partial(
                    TeacherForcingDistr,
                    action_space=self.actor_critic.action_space,
                    num_active_samplers=self.num_active_samplers,
                    approx_steps=self.approx_steps,
                    teacher_forcing=self.training_pipeline.current_stage.teacher_forcing,
                    tracking_info_list=self.tracking_info_list,
                )

        with torch.no_grad():
            agent_input = rollout_storage.agent_input_for_next_step()
            actor_critic_output, memory = self.actor_critic(**agent_input)

            distr = actor_critic_output.distributions
            if dist_wrapper_class is not None:
                distr = dist_wrapper_class(distr=distr, obs=agent_input["observations"])

            # actions = distr.sample() if not self.deterministic_agents else distr.mode()
            actions = distr.sample()

        if self.mode == TRAIN_MODE_STR:
            self.step_count += self.num_active_samplers

        return actions, actor_critic_output, memory, agent_input["observations"]

    @staticmethod
    def _active_memory(memory, keep):
        return memory.sampler_select(keep) if memory is not None else memory

    def collect_step_across_all_task_samplers(
        self,
        rollout_storage_uuid: str,
        uuid_to_storage: Dict[str, ExperienceStorage],
        visualizer=None,
        dist_wrapper_class=None,
    ) -> int:
        rollout_storage = cast(RolloutStorage, uuid_to_storage[rollout_storage_uuid])
        actions, actor_critic_output, memory, _ = self.act(
            rollout_storage=rollout_storage, dist_wrapper_class=dist_wrapper_class,
        )

        # Flatten actions
        flat_actions = su.flatten(self.actor_critic.action_space, actions)

        assert len(flat_actions.shape) == 3, (
            "Distribution samples must include step and task sampler dimensions [step, sampler, ...]. The simplest way"
            "to accomplish this is to pass param tensors (like `logits` in a `CategoricalDistr`) with these dimensions"
            "to the Distribution."
        )

        # Convert flattened actions into list of actions and send them
        outputs: List[RLStepResult] = self.vector_tasks.step(
            su.action_list(self.actor_critic.action_space, flat_actions)
        )

        # Save after task completion metrics
        for step_result in outputs:
            if (
                step_result.info is not None
                and COMPLETE_TASK_METRICS_KEY in step_result.info
            ):
                self.single_process_metrics.append(
                    step_result.info[COMPLETE_TASK_METRICS_KEY]
                )
                del step_result.info[COMPLETE_TASK_METRICS_KEY]

        rewards: Union[List, torch.Tensor]
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device,  # type:ignore
        )

        # We want rewards to have dimensions [sampler, reward]
        if len(rewards.shape) == 1:
            # Rewards are of shape [sampler,]
            rewards = rewards.unsqueeze(-1)
        elif len(rewards.shape) > 1:
            raise NotImplementedError()

        # If done then clean the history of observations.
        masks = (
            1.0
            - torch.tensor(
                dones, dtype=torch.float32, device=self.device,  # type:ignore
            )
        ).view(
            -1, 1
        )  # [sampler, 1]

        npaused, keep, batch = self.remove_paused(observations)

        # TODO self.probe(...) can be useful for debugging (we might want to control it from main?)
        # self.probe(dones, npaused)

        if npaused > 0:
            for s in uuid_to_storage.values():
                if isinstance(s, RolloutStorage):
                    s.sampler_select(keep)

            if self.mode == TRAIN_MODE_STR:
                raise NotImplementedError(
                    "When trying to get a new task from a task sampler (using the `.next_task()` method)"
                    " the task sampler returned `None`. This is not currently supported during training"
                    " (and almost certainly a bug in the implementation of the task sampler or in the "
                    " initialization of the task sampler for training)."
                )

        to_add_to_storage = dict(
            observations=self._preprocess_observations(batch)
            if len(keep) > 0
            else batch,
            memory=self._active_memory(memory, keep),
            actions=flat_actions[0, keep],
            action_log_probs=actor_critic_output.distributions.log_prob(actions)[
                0, keep
            ],
            value_preds=actor_critic_output.values[0, keep],
            rewards=rewards[keep],
            masks=masks[keep],
        )
        for storage in uuid_to_storage.values():
            storage.add(**to_add_to_storage)

        # TODO we always miss tensors for the last action in the last episode of each worker
        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=rollout_storage,
                    vector_task=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        return npaused

    def close(self, verbose=True):
        self._is_closing = True

        if "_is_closed" in self.__dict__ and self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().error(traceback.format_exc())
                else:
                    raise NotImplementedError()

        if "_vector_tasks" in self.__dict__ and self._vector_tasks is not None:
            try:
                logif(
                    f"[{self.mode} worker {self.worker_id}] Closing OnPolicyRLEngine.vector_tasks."
                )
                self._vector_tasks.close()
                logif(f"[{self.mode} worker {self.worker_id}] Closed.")
            except Exception as e:
                logif(
                    f"[{self.mode} worker {self.worker_id}] Exception raised when closing OnPolicyRLEngine.vector_tasks:"
                )
                logif(e)

        self._is_closed = True
        self._is_closing = False

    @property
    def is_closed(self):
        return self._is_closed

    @property
    def is_closing(self):
        return self._is_closing

    def __del__(self):
        self.close(verbose=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=False)

    # Training Engine Properties / Methods
    @property
    def step_count(self):
        return self.training_pipeline.current_stage.steps_taken_in_stage

    @step_count.setter
    def step_count(self, val: int):
        self.training_pipeline.current_stage.steps_taken_in_stage = val

    @property
    def log_interval(self):
        return (
            self.training_pipeline.current_stage.training_settings.metric_accumulate_interval
        )

    @property
    def approx_steps(self):
        if self.is_distributed:
            # the actual number of steps gets synchronized after each rollout
            return (
                self.step_count - self.former_steps
            ) * self.num_workers + self.former_steps
        else:
            return self.step_count  # this is actually accurate

    def train(
        self, 
        checkpoint_file_name: Optional[str] = None, 
        restart_pipeline: bool = False
    ):
        assert (
            self.mode == TRAIN_MODE_STR
        ), "train only to be called from a train instance"

        training_completed_successfully = False
        # noinspection PyBroadException
        try:
            if checkpoint_file_name is not None:
                self.checkpoint_load(checkpoint_file_name, restart_pipeline)

            self.run_pipeline()

            training_completed_successfully = True
        except KeyboardInterrupt:
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] KeyboardInterrupt, exiting."
            )
        except Exception as e:
            get_logger().error(
                f"[{self.mode} worker {self.worker_id}] Encountered {type(e).__name__}, exiting."
            )
            get_logger().exception(traceback.format_exc())
        finally:
            if training_completed_successfully:
                if self.worker_id == 0:
                    self.results_queue.put(("train_stopped", 0))
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}]. Training finished successfully."
                )
            else:
                self.results_queue.put(("train_stopped", 1 + self.worker_id))
            self.close()

    def run_pipeline(
        self,
    ):
        pass

    def update(
        self,
        stage: PipelineStage,
        stage_component: StageComponent,
        storage: ExperienceStorage,
    ):
        if self.is_distributed:
            self.insufficient_data_for_update.set(
                "insufficient_data_for_update", str(0)
            )
            dist.barrier(
                device_ids=None
                if self.device == torch.device("cpu")
                else [self.device.index]
            )
        training_settings = stage_component.training_settings

        loss_names = stage_component.loss_names
        losses = [self.training_pipeline.get_loss(ln) for ln in loss_names]
        loss_weights = [stage.uuid_to_loss_weight[ln] for ln in loss_names]
        loss_update_repeats_list = training_settings.update_repeats
        if isinstance(loss_update_repeats_list, numbers.Integral):
            loss_update_repeats_list = [loss_update_repeats_list] * len(loss_names)

        enough_data_for_update = True
        for current_update_repeat_index in range(
            max(loss_update_repeats_list, default=0)
        ):
            if isinstance(storage, MiniBatchStorageMixin):
                batch_iterator = storage.batched_experience_generator(
                    num_mini_batch=training_settings.num_mini_batch
                )
            elif isinstance(storage, StreamingStorageMixin):
                assert (
                    training_settings.num_mini_batch is None
                    or training_settings.num_mini_batch == 1
                )

                def single_batch_generator(streaming_storage: StreamingStorageMixin):
                    try:
                        yield cast(
                            StreamingStorageMixin, streaming_storage
                        ).next_batch()
                    except EOFError:
                        if streaming_storage.empty():
                            yield None
                        else:
                            cast(
                                StreamingStorageMixin, streaming_storage
                            ).reset_stream()
                            stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ].clear()
                            yield cast(
                                StreamingStorageMixin, streaming_storage
                            ).next_batch()

                batch_iterator = single_batch_generator(streaming_storage=storage)
            else:
                raise NotImplementedError(
                    f"Storage {storage} must be a subclass of `MiniBatchStorageMixin` or `StreamingStorageMixin`."
                )

            for batch in batch_iterator:
                if batch is None:
                    # This should only happen in a `StreamingStorageMixin` when it cannot
                    # generate an initial batch.
                    assert isinstance(storage, StreamingStorageMixin)
                    get_logger().warning(
                        f"Worker {self.worker_id}: could not run update in {storage}, potentially because"
                        f" not enough data has been accumulated to be able to fill an initial batch."
                    )
                    enough_data_for_update = False

                if self.is_distributed:
                    self.insufficient_data_for_update.add(
                        "insufficient_data_for_update",
                        1 * (not enough_data_for_update),
                    )
                    dist.barrier(
                        device_ids=None
                        if self.device == torch.device("cpu")
                        else [self.device.index]
                    )

                    if (
                        int(
                            self.insufficient_data_for_update.get(
                                "insufficient_data_for_update"
                            )
                        )
                        != 0
                    ):
                        enough_data_for_update = False
                        break

                info: Dict[str, float] = {}

                bsize: Optional[int] = None
                total_loss: Optional[torch.Tensor] = None
                actor_critic_output_for_batch: Optional[ActorCriticOutput] = None
                batch_memory = Memory()

                for loss, loss_name, loss_weight, max_update_repeats_for_loss in zip(
                    losses, loss_names, loss_weights, loss_update_repeats_list
                ):
                    if current_update_repeat_index >= max_update_repeats_for_loss:
                        continue
                    
                    if isinstance(loss, AbstractActorCriticLoss):
                        bsize = batch["bsize"]

                        if actor_critic_output_for_batch is None:
                            actor_critic_output_for_batch, _ = self.actor_critic(
                                observations=batch["observations"],
                                memory=batch["memory"],
                                prev_actions=batch["prev_actions"],
                                masks=batch["masks"],
                            )

                        loss_return = loss.loss(
                            step_count=self.step_count,
                            batch=batch,
                            actor_critic_output=actor_critic_output_for_batch,
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
                            model=self.actor_critic,
                            batch=batch,
                            batch_memory=batch_memory,
                            stream_memory=stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ],
                        )
                        current_loss = loss_output.value
                        current_info = loss_output.info
                        per_epoch_info = loss_output.per_epoch_info
                        batch_memory = loss_output.batch_memory
                        stage.stage_component_uuid_to_stream_memory[
                            stage_component.uuid
                        ] = loss_output.stream_memory
                        bsize = loss_output.bsize
                    else:
                        raise NotImplementedError(
                            f"Loss of type {type(loss)} is not supported. Losses must be subclasses of"
                            f" `AbstractActorCriticLoss` or `GenericAbstractLoss`."
                        )

                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

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

                assert total_loss is not None, (
                    f"No {stage_component.uuid} losses specified for training in stage"
                    f" {self.training_pipeline.current_stage_index}"
                )

                total_loss_scalar = total_loss.item()
                info[f"total_loss"] = total_loss_scalar

                self.tracking_info_list.append(
                    TrackingInfo(
                        type=TrackingInfoType.LOSS,
                        info=info,
                        n=bsize,
                        storage_uuid=stage_component.storage_uuid,
                        stage_component_uuid=stage_component.uuid,
                    )
                )

                aggregate_bsize = self.distributed_weighted_sum(bsize, 1)
                to_track = {
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "rollout_epochs": max(loss_update_repeats_list, default=0),
                    "global_batch_size": aggregate_bsize,
                    "worker_batch_size": bsize,
                }
                if training_settings.num_mini_batch is not None:
                    to_track[
                        "rollout_num_mini_batch"
                    ] = training_settings.num_mini_batch

                for k, v in to_track.items():
                    self.tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.UPDATE_INFO,
                            info={k: v},
                            n=bsize,
                            storage_uuid=stage_component.storage_uuid,
                            stage_component_uuid=stage_component.uuid,
                        )
                    )

                self.backprop_step(
                    total_loss=total_loss,
                    max_grad_norm=training_settings.max_grad_norm,
                    local_to_global_batch_size_ratio=bsize / aggregate_bsize,
                )

                stage.stage_component_uuid_to_stream_memory[
                    stage_component.uuid
                ] = detach_recursively(
                    input=stage.stage_component_uuid_to_stream_memory[
                        stage_component.uuid
                    ],
                    inplace=True,
                )

    def backprop_step(
        self,
        total_loss: torch.Tensor,
        max_grad_norm: float,
        local_to_global_batch_size_ratio: float = 1.0,
    ):
        self.optimizer.zero_grad()  # type: ignore
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()

        if self.is_distributed:
            # From https://github.com/pytorch/pytorch/issues/43135
            reductions, all_params = [], []
            for p in self.actor_critic.parameters():
                # you can also organize grads to larger buckets to make all_reduce more efficient
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:  # local_global_batch_size_tuple is not None, since we're distributed:
                        p.grad = p.grad * local_to_global_batch_size_ratio
                    reductions.append(
                        dist.all_reduce(p.grad, async_op=True,)  # sum
                    )  # synchronize
                    all_params.append(p)
            for reduction, p in zip(reductions, all_params):
                reduction.wait()

        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=max_grad_norm,  # type: ignore
        )

        self.optimizer.step()  # type: ignore
    
    def distributed_weighted_sum(
        self,
        to_share: Union[torch.Tensor, float, int],
        weight: Union[torch.Tensor, float, int],
    ):
        """Weighted sum of scalar across distributed workers."""
        if self.is_distributed:
            aggregate = torch.tensor(to_share * weight).to(self.device)
            dist.all_reduce(aggregate)
            return aggregate.item()
        else:
            if abs(1 - weight) > 1e-5:
                get_logger().warning(
                    f"Scaling non-distributed value with weight {weight}"
                )
            return torch.tensor(to_share * weight).item()
    
    def advantage_stats(self, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Computes the mean and variances of advantages (possibly over multiple workers).
        For multiple workers, this method is equivalent to first collecting all versions of
        advantages and then computing the mean and variance locally over that.

        # Parameters

        advantages: Tensors to compute mean and variance over. Assumed to be solely the
         worker's local copy of this tensor, the resultant mean and variance will be computed
         as though _all_ workers' versions of this tensor were concatenated together in
         distributed training.
        """

        # Step count has already been updated with the steps from all workers
        global_rollout_steps = self.step_count - self.former_steps

        if self.is_distributed:
            summed_advantages = advantages.sum()
            dist.all_reduce(summed_advantages)
            mean = summed_advantages / global_rollout_steps

            summed_squares = (advantages - mean).pow(2).sum()
            dist.all_reduce(summed_squares)
            std = (summed_squares / (global_rollout_steps - 1)).sqrt()
        else:
            # noinspection PyArgumentList
            mean, std = advantages.mean(), advantages.std()

        return {"mean": mean, "std": std}

    # Inference Engine Properties / Methods
    def process_checkpoints(
        self,
    ):
        pass

    # Test loop script
    def loop_script(
        self,
    ):  
        finalized = False
        try:
            cur_stage_training_settings = (
                self.training_pipeline.current_stage.training_settings
            )

            rollout_storage = self.training_pipeline.rollout_storage
            uuid_to_storage = self.training_pipeline.current_stage_storage
            self.initialize_storage_and_viz(
                storage_to_initialize=list(uuid_to_storage.values())
            )
            self.tracking_info_list.clear()

            self.last_log = self.training_pipeline.total_steps

            if self.last_save is None:
                self.last_save = self.training_pipeline.total_steps

            while True:
                pipeline_stage_changed = self.training_pipeline.before_rollout(
                    train_metrics=self._last_aggregated_train_task_metrics
                )  # This is `False` at the very start of training, i.e. pipeline starts with a stage initialized

                self._last_aggregated_train_task_metrics.reset()
                training_is_complete = self.training_pipeline.current_stage is None
                
                # `training_is_complete` should imply `pipeline_stage_changed`
                assert pipeline_stage_changed or not training_is_complete

                if self.is_distributed:
                    self.num_workers_done.set("done", str(0))
                    self.num_workers_steps.set("steps", str(0))
                    # Ensure all workers are done before incrementing num_workers_{steps, done}
                    dist.barrier(
                        device_ids=None
                        if self.device == torch.device("cpu")
                        else [self.device.index]
                    )

                self.former_steps = self.step_count
                former_storage_experiences = {
                    k: v.total_experiences
                    for k, v in self.training_pipeline.current_stage_storage.items()
                }

                for step in range(cur_stage_training_settings.num_steps):
                    num_paused = self.collect_step_across_all_task_samplers(
                        rollout_storage_uuid=self.training_pipeline.rollout_storage_uuid,
                        uuid_to_storage=uuid_to_storage,
                    )
                    assert num_paused == 0
                    if self.is_distributed:
                        # Preempt stragglers
                        # Each worker will stop collecting steps for the current rollout whenever a
                        # 100 * distributed_preemption_threshold percentage of workers are finished collecting their
                        # rollout steps, and we have collected at least 25% but less than 90% of the steps.
                        num_done = int(self.num_workers_done.get("done"))
                        if (
                            num_done
                            > self.distributed_preemption_threshold * self.num_workers
                            and 0.25 * cur_stage_training_settings.num_steps
                            <= step
                            < 0.9 * cur_stage_training_settings.num_steps
                        ):
                            get_logger().debug(
                                f"[{self.mode} worker {self.worker_id}] Preempted after {step}"
                                f" steps (out of {cur_stage_training_settings.num_steps})"
                                f" with {num_done} workers done"
                            )
                            break
                
                with torch.no_grad():
                    actor_critic_output, _ = self.actor_critic(
                        **rollout_storage.agent_input_for_next_step()
                    )

                self.training_pipeline.rollout_count += 1
                if self.is_distributed:
                    # Mark that a worker is done collecting experience
                    self.num_workers_done.add("done", 1)
                    self.num_workers_steps.add("steps", self.step_count - self.former_steps)

                    # Ensure all workers are done before updating step counter
                    dist.barrier(
                        device_ids=None
                        if self.device == torch.device("cpu")
                        else [self.device.index]
                    )

                    ndone = int(self.num_workers_done.get("done"))
                    assert (
                        ndone == self.num_workers
                    ), f"# workers done {ndone} != # workers {self.num_workers}"

                    # get the actual step_count
                    self.step_count = (
                        int(self.num_workers_steps.get("steps")) + self.former_steps
                    )

                before_update_info = dict(
                    next_value=actor_critic_output.values.detach(),
                    use_gae=cur_stage_training_settings.use_gae,
                    gamma=cur_stage_training_settings.gamma,
                    tau=cur_stage_training_settings.gae_lambda,
                    adv_stats_callback=self.advantage_stats,
                )

                # Prepare storage for iteration during updates
                for storage in self.training_pipeline.current_stage_storage.values():
                    storage.before_updates(**before_update_info)

                for sc in self.training_pipeline.current_stage.stage_components:
                    component_storage = uuid_to_storage[sc.storage_uuid]

                    # before_update = time.time()

                    self.update(
                        stage=self.training_pipeline.current_stage,
                        stage_component=sc,
                        storage=component_storage,
                    )

                    # after_update = time.time()
                    # delta = after_update - before_update
                    # get_logger().info(
                    #     f"Worker {self.worker_id}: {sc.uuid} took {delta:.2g}s ({sc.training_settings.update_repeats}"
                    #     f" repeats * {sc.training_settings.num_mini_batch} batches)"
                    # )

                for storage in self.training_pipeline.current_stage_storage.values():
                    storage.after_updates()

                
        except KeyboardInterrupt:
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] KeyboardInterrupt, exiting."
            )
        except Exception as e:
            get_logger().error(
                f"[{self.mode} worker {self.worker_id}] Encountered {type(e).__name__}, exiting."
            )
            get_logger().error(traceback.format_exc())
        finally:
            if finalized:
                self.results_queue.put((f"{self.mode}_stopped", 0))
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}] Complete, all checkpoints processed."
                )
            else:
                self.results_queue.put((f"{self.mode}_stopped", self.worker_id + 1))
            self.close(verbose=True)


class TestRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        seed: Optional[int] = None,
        mode: str = "train",
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "default",
        extra_tag: str = "",
        distributed_ip_and_port: str = "127.0.0.1:0",
        machine_id: int = 0,

    ):
        self.config = config
        self.seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        if multiprocessing_start_method == "default":
            if torch.cuda.is_available():
                multiprocessing_start_method = "forkserver"
            else:
                # Spawn seems to play nicer with cpus and debugging
                multiprocessing_start_method = "spawn"
        self.mp_ctx = self.init_context(mp_ctx, multiprocessing_start_method)
        self.extra_tag = extra_tag
        self.mode = mode.lower().strip()

        # assert self.mode in [
        #     TRAIN_MODE_STR,
        #     TEST_MODE_STR,
        # ], "Only 'train' and 'test' modes supported in runner"

        set_seed(self.seed)
        self.queues: Optional[Dict[str, mp.Queue]] = None
        self.processes: Dict[str, List[Union[BaseProcess, mp.Process]]] = defaultdict(
            list
        )
        self._local_start_time_str: Optional[str] = None
        self._is_closed: bool = False

        self.distributed_ip_and_port = distributed_ip_and_port
        self.machine_id = machine_id

    @property
    def local_start_time_str(self) -> str:
        if self._local_start_time_str is None:
            raise RuntimeError(
                "Local start time string does not exist as neither `start_train()` or `start_test()`"
                " has been called on this runner."
            )
        return self._local_start_time_str

    @staticmethod
    def init_context(
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "forkserver",
        valid_start_methods: Tuple[str, ...] = ("forkserver", "spawn", "fork"),
    ):
        if mp_ctx is None:
            assert multiprocessing_start_method in valid_start_methods, (
                f"multiprocessing_start_method must be one of {valid_start_methods}."
                f" Got '{multiprocessing_start_method}'"
            )

            mp_ctx = mp.get_context(multiprocessing_start_method)
        elif multiprocessing_start_method != mp_ctx.get_start_method():
            get_logger().warning(
                f"ignoring multiprocessing_start_method '{multiprocessing_start_method}'"
                f" and using given context with '{mp_ctx.get_start_method()}'"
            )

        return mp_ctx

    def _acquire_unique_local_start_time_string(self) -> str:
        """Creates a (unique) local start time string for this experiment.

        Ensures through file locks that the local start time string
        produced is unique. This implies that, if one has many
        experiments starting in in parallel, at most one will be started
        every second (as the local start time string only records the
        time up to the current second).
        """
        # os.makedirs(self.output_dir, exist_ok=True)
        # start_time_string_lock_path = os.path.abspath(
        #     os.path.join(self.output_dir, ".allenact_start_time_string.lock")
        # )
        # try:
        #     with filelock.FileLock(start_time_string_lock_path, timeout=60):
        #         last_start_time_string_path = os.path.join(
        #             self.output_dir, ".allenact_last_start_time_string"
        #         )
        #         pathlib.Path(last_start_time_string_path).touch()

        #         with open(last_start_time_string_path, "r") as f:
        #             last_start_time_string_list = f.readlines()

        #         while True:
        #             candidate_str = time.strftime(
        #                 "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
        #             )
        #             if (
        #                 len(last_start_time_string_list) == 0
        #                 or last_start_time_string_list[0].strip() != candidate_str
        #             ):
        #                 break
        #             time.sleep(0.2)

        #         with open(last_start_time_string_path, "w") as f:
        #             f.write(candidate_str)

        # except filelock.Timeout as e:
        #     get_logger().exception(
        #         f"Could not acquire the lock for {start_time_string_lock_path} for 60 seconds,"
        #         " this suggests an unexpected deadlock. Please close all AllenAct training processes,"
        #         " delete this lockfile, and try again."
        #     )
        #     raise e
        candidate_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        assert candidate_str is not None
        return candidate_str

    def worker_devices(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode)
        )
        devices = machine_params.devices

        assert all_equal(devices) or all(
            d.index >= 0 for d in devices
        ), f"Cannot have a mix of CPU and GPU devices (`devices == {devices}`)"

        get_logger().info(f"Using {len(devices)} {mode} workers on devices {devices}")
        return devices

    def local_worker_ids(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode, machine_id=self.machine_id)
        )
        ids = machine_params.local_worker_ids

        get_logger().info(
            f"Using local worker ids {ids} (total {len(ids)} workers in machine {self.machine_id})"
        )

        return ids

    @staticmethod
    def init_process(mode: str, id: int, to_close_on_termination: "TestEngine"):
        ptitle(f"{mode}-{id}")

        def create_handler(termination_type: str):
            def handler(_signo, _frame):
                prefix = f"{termination_type} signal sent to worker {mode}-{id}."
                if to_close_on_termination.is_closed:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closed, exiting."
                    )
                    sys.exit(0)
                elif not to_close_on_termination.is_closing:
                    get_logger().info(
                        f"{prefix} Forcing worker {mode}-{id} to close and exiting."
                    )
                    # noinspection PyBroadException
                    try:
                        to_close_on_termination.close(True)
                    except Exception:
                        get_logger().error(
                            f"Error occurred when closing the RL engine used by work {mode}-{id}."
                            f" We cannot recover from this and will simply exit. The exception:"
                        )
                        get_logger().exception(traceback.format_exc())
                        sys.exit(1)
                    sys.exit(0)
                else:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closing, ignoring this signal."
                    )

            return handler

        signal.signal(signal.SIGTERM, create_handler("Termination"))
        signal.signal(signal.SIGINT, create_handler("Interrupt"))

    @staticmethod
    def init_worker(engine_class, args, kwargs):
        mode = kwargs["mode"]
        id = kwargs["worker_id"]

        worker = None
        try:
            worker = engine_class(*args, **kwargs)
        except Exception:
            get_logger().error(f"Encountered Exception. Terminating {mode} worker {id}")
            get_logger().exception(traceback.format_exc())
            kwargs["results_queue"].put((f"{mode}_stopped", 1 + id))
        finally:
            return worker

    staticmethod
    def train_loop(
        id: int = 0,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        *engine_args,
        **engine_kwargs,
    ):
        engine_kwargs["mode"] = TRAIN_MODE_STR
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"train {id} args {engine_kwargs_for_print}")

        trainer: TestEngine = TestRunner.init_worker(
            engine_class=TestEngine, args=engine_args, kwargs=engine_kwargs
        )
        if trainer is not None:
            TestRunner.init_process("Train", id, to_close_on_termination=trainer)
            trainer.train(
                checkpoint_file_name=checkpoint, restart_pipeline=restart_pipeline
            )

    @staticmethod
    def valid_loop(id: int = 0, *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = VALID_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info(f"valid {id} args {engine_kwargs}")

        valid = TestRunner.init_worker(
            engine_class=TestEngine, args=engine_args, kwargs=engine_kwargs
        )
        if valid is not None:
            TestRunner.init_process("Valid", id, to_close_on_termination=valid)
            valid.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def test_loop(id: int = 0, *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = TEST_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info(f"test {id} args {engine_kwargs}")

        test = TestRunner.init_worker(TestEngine, engine_args, engine_kwargs)
        if test is not None:
            TestRunner.init_process("Test", id, to_close_on_termination=test)
            test.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def loop_script(id: int = 0, mode: str = "train", *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = mode
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"{mode} {id} args {engine_kwargs_for_print}")

        engine = TestRunner.init_worker(TestEngine, engine_args, engine_kwargs)
        if engine is not None:
            TestRunner.init_process("LoopScript", id, to_close_on_termination=engine)
            engine.loop_script()

    def _initialize_start_train_or_start_test(self):
        self._is_closed = False

        if self.queues is not None:
            for k, q in self.queues.items():
                try:
                    out = q.get(timeout=1)
                    raise RuntimeError(
                        f"{k} queue was not empty before starting new training/testing (contained {out})."
                        f" This should not happen, please report how you obtained this error"
                        f" by creating an issue at https://github.com/allenai/allenact/issues."
                    )
                except queue.Empty:
                    pass

        self.queues = {
            "results": self.mp_ctx.Queue(),
            "checkpoints": self.mp_ctx.Queue(),
        }

        self._local_start_time_str = self._acquire_unique_local_start_time_string()

    def start_script(
        self,
        max_sampler_processes_per_worker: Optional[int] = None,
    ):
        self._initialize_start_train_or_start_test()
        # devices = self.worker_devices(TRAIN_MODE_STR)
        devices = self.worker_devices(self.mode)
        num_workers = len(devices)

        # Be extra careful to ensure that all models start
        # with the same initializations.
        set_seed(self.seed)
        initial_model_state_dict = self.config.create_model(
            sensor_preprocessor_graph=MachineParams.instance_from(
                self.config.machine_params(self.mode)
            ).sensor_preprocessor_graph
        ).state_dict()

        distributed_port = 0 if num_workers == 1 else self.get_port()
        # worker_ids = self.local_worker_ids(TRAIN_MODE_STR)
        worker_ids = self.local_worker_ids(self.mode)

        # if self.mode == TRAIN_MODE_STR:
        #     worker_fn = self.train_loop
        # elif self.mode == VALID_MODE_STR:
        #     worker_fn = self.valid_loop
        # elif self.mode == TEST_MODE_STR:
        #     worker_fn = self.test_loop
        # else:
        #     raise NotImplementedError
        worker_fn = self.loop_script

        model_hash = None
        for worker_id in worker_ids:
            worker_kwargs = dict(
                id=worker_id,
                experiment_name=self.experiment_name,
                config=self.config,
                results_queue=self.queues["results"],
                checkpoints_queue=self.queues["checkpoints"]
                if self.mode != TRAIN_MODE_STR else None,
                seed=self.seed,
                mp_ctx=self.mp_ctx,
                num_workers=num_workers,
                device=devices[worker_id],
                distributed_ip=self.distributed_ip_and_port.split(":")[0],
                distributed_port=distributed_port,
                max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                initial_model_state_dict=initial_model_state_dict
                if model_hash is None
                else model_hash,
                first_local_worker_id=worker_ids[0],
            )
            worker: BaseProcess = self.mp_ctx.Process(
                target=worker_fn, kwargs=worker_kwargs,
            )
            try:
                worker.start()
            except ValueError as e:
                # If the `initial_model_state_dict` is too large we sometimes
                # run into errors passing it with multiprocessing. In such cases
                # we instead hash the state_dict and confirm, in each engine worker, that
                # this hash equals the model the engine worker instantiates.
                if e.args[0] == "too many fds":
                    model_hash = md5_hash_of_state_dict(initial_model_state_dict)
                    worker_kwargs["initial_model_state_dict"] = model_hash
                    worker = self.mp_ctx.Process(
                        target=worker_fn, kwargs=worker_kwargs,
                    )
                    worker.start()
                else:
                    raise e

            self.processes[self.mode].append(worker)

        get_logger().info(
            f"Started {len(self.processes[self.mode])} {self.mode} processes"
        )

        metrics_file_template: Optional[str] = None
        self.log_and_close(
            start_time_str=self.local_start_time_str,
            nworkers=len(worker_ids),  # TODO num_workers once we forward metrics,
            metrics_file=metrics_file_template,
        )

        return self.local_start_time_str

    def log_and_close(
        self,
        start_time_str: str,
        nworkers: int,
        test_steps: Sequence[int] = (),
        metrics_file: Optional[str] = None,
    ) -> List[Dict]:
        finalized = False

        log_writer: Optional[SummaryWriter] = None
        # if not self.disable_tensorboard:
        #     log_writer = SummaryWriter(
        #         log_dir=self.log_writer_path(start_time_str),
        #         filename_suffix=f"__{self.mode}_{self.local_start_time_str}",
        #     )

        # To aggregate/buffer metrics from trainers/testers
        collected: List[LoggingPackage] = []
        last_train_steps = 0
        last_storage_uuid_to_total_experiences = {}
        last_train_time = time.time()
        # test_steps = sorted(test_steps, reverse=True)
        eval_results: List[Dict] = []
        unfinished_workers = nworkers

        try:
            while True:
                try:
                    package: Union[
                        LoggingPackage, Union[Tuple[str, Any], Tuple[str, Any, Any]]
                    ] = self.queues["results"].get(timeout=1)

                    if isinstance(package, LoggingPackage):
                        pkg_mode = package.mode

                        if pkg_mode == TRAIN_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:

                                collected = sorted(
                                    collected,
                                    key=lambda pkg: (
                                        pkg.training_steps,
                                        *sorted(
                                            pkg.storage_uuid_to_total_experiences.items()
                                        ),
                                    ),
                                )

                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                    and collected[
                                        nworkers - 1
                                    ].storage_uuid_to_total_experiences
                                    == collected[0].storage_uuid_to_total_experiences
                                ):  # ensure all workers have provided the same training_steps and total_experiences
                                    (
                                        last_train_steps,
                                        last_storage_uuid_to_total_experiences,
                                        last_train_time,
                                    ) = self.process_train_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        last_steps=last_train_steps,
                                        last_storage_uuid_to_total_experiences=last_storage_uuid_to_total_experiences,
                                        last_time=last_train_time,
                                    )
                                    collected = collected[nworkers:]
                                elif len(collected) > 2 * nworkers:
                                    get_logger().warning(
                                        f"Unable to aggregate train packages from all {nworkers} workers"
                                        f"after {len(collected)} packages collected"
                                    )
                        elif (
                            pkg_mode == VALID_MODE_STR
                        ):  # they all come from a single worker
                            if (
                                package.training_steps is not None
                            ):  # no validation samplers
                                self.process_eval_package(
                                    log_writer=log_writer,
                                    pkg=package,
                                    all_results=eval_results
                                    if self._collect_valid_results
                                    else None,
                                )

                                if metrics_file is not None:
                                    with open(
                                        metrics_file.format(package.training_steps), "w"
                                    ) as f:
                                        json.dump(
                                            eval_results[-1],
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            "Written valid results file {}".format(
                                                metrics_file.format(
                                                    package.training_steps
                                                ),
                                            )
                                        )

                            if (
                                finalized and self.queues["checkpoints"].empty()
                            ):  # assume queue is actually empty after trainer finished and no checkpoints in queue
                                break
                        elif pkg_mode == TEST_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:
                                collected = sorted(
                                    collected, key=lambda x: x.training_steps
                                )  # sort by num_steps
                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                ):  # ensure nworkers have provided the same num_steps
                                    self.process_test_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        all_results=eval_results,
                                    )

                                    collected = collected[nworkers:]
                                    with open(metrics_file, "w") as f:
                                        json.dump(
                                            eval_results,
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            f"Updated {metrics_file} up to checkpoint"
                                            f" {test_steps[len(eval_results) - 1]}"
                                        )
                        else:
                            get_logger().error(
                                f"Runner received unknown package of type {pkg_mode}"
                            )
                    else:
                        pkg_mode = package[0]

                        if pkg_mode == "train_stopped":
                            if package[1] == 0:
                                finalized = True
                                if not self.running_validation:
                                    get_logger().info(
                                        "Terminating runner after trainer done (no validation)"
                                    )
                                    break
                            else:
                                raise Exception(
                                    f"Train worker {package[1] - 1} abnormally terminated"
                                )
                        elif pkg_mode == "valid_stopped":
                            raise Exception(
                                f"Valid worker {package[1] - 1} abnormally terminated"
                            )
                        elif pkg_mode == "test_stopped":
                            if package[1] == 0:
                                unfinished_workers -= 1
                                if unfinished_workers == 0:
                                    get_logger().info(
                                        "Last tester finished. Terminating"
                                    )
                                    finalized = True
                                    break
                            else:
                                raise RuntimeError(
                                    f"Test worker {package[1] - 1} abnormally terminated"
                                )
                        else:
                            get_logger().error(
                                f"Runner received invalid package tuple {package}"
                            )
                except queue.Empty as _:
                    if all(
                        p.exitcode is not None
                        for p in itertools.chain(*self.processes.values())
                    ):
                        break
        except KeyboardInterrupt:
            get_logger().info("KeyboardInterrupt. Terminating runner.")
        except Exception:
            get_logger().error("Encountered Exception. Terminating runner.")
            get_logger().exception(traceback.format_exc())
        finally:
            if finalized:
                get_logger().info("Done")
            if log_writer is not None:
                log_writer.close()
            self.close()
            return eval_results

    @property
    def experiment_name(self):
        if len(self.extra_tag) > 0:
            return f"{self.config.tag()}_{self.extra_tag}"
        return self.config.tag()

    def close(self, verbose=True):
        if self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().exception(traceback.format_exc())
                else:
                    raise NotImplementedError()

        # First send termination signals
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                if process.is_alive():
                    logif(f"Terminating {process_type} {it}")
                    process.terminate()

        # Now join processes
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                try:
                    logif(f"Joining {process_type} {it}")
                    process.join(1)
                    logif(f"Closed {process_type} {it}")
                except Exception as e:
                    logif(f"Exception raised when closing {process_type} {it}")
                    logif(e)

        self.processes.clear()
        self._is_closed = True
    
    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)


