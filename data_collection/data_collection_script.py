import operator
from collections import defaultdict
from typing import (
    Optional,
    Union,
    Sequence,
    Tuple,
    List,
    Dict,
    Any,
    cast,
)
import os
import sys
import signal
import stringcase
import copy
import filelock
import pathlib
import datetime
import traceback
import random
import time
import queue
import json
import itertools
import argparse
from PIL import Image

from multiprocessing.process import BaseProcess
from multiprocessing.context import BaseContext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributions
import torch.cuda as cuda

import ai2thor

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams, split_processes_onto_devices
from allenact.base_abstractions.misc import RLStepResult, Memory, ActorCriticOutput, GenericAbstractLoss
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, ObservationType, ActionType
from allenact.algorithms.onpolicy_sync.engine import TRAIN_MODE_STR, VALID_MODE_STR, TEST_MODE_STR
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import VectorSampledTasks, SingleProcessVectorSampledTasks, COMPLETE_TASK_METRICS_KEY
from allenact.algorithms.onpolicy_sync.storage import ExperienceStorage, RolloutStorage, RolloutBlockStorage
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int, all_equal, NumpyJSONEncoder
from allenact.utils.experiment_utils import set_seed, TrainingPipeline, LoggingPackage, PipelineStage, ScalarMeanTracker, StageComponent, Builder
from allenact.utils import spaces_utils as su
from allenact.utils.system import init_logging, HUMAN_LOG_LEVELS, get_logger, find_free_port
from allenact.utils.tensor_utils import SummaryWriter, batch_observations, detach_recursively
from allenact.main import load_config

from example_utils import ForkedPdb

from rearrange.environment import RearrangeMode
from rearrange.tasks import RearrangeTaskSampler
from rearrange.constants import THOR_COMMIT_ID, OBJECT_TYPES_WITH_PROPERTIES
from custom.constants import IDX_TO_OBJECT_TYPE, MAP_TYPES_TO_IDX, NUM_OBJECT_TYPES, ORDERED_OBJECT_TYPES

import datagen.datagen_utils as datagen_utils
from data_collection.coco_utils import binary_mask_to_polygon


class DataCollectionAgent:
    
    def __init__(
        self,
        # experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: Optional[
            mp.Queue
        ],  # to write/read (trainer/evaluator) ready checkpoints
        data_dir: str,
        mode: str = "train",
        seed: Optional[int] = None,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_ip: str = "127.0.0.1",
        distributed_port: int = 0,
        max_sampler_processes_per_worker: Optional[int] = None,
        distributed_preemption_threshold: float = 0.7,
        allowed_scenes: Optional[Sequence[str]] = None,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        first_local_worker_id: int = 0,
        **kwargs,
    ):
        self.config = config
        self.results_queue = results_queue
        self.checkpoints_queue = checkpoints_queue
        # self.experiment_name = experiment_name
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        self.mode = mode.lower().strip()
        assert self.mode in [
            TRAIN_MODE_STR,
            VALID_MODE_STR,
            TEST_MODE_STR,
        ], 'Only "train", "valid", "test" modes supported'

        self.mp_ctx = mp_ctx
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = torch.device("cpu") if device == -1 else torch.device(device)
        self.distributed_ip = distributed_ip
        self.distributed_port = distributed_port
        self.allowed_scenes = allowed_scenes
        self.allowed_rearrange_inds_subset = allowed_rearrange_inds_subset
        
        self.seed = seed
        set_seed(self.seed)

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
        
        self.is_distributed = False
        # self.store: Optional[torch.distributed.TCPStore] = None
        if self.num_workers > 1:
        #     self.store = torch.distributed.TCPStore(  # type:ignore
        #         host_name=self.distributed_ip,
        #         port=self.distributed_port,
        #         world_size=self.num_workers,
        #         is_master=self.worker_id == 0,
        #     )
        #     cpu_device = self.device == torch.device("cpu")  # type:ignore

        #     # "gloo" required during testing to ensure that `barrier()` doesn't time out.
        #     backend = "gloo" if cpu_device or self.mode == TEST_MODE_STR else "nccl"
        #     get_logger().debug(
        #         f"Worker {self.worker_id}: initializing distributed {backend} backend with device {self.device}."
        #     )
        #     dist.init_process_group(  # type:ignore
        #         backend=backend,
        #         store=self.store,
        #         rank=self.worker_id,
        #         world_size=self.num_workers,
        #         # During testing, we sometimes found that default timeout was too short
        #         # resulting in the run terminating surprisingly, we increase it here.
        #         timeout=datetime.timedelta(minutes=3000)
        #         if self.mode == TEST_MODE_STR
        #         else dist.default_pg_timeout,
        #     )
            self.is_distributed = True

        self._is_closing: bool = False
        self._is_closed: bool = False

        # if self.is_distributed:
        #     # Tracks how many workers have finished their rollout
        #     self.num_workers_done = torch.distributed.PrefixStore(  # type:ignore
        #         "num_workers_done", self.store
        #     )
        #     # Tracks the number of steps taken by each worker in current rollout
        #     self.num_workers_steps = torch.distributed.PrefixStore(  # type:ignore
        #         "num_workers_steps", self.store
        #     )
        #     self.distributed_preemption_threshold = distributed_preemption_threshold
        #     # Flag for finished worker in current epoch
        #     self.offpolicy_epoch_done = torch.distributed.PrefixStore(  # type:ignore
        #         "offpolicy_epoch_done", self.store
        #     )
        #     # Flag for finished worker in current epoch with custom component
        #     self.insufficient_data_for_update = torch.distributed.PrefixStore(  # type:ignore
        #         "insufficient_data_for_update", self.store
        #     )
        # else:
        #     self.num_workers_done = None
        #     self.num_workers_steps = None
        #     self.distributed_preemption_threshold = 1.0
        #     self.offpolicy_epoch_done = None

        self.first_local_worker_id = first_local_worker_id

        self.rollout_storage: RolloutStorage = Builder(RolloutBlockStorage)()
        self.annotations = []
        self.images = []
        self.image_id = 0
        self.coco_id = 0
        self.remaining_tasks = []
        self.expert_subtask_history = [[] for _ in range(self.num_samplers)]
        self.expert_action_history = [[] for _ in range(self.num_samplers)]
        self.action_history = [[] for _ in range(self.num_samplers)]
        self.dones = [False for _ in range(self.num_samplers)]

    @property
    def vector_tasks(
        self,
    ) -> Union[VectorSampledTasks, SingleProcessVectorSampledTasks]:
        if self._vector_tasks is None and self.num_samplers > 0:
            if self.is_distributed:
                total_processes = sum(
                    self.num_samplers_per_worker
                )
            else:
                total_processes = self.num_samplers
            seeds = self.worker_seeds(
                total_processes,
                initial_seed=self.seed,
            )
            self._vector_tasks = VectorSampledTasks(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(
                    seeds, 
                    self.allowed_scenes, 
                    self.allowed_rearrange_inds_subset
                ),
                multiprocessing_start_method="forkserver"
                if self.mp_ctx is None
                else None,
                mp_ctx=self.mp_ctx,
                max_processes=self.max_sampler_processes_per_worker
            )
        return self._vector_tasks
    
    @property
    def num_active_samplers(self):
        return self.vector_tasks.num_unpaused_tasks

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

    def get_sampler_fn_args(
        self,
        seeds: Optional[List[int]] = None,
        allowed_scenes: Optional[Sequence[str]] = None,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
    ): 
        sampler_devices = self.machine_params.sampler_devices
        fn = self.config.stagewise_task_sampler_args_for_remaining_data
        if self.is_distributed:
            total_processes = sum(self.num_samplers_per_worker)
            process_offset = sum(self.num_samplers_per_worker[: self.worker_id])
        else:
            total_processes = self.num_samplers
            process_offset = 0

        sampler_devices_as_ints: Optional[List[int]] = None
        if self.is_distributed and self.device.index is not None:
            sampler_devices_as_ints = [self.device.index]
        elif sampler_devices is not None:
            sampler_devices_as_ints = [
                -1 if sd.index is None else sd.index for sd in sampler_devices
            ]
        
        return [
            fn(
                stage=self.mode,
                process_ind=process_offset + it,
                total_processes=total_processes,
                devices=sampler_devices_as_ints,
                allowed_scenes=allowed_scenes,
                allowed_rearrange_inds_subset=allowed_rearrange_inds_subset,
                seeds=seeds,
                data_dir=self.data_dir,
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
        # visualizer: Optional[VizSuite] = None,
    ):
        observations = self.vector_tasks.get_observations()
        
        npaused, keep, batch = self.remove_paused(observations)
        observations = self._preprocess_observations(batch) if len(keep) > 0 else batch
        assert npaused == 0, f"{npaused} samplers are paused during initialization."

        num_samplers = len(keep)
        recurrent_memory_specification = (
            self.actor_critic.recurrent_memory_specification
        )
        self.rollout_storage.to(self.device)
        self.rollout_storage.set_partition(
            index=self.worker_id, num_parts=self.num_workers
        )

        self.rollout_storage.initialize(
            observations=observations,
            num_samplers=num_samplers,
            recurrent_memory_specification=recurrent_memory_specification,
            action_space=self.actor_critic.action_space,
        )
        self.set_remaining_tasks(num_samplers)
        self.select_expert_subtask_history(keep)
        self.select_expert_action_history(keep)
        self.select_action_history(keep)

        # if visualizer is not None and len(keep) > 0:
        #     visualizer.collect(vector_task=self.vector_tasks, alive=keep)

        return npaused

    def set_remaining_tasks(self, num_samplers: int):
        self.remaining_tasks = self.vector_tasks.command(
            'sampler_attr', ['total_unique'] * num_samplers
        )

    def select_expert_subtask_history(self, keep: List[int]):
        self.expert_subtask_history = [self.expert_subtask_history[i] for i in keep]

    def select_expert_action_history(self, keep: List[int]):
        self.expert_action_history = [self.expert_action_history[i] for i in keep]

    def select_action_history(self, keep: List[int]):
        self.action_history = [self.action_history[i] for i in keep]

    def get_tasks_num_steps_taken(self, num_samplers: int):
        return self.vector_tasks.call(['num_steps_taken'] * num_samplers)

    def get_tasks_unique_ids(self, num_samplers: int):
        task_specs = self.vector_tasks.command('sampler_attr', ['current_task_spec'] * num_samplers)
        return [spec.unique_id for spec in task_specs]


    def act(self):
        with torch.no_grad():
            agent_input = self.rollout_storage.agent_input_for_next_step()
            actor_critic_output, memory = self.actor_critic(**agent_input)
            self.save_data(**agent_input, **{'updated_memory': memory})
            distr = actor_critic_output.distributions
            actions = distr.sample()
        
        return actions, actor_critic_output, memory, agent_input["observations"]

    @staticmethod
    def _active_memory(memory, keep):
        return memory.sampler_select(keep) if memory is not None else memory

    def collect_step_across_all_task_samplers(
        self,
        visualizer=None,
    ) -> int:
        actions, actor_critic_output, memory, _ = self.act()
        flat_actions = su.flatten(self.actor_critic.action_space, actions)
        assert len(flat_actions.shape) == 3

        unique_ids = self.get_tasks_unique_ids(flat_actions.shape[1])
        num_steps_taken = self.get_tasks_num_steps_taken(flat_actions.shape[1])

        # save current action
        for sampler_id in range(flat_actions.shape[1]):
            filename = f"{num_steps_taken[sampler_id]:03d}"
            sampler_dir = os.path.join(self.data_dir, unique_ids[sampler_id])
            self.action_history[sampler_id].append(
                # observations["expert_action"][0, sampler_id, 0].item()
                flat_actions[0, sampler_id].item()
            )
            self.save_json_data(
                data=self.action_history[sampler_id],
                dirpath=os.path.join(sampler_dir, "action_history"),
                filename=filename
            )
        
        outputs: List[RLStepResult] = self.vector_tasks.step(
            su.action_list(self.actor_critic.action_space, flat_actions)
        )
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

        if npaused > 0:
            self.rollout_storage.sampler_select(keep)
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
        self.rollout_storage.add(**to_add_to_storage)
        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=self.rollout_storage,
                    vector_tasks=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        # refresh history when task is done
        self.expert_action_history = [
            action_history * operator.not_(done) for action_history, done
            in zip(self.expert_action_history, self.dones)
        ]
        self.expert_subtask_history = [
            subtask_history * operator.not_(done) for subtask_history, done
            in zip(self.expert_subtask_history, self.dones)
        ]
        self.action_history = [
            action_history * operator.not_(done) for action_history, done
            in zip(self.action_history, self.dones)
        ]
        self.set_remaining_tasks(len(keep))
        self.select_expert_subtask_history(keep)
        self.select_expert_action_history(keep)
        self.select_action_history(keep)
        self.dones = dones

        return npaused

    def save_data(
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: ActionType,
        masks: torch.FloatTensor,
        updated_memory: Memory,
    ):
        assert masks.shape[0] == 1, f"num_steps should be 1."
        nsteps, nsamplers = masks.shape[:2]

        num_steps_taken = self.get_tasks_num_steps_taken(nsamplers)
        unique_ids = self.get_tasks_unique_ids(nsamplers)

        for sampler_id in range(nsamplers):
            sampler_dir = os.path.join(self.data_dir, unique_ids[sampler_id])
            if not os.path.exists(sampler_dir):
                os.makedirs(sampler_dir, exist_ok=True)
            filename = f"{num_steps_taken[sampler_id]:03d}"

            # Save RGB image data
            for dataname in ("rgb", "unshuffled_rgb"):
                self.save_image_data(
                    imgdata=observations[dataname][0, sampler_id],
                    dirpath=os.path.join(sampler_dir, dataname),
                    filename=filename,
                )

            data_dict = {}
            for dataname in (
                "depth", "unshuffled_depth", "semseg", "unshuffled_semseg", 
                "semmap", "unshuffled_semmap", "inventory",
            ):
                data_dict[dataname] = observations[dataname][0, sampler_id]
            
            for dataname in (
                "instseg", "unshuffled_instseg", "pose", "unshuffled_pose", 
            ):
                # dict observations...
                data_dict[dataname] = {
                    k: v[0, sampler_id]
                    for k, v in observations[dataname].items()
                }

            for map_type in ("Unshuffle", "Walkthrough"):
                data_dict[f"updated_semmap_{map_type.lower()}"] = updated_memory.tensor(
                    key="sem_map"
                )[sampler_id, MAP_TYPES_TO_IDX[map_type]]

            self.save_npz_compress_data(
                dirpath=os.path.join(sampler_dir, "npz_data"),
                filename=filename,
                **data_dict,
            )
            # # Save data
            # for dataname in (
            #     "depth", "unshuffled_depth", "semseg", "unshuffled_semseg", 
            #     "semmap", "unshuffled_semmap", "inventory",
            # ):
            #     self.save_numpy_data(
            #         data=observations[dataname][0, sampler_id],
            #         dirpath=os.path.join(sampler_dir, dataname),
            #         filename=filename
            #     )

            # for dataname in (
            #     "instseg", "unshuffled_instseg", "pose", "unshuffled_pose", 
            # ):
            #     # dict observations...
            #     data = {
            #         k: v[0, sampler_id]
            #         for k, v in observations[dataname].items()
            #     }
            #     self.save_numpy_data(
            #         data=data,
            #         dirpath=os.path.join(sampler_dir, dataname),
            #         filename=filename,
            #     )

            # # Save updated semantic 3D map
            # for map_type in ("Unshuffle", "Walkthrough"):
            #     self.save_numpy_data(
            #         data=updated_memory.tensor(
            #             key="sem_map"
            #         )[:, MAP_TYPES_TO_IDX[map_type]],
            #         dirpath=os.path.join(sampler_dir, f"updated_semmap_{map_type.lower()}"),
            #         filename=filename
            #     )

            # update expert subtask/action history and save
            self.expert_subtask_history[sampler_id].append(
                observations["expert_subtask"][0, sampler_id, 0].item()
            )
            self.save_json_data(
                data=self.expert_subtask_history[sampler_id],
                dirpath=os.path.join(sampler_dir, "expert_subtask_history"),
                filename=filename
            )

            self.expert_action_history[sampler_id].append(
                observations["expert_action"][0, sampler_id, 0].item()
            )
            self.save_json_data(
                data=self.expert_action_history[sampler_id],
                dirpath=os.path.join(sampler_dir, "expert_action_history"),
                filename=filename
            )

            inst_detected = observations["instseg"]["inst_detected"][0, sampler_id]
            add_image = False
            for nonzero_obj_id in inst_detected.nonzero():
                object_id = nonzero_obj_id.item()
                for i in range(inst_detected[object_id]):
                    mask = observations["instseg"]["inst_masks"][0, sampler_id][object_id] & (2 ** i)
                    pos = torch.where(mask)
                    xmin = torch.min(pos[1]).item()
                    xmax = torch.max(pos[1]).item()
                    ymin = torch.min(pos[0]).item()
                    ymax = torch.max(pos[0]).item()
                    width = xmax - xmin
                    height = ymax - ymin
                    # Do not save too much small objects
                    if width < 15 and height < 15:
                        continue
                    poly = binary_mask_to_polygon(mask.detach().cpu().numpy())
                    bbox = [xmin, ymin, width, height]
                    area = width * height
                    # update annotation for annotations
                    data_anno = dict(
                        image_id=self.image_id,
                        id=self.coco_id,
                        category_id=object_id+1,
                        bbox=bbox,
                        area=area,
                        segmentation=poly,
                        iscrowd=0,
                    )
                    self.annotations.append(data_anno)
                    self.coco_id += 1
                    add_image = True
            
            if add_image:
                self.images.append(
                    dict(
                        id=self.image_id,
                        file_name=os.path.relpath(
                            os.path.relpath(os.path.join(sampler_dir, "rgb", f"{filename}.png")),
                            os.path.relpath(self.data_dir)
                        ),
                        height=self.config.SCREEN_SIZE,
                        width=self.config.SCREEN_SIZE,
                    )
                )
                self.image_id += 1

    @staticmethod
    def save_image_data(
        imgdata: torch.Tensor,  # H x W x 3
        dirpath: str,
        filename: str,
    ):
        img = Image.fromarray((imgdata.detach().cpu().numpy() * 255).astype(np.uint8))
        _filename = f"{filename}.png"
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        img_path = os.path.join(dirpath, _filename)
        img.save(img_path)
    
    @staticmethod
    def save_numpy_data(
        data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dirpath: str,
        filename: str,
    ):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        data_path = os.path.join(dirpath, filename)
        if isinstance(data, dict):
            data_dict = {}
            for k, v in data.items():
                data_dict[k] = v.detach().cpu().numpy()
            np.save(data_path, data_dict)
        else:
            np.save(data_path, data.detach().cpu().numpy())

    @staticmethod
    def save_npz_compress_data(
        dirpath: str,
        filename: str,
        **data,
    ):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        data_path = os.path.join(dirpath, filename)
        data_dict = {}
        for k1, v1 in data.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    new_key = f"{k1}_{k2}"
                    data_dict[new_key] = v2.detach().cpu().numpy()
            else:
                data_dict[k1] = v1.detach().cpu().numpy()
        
        np.savez_compressed(data_path, **data_dict)

    @staticmethod
    def save_json_data(
        data: Union[List, Dict],
        dirpath: str,
        filename: str,
    ):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        data_path = os.path.join(dirpath, filename)
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)

    def collect_data(
        self,
    ):
        data_collection_completed = False
        try:
            from pycocotools.coco import COCO
            if os.path.exists(os.path.join(self.data_dir, "annotations.json")):
                coco = COCO(os.path.join(self.data_dir, "annotations.json"))
                self.image_id += len(coco.dataset["images"])
                self.coco_id += len(coco.dataset["annotations"])
                self.images.extend(coco.dataset["images"])
                self.annotations.extend(coco.dataset["annotations"])

            self.data_collection()
            data_collection_completed = True
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
            self.save_json_data(
                data=dict(
                    images=self.images,
                    annotations=self.annotations,
                    categories=[
                        {
                            'id': it + 1,
                            'name': name,
                        }
                        for it, name in enumerate(self.config.ORDERED_OBJECT_TYPES)
                    ],
                ),
                dirpath=self.data_dir,
                filename="annotations.json",
            )
            if data_collection_completed:
                if self.worker_id == 0:
                    self.results_queue.put(("train_stopped", 0))
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}]. Training finished successfully."
                )
            else:
                self.results_queue.put(("train_stopped", 1 + self.worker_id))
            self.close()

    def data_collection(self):
        self.initialize_storage_and_viz()
        while True:
            if (
                not any(self.remaining_tasks)
                and all(self.dones)
            ):
                break
            
            # if self.is_distributed:
            #     self.num_workers_done.set("done", str(0))
            #     self.num_workers_steps.set("steps", str(0))

            #     dist.barrier(
            #         device_ids=None
            #         if self.device == torch.device("cpu")
            #         else [self.device.index]
            #     )
            num_paused = self.collect_step_across_all_task_samplers()
            self.rollout_storage.after_updates()                

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
                pass

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

    

class DataCollectionRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        seed: Optional[int] = None,
        mode: str = "train",
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "default",
        extra_tag: str = "",
        distributed_ip_and_port: str = "127.0.0.1:0",
        machine_id: int = 0,
        allowed_scenes: Sequence[str] = None,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
    ):
        self.config = config
        self.output_dir = output_dir
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
        self.allowed_scenes = allowed_scenes
        self.allowed_rearrange_inds_subset = allowed_rearrange_inds_subset

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
        os.makedirs(self.output_dir, exist_ok=True)
        start_time_string_lock_path = os.path.abspath(
            os.path.join(self.output_dir, ".data_collection_start_time_string.lock")
        )
        try:
            with filelock.FileLock(start_time_string_lock_path, timeout=60):
                last_start_time_string_path = os.path.join(
                    self.output_dir, ".allenact_last_start_time_string"
                )
                pathlib.Path(last_start_time_string_path).touch()

                with open(last_start_time_string_path, "r") as f:
                    last_start_time_string_list = f.readlines()

                while True:
                    candidate_str = time.strftime(
                        "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
                    )
                    if (
                        len(last_start_time_string_list) == 0
                        or last_start_time_string_list[0].strip() != candidate_str
                    ):
                        break
                    time.sleep(0.2)

                with open(last_start_time_string_path, "w") as f:
                    f.write(candidate_str)

        except filelock.Timeout as e:
            get_logger().exception(
                f"Could not acquire the lock for {start_time_string_lock_path} for 60 seconds,"
                " this suggests an unexpected deadlock. Please close all AllenAct training processes,"
                " delete this lockfile, and try again."
            )
            raise e

        assert candidate_str is not None
        # candidate_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        return candidate_str

    def worker_devices(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode)
        )
        devices = machine_params.devices

        assert all_equal(devices) or all(
            d.index >= 0 for d in devices
        ), f"Cannot have a mix of CPU and GPU devices (`devices == {devices}`)"

        get_logger().info(f"Using {len(devices)} workers on devices {devices}")
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

    def init_visualizer(self, mode: str):
        if not self.disable_tensorboard:
            # Note: Avoid instantiating anything in machine_params (use Builder if needed)
            machine_params = MachineParams.instance_from(
                self.config.machine_params(mode)
            )
            self.visualizer = machine_params.visualizer

    @staticmethod
    def init_process(mode: str, id: int, to_close_on_termination: "DataCollectionAgent"):
        def create_handler(termination_type: str):
            prefix = f"{termination_type} signal sent to worker {mode}-{id}."
            def handler(_signo, _frame):
                if to_close_on_termination.is_closed:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closed, exiting."
                    )
                    sys.exit(0)
                elif not to_close_on_termination.is_closing:
                    get_logger().info(
                        f"{prefix} Forcing worker {mode}-{id} to close and exiting."
                    )

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
        except Exception as e:
            get_logger().error(f"Encountered Exception. Terminating worker {id}")
            get_logger().exception(traceback.format_exc())
            kwargs["results_queue"].put((f"{mode}_stopped", 1 + id))
        finally:
            return worker

    @staticmethod
    def data_collection_loop(id: int = 0, mode: str = "train", *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = mode
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"train {id} args {engine_kwargs_for_print}")

        engine = DataCollectionRunner.init_worker(DataCollectionAgent, engine_args, engine_kwargs)
        if engine is not None:
            DataCollectionRunner.init_process("DataCollection", id, to_close_on_termination=engine)
            engine.collect_data()

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

    def get_port(self):
        passed_port = int(self.distributed_ip_and_port.split(":")[1])
        if passed_port == 0:
            assert (
                self.machine_id == 0
            ), "Only runner with `machine_id` == 0 can search for a free port."
            distributed_port = find_free_port(
                self.distributed_ip_and_port.split(":")[0]
            )
        else:
            distributed_port = passed_port

        get_logger().info(
            f"Engines on machine_id == {self.machine_id} using port {distributed_port} and seed {self.seed}"
        )

        return distributed_port

    def start_data_collection(
        self,
        max_sampler_processes_per_worker: Optional[int] = None,
    ):
        self._initialize_start_train_or_start_test()
        devices = self.worker_devices(self.mode)
        get_logger().info(
            f'devices: {devices}'
        )
        num_workers = len(devices)

        # Be extra careful to ensure that all models start
        # with the same initializations.
        set_seed(self.seed)

        distributed_port = 0 if num_workers == 1 else self.get_port()
        worker_ids = self.local_worker_ids(self.mode)
        worker_fn = self.data_collection_loop
        
        for worker_id in worker_ids:
            worker_kwargs = dict(
                id=worker_id,
                mode=self.mode,
                config=self.config,
                data_dir=os.path.join(
                    self.output_dir, self.mode
                ),
                results_queue=self.queues["results"],
                checkpoints_queue=None,
                seed=self.seed,
                mp_ctx=self.mp_ctx,
                num_workers=num_workers,
                device=devices[worker_id],
                distributed_ip=self.distributed_ip_and_port.split(":")[0],
                distributed_port=distributed_port,
                max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                first_local_worker_id=worker_ids[0],
                allowed_scenes=self.allowed_scenes,
                allowed_rearrange_inds_subset=self.allowed_rearrange_inds_subset,
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
                    worker = self.mp_ctx.Process(
                        target=worker_fn, kwargs=worker_kwargs,
                    )
                    worker.start()
                else:
                    raise e

            self.processes[self.mode].append(worker)

        get_logger().info(
            f"Started {len(self.processes[self.mode])} processes"
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

                    print(package)

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
                            pass
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
                    pass

        self.processes.clear()
        self._is_closed = True
    
    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="data_collection", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="the path to experiment config file relative the 'experiment_base' directory"
        " (see the `--experiment_base` flag).",
    )

    parser.add_argument(
        "--config_kwargs",
        type=str,
        default=None,
        required=False,
        help="sometimes it is useful to be able to pass additional key-word arguments"
        " to `__init__` when initializing an experiment configuration. This flag can be used"
        " to pass such key-word arugments by specifying them with json, e.g."
        '\n\t--config_kwargs \'{"gpu_id": 0, "my_important_variable": [1,2,3]}\''
        "\nTo see which arguments are supported for your experiment see the experiment"
        " config's `__init__` function. If the value passed to this function is a file path"
        " then we will try to load this file path as a json object and use this json object"
        " as key-word arguments.",
    )

    parser.add_argument(
        "--extra_tag",
        type=str,
        default="",
        required=False,
        help="Add an extra tag to the experiment when trying out new ideas (will be used"
        " as a subdirectory of the tensorboard path so you will be able to"
        " search tensorboard logs using this extra tag). This can also be used to add an extra"
        " organization when running evaluation (e.g. `--extra_tag running_eval_on_great_idea_12`)",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        type=str,
        default="experiment_output",
        help="experiment output folder",
    )

    parser.add_argument(
        "-s", "--seed", required=False, default=None, type=int, help="random seed",
    )
    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default=os.getcwd(),
        type=str,
        help="experiment configuration base folder (default: working directory)",
    )

    parser.add_argument(
        "-m",
        "--max_sampler_processes_per_worker",
        required=False,
        default=None,
        type=int,
        help="maximal number of sampler processes to spawn for each worker",
    )

    parser.add_argument(
        "-l",
        "--log_level",
        default="info",
        type=str,
        required=False,
        help="sets the log_level. it must be one of {}.".format(
            ", ".join(HUMAN_LOG_LEVELS)
        ),
    )

    parser.add_argument(
        "--distributed_ip_and_port",
        dest="distributed_ip_and_port",
        required=False,
        type=str,
        default="127.0.0.1:0",
        help="IP address and port of listener for distributed process with rank 0."
        " Port number 0 lets runner choose a free port. For more details, please follow the"
        " tutorial https://allenact.org/tutorials/distributed-objectnav-tutorial/.",
    )

    parser.add_argument(
        "--machine_id",
        dest="machine_id",
        required=False,
        type=int,
        default=0,
        help="ID for machine in distributed runs. For more details, please follow the"
        " tutorial https://allenact.org/tutorials/distributed-objectnav-tutorial/",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        required=True,
    )

    parser.add_argument(
        "--allowed_scenes_range",
        nargs="+",
        default=[1, 431],
    )

    parser.add_argument(
        "--allowed_rearrange_inds_subset_range",
        nargs="+",
        default=[0, 50]
    )

    return parser

def get_allowed_scenes(
    mode: str,
    scene_range: List[str],
):
    scenes = datagen_utils.get_scenes(mode)
    scene_range = range(int(scene_range[0]), int(scene_range[1]))
    return [
        scene for scene in scenes
        if int(scene[9:]) in scene_range
    ]

def get_allowed_rearrange_inds_subset(
    rearrange_inds_subset_range: List[str],
):
    rearrange_inds_subset = range(int(rearrange_inds_subset_range[0]), int(rearrange_inds_subset_range[1]))
    return rearrange_inds_subset

def data_collect():
    parser = get_argument_parser()
    args = parser.parse_args()

    init_logging(args.log_level)
    get_logger().info("Running with args {}".format(args))

    cfg, srcs = load_config(args)


    DataCollectionRunner(
        seed=args.seed,
        mode=args.mode,
        config=cfg,
        output_dir=args.output_dir,
        machine_id=args.machine_id,
        distributed_ip_and_port=args.distributed_ip_and_port,
        allowed_scenes=get_allowed_scenes(args.mode, args.allowed_scenes_range),
        allowed_rearrange_inds_subset=get_allowed_rearrange_inds_subset(args.allowed_rearrange_inds_subset_range)
    ).start_data_collection(
        max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
    )


if __name__ == "__main__":
    data_collect()