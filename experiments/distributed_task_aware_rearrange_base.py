from typing import Tuple, Sequence, Optional, Dict, Any
import numpy as np

import torch
from torch import nn, cuda, optim

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import LinearDecay, PipelineStage
from allenact.utils.system import get_logger
from allenact.utils.misc_utils import all_unique
from experiments.task_aware_rearrange_base import TaskAwareRearrangeBaseExperimentConfig


class DistributedTaskAwareRearrangeBaseExperimentConfig(TaskAwareRearrangeBaseExperimentConfig):

    NUM_DISTRIBUTED_NODES: int = 2
    NUM_DEVICES: Sequence[int] = [1, 2]
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = None
    
    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> MachineParams:
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        nprocesses = 1
        devices = (
            list(range(min(nprocesses, num_gpus)))
            if has_gpu
            else [torch.device("cpu")]
        )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )
        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=cls.create_preprocessor_graph(mode=mode),
            # visualizer=self.create_visualizer(mode=self.mode),
        )
        
        if mode == "train":
            devices = sum(
                [
                    list(range(min(cls.num_train_processes(), cls.NUM_DEVICES[idx])))
                    if cls.NUM_DEVICES[idx] > 0 and cuda.is_available()
                    else torch.device("cpu")
                    for idx in range(cls.NUM_DISTRIBUTED_NODES)
                ], []
            )
            params.devices = tuple(
                torch.device("cpu") if d == -1 else torch.device(d) for d in devices
            )
            params.sampler_devices = params.devices

            params.nprocesses = sum(
                [
                    split_processes_onto_devices(
                        cls.num_train_processes() if cuda.is_available() and cls.NUM_DEVICES[idx] > 0 else 1,
                        cls.NUM_DEVICES[idx] if cls.NUM_DEVICES[idx] > 0 else 1
                    )
                    for idx in range(cls.NUM_DISTRIBUTED_NODES)
                ], []
            )

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                    0 <= machine_id < cls.NUM_DISTRIBUTED_NODES
                ), f"machine_id {machine_id} out of range [0, {cls.NUM_DISTRIBUTED_NODES - 1}"
                machine_num_gpus = cuda.device_count()
                machine_has_gpu = machine_num_gpus != 0
                assert (
                    0 <= cls.NUM_DEVICES[machine_id] <= machine_num_gpus
                ), f"Number of devices for machine_id {machine_id} exceeds the number of gpus"

                local_worker_ids = list(
                    range(
                        sum(cls.NUM_DEVICES[:machine_id]),
                        sum(cls.NUM_DEVICES[:machine_id + 1])
                    )
                )
                params.set_local_worker_ids(local_worker_ids)

            # )
        
        elif mode == "valid":
            # nooooo
            params = super().machine_params(mode, **kwargs)
            params.nprocesses = (0, )
        
        return params

    @classmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        return dict(
            named_losses=dict(
                ppo_loss=PPO(clip_decay=LinearDecay(training_steps), **PPOConfig)
            ),
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=training_steps,)
            ],
            num_steps=64,
            num_mini_batch=1,
            update_repeats=3,
            use_lr_decay=True,
            lr=3e-4,
        )

    @classmethod
    def num_train_processes(cls) -> int:
        return 8

    @classmethod
    def tag(cls) -> str:
        return f"TestDistributedVersion"