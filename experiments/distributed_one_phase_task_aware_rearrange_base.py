from typing import Tuple, Sequence, Optional, Dict, Any
import numpy as np

import torch
from torch import nn, cuda, optim

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.experiment_utils import LinearDecay, PipelineStage
from allenact.utils.system import get_logger
from allenact.utils.misc_utils import all_unique
from experiments.distributed_task_aware_rearrange_base import DistributedTaskAwareRearrangeBaseExperimentConfig
from rearrange.tasks import RearrangeTaskSampler


class DistributedOnePhaseTaskAwareRearrangeBaseExperimentConfig(DistributedTaskAwareRearrangeBaseExperimentConfig):

    NUM_DISTRIBUTED_NODES: int = 2
    NUM_DEVICES: Sequence[int] = [1, 2]
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = None
    
    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in sensors
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            **kwargs,
        )