from typing import Tuple, Sequence, Optional, Dict, Any, Type
import gym
import numpy as np

import torch
from torch import nn, cuda, optim
import torchvision

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.experiment_config import (
    MachineParams,
    split_processes_onto_devices,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from baseline_configs.one_phase.one_phase_rgb_base import OnePhaseRGBBaseExperimentConfig
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, UnshuffledRGBRearrangeSensor
from custom.models import TaskAwareOnePhaseRearrangeBaseNetwork
from experiments.task_aware_rearrange_base import TaskAwareRearrangeBaseExperimentConfig
from rearrange.tasks import RearrangeTaskSampler


class OnePhaseTaskAwareRearrangeBaseExperimentConfig(TaskAwareRearrangeBaseExperimentConfig):
    
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        def get_sensor_uuid(stype: Type[Sensor]) -> Optional[str]:
            s = next((s for s in cls.sensors() if isinstance(s, stype)), None,)
            return None if s is None else s.uuid

        walkthrougher_should_ignore_action_mask = [
            any(k in a for k in ["drop", "open", "pickup"]) for a in cls.actions()
        ]

        return TaskAwareOnePhaseRearrangeBaseNetwork(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.UNSHUFFLED_RGB_RESNET_UUID,
            in_walkthrough_phase_uuid=get_sensor_uuid(InWalkthroughPhaseSensor),
            depth_uuid=cls.DEPTH_UUID,
            unshuffled_depth_uuid=cls.UNSHUFFLED_DEPTH_UUID,
            sem_seg_uuid=cls.SEMANTIC_SEGMENTATION_UUID,
            unshuffled_sem_seg_uuid=cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID,
            pose_uuid=cls.POSE_UUID,
            unshuffled_pose_uuid=cls.UNSHUFFLED_POSE_UUID,
            inventory_uuid=cls.INVENTORY_UUID,
            subtask_expert_uuid=cls.SUBTASK_EXPERT_UUID,
            sem_map_uuid=cls.SEMANTIC_MAP_UUID,
            unshuffled_sem_map_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID,
            is_walkthrough_phase_embedding_dim=cls.IS_WALKTHROUGH_PHASE_EMBEDING_DIM,
            rnn_type=cls.RNN_TYPE,
            walkthrougher_should_ignore_action_mask=walkthrougher_should_ignore_action_mask,
            done_action_index=cls.actions().index("done"),
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
        )

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