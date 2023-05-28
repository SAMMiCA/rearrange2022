import torch
import cv2
import torch.nn.functional as functional
import numpy as np
import torchvision.transforms as transforms

from gym.spaces import MultiDiscrete
from abc import ABC
from typing import Optional, Sequence, Dict, Union, Tuple, Any, cast, List
from collections import OrderedDict

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.sensors.vision_sensors import VisionSensor
from allenact_plugins.ithor_plugin.ithor_sensors \
    import RelativePositionChangeTHORSensor

from rearrange.sensors import RGBRearrangeSensor
from rearrange.sensors import UnshuffledRGBRearrangeSensor
from rearrange.sensors import DepthRearrangeSensor

from rearrange.tasks import RearrangeTaskSampler
from rearrange.tasks import UnshuffleTask
from rearrange.tasks import WalkthroughTask

from rearrange.environment import RearrangeTHOREnvironment
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.task import SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from ai2thor.platform import CloudRendering

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


class OnePhaseSegmentationConfig(RearrangeBaseExperimentConfig, ABC):
    """Create a training session using the AI2-THOR Rearrangement task,
    including additional map_depth and semantic segmentation observations
    and expose a task sampling function.

    """

    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
        ),
        UnshuffledRGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
        ),
        DepthRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE
        ),
    ]

    @classmethod
    def make_sampler_fn(cls, stage: str, force_cache_reset: bool,
                        allowed_scenes: Optional[Sequence[str]],
                        seed: int, epochs: int,
                        scene_to_allowed_rearrange_inds:
                        Optional[Dict[str, Sequence[int]]] = None,
                        x_display: Optional[str] = None,
                        sensors: Optional[Sequence[Sensor]] = None,
                        thor_controller_kwargs: Optional[Dict] = None,
                        device: str = 'cuda:0', ground_truth: bool = False,
                        detection_threshold: float = 0.8,
                        **kwargs) -> RearrangeTaskSampler:
        """Helper function that creates an object for sampling AI2-THOR
        Rearrange tasks in walkthrough and unshuffle phases, where additional
        semantic segmentation and map_depth observations are provided.

        Arguments:

        device: str
            specifies the device used by torch during the color lookup
            operation, which can be accelerated when set to a cuda device.

        Returns:

        sampler: RearrangeTaskSampler
            an instance of RearrangeTaskSampler that implements next_task()
            for generating walkthrough and unshuffle tasks successively.

        """

        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        # add a semantic segmentation observation sensor to the list
        sensors = (
            SemanticRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                device=device, ground_truth=ground_truth,
                which_task_env="walkthrough", uuid="semantic",
                detection_threshold=detection_threshold,
            ),
            SemanticRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                device=device, ground_truth=ground_truth,
                which_task_env="unshuffle", uuid="unshuffled_semantic",
                detection_threshold=detection_threshold,
            ),
            *(cls.SENSORS if sensors is None else sensors)
        )

        # allow default controller arguments to be overridden
        controller_kwargs = dict(**cls.THOR_CONTROLLER_KWARGS)
        if thor_controller_kwargs is not None:
            controller_kwargs.update(thor_controller_kwargs)

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
                    "platform": CloudRendering,
                    "renderDepthImage": any(
                        isinstance(sensor, DepthRearrangeSensor)
                        for sensor in sensors
                    ),
                    "renderSemanticSegmentation": any(
                        isinstance(sensor, SemanticRearrangeSensor)
                        for sensor in sensors
                    ),
                    **controller_kwargs,
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
