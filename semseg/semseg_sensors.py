from typing import Dict, Optional, Tuple, Any, Sequence, Union, cast
from abc import ABC
import gym
import numpy as np
from allenact.embodiedai.sensors.vision_sensors import VisionSensor
from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.task import SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from rearrange.environment import RearrangeTHOREnvironment
from rearrange.tasks import (
    UnshuffleTask,
    WalkthroughTask,
    AbstractRearrangeTask,
)
from task_aware_rearrange.constants import UNKNOWN_OBJECT_STR

class SemanticRearrangeSensor(VisionSensor[EnvType, SubTaskType], ABC):
    
    def __init__(
        self,
        ordered_object_types: Sequence[str],
        class_to_color: Dict[str, Tuple[int, ...]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "semantic",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        which_task_env: Optional[str] = None,
        **kwargs: Any,
    ):
        self.which_task_env = which_task_env
        self.ordered_object_types = ordered_object_types
        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        self.num_objects = len(self.ordered_object_types)
        # self.object_type_to_idx[UNKNOWN_OBJECT_STR] = len(self.ordered_object_types)
        # self.num_objects = len(self.ordered_object_types) + 1
        self._class_to_color = class_to_color     # num_objects x 3
        
        super().__init__(**prepare_locals_for_super(locals()))
        
    def _make_observation_space(
        self,
        output_shape: Optional[Tuple[int,...]],
        output_channels: Optional[int],
        unnormalized_infimum: float,
        unnormalized_supremum: float,
    ) -> gym.spaces.MultiDiscrete:
        assert output_shape is None or output_channels is None, (
            "In VisionSensor's config, "
            "only one of output_shape and output_channels can be not None."
        )
        
        shape: Optional[Tuple[int, ...]] = None
        if output_shape is not None:
            shape = output_shape
        elif self._height is not None and output_channels is not None:
            shape = (
                cast(int, self._height),
                cast(int, self._width),
                cast(int, output_channels),
            )
            
        return gym.spaces.MultiDiscrete(
            np.full(shape, self.num_objects)
        )
        
    def get_segmentation(self, semseg_frame: np.ndarray, task):
        seg_frame = semseg_frame
        seg_frame = seg_frame.copy().reshape(self._height, self._width, 1, 3)
        c = np.array(list(self._class_to_color.values()))[1:].reshape(1, 1, self.num_objects - 1, 3)
        padding_dims = ((0, 0), (0, 0), (1, 0))
        class_masks = (abs(seg_frame - c).sum(axis=3) == 0).astype(np.float32)

        return np.pad(
            class_masks,
            padding_dims,
            mode="constant",
            constant_values=0.1
        ).argmax(axis=-1, keepdims=True)
        
    def get_observation(
        self,
        env: RearrangeTHOREnvironment,
        task: Union[WalkthroughTask, UnshuffleTask],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        
        env = (
            task.walkthrough_env
            if isinstance(task, WalkthroughTask)
            else (
                task.walkthrough_env
                if self.which_task_env == "walkthrough"
                else task.unshuffle_env
            )
        )
        
        return self.get_segmentation(env.last_event.semantic_segmentation_frame, task)