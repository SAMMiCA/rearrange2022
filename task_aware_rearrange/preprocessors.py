from collections import OrderedDict
from typing import List, Callable, Optional, Any, cast, Dict, Sequence, Tuple

import os
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.preprocessors.resnet import ResNetEmbedder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils import spaces_utils as su

from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from task_aware_rearrange.voxel_utils import GridParameters, image_to_semantic_maps


class SubtaskActionExpertPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.input_uuids = input_uuids
        self.output_uuid = output_uuid
        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:

        return gym.spaces.Dict(
            [
                ("action_or_policy", gym.spaces.Discrete(len(RearrangeBaseExperimentConfig.actions()))),
                ("expert_success", gym.spaces.Discrete(2)),
            ]
        )

    def to(self, device: torch.device) -> "SubtaskActionExpertPreprocessor":
        self.device = device
        return self

    def flatten_output(self, unflattened):
        return su.flatten(
            self.observation_space,
            su.torch_point(self.observation_space, unflattened),
        )

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        expert_subtask_and_action = obs[self.input_uuids[0]].to(self.device)    # B x 4
        
        return expert_subtask_and_action[..., -2:]


class SubtaskExpertPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.input_uuids = input_uuids
        self.output_uuid = output_uuid
        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:

        return gym.spaces.Dict(
            [
                ("action_or_policy", gym.spaces.Discrete(NUM_SUBTASKS)),
                ("expert_success", gym.spaces.Discrete(2)),
            ]
        )

    def to(self, device: torch.device) -> "SubtaskExpertPreprocessor":
        self.device = device
        return self

    def flatten_output(self, unflattened):
        return su.flatten(
            self.observation_space,
            su.torch_point(self.observation_space, unflattened),
        )

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        expert_subtask_and_action = obs[self.input_uuids[0]].to(self.device)    # B x 4
        
        return expert_subtask_and_action[..., :2]


class Semantic3DMapPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        fov: int,
        # ordered_object_types: Sequence[str],
        # class_to_color: Dict[str, Tuple[int, ...]],
        # class_mapping: List[int],
        num_semantic_classes: int,
        num_additional_channels: int = 3,
        grid_parameters: GridParameters = GridParameters(),
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.fov = fov
        self.grid_parameters = grid_parameters
        # self.ordered_object_types = ordered_object_types
        
        # assert self.ordered_object_types == sorted(self.ordered_object_types)
        # self.class_to_color = class_to_color
        # self.class_mapping = class_mapping
        # self.num_objects = len(self.ordered_object_types) + 1
        self.num_semantic_classes = num_semantic_classes
        self.num_additional_channels = num_additional_channels

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:
        w = int(self.grid_parameters.GRID_SIZE_X / self.grid_parameters.GRID_RES)
        l = int(self.grid_parameters.GRID_SIZE_Y / self.grid_parameters.GRID_RES)
        h = int(self.grid_parameters.GRID_SIZE_Z / self.grid_parameters.GRID_RES)

        return gym.spaces.Box(
            low=0, high=1, dtype=np.bool, 
            shape=(self.num_semantic_classes + self.num_additional_channels, w, l, h),
        )

    def to(self, device: torch.device) -> "Semantic3DMapPreprocessor":
        self.device = device
        return self

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # scene_image = obs[self.input_uuids[0]].to(self.device)  # B x C x H x W
        # obs[self.input_uuids[0]]: B x H x W x 1 LongTensor
        scene_image = torch.as_tensor(obs[self.input_uuids[0]], device=self.device)[..., 0]
        scene_image = F.one_hot(
            scene_image, num_classes=self.num_semantic_classes
        ).permute(0, 3, 1, 2)  # B x C x H x W
        depth_image = obs[self.input_uuids[1]].to(self.device).permute(0, 3, 1, 2)  # B x 1 x H x W
        extrinsics4f = obs[self.input_uuids[2]]["T_world_to_cam"].to(self.device) # B x 1 x 4 x 4
        agent_pos = obs[self.input_uuids[2]]["agent_pos"].to(self.device)

        batch_size = agent_pos.shape[0]

        agent_pos_in_maps = torch.zeros_like(agent_pos, dtype=torch.int32, device=self.device)
        for i in range(3):
            agent_pos_in_maps[:, i:i+1] = (
                (agent_pos[:, i:i+1] - self.grid_parameters.GRID_ORIGIN[i]) / self.grid_parameters.GRID_RES
            ).int()

        agent_pos_maps = []
        for a_pos in agent_pos_in_maps:
            agent_pos_map = torch.zeros((1, *self.observation_space.shape[-3:]), device=self.device)
            agent_pos_map[0, a_pos.long()] = 1.0
            agent_pos_maps.append(agent_pos_map)
        
        agent_pos_maps = torch.stack(agent_pos_maps, dim=0)
        
        sem_maps = image_to_semantic_maps(
            scene=scene_image,
            depth=depth_image,
            extrinsics4f=extrinsics4f,
            hfov_deg=self.fov,
            grid_params=self.grid_parameters
        )

        sem_maps = torch.cat(
            (
                agent_pos_maps,
                sem_maps,
            ),
            dim=1
        )

        return sem_maps.type(torch.bool)