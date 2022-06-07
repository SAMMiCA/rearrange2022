from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.algorithms.onpolicy_sync.policy import FullMemorySpecType, ActionType
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init

from custom.hlsm.image_to_voxel import ImageToVoxels
from custom.hlsm.voxel_3d_observability import Voxel3DObservability
from custom.hlsm.voxel_grid import VoxelGrid


class HLSMSpatialReprModelTest(ActorCriticModel[CategoricalDistr]):

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        depth_uuid: str,
        sem_seg_uuid: str,
        pose_uuid: str,
        inventory_uuid: str,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        fov: float = 90.0,
        rnn_type: str = "LSTM",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        
        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.depth_uuid = depth_uuid
        self.sem_seg_uuid = sem_seg_uuid
        self.pose_uuid = pose_uuid
        self.inventory_uuid = inventory_uuid

        self.fov = fov
        self.im2vx = ImageToVoxels()
        self.vx3Dobs = Voxel3DObservability()


    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return super()._recurrent_memory_specification()

    def forward(
        self, 
        observations: ObservationType, 
        memory: Memory, 
        prev_actions: ActionType, 
        masks: torch.FloatTensor
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        scene_img = observations[self.sem_seg_uuid]
        depth_img = observations[self.depth_uuid]
        extrinsics4f = observations[self.pose_uuid]["T_world_to_cam"]
        inventory = observations[self.inventory_uuid]
        fov = self.fov

        voxel_grid: VoxelGrid = self.im2vx(
            scene=scene_img,
            depth=depth_img,
            extrinsics4f=extrinsics4f,
            hfov_deg=fov,
        )
        voxel_observability_grid, voxel_ray_depths = self.vx3Dobs(
            voxel_grid=voxel_grid,
            extrinsics4f=extrinsics4f,
            depth_image=depth_img,
            hfov_deg=fov,
        )

        voxel_grid_data = 


        return super().forward(observations, memory, prev_actions, masks)