from typing import List, Callable, Optional, Any, cast, Dict, Sequence

import os
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.preprocessors.resnet import ResNetEmbedder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils import spaces_utils as su
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from custom.constants import NUM_MAP_TYPES, NUM_OBJECT_TYPES, NUM_SUBTASK_TYPES
import custom.hlsm.segmentation_definitions as segdef
from custom.hlsm.unets.unet_5 import UNet5
from custom.hlsm.ops.depth_estimate import DepthEstimate
from custom.hlsm.utils.viz import show_image
from custom.hlsm.image_to_voxel import ImageToVoxels
from custom.hlsm.voxel_3d_observability import Voxel3DObservability
from custom.hlsm.voxel_grid import GridParameters, VoxelGrid, DefaultGridParameters
from custom.voxel_utils import image_to_semantic_maps
from example_utils import ForkedPdb

# class SensorPreprocessorGraph(SensorPreprocessorGraph):
#     def get_observations(
#         self, obs: Dict[str, Any], *args: Any, **kwargs: Any
#     ) -> Dict[str, Any]:
#         """Get processed observations.

#         # Returns

#         Collect observations processed from all sensors and return them packaged inside a Dict.
#         """
#         for uuid in self.compute_order[::-1]:
#             if uuid not in obs:
#                 obs[uuid] = self.preprocessors[uuid].process(obs, *args, **kwargs)

#         return {uuid: obs[uuid] for uuid in self.observation_spaces}

class HLSMSegmentationDepthModel(nn.Module):
    TRAINFOR_SEG = "segmentation"
    TRAINFOR_DEPTH = "depth"
    TRAINFOR_BOTH = "both"

    def __init__(
        self, 
        hparams
    ):
        super().__init__()
        self.hidden_state = 128
        self.semantic_channels = segdef.get_num_objects()

        self.params = hparams.get("perception_model", dict())

        # Training hyperparams
        self.train_for = self.params.get("train_for", self.TRAINFOR_BOTH)

        # Inference hyperparams
        self.depth_t_beta = self.params.get("depth_t_beta", 0.5)
        self.seg_t_beta = self.params.get("seg_t_beta", 1.0)

        # Model hyperparams
        self.distr_depth = self.params.get("distributional_depth", True)
        self.depth_bins = self.params.get("depth_bins", 50)
        self.max_depth_m = self.params.get("max_depth", 5.0)

        assert self.train_for in [self.TRAINFOR_SEG, self.TRAINFOR_DEPTH, self.TRAINFOR_BOTH, None]
        print(f"Training perception model for: {self.train_for}")

        self.net = UNet5(self.distr_depth, self.depth_bins)

        self.iter = nn.Parameter(torch.zeros([1], dtype=torch.double), requires_grad=False)

        self.nllloss = nn.NLLLoss(reduce=True, size_average=True)
        self.celoss = nn.CrossEntropyLoss(reduce=True, size_average=True)
        self.mseloss = nn.MSELoss(reduce=True, size_average=True)
        self.act = nn.LeakyReLU()

    def predict(self, rgb_image):
        with torch.no_grad():
            if self.distr_depth:
                seg_pred, depth_pred = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred * self.seg_t_beta)
                depth_pred = torch.exp(depth_pred * self.depth_t_beta)
                depth_pred = depth_pred / (depth_pred.sum(dim=1, keepdim=True))

                depth_pred = DepthEstimate(depth_pred, self.depth_bins, self.max_depth_m)

                # Filter segmentations
                good_seg_mask = seg_pred > 0.3
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
            else:
                seg_pred, depth_pred = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred)

                good_seg_mask = seg_pred > 0.3
                good_depth_mask = (seg_pred > 0.5).sum(dim=1, keepdims=True) * (depth_pred > 0.9)
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
                depth_pred = depth_pred * good_depth_mask

        return seg_pred, depth_pred
    
    def forward_model(self, rgb_image: torch.tensor):
        return self.net(rgb_image)

    def get_name(self) -> str:
        return "HLSM_segmentation_depth_model"

    def loss(self, batch: Dict):
        return self.forward(batch)

    def forward(self, batch: Dict):
        # Inputs
        observations = batch["observations"]
        rgb_image = observations.rgb_image.float()
        seg_gt = observations.semantic_image.float().clone()
        depth_gt = observations.depth_image.float()

        # Switch to a one-hot segmentation representation
        observations.uncompress()
        seg_gt_oh = observations.semantic_image.float()

        b, c, h, w = seg_gt.shape

        # Model forward pass
        seg_pred, depth_pred = self.forward_model(rgb_image)

        # Depth inference and error signal computation
        c = seg_pred.shape[1]
        seg_flat_pred = seg_pred.permute((0, 2, 3, 1)).reshape([b * h * w, c])
        seg_flat_gt = seg_gt.permute((0, 2, 3, 1)).reshape([b * h * w]).long()

        seg_loss = self.nllloss(seg_flat_pred, seg_flat_gt)

        if self.distr_depth:
            depth_flat_pred = depth_pred.permute((0, 2, 3, 1)).reshape([b * h * w, self.depth_bins])
            depth_flat_gt = depth_gt.permute((0, 2, 3, 1)).reshape([b * h * w])
            depth_flat_gt = ((depth_flat_gt / self.max_depth_m).clamp(0, 0.999) * self.depth_bins).long()
            depth_loss = self.nllloss(depth_flat_pred, depth_flat_gt)

            depth_pred_mean = (torch.arange(0, self.depth_bins, 1, device=depth_pred.device)[None, :, None, None] * torch.exp(depth_pred)).sum(dim=1)
            depth_mae = (depth_pred_mean.view([-1]) - depth_flat_gt).abs().float().mean() * (self.max_depth_m / self.depth_bins)
        else:
            depth_flat_pred = depth_pred.reshape([b, h * w])
            depth_flat_gt = depth_gt.reshape([b, h * w])
            depth_loss = self.mseloss(depth_flat_pred, depth_flat_gt)
            depth_mae = (depth_flat_pred - depth_flat_gt).abs().mean()
        
        seg_pred_distr = torch.exp(seg_pred)

        # Loss computation
        if self.train_for is None:
            raise ValueError("train_for hyperparameter not set")
        if self.train_for == self.TRAINFOR_DEPTH:
            loss = depth_loss
        elif self.train_for == self.TRAINFOR_SEG:
            loss = seg_loss
        elif self.train_for == self.TRAINFOR_BOTH:
            loss = seg_loss + depth_loss
        else:
            raise ValueError(f"Unrecognized train_for setting: {self.train_for}")

        # Outputs
        metrics = {}
        metrics["loss"] = loss.item()
        metrics["seg_loss"] = seg_loss.item()
        metrics["depth_loss"] = depth_loss.item()
        metrics["depth_mae"] = depth_mae.item()

        self.iter += 1

        return loss, metrics

    def _real_time_visualization(self, seg_pred_distr, seg_gt_oh, rgb_image, depth_pred, depth_pred_mean, depth_gt):
        def map_colors_for_viz(cdist):
            # Red - 0,  Blue - 1,    bluegreen - 2,    Yellow - 3
            colors = segdef.get_class_color_vector().to(cdist.device).float() / 255.0
            # B x 4 x 3 x H x W
            colors = colors[None, :, :, None, None]
            cdist = cdist[:, :, None, :, :]
            pdist_c = (cdist * colors).sum(dim=1).clamp(0, 1)
            return pdist_c

        if self.iter.item() % 10 == 0:
            with torch.no_grad():
                seg_pred_viz = map_colors_for_viz(seg_pred_distr)[0].permute((1, 2, 0)).detach().cpu().numpy()
                seg_gt_viz = map_colors_for_viz(seg_gt_oh)[0].permute((1, 2, 0)).detach().cpu().numpy()
                rgb_viz = rgb_image[0].permute((1, 2, 0)).detach().cpu().numpy()

                show_image(rgb_viz, "rgb", scale=1, waitkey=1)
                show_image(seg_pred_viz, "seg_pred", scale=1, waitkey=1)
                show_image(seg_gt_viz, "seg_gt", scale=1, waitkey=1)

                if self.distr_depth:
                    depth_pred_amax_viz = depth_pred[0].argmax(0).detach().cpu().numpy()
                    depth_pred_mean_viz = depth_pred_mean[0].detach().cpu().numpy()
                    depth_pred_std = depth_pred[0].std(0).detach().cpu().numpy()
                    depth_gt_viz = depth_gt[0].permute((1, 2, 0)).detach().cpu().numpy()
                    show_image(depth_pred_amax_viz, "depth_pred_amax", scale=1, waitkey=1)
                    show_image(depth_pred_mean_viz, "depth_pred_mean_viz", scale=1, waitkey=1)
                    show_image(depth_pred_std, "depth_pred_std", scale=1, waitkey=1)
                    show_image(depth_gt_viz, "depth_gt", scale=1, waitkey=1)
                else:
                    depth_pred_viz = depth_pred[0].permute((1, 2, 0)).detach().cpu().numpy()
                    depth_gt_viz = depth_gt[0].permute((1, 2, 0)).detach().cpu().numpy()
                    show_image(depth_pred_viz, "depth_pred", scale=1, waitkey=1)
                    show_image(depth_gt_viz, "depth_gt", scale=1, waitkey=1)


class HLSMSegmentationPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        ordered_object_types: Sequence[str] = None,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        hyperparams: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.height = input_height
        self.width = input_width
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types) + 1

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self.hparams = hyperparams if hyperparams else dict()

        self._seg_model: Optional[HLSMSegmentationDepthModel] = None

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0, high=1, shape=(self.num_objects, self.height, self.width), dtype=np.bool,
        )

    @property
    def seg_model(self) -> HLSMSegmentationDepthModel:
        if self._seg_model is None:
            self._seg_model = HLSMSegmentationDepthModel(self.hparams).to(self.device)

        return self._seg_model
    
    def to(self, device: torch.device) -> "HLSMSegmentationPreprocessor":
        self._seg_model = self.seg_model.to(device)
        self.device = device

        return self

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        pred_seg, _ = self.seg_model.predict(x.to(self.device))
        pred_seg_ordered = pred_seg.new_zeros(self._get_observation_space().shape).repeat(x.shape[0], 1, 1, 1)
    
        for id, obj_type in enumerate(self.ordered_object_types):
            if obj_type in segdef.OBJECT_CLASSES:
                hlsm_obj_id = segdef.object_string_to_intid(obj_type)
                pred_seg_ordered[:, id] = pred_seg[:, hlsm_obj_id]
        
        pred_seg_ordered[:, -1] = pred_seg[:, -1]

        return pred_seg_ordered


class VoxelGridPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        fov: int,
        grid_parameters: GridParameters = GridParameters(),
        ordered_object_types: Sequence[str] = None,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.fov = fov
        self.grid_parameters = grid_parameters
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types) + 1

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        self._image_to_voxels = ImageToVoxels()
        self._voxels_3d_observability = Voxel3DObservability()

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:
        w = int(self.grid_parameters.GRID_SIZE_X / self.grid_parameters.GRID_RES)
        l = int(self.grid_parameters.GRID_SIZE_Y / self.grid_parameters.GRID_RES)
        h = int(self.grid_parameters.GRID_SIZE_Z / self.grid_parameters.GRID_RES)

        return gym.spaces.Box(
            low=0, high=1, dtype=np.bool, shape=(self.num_objects + 2, w, l, h),
        )

    def to(self, device: torch.device) -> "VoxelGridPreprocessor":
        self.device = device
        return self

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        scene_image = obs[self.input_uuids[0]].to(self.device)  # B x C x H x W
        depth_image = obs[self.input_uuids[1]].to(self.device).permute(0, 3, 1, 2)  # B x 1 x H x W
        extrinsics4f = obs[self.input_uuids[2]]["T_world_to_cam"].to(self.device) # B x 1 x 4 x 4

        voxel_grid: VoxelGrid = self._image_to_voxels(
            scene=scene_image,
            depth=depth_image,
            extrinsics4f=extrinsics4f,
            hfov_deg=self.fov,
            grid_params=self.grid_parameters
        )
        voxel_observability_grid, voxel_ray_depths = self._voxels_3d_observability(
            voxel_grid=voxel_grid,
            extrinsics4f=extrinsics4f,
            depth_image=depth_image,
            hfov_deg=self.fov
        )
        voxel_centroid_coordinates = voxel_grid.get_centroid_coord_grid()

        # return {
        #     "voxel_grid_data": voxel_grid.data,
        #     "voxel_grid_occupancy": voxel_grid.occupancy,
        #     "voxel_observability_grid_data": voxel_observability_grid.data,
        #     # "voxel_observability_grid_occupancy": voxel_observability_grid.occupancy,
        #     "voxel_ray_depths": voxel_ray_depths,
        #     "voxel_centroid_coordinates": voxel_centroid_coordinates,
        # }
        return torch.cat(
            (voxel_grid.data, voxel_grid.occupancy, voxel_observability_grid.data),
            dim=1
        ).type(torch.bool)


class Semantic3DMapPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        fov: int,
        grid_parameters: GridParameters = GridParameters(),
        ordered_object_types: Sequence[str] = None,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.fov = fov
        self.grid_parameters = grid_parameters
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types) + 1

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
            low=0, high=1, dtype=np.bool, shape=(self.num_objects + 3, w, l, h),
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
        scene_image = obs[self.input_uuids[0]].to(self.device)  # B x C x H x W
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
                ("action_or_policy", gym.spaces.Discrete(
                    (NUM_SUBTASK_TYPES - 1) * NUM_OBJECT_TYPES * NUM_MAP_TYPES + 1
                )),
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
