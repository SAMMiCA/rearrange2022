# From https://github.com/valtsblukis/hlsm

import torch.nn as nn
import torch

from custom.hlsm.image_to_pc import ImageToPointcloud
from custom.hlsm.pc_to_voxel import PointcloudToVoxels


class ImageToVoxels(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_to_pointcloud = ImageToPointcloud()
        self.pointcloud_to_voxels = PointcloudToVoxels()

    def forward(self, scene, depth, extrinsics4f, hfov_deg, grid_params, mark_agent=False):
        # CPU doesn't support most of the half-precision operations.
        if scene.device == "cpu":
            scene = scene.float()
            depth = depth.float()
        point_coords, img = self.image_to_pointcloud(scene, depth, extrinsics4f, hfov_deg)
        voxel_grid = self.pointcloud_to_voxels(point_coords, img, grid_params)
        return voxel_grid