# From https://github.com/valtsblukis/hlsm

import numpy as np
import torch
import torch.nn as nn
import imageio

from custom.hlsm.utils.projection_utils import make_pinhole_camera_matrix
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.depth import depth_to_3d



class ImageToPointcloud(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        camera_image,
        depth_image,
        extrinsics4f,
        hfov_deg,
        min_depth=0.7,
        max_depth=2.5
    ):
        b, c, h, w = camera_image.shape
        dev = camera_image.device

        intrinsics = make_pinhole_camera_matrix(
            height_px=h,
            width_px=w,
            hfov_deg=hfov_deg,
        )
        intrinsics = intrinsics.to(dev)
        intrinsics = intrinsics[None, :, :].repeat((b, 1, 1))

        # Extrinsics project world points to camera
        extrinsics = extrinsics4f.to(dev).float()

        # Inverse extrinsics project camera points to the world
        inverse_extrinsics = extrinsics.inverse()

        if inverse_extrinsics.shape[0] == 1 and b > 1:
            inverse_extrinsics = inverse_extrinsics.repeat((b, 1, 1))

        # Points3D - 1 x 3 x H x W grid of coordinates
        pts_3d_wrt_cam = depth_to_3d(
            depth=depth_image,
            camera_matrix=intrinsics,
            normalize_points=False
        )

        has_depth = (depth_image > min_depth) * (depth_image < max_depth)


        # Project to world reference frame by applying the extrinsic homogeneous transformation matrix
        homo_ones = torch.ones_like(pts_3d_wrt_cam[:, 0:1, :, :])
        homo_pts_3d_wrt_cam = torch.cat([pts_3d_wrt_cam, homo_ones], dim=1)
        homo_pts_3d_wrt_world = torch.einsum("bxhw,byx->byhw", homo_pts_3d_wrt_cam, inverse_extrinsics)
        # homo_pts_3d_wrt_world = torch.einsum("byx,bxhw->byhw", extrinsics, homo_pts_3d_wrt_cam)
        pts_3d_wrt_world = homo_pts_3d_wrt_world[:, :3, :, :]

        pts_3d_wrt_world = pts_3d_wrt_world * has_depth
        camera_image = camera_image * has_depth

        return pts_3d_wrt_world, camera_image
