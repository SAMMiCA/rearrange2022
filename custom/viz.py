from typing import Sequence, Tuple
import torch
import open3d as o3d
import numpy as np

from allenact.utils.viz_utils import AbstractViz

import custom.hlsm.segmentation_definitions as segdef
from custom.hlsm.utils.render3d import show_geometries
from custom.hlsm.voxel_grid import DefaultGridParameters, VoxelGrid


def object_intid_to_color(intid: int, object_types: Sequence[str]) -> Tuple[int, int, int]:
    if intid < len(object_types):
        object_str = object_types[intid]
        if object_str in segdef.OBJECT_CLASSES:
            color = segdef.object_string_to_color(object_str)
        else:
            color = (0, 0, 0)
    elif intid == len(object_types):
        object_str = "Unknown"
        color = segdef.object_string_to_color(object_str)
    else:
        raise ValueError("Wrong intid")
    
    return color


def intid_tensor_to_rgb(data: torch.tensor, object_types: Sequence[str]) -> torch.tensor:
    b, nc = data.shape[:2]

    data = data.float()

    num_spatial_dims = len(data.shape) - 2
    rgb_tensor = data.new_zeros((b, 3, *data.shape[2:]))
    count_tensor = data.new_zeros((b, 1, *data.shape[2:]))

    for c in range(nc):
        c_slice = data[:, c:c+1]
        c_count_slice = c_slice > 0.01
        rgb_color = object_intid_to_color(c, object_types)

        rgb_color = torch.tensor(rgb_color, device=data.device).unsqueeze(0)

        for _ in range(num_spatial_dims):
            rgb_color = rgb_color.unsqueeze(2)
        rgb_slice = c_slice * rgb_color
        count_tensor += c_count_slice
        rgb_tensor += rgb_slice

    rgb_avg_tensor = rgb_tensor / (count_tensor + 1e-10)
    return rgb_avg_tensor / 255


def voxel_tensors_to_geometry(
    data: torch.tensor,
    occupancy: torch.tensor,
    centroid_coords: torch.tensor,
    voxel_size: float,
):
    coord_grid = centroid_coords
    occupied_mask = occupancy > 0.5

    occupied_coords = coord_grid[occupied_mask.repeat((1, 3, 1, 1, 1))]
    occupied_data = data[occupied_mask.repeat((1, data.shape[1], 1, 1, 1))]

    pcd = o3d.geometry.PointCloud()
    np_points = occupied_coords.view([3, -1]).permute((1, 0)).detach().cpu().numpy()
    centroid = np_points.sum(0) / (np_points.shape[0] + 1e-10)

    np_colors = occupied_data.view([3, -1]).permute((1, 0)).detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    o3dvoxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    return o3dvoxels, centroid

def view_voxel_grid(
    data: torch.tensor,
    occupancy: torch.tensor,
    centroid_coords: torch.tensor,
    voxel_size: float = DefaultGridParameters.GRID_RES,
):
    geometry, centroid = voxel_tensors_to_geometry(
        data=data, occupancy=occupancy, centroid_coords=centroid_coords, voxel_size=voxel_size
    )

    show_geometries(geometry)

def render_voxel_grid(
    data: torch.tensor,
    occupancy: torch.tensor,
    centroid_coords: torch.tensor,
    voxel_size: float = DefaultGridParameters.GRID_RES,
    animate: bool = False,
):
    geometry, centroid = voxel_tensors_to_geometry(
        data=data, occupancy=occupancy, centroid_coords=centroid_coords, voxel_size=voxel_size
    )

    frame_or_frames = render_geometries(geometry)

    return frame_or_frames


def render_geometries(geometry, animate=False, num_frames=18, centroid=None, visible=False):
    vis = o3d.visualization.Visualizer(visible=visible)
    vis.create_window()

    vis.add_geometry(geometry)
    vis.update_geometry(geometry)
    vis.update_renderer()

    # change view point 

    p = vis.capture_screen_float_buffer(do_render=True)
    f = np.asarray(p)
    vis.destroy_window()

    return f
