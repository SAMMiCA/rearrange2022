# From https://github.com/valtsblukis/hlsm

import torch
import torch.nn as nn
from custom.hlsm.voxel_grid import GridParameters, VoxelGrid
from custom.hlsm.utils.projection_utils import scatter_add_and_pool

# This many points need to land within a voxel for that voxel to be considered "occupied".
# Too low, and noisy depth readings can generate obstacles.
# Too high, and objects far away don't register in the map
MIN_POINTS_PER_VOXEL = 10


class PointcloudToVoxels(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        point_coordinates: torch.Tensor,
        point_attributes: torch.Tensor,
        grid_params: GridParameters,
    ):
        dtype = torch.float32 if str(point_coordinates.device) == "cpu" else torch.half
        point_attributes = point_attributes.type(dtype)

        voxelgrid = VoxelGrid.create_empty(
            batch_size=point_coordinates.shape[0],
            channels=point_attributes.shape[1],
            params=grid_params,
            device=point_coordinates.device,
            data_dtype=dtype,
        )

        b, c, w, l, h = voxelgrid.data.shape

        # Compute which voxel coordinates (integer) each point falls within
        pt_in_vx_f = (point_coordinates - voxelgrid.origin[:, :, None, None]) / voxelgrid.voxel_size
        pt_in_vx = (pt_in_vx_f).long()

        # Compute a mask of which points land within voxel grid bounds
        min_bounds, max_bounds = voxelgrid.get_integer_bounds()
        pt_in_bound_mask = torch.logical_and(
            pt_in_vx >= min_bounds[None, :, None, None],
            pt_in_vx < max_bounds[None, :, None, None]
        )
        pt_in_bound_mask = pt_in_bound_mask.min(dim=1, keepdim=True).values
        num_oob_pts = (pt_in_bound_mask.int() == 0).int().sum().detach().cpu().item()
        # if num_oob_pts > 20000:
        #     print(f"Number of OOB points: {num_oob_pts}")

        # Convert coordinates into a flattened voxel grid
        pt_in_vx_flat = pt_in_vx[:, 0] * l * h + pt_in_vx[:, 1] * h + pt_in_vx[:, 2]

        # Flatten spatial coordinates so that we can run the scatter operation
        vxdata_flat = voxelgrid.data.view([b, c, -1])
        pt_data_flat = point_attributes.view([b, c, -1])
        pt_in_vx_flat = pt_in_vx_flat.view([b, 1, -1])
        pt_in_bound_mask_flat = pt_in_bound_mask.view([b, 1, -1])

        voxeldata_new_pooled, voxeloccupancy_new_pooled = scatter_add_and_pool(
            vxdata_flat,
            pt_data_flat,
            pt_in_bound_mask_flat,
            pt_in_vx_flat,
            pool="max",
            occupancy_threshold=MIN_POINTS_PER_VOXEL,
        )

        # Convert dtype to save space
        voxeldata_new_pooled = voxeldata_new_pooled.type(dtype)

        # Unflatten the results
        voxeldata_new_pooled = voxeldata_new_pooled.view([b, c, w, l, h])
        voxeloccupancy_new_pooled = voxeloccupancy_new_pooled.view([b, 1, w, l, h])

        voxelgrid_new = VoxelGrid(
            voxeldata_new_pooled,
            voxeloccupancy_new_pooled,
            voxelgrid.voxel_size,
            voxelgrid.origin
        )

        return voxelgrid_new