# From https://github.com/valtsblukis/hlsm

from typing import Tuple, Union
import torch
import torch.nn as nn

from kornia.geometry.depth import depth_to_3d
from custom.hlsm.utils.projection_utils import make_pinhole_camera_matrix, project_3d_camera_points_to_2d_pixels, project_3d_points

from custom.hlsm.voxel_grid import DefaultGridParameters, GridParameters, VoxelGrid
from custom.hlsm.utils.projection_utils import scatter_add_and_pool


def image_to_voxels(
    scene: torch.Tensor,
    depth: torch.Tensor,
    extrinsics4f: torch.Tensor,
    hfov_deg: Union[int, float],
    grid_params: GridParameters,
):
    if scene.device == "cpu":
        scene = scene.float()
        depth = depth.float()
    
    point_coords, img = image_to_pointcloud(
        camera_image=scene,
        depth_image=depth,
        extrinsics4f=extrinsics4f,
        hfov_deg=hfov_deg
    )

    return pointcloud_to_voxel(
        point_coordinates=point_coords,
        point_attributes=img,
        grid_params=grid_params,
    )


def image_to_pointcloud(
    camera_image: torch.Tensor,
    depth_image: torch.Tensor,
    extrinsics4f: torch.Tensor,
    hfov_deg: Union[int, float],
    min_depth: float = 0.7,
    max_depth: float = 2.5
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


def pointcloud_to_voxel(
    point_coordinates: torch.Tensor,
    point_attributes: torch.Tensor,
    grid_params: GridParameters,
    min_points_per_voxel: int = 10,
):
    dtype = torch.float32 if str(point_coordinates.device) == "cpu" else torch.half
    point_attributes = point_attributes.type(dtype)

    voxel_data, voxel_occupancy, voxel_size, voxel_origin = create_empty_voxel_data(
        batch_size=point_coordinates.shape[0],
        channels=point_attributes.shape[1],
        params=grid_params,
        device=point_coordinates.device,
        data_dtype=dtype,
    )

    b, c, w, l, h = voxel_data.shape

    # Compute which voxel coordinates (integer) each point falls within
    pt_in_vx_f = (point_coordinates - voxel_origin[:, :, None, None]) / voxel_size
    pt_in_vx = (pt_in_vx_f).long()

    # Compute a mask of which points land within voxel grid bounds
    min_bounds, max_bounds = get_integer_bounds(voxel_data=voxel_data)
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
    vxdata_flat = voxel_data.view([b, c, -1])
    pt_data_flat = point_attributes.view([b, c, -1])
    pt_in_vx_flat = pt_in_vx_flat.view([b, 1, -1])
    pt_in_bound_mask_flat = pt_in_bound_mask.view([b, 1, -1])

    voxel_data_new_pooled, voxel_occupancy_new_pooled = scatter_add_and_pool(
        vxdata_flat,
        pt_data_flat,
        pt_in_bound_mask_flat,
        pt_in_vx_flat,
        pool="max",
        occupancy_threshold=min_points_per_voxel,
    )

    # Convert dtype to save space
    voxel_data_new_pooled = voxel_data_new_pooled.type(dtype)

    # Unflatten the results
    voxel_data_new_pooled = voxel_data_new_pooled.view([b, c, w, l, h])
    voxel_occupancy_new_pooled = voxel_occupancy_new_pooled.view([b, 1, w, l, h])

    return voxel_data_new_pooled, voxel_occupancy_new_pooled, voxel_size, voxel_origin


def create_empty_voxel_data(
    batch_size: int = 1,
    channels: int = 1,
    params: GridParameters = DefaultGridParameters(),
    device: Union[torch.device, str, int] = "cpu",
    data_dtype: torch.dtype = torch.float32,
):
    w = int(params.GRID_SIZE_X / params.GRID_RES)
    l = int(params.GRID_SIZE_Y / params.GRID_RES)
    h = int(params.GRID_SIZE_Z / params.GRID_RES)

    data = torch.zeros([batch_size, channels, w, l, h], device=device, dtype=data_dtype)
    occupancy = torch.zeros([batch_size, 1, w, l, h], device=device, dtype=data_dtype)
    voxel_size = params.GRID_RES
    origin = torch.tensor([params.GRID_ORIGIN], device=device)

    return data, occupancy, voxel_size, origin


def get_centroid_coord_grid_voxel(
    voxel_data: torch.Tensor,
    voxel_occupancy: torch.Tensor,
    voxel_size: float,
    voxel_origin: Tuple[float, float, float],
):
    b, c, w, l, h = voxel_data.shape
    device = voxel_data.device

    # TODO: Assume all origins in the batch are the same
    # Voxel coordinates are taken as the center coordinates of each voxel
    xrng = torch.arange(0, w, device=device).float() * voxel_size + voxel_origin[0, 0] + voxel_size * 0.5
    yrng = torch.arange(0, l, device=device).float() * voxel_size + voxel_origin[0, 1] + voxel_size * 0.5
    zrng = torch.arange(0, h, device=device).float() * voxel_size + voxel_origin[0, 2] + voxel_size * 0.5

    xrng = xrng[:, None, None].repeat((1, l, h))
    yrng = yrng[None, :, None].repeat((w, 1, h))
    zrng = zrng[None, None, :].repeat((w, l, 1))

    grid = torch.stack([xrng, yrng, zrng]).unsqueeze(0).repeat((b, 1, 1, 1, 1))
    return grid


def get_integer_bounds(
    voxel_data: torch.Tensor,
):
    device = voxel_data.device
    b, c, w, l, h = voxel_data.shape
    min_bounds = torch.tensor([0, 0, 0], device=device)
    max_bounds = torch.tensor([w, l, h], device=device)
    return min_bounds, max_bounds


def voxel_3d_observability(
    voxel_data: torch.Tensor,
    voxel_occupancy: torch.Tensor,
    voxel_size: float,
    voxel_origin: Tuple[float, float, float],
    extrinsics4f: torch.Tensor, # B x 1 x 4 x 4
    depth_image: torch.Tensor, # B x 1 x H x W
    hfov_deg: float
):
    b, c, ih, iw = depth_image.shape
    _, _, w, l, h = voxel_data.shape
    device = depth_image.device
    dtype = torch.float32 if str(voxel_data.device) == "cpu" else torch.half
    extrinsics4f = extrinsics4f.to(device).float()

    # Represent voxel grid by a point cloud of voxel centers
    voxel_coordinates_3d_world_meters = get_centroid_coord_grid_voxel(
        voxel_data=voxel_data,
        voxel_occupancy=voxel_occupancy,
        voxel_size=voxel_size,
        voxel_origin=voxel_origin,
    )

    # Project points into camera pixels
    voxel_coordinates_3d_cam_meters = project_3d_points(
        extrinsics4f, voxel_coordinates_3d_world_meters
    )
    voxel_coordinates_cam_pixels, pixel_z = project_3d_camera_points_to_2d_pixels(
        ih, iw, hfov_deg, voxel_coordinates_3d_cam_meters
    )

    # Compute a mask indicating which voxels are within camera FOV and in front of the camera
    voxels_in_image_bounds = (
        (voxel_coordinates_cam_pixels[:, 0:1, :, :, :] > 0) *
        (voxel_coordinates_cam_pixels[:, 0:1, :, :, :] < ih) *
        (voxel_coordinates_cam_pixels[:, 1:2, :, :, :] > 0) *
        (voxel_coordinates_cam_pixels[:, 1:2, :, :, :] < iw) *
        (pixel_z > 0)
    )

    # Map all out-of-bounds pixels to pixel (0, 0) in the image (just as a dummy value that's in-bounds)
    voxel_coordinates_cam_pixels = (voxel_coordinates_cam_pixels * voxels_in_image_bounds.int()).int()
    voxel_coordinates_cam_pixels = voxel_coordinates_cam_pixels[:, 0:2, :, :, :] # drop the z coordinate

    # compute coordinates of each voxel into a 1D flattened image
    voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_cam_pixels[:, 1:2, :, :, :] * iw + voxel_coordinates_cam_pixels[:, 0:1, :, :, :]
    flat_voxel_coordinates_in_flat_cam_pixels = voxel_coordinates_in_flat_cam_pixels.view([b, -1]).long()
    flat_depth_image = depth_image.view([b, -1])

    # Gather the depth image values corresponding to each "voxel"
    flat_voxel_ray_bounce_depths = torch.stack(
        [
            torch.index_select(
                flat_depth_image[i], 
                dim=0,
                index=flat_voxel_coordinates_in_flat_cam_pixels[i]
            )
            for i in range(b)
        ]
    )

    # Depth where a ray cast from the camera to this voxel hits an object in the depth image
    voxel_ray_bounce_depths = flat_voxel_ray_bounce_depths.view([b, 1, w, l, h]) # Unflatten
    voxel_depths = voxel_coordinates_3d_cam_meters[:, 2:3, :, :, :]

    # Compute which voxels are observed by this camera taking depth image into account:
    #       All voxels along camera rays that hit an obstacle are considered observed
    #       Consider a voxel that's immediately behind an observed point as observed
    #       ... to make sure that we consider the voxels that contain objects as observed
    depth_bleed_tolerance = voxel_size / 2
    voxel_observability_mask = torch.logical_and(
        voxel_depths <= voxel_ray_bounce_depths + depth_bleed_tolerance,
        voxels_in_image_bounds
    ).long()

    # Consider all voxels that contain stuff as observed
    voxel_observability_mask = torch.max(
        voxel_observability_mask,
        (
            voxel_data.max(1, keepdim=True).values > 0.2
        ).long()
    )
    voxel_observability_mask = voxel_observability_mask.long()    

    return voxel_observability_mask.type(dtype), voxel_ray_bounce_depths


def image_to_semantic_maps(
    scene: torch.Tensor,
    depth: torch.Tensor,
    extrinsics4f: torch.Tensor,
    hfov_deg: Union[int, float],
    grid_params: GridParameters,
):
    voxel_data, voxel_occupancy, voxel_size, voxel_origin = image_to_voxels(
        scene=scene,
        depth=depth,
        extrinsics4f=extrinsics4f,
        hfov_deg=hfov_deg,
        grid_params=grid_params,
    )
    voxel_observability_mask, voxel_ray_depths = voxel_3d_observability(
        voxel_data=voxel_data, 
        voxel_occupancy=voxel_occupancy, 
        voxel_size=voxel_size, 
        voxel_origin=voxel_origin,
        extrinsics4f=extrinsics4f,
        depth_image=depth,
        hfov_deg=hfov_deg
    )

    semantic_maps = torch.cat(
        (
            voxel_data,
            voxel_occupancy,
            voxel_observability_mask,
        ),
        dim=1
    ).type(torch.bool)

    return semantic_maps


def update_semantic_map(
    sem_map: torch.Tensor,
    sem_map_prev: torch.Tensor,
    map_mask: torch.Tensor,
):
    """
    sem_map [nsampler, nchannels, width, length, height]: sem_maps[step]
    sem_map_prev [nsampler, nchannels, width, length, height]
    map_mask  [nsampler, 1, 1, 1, 1]: map_masks[step]
    """
    sem_map_prev = sem_map_prev * map_mask
    
    # update agent_position_map
    sem_map_prev[:, 0:1] = sem_map[:, 0:1]
    
    # update voxel_data based on current voxel_observability
    sem_map_prev[:, 1:-2] = (
        sem_map_prev[:, 1:-2] * ~sem_map[:, -1:]
        + sem_map[:, 1:-2] * sem_map[:, -1:]
    )
    
    # update voxel_occupancy based on current voxel_observability
    sem_map_prev[:, -2:-1] = (
        sem_map_prev[:, -2:-1] * ~sem_map[:, -1:]
        + sem_map[:, -2:-1] * sem_map[:, -1:]
    )

    # update volxe_observability via max pooling
    sem_map_prev[:, -1:] = torch.max(
        sem_map[:, -1:], sem_map_prev[:, -1:]
    )

    return sem_map_prev
