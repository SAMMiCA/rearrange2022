# %%
from typing import List, Optional
import numpy as np
import torch

from custom.hlsm.voxel_grid import DefaultGridParameters
from experiments.one_phase_task_aware_rearrange_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor, include_object_data


NUM_PROCESSES = 4
DEVICE = "cpu"
TRAIN_MODE_STR = "train"
VALID_MODE_STR = "valid"
TEST_MODE_STR = "test"


# def get_sampler_fn_args(mode: str, seeds: Optional[List[int]] = None):
#     if mode == TRAIN_MODE_STR:
#         fn = SensorTestExperimentConfig.train_task_sampler_args
#     elif mode == VALID_MODE_STR:
#         fn = SensorTestExperimentConfig.valid_task_sampler_args
#     elif mode == TEST_MODE_STR:
#         fn = SensorTestExperimentConfig.test_task_sampler_args
#     else:
#         raise NotImplementedError(
#             f"mode must be one of ('train', 'valid', 'test')"
#         )
    
#     total_processes = NUM_PROCESSES
#     device = torch.device(DEVICE)
    
#     sampler_devices_as_ints: Optional[List[int]] = None


#     if mode == TEST_MODE_STR and device.index is not None:
#         sampler_devices_as_ints = [device.index]
#     elif sampler_devices

task_sampler_params = OnePhaseTaskAwareRearrangeBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1
)
one_phase_rgb_task_sampler: RearrangeTaskSampler = (
    OnePhaseTaskAwareRearrangeBaseExperimentConfig.make_sampler_fn(
        **task_sampler_params, force_cache_reset=False, epochs=1
    )
)
#%%
manual = False
for _ in range(14):
    task = one_phase_rgb_task_sampler.next_task()
obs = task.get_observations()

# import pdb; pdb.set_trace()131313
# while soap_id not in task.env.last_event.instance_masks.keys():
while True:
    # act = task.action_space.sample()
    if manual:
        act = int(input(f'input = '))
        while act == -1:
            import pdb; pdb.set_trace()
            act = int(input(f'input = '))
    else:
        act = int(obs['subtask_and_action_expert'][2])

    obs = task.step(act).observation
    if act == 0:
        print(task.metrics())
        import pdb; pdb.set_trace()
        task = one_phase_rgb_task_sampler.next_task()
        obs = task.get_observations()
import pdb; pdb.set_trace()
############################################################################################
# from allenact.base_abstractions.sensor import SensorSuite
# from allenact.base_abstractions.experiment_config import MachineParams
# from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
# from allenact.utils.experiment_utils import Builder


# preprocessors = []
# additional_output_uuids = ["rgb", "semseg", "depth", "pose"]

# # preprocessors.append(
# #     SensorTestExperimentConfig.create_hlsm_segmentation_builder(
# #         in_uuid="rgb",
# #         out_uuid="semseg_pred"
# #     )
# # )
# preprocessors.append(
#     SensorTestExperimentConfig.create_voxel_preprocessor_builder(
#         in_uuids=["semseg", "depth", "pose"],
#         out_uuid="voxel"
#     )
# )

# sensor_preprocessor_graph = Builder(
#     SensorPreprocessorGraph,
#     {
#         "source_observation_spaces": SensorSuite(task_sampler_params['sensors']).observation_spaces,
#         "preprocessors": preprocessors,
#         "additional_output_uuids": additional_output_uuids,
#     }
# )()

# obs_b = dict()
# for k, v in obs.items():
#     if isinstance(v, dict):
#         obs_b[k] = dict()
#         for k1, v1 in v.items():
#             obs_b[k][k1] = torch.from_numpy(v1.copy()).unsqueeze(0).float()
#     else:
#         obs_b[k] = torch.from_numpy(v.copy()).unsqueeze(0).float()

# processed_obs_b = sensor_preprocessor_graph.get_observations(obs_b)
############################################################################################

############################################################################################

import os
experimental_base = os.getcwd()
rel_base_dir = os.path.relpath(
    os.path.abspath(experimental_base), os.getcwd()
)
rel_base_dot_path = rel_base_dir.replace("/", ".")
if rel_base_dot_path == ".":
    rel_base_dot_path = ""

exp_dot_path = "experiments/sensor_test.py"
if exp_dot_path[-3:] == ".py":
    exp_dot_path = exp_dot_path[:-3]
exp_dot_path = exp_dot_path.replace("/", ".")

module_path = (
    f"{rel_base_dot_path}.{exp_dot_path}"
    if len(rel_base_dot_path) != 0
    else exp_dot_path
)

from typing import Optional, List, Dict, Any, DefaultDict, Union, cast
from collections import defaultdict

def find_sub_modules(path: str, module_list: Optional[List] = None):
    if module_list is None:
        module_list = []

    path = os.path.abspath(path)
    if path[-3:] == ".py":
        module_list.append(path)
    elif os.path.isdir(path):
        contents = os.listdir(path)
        if any(key in contents for key in ["__init__.py", "setup.py"]):
            new_paths = [os.path.join(path, f) for f in os.listdir(path)]
            for new_path in new_paths:
                find_sub_modules(new_path, module_list)
    return module_list
    
import importlib
try:
    importlib.invalidate_caches()
    module = importlib.import_module(module_path)
except ModuleNotFoundError as e:
    if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
        raise e
    all_sub_modules = set(find_sub_modules(os.getcwd()))
    desired_config_name = module_path.split(".")[-1]
    relevant_submodules = [
        sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
    ]
    raise ModuleNotFoundError(
        f"Could not import experiment '{module_path}', are you sure this is the right path?"
        f" Possibly relevant files include {relevant_submodules}."
        f" Note that the experiment must be reachable along your `PYTHONPATH`, it might"
        f" be helpful for you to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your"
        f" project's top level directory."
    ) from e

import inspect
import numbers
from allenact.base_abstractions.experiment_config import ExperimentConfig

experiments = [
    m[1]
    for m in inspect.getmembers(module, inspect.isclass)
    if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
]

config = experiments[0]()

mode = "valid"
device = "cpu"
machine_params = config.machine_params(mode)
create_model_kwargs = {}
sensor_preprocessor_graph = None
if machine_params.sensor_preprocessor_graph is not None:
    sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(device)
    create_model_kwargs["sensor_preprocessor_graph"] = sensor_preprocessor_graph
create_model_kwargs["mode"] = mode


actor_critic = config.create_model(**create_model_kwargs).to(device)

training_pipeline = config.training_pipeline()
rollout_storage_uuid = training_pipeline.rollout_storage_uuid
uuid_to_storage = training_pipeline.current_stage_storage
rollout_storage = uuid_to_storage[rollout_storage_uuid]

# initialize storage
def to_tensor(v) -> torch.Tensor:
    """Return a torch.Tensor version of the input.

    # Parameters

    v : Input values that can be coerced into being a tensor.

    # Returns

    A tensor version of the input.
    """
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(
            v, dtype=torch.int64 if isinstance(v, numbers.Integral) else torch.float
        )

def batch_observations(
    observations: List[Dict],
    device: Optional[torch.device] = None,
):
    def dict_from_observation(
        observation: Dict[str, Any]
    ) -> Dict[str, Union[Dict, List]]:
        batch_dict: DefaultDict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                batch_dict[sensor] = dict_from_observation(observation[sensor])
            else:
                batch_dict[sensor].append(to_tensor(observation[sensor]))

        return batch_dict

    def fill_dict_from_observations(
        input_batch: Any, observation: Dict[str, Any]
    ) -> None:
        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                fill_dict_from_observations(input_batch[sensor], observation[sensor])
            else:
                input_batch[sensor].append(to_tensor(observation[sensor]))

    def dict_to_batch(input_batch: Any) -> None:
        for sensor in input_batch:
            if isinstance(input_batch[sensor], Dict):
                dict_to_batch(input_batch[sensor])
            else:
                input_batch[sensor] = torch.stack(
                    [batch.to(device=device) for batch in input_batch[sensor]], dim=0
                )

    if len(observations) == 0:
        return cast(Dict[str, Union[Dict, torch.Tensor]], observations)

    batch = dict_from_observation(observations[0])

    for obs in observations[1:]:
        fill_dict_from_observations(batch, obs)

    dict_to_batch(batch)

    return cast(Dict[str, Union[Dict, torch.Tensor]], batch)

batch = batch_observations([obs])
preprocessed_obs = sensor_preprocessor_graph.get_observations(batch) if sensor_preprocessor_graph else batch

rollout_storage.initialize(
    observations=preprocessed_obs,
    num_samplers=1,
    recurrent_memory_specification=actor_critic.recurrent_memory_specification,
    action_space=actor_critic.action_space,
)
agent_input = rollout_storage.agent_input_for_next_step()
actor_critic_output, memory = actor_critic(**agent_input)

############################################################################################

############################################################################################

# # VIS Test - w.r.t. Camera coordinate system
# import open3d as o3d
# import matplotlib.pyplot as plt
# rgb_raw = (obs['rgb'] * 255).astype(np.uint8)               # H x W x 3
# depth_raw_u16 = (obs['depth'] * 1000).astype(np.uint16)     # H x W x 1

# rgb_o = o3d.geometry.Image(rgb_raw)
# depth_o = o3d.geometry.Image(depth_raw_u16)
# rgbd_o = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o, depth_o, convert_rgb_to_intensity=False)

# intrinsic_o = o3d.camera.PinholeCameraIntrinsic(
#     width=224, height=224, fx=112, fy=112, cx=112, cy=112
# )

# pcd_o = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_o, intrinsic_o
# )

# axes = np.zeros((3, 20, 3))
# axes_c = np.zeros((3, 20, 3))
# for ax in range(3):
#     axes[ax, :, ax] = np.linspace(0, 0.1, 20)
#     axes_c[ax, :, ax] = 1.0

# pcd_o.points.extend(axes.reshape(-1, 3))
# pcd_o.colors.extend(axes_c.reshape(-1, 3))

# o3d.visualization.draw_geometries([pcd_o])

# # VIS Test - w.r.t. Camera coordinate system
# extrinsic = obs['pose']['T_world_to_cam']
# extrinsic_t = torch.from_numpy(extrinsic)
# inverse_extrinsic_t = extrinsic_t.inverse()

# pcd_o_t = o3d.geometry.PointCloud(pcd_o)
# pcd_o_t.transform(inverse_extrinsic_t.detach().cpu().numpy())

# pcd_o_t.points.extend(axes.reshape(-1, 3))
# pcd_o_t.colors.extend(axes_c.reshape(-1, 3))

# from custom.hlsm.utils.projection_utils import make_pinhole_camera_matrix
# from kornia.geometry.depth import depth_to_3d

# h, w = rgb_raw.shape[:2]
# intrinsic_k = make_pinhole_camera_matrix(
#     height_px=h, width_px=w, hfov_deg=90
# )
# f_rgb = np.where(obs['depth'] < 3.0, obs['rgb'], 1)
# f_depth = np.where(obs['depth'] < 3.0, obs['depth'], 0.0)
# f_depth_t = torch.from_numpy(f_depth).unsqueeze(0).permute(0, 3, 1, 2)
# intrinsic_k_t = intrinsic_k.unsqueeze(0)
# pts_3d_wrt_cam = depth_to_3d(
#     depth=f_depth_t, camera_matrix=intrinsic_k_t, normalize_points=False
# )

# # axis
# cam_axes_t = torch.from_numpy(axes.reshape(-1, 3)).unsqueeze(0).float().permute(0, 2, 1)
# homo_ones_axes = torch.ones_like(cam_axes_t[:, 0:1, :])
# homo_cam_axes_wrt_cam = torch.cat([cam_axes_t, homo_ones_axes], dim=1)
# homo_cam_axes_wrt_world = torch.einsum("bxk,byx->byk", homo_cam_axes_wrt_cam, inverse_extrinsic_t.float().unsqueeze(0))

# homo_ones = torch.ones_like(pts_3d_wrt_cam[:, 0:1, :, :])
# homo_pts_3d_wrt_cam = torch.cat([pts_3d_wrt_cam, homo_ones], dim=1)
# homo_pts_3d_wrt_world = torch.einsum("bxhw,byx->byhw", homo_pts_3d_wrt_cam, inverse_extrinsic_t.float().unsqueeze(0))

# pcd_k = o3d.geometry.PointCloud()
# pcd_k.colors = o3d.utility.Vector3dVector(f_rgb.reshape(-1, 3))
# pcd_k.points = o3d.utility.Vector3dVector(homo_pts_3d_wrt_world[:, :-1].squeeze(0).permute(1, 2, 0).view(-1, 3).detach().cpu().numpy())

# pcd_k.points.extend(homo_cam_axes_wrt_world.squeeze(0)[:-1].permute(1, 0).view(-1, 3).detach().cpu().numpy())
# pcd_k.colors.extend(axes_c.reshape(-1, 3))

# pcd_k.points.extend(axes.reshape(-1, 3))
# pcd_k.colors.extend(axes_c.reshape(-1, 3))

# o3d.visualization.draw_geometries([pcd_k])

############################################################################################

############################################################################################

# from custom.viz import intid_tensor_to_rgb, view_voxel_grid, render_voxel_grid, voxel_tensors_to_geometry
# from custom.hlsm.voxel_grid import DefaultGridParameters

# print(f"task.env.last_event\n {task.env.last_event}")
# print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
# print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")
# print(f"obs['pose']['agent_pos']\n {obs['pose']['agent_pos']}")

# o_grid_data = processed_obs_b['voxel']['voxel_grid_data']
# o_grid_occupancy = processed_obs_b['voxel']['voxel_grid_occupancy']
# o_obs_grid_data = processed_obs_b['voxel']['voxel_observability_grid_data']
# # o_obs_grid_occupancy = processed_obs_b['voxel']['voxel_observability_grid_occupancy']
# o_centroid_coords = processed_obs_b['voxel']['voxel_centroid_coordinates']

# o_rgb_voxel_data = intid_tensor_to_rgb(
#     data=o_grid_data,
#     object_types=SensorTestExperimentConfig.ORDERED_OBJECT_TYPES,
# )

# a_grid_data = o_grid_data.clone()
# a_grid_occupancy = o_grid_occupancy.clone()
# a_obs_grid_data = o_obs_grid_data.clone()

# base_color = torch.tensor([1.0, 1.0, 1.0], device=o_rgb_voxel_data.device)[None, :, None, None, None]
# a_rgb_voxel_data = intid_tensor_to_rgb(
#     data=a_grid_data,
#     object_types=SensorTestExperimentConfig.ORDERED_OBJECT_TYPES,
# )

# a_rgb_voxel_data = a_rgb_voxel_data * 0.5 + base_color * 0.5
# a_rgb_voxel_data = a_rgb_voxel_data * (1 - o_grid_occupancy) + o_rgb_voxel_data * o_grid_occupancy
# a_rgb_voxel_occupancy = a_grid_occupancy.clone()

# a_pos = processed_obs_b['pose']['agent_pos'][0].detach().cpu()
# a_x, a_y, a_z = a_pos[0].item(), a_pos[1].item(), a_pos[2].item()
# a_x_vx = int((a_x - DefaultGridParameters.GRID_ORIGIN[0]) / DefaultGridParameters.GRID_RES)
# a_y_vx = int((a_y - DefaultGridParameters.GRID_ORIGIN[1]) / DefaultGridParameters.GRID_RES)
# a_z_vx = int((a_z - DefaultGridParameters.GRID_ORIGIN[2]) / DefaultGridParameters.GRID_RES)

# a_rgb_voxel_data[0, :, a_x_vx, a_y_vx, :a_z_vx] = torch.tensor([[0], [0], [0]], device=o_rgb_voxel_data.device).repeat(1, a_z_vx)
# a_rgb_voxel_occupancy[0, :, a_x_vx, a_y_vx, :a_z_vx] = 1.0

# o_x_vx = int((0.0 - DefaultGridParameters.GRID_ORIGIN[0]) / DefaultGridParameters.GRID_RES)
# o_y_vx = int((0.0 - DefaultGridParameters.GRID_ORIGIN[1]) / DefaultGridParameters.GRID_RES)
# o_z_vx = int((0.0 - DefaultGridParameters.GRID_ORIGIN[2]) / DefaultGridParameters.GRID_RES)
# a_rgb_voxel_occupancy[0, :, o_x_vx, o_y_vx, o_z_vx] = 0.0

# # Indicate world coordinate origin & axes
# # origin
# a_rgb_voxel_data[0, :, o_x_vx, o_y_vx, o_z_vx] = torch.tensor([1.0, 1.0, 1.0], device=o_rgb_voxel_data.device)
# # x-axis
# a_rgb_voxel_data[0, :, o_x_vx:o_x_vx+3, o_y_vx, o_z_vx] = torch.tensor([[1], [0], [0]], device=o_rgb_voxel_data.device).repeat(1, 3)
# a_rgb_voxel_occupancy[0, :, o_x_vx:o_x_vx+3, o_y_vx, o_z_vx] = 1.0
# # y-axis
# a_rgb_voxel_data[0, :, o_x_vx, o_y_vx:o_y_vx+3, o_z_vx] = torch.tensor([[0], [1], [0]], device=o_rgb_voxel_data.device).repeat(1, 3)
# a_rgb_voxel_occupancy[0, :, o_x_vx, o_y_vx:o_y_vx+3, o_z_vx] = 1.0
# # z-axis
# a_rgb_voxel_data[0, :, o_x_vx, o_y_vx, o_z_vx:o_z_vx+3] = torch.tensor([[0], [0], [1]], device=o_rgb_voxel_data.device).repeat(1, 3)
# a_rgb_voxel_occupancy[0, :, o_x_vx, o_y_vx, o_z_vx:o_z_vx+3] = 1.0

# geom, centroid = voxel_tensors_to_geometry(
#     data=a_rgb_voxel_data,
#     occupancy=a_rgb_voxel_occupancy,
#     centroid_coords=o_centroid_coords,
#     voxel_size=DefaultGridParameters.GRID_RES
# )

# view_voxel_grid(
#     data=a_rgb_voxel_data,
#     occupancy=a_rgb_voxel_occupancy,
#     centroid_coords=o_centroid_coords,
#     voxel_size=DefaultGridParameters.GRID_RES
# )

# f = render_voxel_grid(
#     data=a_rgb_voxel_data,
#     occupancy=a_rgb_voxel_occupancy,
#     centroid_coords=o_centroid_coords,
#     voxel_size=DefaultGridParameters.GRID_RES
# )


############################################################################################

############################################################################################

# while True:
#     act_ind = int(input("action index: "))
#     obs = task.step(act_ind).observation
#     obs_b = dict()
#     for k, v in obs.items():
#         if isinstance(v, dict):
#             obs_b[k] = dict()
#             for k1, v1 in v.items():
#                 obs_b[k][k1] = torch.from_numpy(v1.copy()).unsqueeze(0).float()
#         else:
#             obs_b[k] = torch.from_numpy(v.copy()).unsqueeze(0).float()

#     processed_obs_b = sensor_preprocessor_graph.get_observations(obs_b)

#     print(f"task.env.last_event\n {task.env.last_event}")
#     print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
#     print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")
#     print(f"obs['pose']['agent_pos']\n {obs['pose']['agent_pos']}")

#     o_grid_data = processed_obs_b['voxel']['voxel_grid_data']
#     o_grid_occupancy = processed_obs_b['voxel']['voxel_grid_occupancy']
#     o_obs_grid_data = processed_obs_b['voxel']['voxel_observability_grid_data']
#     # o_obs_grid_occupancy = processed_obs_b['voxel']['voxel_observability_grid_occupancy']
#     o_centroid_coords = processed_obs_b['voxel']['voxel_centroid_coordinates']
    
#     # update all data
#     a_grid_data = a_grid_data * (1 - o_obs_grid_data) + o_grid_data * o_obs_grid_data
#     a_grid_occupancy = a_grid_occupancy * (1 - o_obs_grid_data) + o_grid_occupancy * o_obs_grid_data
#     a_obs_grid_data = torch.max(o_obs_grid_data, a_obs_grid_data)

#     o_rgb_voxel_data = intid_tensor_to_rgb(
#         data=o_grid_data,
#         object_types=SensorTestExperimentConfig.ORDERED_OBJECT_TYPES,
#     )
#     a_rgb_voxel_data = intid_tensor_to_rgb(
#         data=a_grid_data,
#         object_types=SensorTestExperimentConfig.ORDERED_OBJECT_TYPES,
#     )
#     a_rgb_voxel_data = a_rgb_voxel_data * 0.5 + base_color * 0.5
#     a_rgb_voxel_data = a_rgb_voxel_data * (1 - o_grid_occupancy) + o_rgb_voxel_data * o_grid_occupancy
#     a_rgb_voxel_occupancy = a_grid_occupancy.clone()

#     a_pos = processed_obs_b['pose']['agent_pos'][0].detach().cpu()
#     a_x, a_y, a_z = a_pos[0].item(), a_pos[1].item(), a_pos[2].item()
#     a_x_vx = int((a_x - DefaultGridParameters.GRID_ORIGIN[0]) / DefaultGridParameters.GRID_RES)
#     a_y_vx = int((a_y - DefaultGridParameters.GRID_ORIGIN[1]) / DefaultGridParameters.GRID_RES)
#     a_z_vx = int((a_z - DefaultGridParameters.GRID_ORIGIN[2]) / DefaultGridParameters.GRID_RES)

#     a_rgb_voxel_data[0, :, a_x_vx, a_y_vx, :a_z_vx] = torch.tensor([[0], [0], [0]], device=o_rgb_voxel_data.device).repeat(1, a_z_vx)
#     a_rgb_voxel_occupancy[0, :, a_x_vx, a_y_vx, :a_z_vx] = 1.0
#     a_rgb_voxel_occupancy[0, :, o_x_vx, o_y_vx, o_z_vx] = 0.0

#     view_voxel_grid(
#         data=a_rgb_voxel_data,
#         occupancy=a_rgb_voxel_occupancy,
#         centroid_coords=o_centroid_coords,
#         voxel_size=DefaultGridParameters.GRID_RES
#     )

############################################################################################

############################################################################################

# from custom.hlsm.image_to_pc import ImageToPointcloud
# i2pc = ImageToPointcloud()
# pts_3d_wrt_world, ci = i2pc.forward(
#     camera_image=obs_b['rgb'].permute(0, 3, 1, 2),
#     depth_image=obs_b['depth'].permute(0, 3, 1, 2),
#     extrinsics4f=obs_b['pose']['T_world_to_cam'],
#     hfov_deg=90,
#     min_depth=0.7
# )

############################################################################################

############################################################################################

# from custom.hlsm.utils.projection_utils import make_pinhole_camera_matrix
# ci = obs_b['rgb'].permute(0, 3, 1, 2)
# di = obs_b['depth'].permute(0, 3, 1, 2)
# b, c, h, w = ci.shape
# dev = ci.device

# intrinsics = make_pinhole_camera_matrix(
#     height_px=h,
#     width_px=w,
#     hfov_deg=90
# )
# intrinsics = intrinsics.to(dev)
# intrinsics = intrinsics[None, :, :].repeat((b, 1, 1))

# extrinsics_test = obs_b['pose']['T_world_to_cam']
# inverse_extrinsics_test = extrinsics_test.inverse()

# from kornia.geometry.depth import depth_to_3d
# pts_3d_wrt_cam = depth_to_3d(
#     depth=di,
#     camera_matrix=intrinsics,
#     normalize_points=False
# )

# import os
# import custom.hlsm.utils.render3d as r3d
# outdir = os.path.join(os.getcwd(), 'test')
# os.makedirs(outdir, exist_ok=True)

# homo_ones = torch.ones_like(pts_3d_wrt_cam[:, 0:1, :, :])
# homo_pts_3d_wrt_cam = torch.cat([pts_3d_wrt_cam, homo_ones], dim=1)
# homo_pts_3d_wrt_world = torch.einsum("bxhw,byx->byhw", homo_pts_3d_wrt_cam, inverse_extrinsics_test)
# pts_3d_wrt_world = homo_pts_3d_wrt_world[:, :3, :, :]

# # from custom.hlsm.utils.utils import save_gif, standardize_image
# # import imageio
# # img_w = r3d.render_aligned_point_cloud(pts_3d_wrt_world, ci, animate=True)
# # img_c = r3d.render_aligned_point_cloud(pts_3d_wrt_cam, ci, animate=True)
# # save_gif(img_w, os.path.join(outdir, "pc_test_g.gif"))
# # save_gif(img_c, os.path.join(outdir, "pc_test_c.gif"))
# # imageio.imsave(os.path.join(outdir, "pc_test_scene.png"), standardize_image(ci[0]))
# # imageio.imsave(os.path.join(outdir, "pc_test_depth.png"), standardize_image(di[0]))

# has_depth = di > 0.7
# pts_3d_wrt_world = pts_3d_wrt_world * has_depth
# ci = ci * has_depth

############################################################################################

############################################################################################
# import numpy as np

# print(f"task.env.last_event\n {task.env.last_event}")
# print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
# print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")

# print(f"obs['pose']['cam_pos_enu']\n {obs['pose']['cam_pos_enu']}")
# print(f"obs['pose']['rot_3d_enu_deg']\n {obs['pose']['rot_3d_enu_deg']}")
# print(f"obs['pose']['cam_horizon_deg']\n {obs['pose']['cam_horizon_deg']}")

# print(f"T_u2w\n {obs['pose']['T_unity_to_world']}")
# print(f"T_w2c\n {obs['pose']['T_world_to_cam']}")
# print(f"T_c2w: inv(T_w2c) \n {np.linalg.inv(obs['pose']['T_world_to_cam'])}")
# # print(f"obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']\n {obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']}")
# print(f"T_u2c: T_w2c @ T_u2w \n {obs['pose']['T_world_to_cam'] @ obs['pose']['T_unity_to_world']}")

# print(f"T_c2u: inv(T_u2c) \n {np.linalg.inv(obs['pose']['T_world_to_cam'] @ obs['pose']['T_unity_to_world'])}")

# while True:
#     act_ind = int(input("action index: "))
#     obs = task.step(act_ind).observation
#     print(f"task.env.last_event\n {task.env.last_event}")
#     print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
#     print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")

#     print(f"obs['pose']['cam_pos_enu']\n {obs['pose']['cam_pos_enu']}")
#     print(f"obs['pose']['rot_3d_enu_deg']\n {obs['pose']['rot_3d_enu_deg']}")
#     print(f"obs['pose']['cam_horizon_deg']\n {obs['pose']['cam_horizon_deg']}")

#     print(f"T_u2w\n {obs['pose']['T_unity_to_world']}")
#     print(f"T_w2c\n {obs['pose']['T_world_to_cam']}")
#     print(f"T_c2w: inv(T_w2c) \n {np.linalg.inv(obs['pose']['T_world_to_cam'])}")
#     # print(f"obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']\n {obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']}")
#     print(f"T_u2c: T_w2c @ T_u2w \n {obs['pose']['T_world_to_cam'] @ obs['pose']['T_unity_to_world']}")
#     print(f"T_c2u: inv(T_u2c) \n {np.linalg.inv(obs['pose']['T_world_to_cam'] @ obs['pose']['T_unity_to_world'])}")

############################################################################################

############################################################################################

# import math
# import numpy as np
# import torch
# from typing import Union, Tuple
# from transforms3d import euler

# IDX_TO_ACTION_TYPE = {
#     0: "RotateLeft",
#     1: "RotateRight",
#     2: "MoveAhead",
#     3: "LookUp",
#     4: "LookDown",
#     5: "OpenObject",
#     6: "CloseObject",
#     7: "PickupObject",
#     8: "PutObject",
#     9: "ToggleObjectOn",
#     10: "ToggleObjectOff",
#     11: "SliceObject",
#     12: "Stop"
# }
# # TODO: Reinstate Stop action as an action type

# ACTION_TYPE_TO_IDX = {v:k for k,v in IDX_TO_ACTION_TYPE.items()}
# ACTION_TYPES = [IDX_TO_ACTION_TYPE[i] for i in range(len(IDX_TO_ACTION_TYPE))]

# NAV_ACTION_TYPES = [
#     "RotateLeft",
#     "RotateRight",
#     "MoveAhead",
#     "LookUp",
#     "LookDown"
# ]

# INTERACT_ACTION_TYPES = [
#     "OpenObject",
#     "CloseObject",
#     "PickupObject",
#     "PutObject",
#     "ToggleObjectOn",
#     "ToggleObjectOff",
#     "SliceObject"
# ]


# class AlfredAction():
#     def __init__(self,
#                  action_type: str,
#                  argument_mask : torch.tensor):
#         super().__init__()
#         self.action_type = action_type
#         self.argument_mask = argument_mask

#     def to(self, device):
#         self.argument_mask = self.argument_mask.to(device) if self.argument_mask is not None else None
#         return self

#     @classmethod
#     def stop_action(cls):
#         return cls("Stop", cls.get_empty_argument_mask())

#     @classmethod
#     def get_empty_argument_mask(cls) -> torch.tensor:
#         return torch.zeros((300, 300))

#     @classmethod
#     def get_action_type_space_dim(cls) -> int:
#         return len(ACTION_TYPE_TO_IDX)

#     @classmethod
#     def action_type_str_to_intid(cls, action_type_str : str) -> int:
#         return ACTION_TYPE_TO_IDX[action_type_str]

#     @classmethod
#     def action_type_intid_to_str(cls, action_type_intid : int) -> str:
#         return IDX_TO_ACTION_TYPE[action_type_intid]

#     @classmethod
#     def get_interact_action_list(cls):
#         return INTERACT_ACTION_TYPES

#     @classmethod
#     def get_nav_action_list(cls):
#         return NAV_ACTION_TYPES

#     def is_valid(self):
#         if self.action_type in NAV_ACTION_TYPES:
#             return True
#         elif self.argument_mask is None:
#             print("AlfredAction::is_valid -> missing argument mask")
#             return False
#         elif self.argument_mask.sum() < 1:
#             print("AlfredAction::is_valid -> empty argument mask")
#             return False
#         return True

#     def type_intid(self):
#         return self.action_type_str_to_intid(self.action_type)

#     def type_str(self):
#         return self.action_type

#     def to_alfred_api(self) -> Tuple[str, Union[None, np.ndarray]]:
#         if self.action_type in NAV_ACTION_TYPES:
#             argmask_np = None
#         else: # Interaction action needs a mask
#             if self.argument_mask is not None:
#                 if isinstance(self.argument_mask, torch.Tensor):
#                     argmask_np = self.argument_mask.detach().cpu().numpy()
#                 else:
#                     argmask_np = self.argument_mask
#             else:
#                 argmask_np = None
#         return self.action_type, argmask_np

#     def is_stop(self):
#         return self.action_type == "Stop"

#     def __eq__(self, other: "AlfredAction"):
#         return self.action_type == other.action_type and self.argument_mask == other.argument_mask

#     def __str__(self):
#         return f"AA: {self.action_type}"

#     def represent_as_image(self):
#         if self.argument_mask is None:
#             return torch.zeros((1, 300, 300))
#         else:
#             return self.argument_mask


# class PoseInfo():
#     """
#     Given all the different inputs from AI2Thor event, constructs a pose matrix and a position vector
#     to add to the observation.
#     """

#     def __init__(self,
#                  cam_horizon_deg,
#                  cam_pos_enu,
#                  rot_3d_enu_deg,
#                  ):
#         self.cam_horizon_deg = cam_horizon_deg
#         self.cam_pos_enu = cam_pos_enu
#         self.rot_3d_enu_deg = rot_3d_enu_deg

#     def is_close(self, pi: "PoseInfo"):
#         horizon_close = math.isclose(self.cam_horizon_deg, pi.cam_horizon_deg, abs_tol=1e-3, rel_tol=1e-3)
#         cam_pos_close = [math.isclose(a, b) for a,b in zip(self.cam_pos_enu, pi.cam_pos_enu)]
#         rot_close = [math.isclose(a, b) for a,b in zip(self.rot_3d_enu_deg, pi.rot_3d_enu_deg)]
#         all_close = horizon_close and cam_pos_close[0] and cam_pos_close[1] and cam_pos_close[2] and rot_close[0] and rot_close[1] and rot_close[2]
#         return all_close

#     @classmethod
#     def from_ai2thor_event(cls, event):
#         # Unity uses a left-handed coordinate frame with X-Z axis on ground, Y axis pointing up.
#         # We want to convert to a right-handed coordinate frame with X-Y axis on ground, and Z axis pointing up.
#         # To do this, all you have to do is swap Y and Z axes.

#         cam_horizon_deg = event.metadata['agent']['cameraHorizon']

#         # Translation from world origin to camera/agent position
#         cam_pos_dict_3d_unity = event.metadata['cameraPosition']
#         # Remap Unity left-handed frame to ENU right-handed frame 
#         cam_pos_enu = [cam_pos_dict_3d_unity['z'],
#                        -cam_pos_dict_3d_unity['x'],
#                        cam_pos_dict_3d_unity['y']]

#         # ... rotation to agent frame (x-forward, y-left, z-up)
#         rot_dict_3d_unity = event.metadata['agent']['rotation']
#         rot_3d_enu_deg = [-rot_dict_3d_unity['z'], rot_dict_3d_unity['x'], -rot_dict_3d_unity['y']]

#         return PoseInfo(cam_horizon_deg=cam_horizon_deg,
#                         cam_pos_enu=cam_pos_enu,
#                         rot_3d_enu_deg=rot_3d_enu_deg)

#     @classmethod
#     def create_new_initial(cls):
#         cam_horizon_deg = 30.0
#         cam_pos_enu = [0.0, 0.0, 1.576]
#         rot_3d_enu_deg = [0.0, 0.0, 0.0]
#         return PoseInfo(cam_horizon_deg=cam_horizon_deg,
#                         cam_pos_enu=cam_pos_enu,
#                         rot_3d_enu_deg=rot_3d_enu_deg)

#     def simulate_successful_action(self, action):
#         MOVE_STEP = 0.25
#         PITCH_STEP = 15
#         YAW_STEP = 90

#         if action.action_type == "RotateLeft":
#             self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] - YAW_STEP) % 360
#         elif action.action_type == "RotateRight":
#             self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] + YAW_STEP) % 360
#         elif action.action_type in ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack"):
#             # TODO: Solve this with a geometry equation instead
#             if action.action_type == "MoveAhead":
#                 step = np.array([MOVE_STEP, 0.0])
#             elif action.action_type == "MoveLeft":
#                 step = np.array([0.0, MOVE_STEP])
#             elif action.action_type == "MoveBack":
#                 step = np.array([-MOVE_STEP, 0.0])
#             elif action.action_type == "MoveRight":
#                 step = np.array([0.0, -MOVE_STEP])
#             else:
#                 raise ValueError("Wrong Action")

#             theta = self.rot_3d_enu_deg[2] / 180.0 * np.pi # [rad]
#             ct = np.cos(theta)
#             st = np.sin(theta)
#             rot_mat = np.array(
#                 [
#                     [ct, -st],
#                     [st, ct]
#                 ]
#             )
#             self.cam_pos_enu[0:2] += (rot_mat @ step.reshape(-1, 1)).reshape(-1)

#             # if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
#             #     self.cam_pos_enu[1] += MOVE_STEP
#             # elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
#             #     self.cam_pos_enu[1] -= MOVE_STEP
#             # elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
#             #     self.cam_pos_enu[0] -= MOVE_STEP
#             # elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
#             #     self.cam_pos_enu[0] += MOVE_STEP
#             # else:
#             #     raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
#         elif action.action_type == "LookDown":
#             self.cam_horizon_deg = self.cam_horizon_deg + PITCH_STEP
#         elif action.action_type == "LookUp":
#             self.cam_horizon_deg = self.cam_horizon_deg - PITCH_STEP

#     def get_agent_pos(self, device="cpu"):
#         cam_pos = [
#             self.cam_pos_enu[0],
#             self.cam_pos_enu[1],
#             self.cam_pos_enu[2]
#         ]
#         cam_pos = torch.tensor(cam_pos, device=device, dtype=torch.float32)
#         return cam_pos

#     def get_pose_mat(self):
#         cam_pos_enu = torch.tensor(self.cam_pos_enu)
#         # Rotation matrix from unity frame to world frame
#         T_unity_to_world = np.array(
#             [
#                 [0, -1, 0, 0],
#                 [0, 0, 1, 0],
#                 [1, 0, 0, 0],
#                 [0, 0, 0, 1]
#             ]
#         )

#         # Translation from world origin to camera/agent position
#         T_world_to_agent_pos = np.array(
#             [
#                 [1, 0, 0, cam_pos_enu[0]],
#                 [0, 1, 0, cam_pos_enu[1]],
#                 [0, 0, 1, cam_pos_enu[2]],
#                 [0, 0, 0, 1]
#             ]
#         )

#         # ... rotation to agent frame (x-forward, y-left, z-up)
#         rot_3d_enu_rad = [math.radians(r) for r in self.rot_3d_enu_deg]
#         R_agent = euler.euler2mat(rot_3d_enu_rad[0], rot_3d_enu_rad[1], rot_3d_enu_rad[2])
#         T_agent_pos_to_agent = np.asarray([[R_agent[0, 0], R_agent[0, 1], R_agent[0, 2], 0],
#                                            [R_agent[1, 0], R_agent[1, 1], R_agent[1, 2], 0],
#                                            [R_agent[2, 0], R_agent[2, 1], R_agent[2, 2], 0],
#                                            [0, 0, 0, 1]])

#         # .. transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch
#         R_agent_to_camflat = euler.euler2mat(math.radians(-90), math.radians(0), math.radians(-90))
#         T_agent_to_camflat = np.asarray(
#             [
#                 [R_agent_to_camflat[0, 0], R_agent_to_camflat[0, 1], R_agent_to_camflat[0, 2], 0],
#                 [R_agent_to_camflat[1, 0], R_agent_to_camflat[1, 1], R_agent_to_camflat[1, 2], 0],
#                 [R_agent_to_camflat[2, 0], R_agent_to_camflat[2, 1], R_agent_to_camflat[2, 2], 0],
#                 [0, 0, 0, 1]
#             ]
#         )

#         # .. transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch
#         R_camflat_to_cam = euler.euler2mat(math.radians(-self.cam_horizon_deg), 0, 0)
#         T_camflat_to_cam = np.asarray(
#             [
#                 [R_camflat_to_cam[0, 0], R_camflat_to_cam[0, 1], R_camflat_to_cam[0, 2], 0],
#                 [R_camflat_to_cam[1, 0], R_camflat_to_cam[1, 1], R_camflat_to_cam[1, 2], 0],
#                 [R_camflat_to_cam[2, 0], R_camflat_to_cam[2, 1], R_camflat_to_cam[2, 2], 0],
#                 [0, 0, 0, 1]
#             ]
#         )

#         # compose into a transform from world to camera
#         T_world_to_cam = T_world_to_agent_pos @ T_agent_pos_to_agent @ T_agent_to_camflat @ T_camflat_to_cam
#         T_world_to_cam = torch.from_numpy(T_world_to_cam).unsqueeze(0)
#         return T_world_to_cam, T_unity_to_world, T_world_to_agent_pos, T_agent_pos_to_agent, T_agent_to_camflat, T_camflat_to_cam


# print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
# print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")
# pi = PoseInfo.from_ai2thor_event(task.env.last_event)

# obs = task.get_observations()
# print(f"task.env.last_event\n {task.env.last_event}")
# print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
# print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")

# print(f"obs['pose']['cam_pos_enu']\n {obs['pose']['cam_pos_enu']}")
# print(f"obs['pose']['rot_3d_enu_deg']\n {obs['pose']['rot_3d_enu_deg']}")
# print(f"obs['pose']['cam_horizon_deg']\n {obs['pose']['cam_horizon_deg']}")

# print(f"obs['pose']['cam_pos_enu_rel']\n {obs['pose']['cam_pos_enu_rel']}")
# print(f"obs['pose']['rot_3d_enu_deg_rel']\n {obs['pose']['rot_3d_enu_deg_rel']}")
# print(f"obs['pose']['cam_horizon_deg_rel']\n {obs['pose']['cam_horizon_deg_rel']}")

# print(f"obs['pose']['T_world_to_cam']\n {obs['pose']['T_world_to_cam']}")
# print(f"obs['pose']['T_world_rel_to_cam']\n {obs['pose']['T_world_rel_to_cam']}")
# print(f"obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']\n {obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']}")
# print(f"obs['pose']['T_unity_to_world_rel'] @ obs['pose']['T_world_rel_to_cam']\n {obs['pose']['T_unity_to_world_rel'] @ obs['pose']['T_world_rel_to_cam']}")


# while True:
#     act_ind = int(input("action index: "))
#     obs = task.step(act_ind).observation
#     print(f"task.env.last_event\n {task.env.last_event}")
#     print(f"task.env.last_event.metadata['agent']\n{task.env.last_event.metadata['agent']}")
#     print(f"task.env.last_event.metadata['cameraPosition']\n{task.env.last_event.metadata['cameraPosition']}")

#     print(f"obs['pose']['cam_pos_enu']\n {obs['pose']['cam_pos_enu']}")
#     print(f"obs['pose']['rot_3d_enu_deg']\n {obs['pose']['rot_3d_enu_deg']}")
#     print(f"obs['pose']['cam_horizon_deg']\n {obs['pose']['cam_horizon_deg']}")

#     print(f"obs['pose']['cam_pos_enu_rel']\n {obs['pose']['cam_pos_enu_rel']}")
#     print(f"obs['pose']['rot_3d_enu_deg_rel']\n {obs['pose']['rot_3d_enu_deg_rel']}")
#     print(f"obs['pose']['cam_horizon_deg_rel']\n {obs['pose']['cam_horizon_deg_rel']}")

#     print(f"obs['pose']['T_world_to_cam']\n {obs['pose']['T_world_to_cam']}")
#     print(f"obs['pose']['T_world_rel_to_cam']\n {obs['pose']['T_world_rel_to_cam']}")
#     print(f"obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']\n {obs['pose']['T_unity_to_world'] @ obs['pose']['T_world_to_cam']}")
#     print(f"obs['pose']['T_unity_to_world_rel'] @ obs['pose']['T_world_rel_to_cam']\n {obs['pose']['T_unity_to_world_rel'] @ obs['pose']['T_world_rel_to_cam']}")

############################################################################################