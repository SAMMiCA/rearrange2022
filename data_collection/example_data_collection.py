#%%
import json
import operator
from typing import List, Optional, Dict, Any, DefaultDict, Union, cast
import os
import numpy as np
import torch
import importlib
import inspect
import multiprocessing as mp
from PIL import Image

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
from allenact.utils import spaces_utils as su
from data_collection.coco_utils import binary_mask_to_polygon
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from example_utils import advantage_stats, collect_step_across_all_task_samplers, find_sub_modules, get_sampler_fn_args, to_tensor, batch_observations, set_seed, worker_seed
from experiments.distributed_task_aware_rearrange_data_collection import DistributedTaskAwareRearrangeDataCollectionExperimentConfig


NUM_PROCESSES = 2
MP_START_METHOD = "forkserver"
DEVICE = "cuda"
MODE = "train"
SEED = 0
DATA_SAVE_DIR = "data_collection/data"


experimental_base = ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
rel_base_dir = os.path.relpath(experimental_base, os.getcwd())
rel_base_dot_path = rel_base_dir.replace("/", ".")
if rel_base_dot_path == ".":
    rel_base_dot_path = ""

exp_dot_path = "experiments/task_aware_rearrange_data_collection.py"
if exp_dot_path[-3:] == ".py":
    exp_dot_path = exp_dot_path[:-3]
exp_dot_path = exp_dot_path.replace("/", ".")

save_dir = os.path.join(experimental_base, DATA_SAVE_DIR, MODE)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

module_path = (
    f"{rel_base_dot_path}.{exp_dot_path}"
    if len(rel_base_dot_path) != 0
    else exp_dot_path
)
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

experiments = [
    m[1]
    for m in inspect.getmembers(module, inspect.isclass)
    if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
]
config = experiments[0]()
machine_params = config.machine_params(MODE)
sensor_preprocessor_graph = None
if machine_params.sensor_preprocessor_graph is not None:
    sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(DEVICE)

set_seed(SEED)

mp_ctx = mp.get_context(MP_START_METHOD)
seeds = worker_seed(
    nprocs=NUM_PROCESSES,
    init_seed=SEED
)
sampler_fn_args = get_sampler_fn_args(
    config=config,
    machine_params=machine_params,
    mode=MODE, 
    device=DEVICE,
    num_procs=config.NUM_PROCESSES,
    seeds=seeds,
)
for sampler_fn_arg in sampler_fn_args:
    sampler_fn_arg["epochs"] = 1

image_size = config.SCREEN_SIZE


if __name__ == "__main__":
    vector_tasks = VectorSampledTasks(
        make_sampler_fn=config.make_sampler_fn,
        sampler_fn_args=sampler_fn_args,
        multiprocessing_start_method=None,
        mp_ctx=mp_ctx
    )
    remaining_tasks = vector_tasks.command('sampler_attr', ['total_unique'] * NUM_PROCESSES)

    image_id = 0
    coco_id = 0
    annos = []
    images = []

    stop = False

    expert_subtasks = np.array([[] for _ in range(NUM_PROCESSES)])
    expert_actions = np.array([[] for _ in range(NUM_PROCESSES)])

    observations = vector_tasks.get_observations()
    
    while any(remaining_tasks):
        paused, keep, running = [], [], []
        for it, obs in enumerate(observations):
            if obs is None:
                paused.append(it)
            else:
                keep.append(it)
                running.append(obs)
        
        for p in reversed(paused):
            vector_tasks.pause_at(p)
        
        remaining_tasks = vector_tasks.command('sampler_attr', ['length'] * len(keep))
        if not any(remaining_tasks):
            break

        unique_ids = vector_tasks.command('sampler_attr', ['current_task_spec.unique_id'] * len(keep))
        act_spaces = vector_tasks.attr(['action_space'] * len(keep))
        act_space = act_spaces[0]
        num_steps_taken = vector_tasks.call(['num_steps_taken'] * len(keep))

        batch_obs = batch_observations(running)
        preproc_obs = sensor_preprocessor_graph.get_observations(batch_obs) if sensor_preprocessor_graph else batch_obs
        
        # collect data
        for sampler_id in range(len(keep)):
        
            sampler_dir = os.path.join(save_dir, unique_ids[sampler_id])
            if not os.path.exists(sampler_dir):
                os.makedirs(sampler_dir)
            file_name = f"{num_steps_taken[sampler_id]:03d}"
            
            # save RGB
            rgb = preproc_obs["rgb"][sampler_id]    # height x width x 3
            im_rgb = Image.fromarray((rgb.detach().cpu().numpy() * 255).astype(np.uint8))
            rgb_file_name = f"{file_name}.png"
            im_name = os.path.join(sampler_dir, 'rgb', rgb_file_name)
            if not os.path.exists(os.path.dirname(im_name)):
                os.makedirs(os.path.dirname(im_name))
            im_rgb.save(im_name)

            u_rgb = preproc_obs["unshuffled_rgb"][sampler_id]    # height x width x 3
            im_u_rgb = Image.fromarray((u_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
            u_rgb_file_name = f"{file_name}.png"
            im_u_name = os.path.join(sampler_dir, 'unshuffled_rgb', u_rgb_file_name)
            if not os.path.exists(os.path.dirname(im_u_name)):
                os.makedirs(os.path.dirname(im_u_name))
            im_u_rgb.save(im_u_name)

            # save Depth
            depth = preproc_obs["depth"][sampler_id]    # height x width x 1
            depth_path = os.path.join(sampler_dir, 'depth', file_name)
            if not os.path.exists(os.path.dirname(depth_path)):
                os.makedirs(os.path.dirname(depth_path))
            np.save(depth_path, depth.detach().cpu().numpy())

            u_depth = preproc_obs["unshuffled_depth"][sampler_id]    # height x width x 1
            u_depth_path = os.path.join(sampler_dir, 'unshuffled_depth', file_name)
            if not os.path.exists(os.path.dirname(u_depth_path)):
                os.makedirs(os.path.dirname(u_depth_path))
            np.save(u_depth_path, u_depth.detach().cpu().numpy())

            # save semseg / instseg
            semseg = preproc_obs['semseg'][sampler_id]  # (num_object + 1) x height x width
            semseg_path = os.path.join(sampler_dir, 'semseg', file_name)
            if not os.path.exists(os.path.dirname(semseg_path)):
                os.makedirs(os.path.dirname(semseg_path))
            np.save(semseg_path, semseg.detach().cpu().numpy())

            u_semseg = preproc_obs['unshuffled_semseg'][sampler_id]  # (num_object + 1) x height x width
            u_semseg_path = os.path.join(sampler_dir, 'unshuffled_semseg', file_name)
            if not os.path.exists(os.path.dirname(u_semseg_path)):
                os.makedirs(os.path.dirname(u_semseg_path))
            np.save(u_semseg_path, u_semseg.detach().cpu().numpy())

            instseg = preproc_obs['instseg']['inst_masks'][sampler_id]  # num_object x height x width
            instseg_path = os.path.join(sampler_dir, 'instseg', file_name)
            if not os.path.exists(os.path.dirname(instseg_path)):
                os.makedirs(os.path.dirname(instseg_path))
            np.save(instseg_path, instseg.detach().cpu().numpy())

            # save semantic map
            semmap = preproc_obs['semmap'][sampler_id]      # (num_object + 1 + 1 + 1 + 1) x v_width x v_length x v_height
            semmap_path = os.path.join(sampler_dir, 'semmap', file_name)
            if not os.path.exists(os.path.dirname(semmap_path)):
                os.makedirs(os.path.dirname(semmap_path))
            np.save(semmap_path, semmap.detach().cpu().numpy())

            # save expert subtask & action & inventory
            expert_subtask = preproc_obs['expert_subtask'][sampler_id][0].item()
            expert_subtasks[sampler_id].append(expert_subtask)
            expert_subtask_path = os.path.join(sampler_dir, 'expert_subtask', file_name)
            if not os.path.exists(os.path.dirname(expert_subtask_path)):
                os.makedirs(os.path.dirname(expert_subtask_path))
            with open(f"{expert_subtask_path}.json", "w") as f:
                json.dump(expert_subtasks[sampler_id], f, indent=4)

            expert_action = preproc_obs['expert_action'][sampler_id][0].item()
            expert_actions[sampler_id].append(expert_action)
            expert_action_path = os.path.join(sampler_dir, 'expert_action', file_name)
            if not os.path.exists(os.path.dirname(expert_action_path)):
                os.makedirs(os.path.dirname(expert_action_path))
            with open(f"{expert_action_path}.json", "w") as f:
                json.dump(expert_actions[sampler_id], f, indent=4)

            inventory = preproc_obs['inventory'][sampler_id]
            inventory_path = os.path.join(sampler_dir, 'inventory', file_name)
            if not os.path.exists(os.path.dirname(inventory_path)):
                os.makedirs(os.path.dirname(inventory_path))
            np.save(inventory_path, inventory.detach().cpu().numpy())

            # import pdb; pdb.set_trace()
            inst_detected = preproc_obs['instseg']['inst_detected'][sampler_id]
            for nonzero_id in inst_detected.nonzero():
                object_id = nonzero_id.item()
                add_image = False
                for i in range(inst_detected[object_id]):
                    mask = instseg[object_id] & (2 ** i)
                    pos = torch.where(mask)
                    xmin = torch.min(pos[1]).item()
                    xmax = torch.max(pos[1]).item()
                    ymin = torch.min(pos[0]).item()
                    ymax = torch.max(pos[0]).item()
                    width = xmax - xmin
                    height = ymax - ymin
                    if width < 15 and height < 15:
                        continue
                    poly = binary_mask_to_polygon(mask.detach().cpu().numpy())
                    bbox = [xmin, ymin, width, height]
                    area = width * height

                    # update annotation for annotations
                    data_anno = dict(
                        image_id=image_id,
                        id=coco_id,
                        category_id=object_id+1,
                        bbox=bbox,
                        area=area,
                        segmentation=poly,
                        iscrowd=0,
                    )
                    annos.append(data_anno)
                    coco_id += 1
                    add_iamge = True
                
            if add_image:
                # update annotation for image
                images.append(
                    dict(
                        id=image_id,
                        file_name=im_name,
                        height=image_size,
                        width=image_size,
                    )
                )
                image_id += 1

        step_actions = preproc_obs['expert_action'][..., 0]
        flat_actions = su.flatten(act_space, step_actions)
        outputs = vector_tasks.step(
            su.action_list(act_space, flat_actions[None, ...])
        )
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        not_dones = list(map(operator.not_, dones))
        expert_subtasks = [expert_subtasks[i] * not_dones[i] for i in range(len(keep))]
        expert_actions = [expert_actions[i] * not_dones[i] for i in range(len(keep))]

    coco_format_json = dict(
        images=images,
        annotations=annos,
        categories=[
            {'id': i+1, 'name': t} 
            for i, t in enumerate(DistributedTaskAwareRearrangeDataCollectionExperimentConfig.ORDERED_OBJECT_TYPES)
        ],
    )
    with open(os.path.join(save_dir, "anno.json"), 'w') as f:
        json.dump(coco_format_json, f)
    
    vector_tasks.close()

    # Pure observations
    # while True:
    #     paused, keep, running = [], [], []
    #     for it, obs in enumerate(observations):
    #         if obs is None:
    #             paused.append(it)
    #         else:
    #             keep.append(it)
    #             running.append(obs)
        
    #     for p in reversed(paused):
    #         vector_tasks.pause_at(p)

    #     expert_actions = []
    #     for obs in running:
    #         instseg = obs["instseg"]
    #         add_image = False
    #         for idx in instseg["inst_detected"].nonzero()[0]:
    #             for i in range(instseg["inst_detected"][idx]):
    #                 mask = instseg["inst_masks"][idx] & (2 ** i)
    #                 poly = binary_mask_to_polygon(mask)
    #                 pos = np.where(mask)
    #                 if pos[0].size == 0:
    #                     import pdb; pdb.set_trace()
    #                 xmin = np.min(pos[1])
    #                 xmax = np.max(pos[1])
    #                 ymin = np.min(pos[0])
    #                 ymax = np.max(pos[0])
    #                 width = xmax - xmin
    #                 height = ymax - ymin
    #                 if width < 15 and height < 15:
    #                     continue
    #                 area = width * height
    #                 bbox = ([xmin, ymin, width, height])

    #                 data_anno = dict(
    #                     image_id=image_id,
    #                     id=coco_id,
    #                     category_id=idx+1,
    #                     bbox=bbox,
    #                     area=area,
    #                     segmentation=poly,
    #                     iscrowd=0,
    #                 )
    #                 annos.append(data_anno)
    #                 coco_id += 1
    #                 add_image = True
            
    #         if add_image:
    #             image_rgb = Image.fromarray((obs["rgb"] * 255).astype(np.uint8))
    #             image_name = os.path.join(image_dir, f"{image_id:06d}.png")
    #             image_rgb.save(image_name)
    #             images.append(
    #                 dict(
    #                     id=image_id,
    #                     file_name=image_name,
    #                     height=image_size,
    #                     width=image_size,
    #                 )
    #             )
    #             image_id += 1
            
    #         expert_actions.append(obs['expert_subtask_action'][-2])
        
    #     outputs = vector_tasks.step(expert_actions)
    #     observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

    #     if any(dones):
    #         import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # vector_tasks.close()

    # batch_obs = batch_observations(running)
    # preproc_obs = sensor_preprocessor_graph.get_observations(batch_obs) if sensor_preprocessor_graph else batch_obs

    # import pdb; pdb.set_trace()

    
    
# %%
# with torch.no_grad():
#     actor_critic_output, _ = actor_critic(
#         **rollout_storage.agent_input_for_next_step()
#     )

# before_update_info = dict(
#     next_value=actor_critic_output.values.detach(),
#     use_gae=cur_stage_training_settings.use_gae,
#     gamma=cur_stage_training_settings.gamma,
#     tau=cur_stage_training_settings.gae_lambda,
#     adv_stats_callback=advantage_stats,
# )

# rollout_storage.before_updates(**before_update_info)
# batch_iterator = rollout_storage.batched_experience_generator(
#     num_mini_batch=cur_stage_training_settings.num_mini_batch
# )

# batch = next(batch_iterator)
# # %%
# actor_critic_output_for_batch, _ = actor_critic(
#     observations=batch["observations"],
#     memory=batch["memory"],
#     prev_actions=batch["prev_actions"],
#     masks=batch["masks"],
# )
# # %%
