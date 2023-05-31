import numpy as np
import argparse

import json
import gzip
import os
from tqdm import tqdm

from utils.experiment_utils import NumpyJSONEncoder
from experiments.two_phase.two_phase_test_config import TwoPhaseTestConfig as Config
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from task_aware_rearrange.subtasks import IDX_TO_SUBTASK, SUBTASKS
from transforms3d import euler
from task_aware_rearrange.sensors import PoseSensor


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument('--visualize', action='store_true', )
    parser.add_argument('--verbose', action='store_true', )
    
    args = parser.parse_args()
    
    return args


def stagewise_example(stage: str = "train", verbose: bool = False):
    task_sampler_params = Config.stagewise_task_sampler_args(
        stage=stage, process_ind=0, total_processes=1,
    )
    task_sampler: RearrangeTaskSampler = Config.make_sampler_fn(
        **task_sampler_params,
        force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
        only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
        epochs=1,
    )

    how_many_unique_datapoints = task_sampler.total_unique
    num_tasks_to_do = how_many_unique_datapoints
    
    if verbose:
        print(
            f"Sampling {num_tasks_to_do} tasks from the Two-Phase {stage} dataset"
            f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
        )
    
    leaderboard_submission = {}
    all_metrics = {}
    for i_task in tqdm(range(num_tasks_to_do), total=num_tasks_to_do, desc=f"Stage: {stage}"):
        if verbose: 
            print(f"\nStarting task {i_task}")
        walkthorugh_expert_subtasks = []
        walkthrough_expert_actions = []
        walkthrough_task = task_sampler.next_task()
        if verbose:
            print(f"======== WALKTHROUGH TASK START :) ========")
            print(
                f"Sampled task is from the "
                f" '{task_sampler.current_task_spec.stage}' stage and has"
                f" unique id '{task_sampler.current_task_spec.unique_id}'"
            )
        assert isinstance(walkthrough_task, WalkthroughTask)
        observations = walkthrough_task.get_observations()
        while not walkthrough_task.is_done():
            expert_obs = observations['expert_subtask_action']
            act = expert_obs[2] if expert_obs[3] else walkthrough_task.action_space.sample()
            walkthrough_expert_actions.append(walkthrough_task.action_names()[expert_obs[2]])
            walkthorugh_expert_subtasks.append(
                ''.join([arg for arg in IDX_TO_SUBTASK[expert_obs[0]] if arg is not None])
            )
            if verbose:
                print(f'action: {walkthrough_task.action_names()[act]}')
                print(f'seen pickupables: {walkthrough_task.seen_pickupable_objects}')
                print(f'seen openables: {walkthrough_task.seen_openable_objects}')
                print(
                    f'expert subtask: {SUBTASKS[expert_obs[0]][0]}[{expert_obs[0]}] '
                    f'expert action: {walkthrough_task.action_names()[expert_obs[2]]}[{expert_obs[2]}]'
                )
            step_results = walkthrough_task.step(act)
            observations = step_results.observation
            if visualize:
                import matplotlib.pylab as plt
                # %matplotlib inline

                rgb = (observations['rgb'] * 255).astype(np.uint8)
                plt.imshow(rgb)
        
        walkthrough_expert_actions.append(walkthrough_task.action_names()[observations['expert_subtask_action'][2]])
        walkthorugh_expert_subtasks.append(
            ''.join([arg for arg in IDX_TO_SUBTASK[observations['expert_subtask_action'][0]] if arg is not None])
        )
        if verbose:
            print(f"======== WALKTHROUGH TASK DONE :) ========")
            print(f"======== UNSHUFFLE TASK START :) ========")
        unshuffle_expert_subtasks = []
        unshuffle_expert_actions = []
        unshuffle_task: UnshuffleTask = task_sampler.next_task()
        observations = unshuffle_task.get_observations()
        while not unshuffle_task.is_done():
            expert_obs = observations['expert_subtask_action']
            act = expert_obs[2]
            unshuffle_expert_actions.append(unshuffle_task.action_names()[expert_obs[2]])
            unshuffle_expert_subtasks.append(
                ''.join([arg for arg in IDX_TO_SUBTASK[expert_obs[0]] if arg is not None])
            )
            if verbose:
                print(f'action: {unshuffle_task.action_names()[act]}')
                print(
                    f'expert subtask: {SUBTASKS[expert_obs[0]][0]}[{expert_obs[0]}] '
                    f'expert action: {unshuffle_task.action_names()[expert_obs[2]]}[{expert_obs[2]}]'
                )
            step_results = unshuffle_task.step(act)
            observations = step_results.observation
            if visualize:
                import matplotlib.pylab as plt
                # %matplotlib inline

                rgb = (observations['rgb'] * 255).astype(np.uint8)
                plt.imshow(rgb)
        unshuffle_expert_actions.append(unshuffle_task.action_names()[observations['expert_subtask_action'][2]])
        unshuffle_expert_subtasks.append(
            ''.join([arg for arg in IDX_TO_SUBTASK[observations['expert_subtask_action'][0]] if arg is not None])
        )
        metrics = unshuffle_task.metrics()
        if verbose:
            print(f"======== UNSHUFFLE TASK DONE :) ========")
            print(f"Both phases complete, metrics: '{metrics}'")
        
        task_info = metrics["task_info"]
        del metrics["task_info"]
        leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
        task_info["walkthrough_expert_subtasks"] = walkthorugh_expert_subtasks
        task_info["walkthrough_expert_actions"] = walkthrough_expert_actions
        task_info["unshuffle_expert_subtasks"] = unshuffle_expert_subtasks
        task_info["unshuffle_expert_actions"] = unshuffle_expert_actions
        all_metrics[task_info["unique_id"]] = {**task_info, **metrics}
        
    print(f"\nFinished {num_tasks_to_do} Two-Phase tasks.")
    task_sampler.close()


    save_path = f"./two_phase_subtask_expert_res_{stage}.json.gz"
    save_path_all = f"./two_phase_subtask_expert_res_all_{stage}.json.gz"
    if os.path.exists(os.path.dirname(save_path)):
        print(f"Saving example submission file to {save_path}")
        submission_json_str = json.dumps(leaderboard_submission, cls=NumpyJSONEncoder)
        with gzip.open(save_path, "w") as f:
            f.write(submission_json_str.encode("utf-8"))
    else:
        print(
            f"If you'd like to save an example leaderboard submission, you'll need to edit"
            "`/YOUR/FAVORITE/SAVE/PATH/` so that it references an existing directory."
        )
    if os.path.exists(os.path.dirname(save_path_all)):
        print(f"Saving example submission file to {save_path_all}")
        all_json_str = json.dumps(all_metrics, cls=NumpyJSONEncoder)
        with gzip.open(save_path_all, "w") as f:
            f.write(all_json_str.encode("utf-8"))
    else:
        print(
            f"If you'd like to save an example leaderboard submission, you'll need to edit"
            "`/YOUR/FAVORITE/SAVE/PATH/` so that it references an existing directory."
        )

if __name__ == "__main__":
    args = parse_args()

    visualize = args.visualize
    stages = args.stages
    verbose = args.verbose

    for stage in tqdm(stages):
        stagewise_example(stage=stage, verbose=verbose)
    


#%%

# #%%
# # ===================== TEST FOR SENSORS ============================
# wtask = task_sampler.next_task()
# obs = []
# obss = []
# sr = []
# srobs = []

# #%%

# print(f"num_steps_taken: {wtask.num_steps_taken()}")
# obs.append(wtask.get_observations())
# key_list = list(obs[-1].keys())

# print(f"OBS")
# print(f"\tPose Sensor")
# print(f"\t\tagent_pos: {obs[-1]['pose']['agent_pos']}")
# print(f"\t\tagent_rot: {obs[-1]['pose']['rot_3d_enu_deg']}")
# allocentric_pos = np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"]) @ np.append(obs[-1]['pose']['agent_pos'], 1)
# print(f"\t\tAllocentric Position: {allocentric_pos}")
# allocentric_rot = np.degrees(
#     np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"][:3, :3]) @ (
#         euler.mat2euler(
#             obs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
#                 PoseSensor.create_new_transformation_matrix()
#             )
#         )
#     )
# ) - np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"])[:3, :3] @ obs[-1]['pose']['rot_3d_enu_deg'] % 360
# print(f"\t\tAllocentric Rotation: {allocentric_rot}")
# # allocentric_rot = np.degrees(
# #     euler.mat2euler(
# #         obs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
# #             PoseSensor.create_new_transformation_matrix()
# #         )
# #     )
# # ) + obs[-1]["pose"]["rot_3d_enu_deg"]

# print(f"\tRel. Pose Sensor")
# print(f"\t\tpos: {obs[-1]['test_pos']['last_allocentric_position']}, dxdzdr: {obs[-1]['test_pos']['dx_dz_dr']}")

# print(f"oracle pos: {wtask.env.last_event.metadata['agent']['position']}")
# print(f"oracle rot: {wtask.env.last_event.metadata['agent']['rotation']}")

# print(f"allo[x] {allocentric_pos[0]} | oracle[x] {wtask.env.last_event.metadata['agent']['position']['x']}")
# print(f"allo[z] {allocentric_pos[2]} | oracle[z] {wtask.env.last_event.metadata['agent']['position']['z']}")
# print(f"allo[r] {allocentric_rot[1]} | oracle[r] {wtask.env.last_event.metadata['agent']['rotation']['y']}")

# obss.append([])
# i = 0
# #%%
# print(f"Test repeatition of get_observations()")
# print(f"Trial {i}-th")
# obss[-1].append(wtask.get_observations())
# same_dict = dict()
# for k, v in obss[-1][-1].items():
#     if not isinstance(v, dict):
#         same_dict[k] = (obs[-1][k] == obss[-1][-1][k]).all()
#     else:
#         same_dict[k] = dict()
#         for k_, v_ in obss[-1][-1][k].items():
#             same_dict[k][k_] = (obs[-1][k][k_] == obss[-1][-1][k][k_]).all()
#         same_dict[k] = all(list(same_dict[k].values()))

# print(f"ALL SAME = {all(list(same_dict.values()))}")

# print(f"OBSS[-1]")
# print(f"\tPose Sensor")
# print(f"\t\tagent_pos: {obss[-1][-1]['pose']['agent_pos']}")
# print(f"\t\tagent_rot: {obs[-1]['pose']['rot_3d_enu_deg']}")

# allocentric_pos = np.linalg.inv(obss[-1][-1]["pose"]["T_unity_to_world"]) @ np.append(obss[-1][-1]['pose']['agent_pos'], 1)
# print(f"\t\tAllocentric Position: {allocentric_pos}")
# allocentric_rot = np.degrees(
#     np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"][:3, :3]) @ (
#         euler.mat2euler(
#             obs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
#                 PoseSensor.create_new_transformation_matrix()
#             )
#         )
#     )
# ) - np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"])[:3, :3] @ obs[-1]['pose']['rot_3d_enu_deg'] % 360
# print(f"\t\tAllocentric Rotation: {allocentric_rot}")

# print(f"\tRel. Pose Sensor")
# print(f"\t\tpos: {obss[-1][-1]['test_pos']['last_allocentric_position']}, dxdzdr: {obss[-1][-1]['test_pos']['dx_dz_dr']}")

# print(f"oracle pos: {wtask.env.last_event.metadata['agent']['position']}")

# print(f"allo[x] {allocentric_pos[0]} | oracle[x] {wtask.env.last_event.metadata['agent']['position']['x']}")
# print(f"allo[z] {allocentric_pos[2]} | oracle[z] {wtask.env.last_event.metadata['agent']['position']['z']}")
# print(f"allo[r] {allocentric_rot[1]} | oracle[r] {wtask.env.last_event.metadata['agent']['rotation']['y']}")

# i += 1
# #%%
# sr.append(wtask.step(6))
# srobs.append(sr[-1].observation)
# obs.append(wtask.get_observations())
# obss.append([])
# check_same_obs = [
#     (obs[-1][k] == srobs[-1][k]).all() if not isinstance(obs[-1][k], dict)
#     else (all([(obs[-1][k][k_] == srobs[-1][k][k_]).all() for k_ in obs[-1][k].keys()]))
#     for k in key_list 
# ]
# print(f'\tcheck: {check_same_obs}')

# print(f'SROBS')
# print(f"\tPose Sensor")
# print(f"\t\tagent_pos: {srobs[-1]['pose']['agent_pos']}")
# print(f"\t\tagent_rot: {srobs[-1]['pose']['rot_3d_enu_deg']}")
# allocentric_pos = np.linalg.inv(srobs[-1]["pose"]["T_unity_to_world"]) @ np.append(srobs[-1]['pose']['agent_pos'], 1)
# print(f"\t\tAllocentric Position: {allocentric_pos}")
# allocentric_rot = np.degrees(
#     np.linalg.inv(srobs[-1]["pose"]["T_unity_to_world"][:3, :3]) @ (
#         euler.mat2euler(
#             srobs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
#                 PoseSensor.create_new_transformation_matrix()
#             )
#         )
#     )
# ) - np.linalg.inv(srobs[-1]["pose"]["T_unity_to_world"])[:3, :3] @ srobs[-1]['pose']['rot_3d_enu_deg'] % 360
# print(f"\t\tAllocentric Rotation: {allocentric_rot}")

# print(f"\tRel. Pose Sensor")
# print(f"\t\tpos: {srobs[-1]['test_pos']['last_allocentric_position']}, dxdzdr: {srobs[-1]['test_pos']['dx_dz_dr']}")

# print(f'OBS')
# print(f"\tPose Sensor")
# print(f"\t\tagent_pos: {obs[-1]['pose']['agent_pos']}")
# print(f"\t\tagent_rot: {obs[-1]['pose']['rot_3d_enu_deg']}")

# allocentric_pos = np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"]) @ np.append(obs[-1]['pose']['agent_pos'], 1)
# print(f"\t\tAllocentric Position: {allocentric_pos}")
# allocentric_rot = np.degrees(
#     np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"][:3, :3]) @ (
#         euler.mat2euler(
#             obs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
#                 PoseSensor.create_new_transformation_matrix()
#             )
#         )
#     )
# ) - np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"])[:3, :3] @ obs[-1]['pose']['rot_3d_enu_deg'] % 360
# print(f"\t\tAllocentric Rotation: {allocentric_rot}")

# print(f"\tRel. Pose Sensor")
# print(f"\t\tpos: {obs[-1]['test_pos']['last_allocentric_position']}, dxdzdr: {obs[-1]['test_pos']['dx_dz_dr']}")

# print(f"oracle pos: {wtask.env.last_event.metadata['agent']['position']}")

# print(f"allo[x] {allocentric_pos[0]} | oracle[x] {wtask.env.last_event.metadata['agent']['position']['x']}")
# print(f"allo[z] {allocentric_pos[2]} | oracle[z] {wtask.env.last_event.metadata['agent']['position']['z']}")
# print(f"allo[r] {allocentric_rot[1]} | oracle[r] {wtask.env.last_event.metadata['agent']['rotation']['y']}")


# i = 0

# #%%
# print(f"Test repeatition of get_observations() for obs[{len(obs) - 1}]")
# print(f"Trial {i}-th")
# obss[-1].append(wtask.get_observations())
# same_dict = dict()
# for k, v in obss[-1][-1].items():
#     if not isinstance(v, dict):
#         same_dict[k] = (obs[-1][k] == obss[-1][-1][k]).all()
#     else:
#         same_dict[k] = dict()
#         for k_, v_ in obss[-1][-1][k].items():
#             same_dict[k][k_] = (obs[-1][k][k_] == obss[-1][-1][k][k_]).all()
#         same_dict[k] = all(list(same_dict[k].values()))

# print(f"ALL SAME = {all(list(same_dict.values()))}")

# print(f"OBSS[-1]")
# print(f"\tPose Sensor")
# print(f"\t\tagent_pos: {obss[-1][-1]['pose']['agent_pos']}")
# print(f"\t\tagent_rot: {obs[-1]['pose']['rot_3d_enu_deg']}")

# allocentric_pos = np.linalg.inv(obss[-1][-1]["pose"]["T_unity_to_world"]) @ np.append(obss[-1][-1]['pose']['agent_pos'], 1)
# print(f"\t\tAllocentric Position: {allocentric_pos}")
# allocentric_rot = np.degrees(
#     np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"][:3, :3]) @ (
#         euler.mat2euler(
#             obs[-1]["pose"]["T_unity_to_world"] @ np.linalg.inv(
#                 PoseSensor.create_new_transformation_matrix()
#             )
#         )
#     )
# ) - np.linalg.inv(obs[-1]["pose"]["T_unity_to_world"])[:3, :3] @ obs[-1]['pose']['rot_3d_enu_deg'] % 360
# print(f"\t\tAllocentric Rotation: {allocentric_rot}")

# print(f"\tRel. Pose Sensor")
# print(f"\t\tpos: {obss[-1][-1]['test_pos']['last_allocentric_position']}, dxdzdr: {obss[-1][-1]['test_pos']['dx_dz_dr']}")

# print(f"oracle pos: {wtask.env.last_event.metadata['agent']['position']}")

# print(f"allo[x] {allocentric_pos[0]} | oracle[x] {wtask.env.last_event.metadata['agent']['position']['x']}")
# print(f"allo[z] {allocentric_pos[2]} | oracle[z] {wtask.env.last_event.metadata['agent']['position']['z']}")
# print(f"allo[r] {allocentric_rot[1]} | oracle[r] {wtask.env.last_event.metadata['agent']['rotation']['y']}")

# i += 1

# #%%
# # ===================== TEST FOR SENSORS DONE!!!============================

# %%
