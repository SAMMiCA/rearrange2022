import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from allenact.utils.misc_utils import NumpyJSONEncoder
from allenact.utils.system import get_logger, init_logging, _new_logger

from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from experiments.one_phase.expert_data_collection.one_phase_expert_data_collection import OnePhaseExpertDataCollectionExpConfig
from task_aware_rearrange.subtasks import IDX_TO_SUBTASK


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', type=str, default="train")
    parser.add_argument('--visualize', action='store_true', )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # init_logging("info")
    args = parse_args()
    stage = args.stage
    visualize = args.visualize
    
    task_sampler_params = OnePhaseExpertDataCollectionExpConfig.stagewise_task_sampler_args(
        stage=stage, process_ind=0, total_processes=1, devices=[0]
    )
    task_sampler_params['thor_controller_kwargs'].update(OnePhaseExpertDataCollectionExpConfig.THOR_CONTROLLER_KWARGS)
    one_phase_rgb_task_sampler: RearrangeTaskSampler = (
        OnePhaseExpertDataCollectionExpConfig.make_sampler_fn(
            **task_sampler_params, force_cache_reset=False, epochs=1,
        )
    )

    how_many_unique_datapoints = one_phase_rgb_task_sampler.total_unique
    num_tasks_to_do = how_many_unique_datapoints
    print(
        f"\n\nSampling {num_tasks_to_do} tasks from the One-Phase VALIDATION dataset"
        f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
    )

    if visualize:
        fig = plt.figure()
        plt.ion()
        plt.show()
        ax = []
        ax.append(fig.add_subplot(1, 2, 1))
        ax.append(fig.add_subplot(1, 2, 2))
        ax[0].set_title('RawRGB-Unshuffle')
        ax[1].set_title('RawRGB-Walkthrough')

    my_leaderboard_submission = {}
    for i_task in range(num_tasks_to_do):
        print(f"\nStarting task {i_task}")
        task_subtasks = []
        
        obs_u_rgb = []
        obs_w_rgb = []
        obs_u_depth = []
        obs_w_depth = []
        obs_u_pose = dict(
            agent_pos_rot_horizon=[],
            agent_pos_unity=[],
            T_w2c=[],
            T_u2w=[],
        )
        obs_w_pose = dict(
            agent_pos_rot_horizon=[],
            agent_pos_unity=[],
            T_w2c=[],
            T_u2w=[],
        )
        obs_u_semseg = []
        obs_w_semseg = []
        obs_inven = []
        obs_expert = []

        # Get the next task from the task sampler, for One Phase Rearrangement
        # there is only the unshuffle phase (walkthrough happens at the same time implicitly).
        unshuffle_task: UnshuffleTask = one_phase_rgb_task_sampler.next_task()
        print(
            f"Sampled task is from the "
            f" '{one_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
            f" unique id '{one_phase_rgb_task_sampler.current_task_spec.unique_id}'"
        )

        observations = unshuffle_task.get_observations()
        # import pdb; pdb.set_trace()
        while not unshuffle_task.is_done():
            esa = observations[OnePhaseExpertDataCollectionExpConfig.EXPERT_SUBTASK_ACTION_UUID]
            raw_rgb = observations[OnePhaseExpertDataCollectionExpConfig.EGOCENTRIC_RAW_RGB_UUID]
            raw_w_rgb = observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_RAW_RGB_UUID]
            if visualize:
                ax[0].imshow(raw_rgb)
                ax[1].imshow(raw_w_rgb)

                plt.draw()
                plt.pause(0.001)

            obs_u_rgb.append((raw_rgb * 255).astype(np.uint8))
            obs_w_rgb.append((raw_w_rgb * 255).astype(np.uint8))
            obs_u_depth.append(observations[OnePhaseExpertDataCollectionExpConfig.DEPTH_UUID])
            obs_w_depth.append(observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_DEPTH_UUID])
            obs_u_pose["agent_pos_rot_horizon"].append(
                [
                    *observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["agent_pos"],
                    observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["rot_3d_enu_deg"][-1],
                    observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["cam_horizon_deg"].item(),
                ]
            )
            obs_u_pose["agent_pos_unity"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["agent_pos_unity"]
            )
            obs_u_pose["T_w2c"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["T_world_to_cam"]
            )
            obs_u_pose["T_u2w"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.POSE_UUID]["T_unity_to_world"]
            )
            obs_w_pose["agent_pos_rot_horizon"].append(
                [
                    *observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["agent_pos"],
                    observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["rot_3d_enu_deg"][-1],
                    observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["cam_horizon_deg"].item(),
                ]
            )
            obs_w_pose["agent_pos_unity"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["agent_pos_unity"]
            )
            obs_w_pose["T_w2c"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["T_world_to_cam"]
            )
            obs_w_pose["T_u2w"].append(
                observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_POSE_UUID]["T_unity_to_world"]
            )
            
            _u_semseg = observations[OnePhaseExpertDataCollectionExpConfig.SEMANTIC_SEGMENTATION_UUID]
            _w_semseg = observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID]

            u_semseg = np.zeros(_u_semseg.shape[1:], dtype=np.uint8)
            w_semseg = np.zeros(_w_semseg.shape[1:], dtype=np.uint8)
            for i in range(_u_semseg.shape[0]):
                u_semseg += i * _u_semseg[i].astype(np.uint8)
                w_semseg += i * _w_semseg[i].astype(np.uint8)

            if (
                u_semseg.max() > 72
                or w_semseg.max() > 72
            ):
                print("A?")
                import pdb; pdb.set_trace()

            obs_u_semseg.append(u_semseg)
            obs_w_semseg.append(w_semseg)

            obs_inven.append(observations[OnePhaseExpertDataCollectionExpConfig.INVENTORY_UUID])
            obs_expert.append(observations[OnePhaseExpertDataCollectionExpConfig.EXPERT_SUBTASK_ACTION_UUID])

            subtask_ind = esa[0]
            task_subtasks.append(IDX_TO_SUBTASK[subtask_ind])
            action_ind = esa[-2]
            if unshuffle_task.num_steps_taken() % 10 == 0:
                print(
                    f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                    f" taking action {unshuffle_task.action_names()[action_ind]}"
                )
            observations = unshuffle_task.step(action=action_ind).observation

        if visualize:
            raw_rgb = observations[OnePhaseExpertDataCollectionExpConfig.EGOCENTRIC_RAW_RGB_UUID]
            raw_w_rgb = observations[OnePhaseExpertDataCollectionExpConfig.UNSHUFFLED_RAW_RGB_UUID]
            ax[0].imshow(raw_rgb)
            ax[1].imshow(raw_w_rgb)

            plt.draw()
            plt.pause(0.001)

        metrics = unshuffle_task.metrics()
        metrics["task_info"]["unshuffle_subtasks"] = task_subtasks
        obs = {
            'unshuffle_rgb': np.stack(obs_u_rgb),           # (nsteps, 224, 224, 3)
            'walkthrough_rgb': np.stack(obs_w_rgb),         # (nsteps, 224, 224, 3)
            'unshuffle_depth': np.stack(obs_u_depth),       # (nsteps, 224, 224, 1)
            'walkthrough_depth': np.stack(obs_w_depth),     # (nsteps, 224, 224, 1)
            'unshuffle_pos_rot_horizon': np.stack(obs_u_pose["agent_pos_rot_horizon"]),     # (nsteps, 5)
            'unshuffle_pos_unity': np.stack(obs_u_pose["agent_pos_unity"]),                 # (nsteps, 3)
            'unshuffle_Tw2c': np.stack(obs_u_pose["T_w2c"]),                                # (nsteps, 4, 4)
            'unshuffle_Tu2w': np.stack(obs_u_pose["T_u2w"]),                                # (nsteps, 4, 4)
            'walkthrough_pos_rot_horizon': np.stack(obs_w_pose["agent_pos_rot_horizon"]),   # (nsteps, 5)
            'walkthrough_pos_unity': np.stack(obs_w_pose["agent_pos_unity"]),               # (nsteps, 3)
            'walkthrough_Tw2c': np.stack(obs_w_pose["T_w2c"]),                              # (nsteps, 4, 4)
            'walkthrough_Tu2w': np.stack(obs_w_pose["T_u2w"]),                              # (nsteps, 4, 4)
            'unshuffle_semseg': np.stack(obs_u_semseg),     # (nsteps, 73, 224, 224)
            'walkthrough_semseg': np.stack(obs_w_semseg),   # (nsteps, 73, 224, 224)
            'inventory': np.stack(obs_inven),               # (nsteps, 73)
            'expert_subtask_action': np.stack(obs_expert),  # (nsteps, 4)
        }
        # print(f"Both phases complete, metrics: '{metrics}'")
        save_obs_dir_path = f'./expert_data/{stage}/{metrics["task_info"]["unique_id"]}'
        if not os.path.exists(save_obs_dir_path):
            os.makedirs(save_obs_dir_path)
        
        for k, v in obs.items():
            np.save(os.path.join(save_obs_dir_path, k), v)

        print(f"Both phases complete....")

        task_info = metrics["task_info"]
        del metrics["task_info"]
        my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
        # import pdb; pdb.set_trace()
        
    import json
    import gzip
    import os

    save_path = f"./expert_data_{stage}.json.gz"
    if os.path.exists(os.path.dirname(save_path)):
        print(f"Saving expert data file to {save_path}")
        submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
        with gzip.open(save_path, "w") as f:
            f.write(submission_json_str.encode("utf-8"))
    else:
        print(
            f"If you'd like to save an example leaderboard submission, you'll need to edit"
            "`/YOUR/FAVORITE/SAVE/PATH/` so that it references an existing directory."
        )

    one_phase_rgb_task_sampler.close()

    print(f"\nFinished {num_tasks_to_do} One-Phase tasks.")
