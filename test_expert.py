import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from allenact.utils.misc_utils import NumpyJSONEncoder
from allenact.utils.system import get_logger, init_logging, _new_logger
from allenact.utils.tensor_utils import batch_observations
from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from experiments.test_exp import ExpertTestExpConfig
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
    
    # # simple metric
    # simple_metrics = dict(
    #     num_tasks=0,
    #     ep_length=0.0,
    #     reward=0.0,
    #     start_energy=0.0,
    #     end_energy=0.0,
    #     success=0.0,
    #     prop_fixed=0.0,
    #     prop_fixed_strict=0.0,
    #     num_misplaced=0.0,
    #     num_newly_misplaced=0.0,
    #     num_initially_misplaced=0.0,
    #     num_fixed=0.0,
    #     num_broken=0.0,
    # )

    task_sampler_params = ExpertTestExpConfig.stagewise_task_sampler_args(
        stage=stage, process_ind=0, total_processes=1, devices=[0]
    )
    task_sampler_params['thor_controller_kwargs'].update(ExpertTestExpConfig.THOR_CONTROLLER_KWARGS)
    one_phase_rgb_task_sampler: RearrangeTaskSampler = (
        ExpertTestExpConfig.make_sampler_fn(
            **task_sampler_params, force_cache_reset=False, epochs=1,
        )
    )
    sensor_preprocessor_graph = ExpertTestExpConfig.create_preprocessor_graph(mode="train")().to(
        torch.device(0)
    )
    agent_model = ExpertTestExpConfig.create_model(**{"sensor_preprocessor_graph": sensor_preprocessor_graph}).to(
        torch.device(0)
    )
    # RolloutStorages
    training_pipeline = ExpertTestExpConfig.training_pipeline()
    rollout_storage = training_pipeline.rollout_storage
    uuid_to_storage = training_pipeline.current_stage_storage
    recurrent_memory_specification = agent_model.recurrent_memory_specification

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

        # Get the next task from the task sampler, for One Phase Rearrangement
        # there is only the unshuffle phase (walkthrough happens at the same time implicitly).
        unshuffle_task: UnshuffleTask = one_phase_rgb_task_sampler.next_task()
        print(
            f"Sampled task is from the "
            f" '{one_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
            f" unique id '{one_phase_rgb_task_sampler.current_task_spec.unique_id}'"
        )

        print(f"Observations from Environments")
        observations = unshuffle_task.get_observations()
        # for k, v in observations.items():
        #     if not isinstance(v, dict):
        #         print(
        #             f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
        #             + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #         )
        #     else:
        #         print(f'KEY [{k}] | TYPE [{type(v)}]')
        #         for k1, v1 in v.items():
        #             print(
        #                 f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
        #                 + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #             )

        print(f"BATCHED Observations")
        batch = batch_observations([observations], device=torch.device(0))
        # for k, v in batch.items():
        #     if not isinstance(v, dict):
        #         print(
        #             f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
        #             + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #         )
        #     else:
        #         print(f'KEY [{k}] | TYPE [{type(v)}]')
        #         for k1, v1 in v.items():
        #             print(
        #                 f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
        #                 + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #             )
        
        print(f"PREPROCESSED Observations")
        preprocessed_obs = sensor_preprocessor_graph.get_observations(batch)
        # for k, v in preprocessed_obs.items():
        #     if not isinstance(v, dict):
        #         print(
        #             f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
        #             + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #         )
        #     else:
        #         print(f'KEY [{k}] | TYPE [{type(v)}]')
        #         for k1, v1 in v.items():
        #             print(
        #                 f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
        #                 + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
        #             )

        rollout_storage.to(torch.device(0))
        rollout_storage.set_partition(index=0, num_parts=1)
        rollout_storage.initialize(
            observations=preprocessed_obs,
            num_samplers=1,
            recurrent_memory_specification=recurrent_memory_specification,
            action_space=agent_model.action_space,
        )

        agent_input = rollout_storage.agent_input_for_next_step()
        ac_out, memory = agent_model(**agent_input)

        import pdb; pdb.set_trace()
        while not unshuffle_task.is_done():
            esa = observations[ExpertTestExpConfig.EXPERT_SUBTASK_ACTION_UUID]
            raw_rgb = observations[ExpertTestExpConfig.EGOCENTRIC_RAW_RGB_UUID]
            raw_w_rgb = observations[ExpertTestExpConfig.UNSHUFFLED_RAW_RGB_UUID]
            if visualize:
                ax[0].imshow(raw_rgb)
                ax[1].imshow(raw_w_rgb)

                plt.draw()
                plt.pause(0.001)
            # Take a random action
            # print(
            #     f"(step {unshuffle_task.num_steps_taken()}) Expert Subtask: {IDX_TO_SUBTASK[esa[0]]}"
            # )
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
            raw_rgb = observations[ExpertTestExpConfig.EGOCENTRIC_RAW_RGB_UUID]
            raw_w_rgb = observations[ExpertTestExpConfig.UNSHUFFLED_RAW_RGB_UUID]
            ax[0].imshow(raw_rgb)
            ax[1].imshow(raw_w_rgb)

            plt.draw()
            plt.pause(0.001)

        metrics = unshuffle_task.metrics()
        metrics["task_info"]["unshuffle_subtasks"] = task_subtasks
        print(f"Both phases complete, metrics: '{metrics}'")

        task_info = metrics["task_info"]
        del metrics["task_info"]
        my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
        
    import json
    import gzip
    import os

    save_path = f"./subtask_expert_res_{stage}.json.gz"
    if os.path.exists(os.path.dirname(save_path)):
        print(f"Saving example submission file to {save_path}")
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
