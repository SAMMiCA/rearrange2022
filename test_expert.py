import matplotlib.pyplot as plt

from allenact.utils.misc_utils import NumpyJSONEncoder
from allenact.utils.system import get_logger, init_logging, _new_logger

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from experiments.test_exp import ExpertTestExpConfig
from task_aware_rearrange.subtasks import IDX_TO_SUBTASK


if __name__ == "__main__":
    # init_logging("info")
    visualize = False
    
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
        stage="combined", process_ind=0, total_processes=1, devices=[0]
    )
    task_sampler_params['thor_controller_kwargs'].update(ExpertTestExpConfig.THOR_CONTROLLER_KWARGS)
    one_phase_rgb_task_sampler: RearrangeTaskSampler = (
        ExpertTestExpConfig.make_sampler_fn(
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

        # Get the next task from the task sampler, for One Phase Rearrangement
        # there is only the unshuffle phase (walkthrough happens at the same time implicitly).
        unshuffle_task: UnshuffleTask = one_phase_rgb_task_sampler.next_task()
        print(
            f"Sampled task is from the "
            f" '{one_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
            f" unique id '{one_phase_rgb_task_sampler.current_task_spec.unique_id}'"
        )
        # if i_task < 5:
        #     continue

        observations = unshuffle_task.get_observations()
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
        print(f"Both phases complete, metrics: '{metrics}'")

        task_info = metrics["task_info"]
        del metrics["task_info"]
        my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
        
    import json
    import gzip
    import os

    save_path = "./test_subtask_submission_combined.json.gz"
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
