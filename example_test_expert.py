#%%
from experiments.two_phase.two_phase_test_config import TwoPhaseTestConfig as Config
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from task_aware_rearrange.subtasks import SUBTASKS
import numpy as np

# First let's generate our task sampler that will let us run through all of the
# data points in our training set.

task_sampler_params = Config.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1,
)
task_sampler: RearrangeTaskSampler = Config.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
#%%
import matplotlib.pylab as plt
%matplotlib inline

walkthrough_task = task_sampler.next_task()
observations = walkthrough_task.get_observations()
rgb = (observations['rgb'] * 255).astype(np.uint8)
plt.imshow(rgb)
print(f'seen pickupables: {walkthrough_task.seen_pickupable_objects}')
print(f'seen openables: {walkthrough_task.seen_openable_objects}')
expert_obs = observations['expert_subtask_action']
print(
    f'expert subtask: {SUBTASKS[expert_obs[0]][0]}[{expert_obs[0]}] '
    f'expert action: {walkthrough_task.action_names()[expert_obs[2]]}[{expert_obs[2]}]'
)


#%%
act = int(input())
print(f'action: {walkthrough_task.action_names()[act]}')
walkthrough_task.step(act)
observations = walkthrough_task.get_observations()
rgb = (observations['rgb'] * 255).astype(np.uint8)
plt.imshow(rgb)
print(f'seen pickupables: {walkthrough_task.seen_pickupable_objects}')
print(f'seen openables: {walkthrough_task.seen_openable_objects}')
# %%
