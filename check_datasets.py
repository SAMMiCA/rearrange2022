"""Inference loop for the AI2-THOR object rearrangement task."""
from collections import OrderedDict
from tqdm import tqdm

from allenact.utils.misc_utils import NumpyJSONEncoder
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from experiments.two_phase.two_phase_ta_base import TwoPhaseTaskAwareRearrangeExperimentConfig
from task_aware_rearrange.constants import ORDERED_OBJECT_TYPES


if __name__ == "__main__":
    
    pickupables = OrderedDict()
    openables = OrderedDict()
    pushables = OrderedDict()
    rearranges = OrderedDict()
    obj2scenes = OrderedDict()
    miscs = OrderedDict()
    
    pickupables["merged"] = OrderedDict()
    openables["merged"] = OrderedDict()
    pushables["merged"] = OrderedDict()
    rearranges["merged"] = OrderedDict()
    obj2scenes["merged"] = OrderedDict()
    miscs["merged"] = OrderedDict()
    
    for stage in tqdm(("train", "val", "test"), desc=f"stagewise", position=0):
        task_sampler_params = TwoPhaseTaskAwareRearrangeExperimentConfig.stagewise_task_sampler_args(
            stage=stage,
            process_ind=0,
            total_processes=1,
            devices=[0],
        )
        
        task_sampler_params['thor_controller_kwargs'].update(
            dict(
                renderSemanticSegmentation=True,
                renderInstanceSegmentation=True,
            )
        )

        task_sampler: RearrangeTaskSampler = TwoPhaseTaskAwareRearrangeExperimentConfig.make_sampler_fn(
            **task_sampler_params,
            force_cache_reset=True,
            only_one_unshuffle_per_walkthrough=True,
            epochs=1,
        )
        
        pickupable_map = OrderedDict()
        openable_map = OrderedDict()
        pushable_map = OrderedDict()
        rearrange_map = OrderedDict()
        obj2scene = dict()
        misc_map = OrderedDict()
        
        for task_id in tqdm(range(task_sampler.total_unique), desc=f" task", position=1, leave=False):
            wtask = task_sampler.next_task()
            assert isinstance(wtask, WalkthroughTask)
            wtask.step(action=wtask.action_names().index('done'))

            utask = task_sampler.next_task()
            assert isinstance(utask, UnshuffleTask)
            utask.step(action=utask.action_names().index('done'))
            
            uposes, wposes, _ = utask.env.poses

            for upose, wpose in zip(uposes, wposes):
                if upose["type"] in ORDERED_OBJECT_TYPES:
                    scene = utask.env.scene
                    if upose["type"] not in obj2scene:
                        obj2scene[upose["type"]] = []
                    if scene not in obj2scene[upose["type"]]:
                        obj2scene[upose["type"]].append(scene)
                    
                    color = utask.env.controller.last_event.object_id_to_color[upose["type"]]
                    rearrange_map[upose["type"]] = color
                    
                if not utask.env.are_poses_equal(upose, wpose):
                    color = utask.env.controller.last_event.object_id_to_color[upose["type"]]
                    
                    if wpose["pickupable"]:
                        pickupable_map[upose["type"]] = color
                    elif wpose["openness"] is not None:
                        openable_map[upose["type"]] = color
                    else:
                        pushable_map[upose["type"]] = color
            
            for k, v in utask.env.last_event.object_id_to_color.items():
                if len(k.split('|')) > 1:
                    continue
                misc_map[k] = v

        print(
            f"\n\t = All tasks in {stage} dataset have been processed. = \n"
        )
        print(f"\t\tPickupable objects ({len(pickupable_map)} objects): {pickupable_map}")
        print(f"\t\tOpenable objects ({len(openable_map)} objects): {openable_map}")
        print(f"\t\tPushable objects ({len(pushable_map)} objects): {pushable_map}")
        print(f"\t\tExisting objects ({len(rearrange_map)} objects): {rearrange_map}")
        print(f"\t\tMisc objects ({len(misc_map)} objects): {misc_map}")
        print(f"\t\tObjects to Scenes ({len(obj2scene)} objects): {obj2scene}")
        
        pickupables[stage] = pickupable_map
        openables[stage] = openable_map
        pushables[stage] = pushable_map
        rearranges[stage] = rearrange_map
        obj2scenes[stage] = obj2scene
        miscs[stage] = misc_map
        
        pickupables["merged"].update(pickupable_map)
        openables["merged"].update(openable_map)
        pushables["merged"].update(pushable_map)
        rearranges["merged"].update(rearrange_map)
        obj2scenes["merged"].update(obj2scene)
        miscs["merged"].update(misc_map)
        
        task_sampler.close()
    
    print(f"\nAll Stages Done. \n")
    print(f"\tPickupable objects ({len(pickupables['merged'])} objects): {pickupables['merged']}")
    print(f"\tOpenable objects ({len(openables['merged'])} objects): {openables['merged']}")
    print(f"\tPushable objects ({len(pushables['merged'])} objects): {pushables['merged']}")
    print(f"\tExisting Rearrange objects ({len(rearranges['merged'])} objects): {rearranges['merged']}")
    print(f"\tMisc objects ({len(miscs['merged'])} objects): {miscs['merged']}")
    print(f"\tObjects to Scenes ({len(obj2scenes['merged'])} objects): {obj2scenes['merged']}")
    
    import json
    with open("data_2022.json", "w") as f:
        json.dump(
            dict(
                pickupables=pickupables,
                openables=openables,
                pushables=pushables,
                rearranges=rearranges,
                obj2scenes=obj2scenes,
                miscs=miscs,
            ),
            f,
            cls=NumpyJSONEncoder,
            indent=4
        )
    import pdb; pdb.set_trace()