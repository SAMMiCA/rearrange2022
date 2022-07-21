from collections import defaultdict, OrderedDict
import copy
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Dict,
    TYPE_CHECKING,
    TypeVar,
)
import numpy as np
import torch
import math
import stringcase
import gym
import gym.spaces as gyms
import networkx as nx
from transforms3d import euler
from allenact.base_abstractions.task import Task
from allenact.base_abstractions.sensor import Sensor, AbstractExpertActionSensor
from allenact.utils.system import get_logger
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor, include_object_data
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from rearrange.environment import RearrangeMode, RearrangeTHOREnvironment
from rearrange.expert import ShortestPathNavigatorTHOR
from rearrange.tasks import AbstractRearrangeTask, UnshuffleTask, WalkthroughTask
from rearrange.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS, STEP_SIZE
from example_utils import ForkedPdb
from task_aware_rearrange.subtasks import NUM_SUBTASKS, SUBTASK_TO_IDX, Subtask


def _are_agent_locations_equal(
    ap0: Dict[str, Union[float, int, bool]],
    ap1: Dict[str, Union[float, int, bool]],
    ignore_standing: bool,
    tol=1e-2,
    ignore_y: bool = True,
):
    """Determines if two agent locations are equal up to some tolerance."""

    def rot_dist(r0: float, r1: float):
        diff = abs(r0 - r1) % 360
        return min(diff, 360 - diff)

    return (
        all(
            abs(ap0[k] - ap1[k]) <= tol
            for k in (["x", "z"] if ignore_y else ["x", "y", "z"])
        )
        and rot_dist(ap0["rotation"], ap1["rotation"]) <= tol
        and rot_dist(ap0["horizon"], ap1["horizon"]) <= tol
        and (ignore_standing or (ap0["standing"] == ap1["standing"]))
    )


class OnePhaseSubtaskAndActionExpertSensor(AbstractExpertActionSensor):

    ACTION_LABEL = "action"
    SUBTASK_LABEL = "subtask"

    def __init__(
        self,
        action_space: Optional[Union[gyms.Dict, Tuple[int, int]]] = None,
        uuid: str = "subtask_and_action_expert",
        expert_args: Optional[Dict[str, Any]] = None,
        nsubtasks: Optional[int] = None,
        nactions: Optional[int] = None,
        use_dict_as_groups: bool = True,
        max_priority_per_object: int = 3,
        verbose: bool = False,
        **kwargs: Any,
    ):
        if isinstance(action_space, tuple):
            action_space = gyms.Dict(
                OrderedDict(
                    [
                        (self.SUBTASK_LABEL, gyms.Discrete(action_space[0])),
                        (self.ACTION_LABEL, gyms.Discrete(action_space[1])),
                    ]
                )
            )
        elif action_space is None:
            assert (
                nactions is not None and nsubtasks is not None
            ), "One of `action_space` or (`nactions`, `nsubtasks`) must be not `None`."
            action_space = gyms.Dict(
                OrderedDict(
                    [
                        (self.SUBTASK_LABEL, gyms.Discrete(nsubtasks)),
                        (self.ACTION_LABEL, gyms.Discrete(nactions)),
                    ]
                )
            )
        super().__init__(action_space, uuid, expert_args, None, use_dict_as_groups, **kwargs)

        self.max_priority_per_object = max_priority_per_object
        self.verbose = verbose

        self.expert_action_list: List[Optional[int]] = []
        self.expert_subtask_list: List[Optional[int]] = []

        self._last_held_object_name: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self._name_of_object_we_wanted_to_pickup: Optional[str] = None
        self.object_name_to_priority: defaultdict = defaultdict(self.default_priority)
        
        self._objects_to_rearrange: Optional[Dict[str, Any]] = None
        self.seen_obj_names_unshuffle: Dict[str, Any] = dict()
        self.seen_obj_names_walkthrough: Dict[str, Any] = dict()

        self._last_subtask: Subtask = Subtask()

    @staticmethod
    def default_priority():
        return 0

    def reset(self) -> None:
        self.expert_action_list = []
        self.expert_subtask_list = []

        self._last_held_object_name = None
        self._last_to_interact_object_pose = None
        self._name_of_object_we_wanted_to_pickup = None
        self.object_name_to_priority = defaultdict(self.default_priority)
        
        self._objects_to_rearrange = None
        self.seen_obj_names_unshuffle = dict()
        self.seen_obj_names_walkthrough = dict()

        self._last_subtask.reset()

    def expert_action(self, task: UnshuffleTask) -> int:
        """Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        """
        assert task.num_steps_taken() == len(self.expert_action_list) - 1
        return self.expert_action_list[-1]

    def expert_subtask(self, task: UnshuffleTask) -> int:

        assert task.num_steps_taken() == len(self.expert_subtask_list) - 1
        return self.expert_subtask_list[-1]

    @staticmethod
    def get_object_names_from_current_view(
        task,
        env_type,
        rearrange_targets,
    ):
        if env_type == "Unshuffle":
            env = task.env
            
        elif env_type == "Walkthrough":
            # match agent location
            unshuffle_loc = task.env.get_agent_location()
            walkthrough_agent_loc = task.walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                task.walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
            env = task.walkthrough_env        

        else:
            raise NotImplementedError

        with include_object_data(env.controller):
            metadata = env.controller.last_event.metadata
            visible_objects = env.controller.last_event.instance_masks.keys()
            visible_rearrange_objects = [
                o['name'] 
                for o in metadata['objects']
                if (
                    o['name'] in rearrange_targets
                    and o['objectId'] in visible_objects
                )
            ]

        return visible_rearrange_objects    

    @staticmethod
    def _invalidate_interactable_loc_for_pose(
        env,
        location: Dict[str, Any], 
        obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        
        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env.controller
        )
        for i, loc in enumerate([*interactable_positions]):
            if (
                env.shortest_path_navigator.get_key(loc)
                == env.shortest_path_navigator.get_key(location)
                and loc["standing"] == location["standing"]
            ):
                interactable_positions.pop(i)
                return True
        return False

    def get_observation(
        self, 
        env: IThorEnvironment, 
        task: UnshuffleTask, 
        *args: Any, 
        **kwargs: Any
    ) -> Union[OrderedDict, Tuple]:

        if task.num_steps_taken() == 0:
            self.reset()
            if not hasattr(env, "shortest_path_navigator"):
                env.shortest_path_navigator = ShortestPathNavigatorTHOR(
                    controller=env.controller,
                    grid_size=STEP_SIZE,
                    include_move_left_right=all(
                        f"move_{k}" in task.action_names() for k in ["left", "right"]
                    ),
                )
            env.shortest_path_navigator.on_reset()

            # for two-phase
            if task.object_names_seen_in_walkthrough is not None:
                c = env.controller
                with include_object_data(c):
                    for o in c.last_event.metadata['objects']:
                        if o['name'] not in task.object_names_seen_in_walkthrough:
                            self.object_name_to_priority[o["name"]] = (
                                self.max_priority_per_object + 1
                            )

        last_action = task.actions_taken[-1] if task.actions_taken else None    # string
        last_action_success = task.actions_taken_success[-1] if task.actions_taken_success else None
        last_action_ind = task.action_names().index(last_action) if last_action else None

        if task.is_done():
            return self.flatten_output(self._zeroed_observation)

        actions_or_policies = OrderedDict()
        subtask, action = self.query_expert(task)
        actions_or_policies[self.SUBTASK_LABEL] = OrderedDict(
            [
                (self.ACTION_POLICY_LABEL, subtask),
                (self.EXPERT_SUCCESS_LABEL, True if subtask is not None else False),
            ]
        )
        actions_or_policies[self.ACTION_LABEL] = OrderedDict(
            [
                (self.ACTION_POLICY_LABEL, action),
                (self.EXPERT_SUCCESS_LABEL, True if action is not None else False),
            ]
        )

        return self.flatten_output(
            actions_or_policies
            if self.use_groups
            else actions_or_policies[self._NO_GROUPS_LABEL]
        )

    def query_expert(
        self, 
        task: UnshuffleTask, 
        expert_sensor_group_name: Optional[str] = None,
    ) -> Tuple[Any, bool]:
        env = task.unshuffle_env

        last_action = task.actions_taken[-1] if task.actions_taken else None    # string
        last_action_success = task.actions_taken_success[-1] if task.actions_taken_success else None
        last_action_ind = task.action_names().index(last_action) if last_action else None

        # Update
        self.update(
            task=task, 
            shortest_path_navigator=env.shortest_path_navigator, 
            action_taken=last_action_ind, 
            action_success=last_action_success
        )

        return self.expert_subtask(task=task), self.expert_action(task)

    def update(
        self,
        task: UnshuffleTask, 
        shortest_path_navigator: ShortestPathNavigatorTHOR,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        """
        update and generate expert subtask
        """
        _, goal_poses, cur_poses = task.env.poses
        if self.verbose:
            get_logger().info(
                f'==== [STEP {task.num_steps_taken()}] ==== '
            )
            get_logger().info(
                f'  ** IN update() method'
            )

        if task.num_steps_taken() == 0:
            # At the first step of the task
            self._objects_to_rearrange = {
                goal_poses[id]["name"]: goal_poses[id]
                for id in task.start_energies.nonzero()[0]
            }
            if self.verbose:
                get_logger().info(
                    f'    set objects to be rearranged: {self._objects_to_rearrange.keys()}'
                )
        
        if self.verbose:
            get_logger().info(
                f'    priorities: {self.object_name_to_priority}'
            )
            
        # update objects has been seen
        for gp, cp in zip(goal_poses, cur_poses):
            if (
                (gp["broken"] == cp["broken"] == False)
                and self.object_name_to_priority[gp["name"]]
                <= self.max_priority_per_object
                and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
            ):
                # If something has moved from its original position,
                # it should be added to the list of objects to be rearranged.
                if cp["name"] not in self._objects_to_rearrange:
                    self._objects_to_rearrange[cp["name"]] = gp

                # Store newly detected objects to rearrange
                if cp["name"] in self.get_object_names_from_current_view(
                    task=task, env_type="Unshuffle", rearrange_targets=self._objects_to_rearrange
                ):
                    self.seen_obj_names_unshuffle[gp["name"]] = cp
                if gp["name"] in self.get_object_names_from_current_view(
                    task=task, env_type="Walkthrough", rearrange_targets=self._objects_to_rearrange
                ):
                    self.seen_obj_names_walkthrough[gp["name"]] = gp
            elif (
                (gp["broken"] == cp["broken"] == False)
                and RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                and (
                    gp["name"] in self.seen_obj_names_unshuffle
                    or gp["name"] in self.seen_obj_names_walkthrough
                )
            ):
                # Delete the object that has been rearranged from stored memory.
                if gp["name"] in self.seen_obj_names_unshuffle:
                    del self.seen_obj_names_unshuffle[gp["name"]]
                if gp["name"] in self.seen_obj_names_walkthrough:
                    del self.seen_obj_names_walkthrough[gp["name"]]
        
        if self.verbose:
            get_logger().info(
                f'    seen objects updated... '
            )
            get_logger().info(
                f'      unshuffle env.: {self.seen_obj_names_unshuffle.keys()}'
            )
            get_logger().info(
                f'      walkthrough env.: {self.seen_obj_names_walkthrough.keys()}'
            )

        held_object = task.env.held_object
        if held_object:
            self._last_held_object_name = held_object['name']
            if self.verbose:
                get_logger().info(
                    f'    agent is holding object [{held_object["name"]}]... '
                )
        if self.verbose:
            get_logger().info(
                f'    Last Expert Subtask: {self._last_subtask}[{self._last_subtask.get_subtask_idx()}]'
            )
            last_expert_action = None if len(self.expert_action_list) == 0 else self.expert_action_list[-1]
            last_expert_action_str = None if last_expert_action is None else task.action_names()[last_expert_action]
            get_logger().info(
                f'    Last Expert Action: {last_expert_action_str}[{last_expert_action}]'
            )
            get_logger().info(
                f'    Last Action Taken: {task.action_names()[action_taken] if action_taken else None}[{action_taken}] | Success: {action_success}'
            )
        if action_taken is not None:
            assert action_success is not None

            action_names = task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            assert self._last_subtask is not None
            last_expert_subtask = copy.deepcopy(self._last_subtask)
            was_interact_subtask = last_expert_subtask.is_interact_subtask()
            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if agent_took_expert_action:
                # when the agent took expert action
                if self.verbose:
                    get_logger().info(
                        f'      Agent took expert action!'
                    )
                if "open_by_type" in action_str:
                    self.object_name_to_priority[
                        self._last_to_interact_object_pose["name"]
                    ] += 1
                if "drop_held_object_with_snap" in action_str:
                    if self._name_of_object_we_wanted_to_pickup is not None:
                        self.object_name_to_priority[
                            self._name_of_object_we_wanted_to_pickup
                        ] += 1
                    else:
                        self.object_name_to_priority[self._last_held_object_name] += 1

                if action_success:
                    if self.verbose:
                        get_logger().info(
                            f'      Action succeeded!'
                        )
                    if was_interact_subtask:
                        assert action_str == last_expert_subtask.get_expert_action_str()
                        assert self._last_to_interact_object_pose is not None
                        if self.verbose:
                            get_logger().info(
                                f'      Interaction correctly done!'
                            )
                        
                        if last_expert_subtask.subtask_type == "PickupObject":
                            self._name_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose[
                                "name"
                            ]
                            self._last_to_interact_object_pose["target_map"] = "Walkthrough"
                            for k in ["position", "rotation"]:
                                self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                                    held_object["name"]
                                ][k]

                        elif (set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)):
                            if self.verbose:
                                for obj_name in (
                                    set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                                ):
                                    get_logger().info(
                                        f'      priority[{obj_name}]: {self.object_name_to_priority[obj_name]}'
                                    )
                                get_logger().info(
                                    self._objects_to_rearrange.keys()
                                )
                            target_obj_name = None
                            failed_places_and_min_dist = (float("inf"), float("inf"))
                            for obj_name in (
                                set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                            ):
                                if self.object_name_to_priority[obj_name] < self.max_priority_per_object:
                                    priority = self.object_name_to_priority[obj_name]
                                    position = next(
                                        (
                                            obj["position"]
                                            for obj in task.env.last_event.metadata["objects"]
                                            if obj["name"] == obj_name
                                        ),
                                        None
                                    )
                                    priority_and_dist_to_object = (
                                        priority,
                                        IThorEnvironment.position_dist(
                                            task.env.get_agent_location(), position, ignore_y=True, l1_dist=True
                                        )
                                    )
                                    if priority_and_dist_to_object < failed_places_and_min_dist:
                                        failed_places_and_min_dist = priority_and_dist_to_object
                                        target_obj_name = obj_name

                            self._last_to_interact_object_pose = None
                            if target_obj_name is not None:
                                # if target_obj_name in self.seen_obj_names_unshuffle:
                                #     target_obj = self.seen_obj_names_unshuffle[target_obj_name]
                                # else:
                                #     target_obj = self.seen_obj_names_walkthrough[target_obj_name]
                                #     # objectId and positions should be replaced.
                                #     for k in ["position", "rotation"]:
                                #         target_obj[k] = next(
                                #             (
                                #                 obj[k] 
                                #                 for obj in task.env.last_event.metadata["objects"]
                                #                 if obj["name"] == target_obj["name"]
                                #             ),
                                #             None
                                #         )
                                                
                                #     target_obj["objectId"] = next(
                                #         (
                                #             obj["objectId"]
                                #             for obj in task.env.last_event.metadata["objects"]
                                #             if obj["name"] == target_obj["name"]
                                #         ),
                                #         None
                                #     )
                                #     if target_obj["objectId"] is None:
                                #         raise Exception
                                target_obj = next(
                                    (
                                        cp for cp in cur_poses
                                        if cp["name"] == target_obj_name
                                    ),
                                    None
                                )
                                assert target_obj is not None
                            
                                self._last_to_interact_object_pose = target_obj
                                self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                        
                        else:
                            self._last_to_interact_object_pose = None

                        self._last_subtask.next_subtask(
                            obj_type=self._last_to_interact_object_pose["type"] if (
                                self._last_to_interact_object_pose and "type" in self._last_to_interact_object_pose
                            ) else None, 
                            target_map=self._last_to_interact_object_pose["target_map"] if (
                                self._last_to_interact_object_pose and "target_map" in self._last_to_interact_object_pose
                            ) else None,
                        )

                    elif "Explore" in last_expert_subtask.subtask_type:
                        if (set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)):
                            # If found something different,
                            # Go to there...
                            # found object should be assigned to Goto Subtask...
                            target_obj_name = None
                            failed_places_and_min_dist = (float("inf"), float("inf"))
                            for obj_name in (
                                set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                            ):
                                if self.object_name_to_priority[obj_name] < self.max_priority_per_object:
                                    priority = self.object_name_to_priority[obj_name]
                                    position = next(
                                        (
                                            obj["position"]
                                            for obj in task.env.last_event.metadata["objects"]
                                            if obj["name"] == obj_name
                                        ),
                                        None
                                    )
                                    priority_and_dist_to_object = (
                                        priority,
                                        IThorEnvironment.position_dist(
                                            task.env.get_agent_location(), position, ignore_y=True, l1_dist=True
                                        )
                                    )
                                    if priority_and_dist_to_object < failed_places_and_min_dist:
                                        failed_places_and_min_dist = priority_and_dist_to_object
                                        target_obj_name = obj_name
                            
                            if target_obj_name is not None:
                                # if target_obj_name in self.seen_obj_names_unshuffle:
                                #     target_obj = self.seen_obj_names_unshuffle[target_obj_name]
                                # else:
                                #     target_obj = self.seen_obj_names_walkthrough[target_obj_name]
                                #     for k in ["position", "rotation"]:
                                #         target_obj[k] = next(
                                #             (
                                #                 obj[k] 
                                #                 for obj in task.env.last_event.metadata["objects"]
                                #                 if obj["name"] == target_obj["name"]
                                #             ),
                                #             None
                                #         )
                                #     target_obj["objectId"] = next(
                                #         (
                                #             obj["objectId"]
                                #             for obj in task.env.last_event.metadata["objects"]
                                #             if obj["name"] == target_obj["name"]
                                #         ),
                                #         None
                                #     )
                                #     if target_obj["objectId"] is None:
                                #         raise Exception
                                target_obj = next(
                                    (
                                        cp for cp in cur_poses
                                        if cp["name"] == target_obj_name
                                    ),
                                    None
                                )
                                assert target_obj is not None
                                self._last_to_interact_object_pose = target_obj
                                self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                                if self.verbose:
                                    get_logger().info(
                                        f'      Found object to rearrage! Target: {target_obj["name"]}'
                                    )

                                self._last_subtask.next_subtask(
                                    obj_type=target_obj["type"],
                                    target_map="Unshuffle",
                                )

                    elif "Goto" in last_expert_subtask.subtask_type:
                        assert (
                            self._last_to_interact_object_pose is not None
                            and "target_map" in self._last_to_interact_object_pose
                        )
                        if (
                            held_object is not None 
                            and self._last_to_interact_object_pose["name"] != held_object["name"]
                        ):
                            self._last_to_interact_object_pose = copy.deepcopy(held_object)
                            self._last_to_interact_object_pose["target_map"] = "Walkthrough"

                        if self._last_to_interact_object_pose["target_map"] == "Walkthrough":
                            for k in ["position", "rotation"]:
                                self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                                    self._last_to_interact_object_pose["name"]
                                ][k]

                        vis_objs = self.get_object_names_from_current_view(
                            task=task, env_type=self._last_to_interact_object_pose["target_map"], rearrange_targets=self._objects_to_rearrange
                        )
                        if self._last_to_interact_object_pose["name"] in vis_objs:
                            if self.verbose:
                                get_logger().info(
                                    f'      Target object is visible in the current view frame...'
                                )
                            if self.object_name_to_priority[self._last_to_interact_object_pose["name"]] > self.max_priority_per_object:
                                # it is ignored when predict the next subtask.
                                if self.verbose:
                                    get_logger().info(
                                        f'      Exceeded maximum number of trial for {self._last_to_interact_object_pose["name"]}'
                                    )
                                    get_logger().info(
                                        f'      Let the agent start Explore...'
                                    )
                                self._last_subtask.set_subtask("Explore", None, None)
                            else:
                                if self.verbose:
                                    get_logger().info(
                                        f'      Let the agent do proper interaction...'
                                    )
                                self._last_subtask.next_subtask(held_object=held_object)
                        else:
                            if self.verbose:
                                get_logger().info(
                                    f'      Keep going to target: {self._last_to_interact_object_pose["name"]}'
                                )
                            obj_type = (
                                self._last_to_interact_object_pose["type"] 
                                if "type" in self._last_to_interact_object_pose else 
                                self._last_to_interact_object_pose["objectType"]
                            )
                            self._last_subtask.set_subtask(
                                subtask_type="Goto",
                                obj_type=obj_type,
                                target_map=self._last_to_interact_object_pose["target_map"]
                            )
                else:
                    if self.verbose:
                        get_logger().info(
                            f'      Action failed...'
                        )
                    if was_interact_subtask:
                        assert action_str == last_expert_subtask.get_expert_action_str()
                        assert self._last_to_interact_object_pose is not None
                        if self.verbose:
                            get_logger().info(
                                f'      current agent position invalidated for object {self._last_to_interact_object_pose["name"]}!'
                            )
                        self._invalidate_interactable_loc_for_pose(
                            env=task.unshuffle_env,
                            location=task.unshuffle_env.get_agent_location(),
                            obj_pose=self._last_to_interact_object_pose,
                        )
                        # if self.verbose:
                        #     get_logger().info(
                        #         f'      Then let the agent to go to target object {self._last_to_interact_object_pose["name"]} in another position'
                        #     )
                        # self._last_subtask.set_subtask(
                        #     subtask_type="Goto",
                        #     obj_type=self._last_to_interact_object_pose["type"],
                        #     target_map=self._last_to_interact_object_pose["target_map"],
                        # )
                        if (set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)):
                            if self.verbose:
                                for obj_name in (
                                    set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                                ):
                                    get_logger().info(
                                        f'      priority[{obj_name}]: {self.object_name_to_priority[obj_name]}'
                                    )
                                get_logger().info(
                                    self._objects_to_rearrange.keys()
                                )
                            target_obj_name = None
                            failed_places_and_min_dist = (float("inf"), float("inf"))
                            for obj_name in (
                                set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                            ):
                                if self.object_name_to_priority[obj_name] < self.max_priority_per_object:
                                    priority = self.object_name_to_priority[obj_name]
                                    position = next(
                                        (
                                            obj["position"]
                                            for obj in task.env.last_event.metadata["objects"]
                                            if obj["name"] == obj_name
                                        ),
                                        None
                                    )
                                    priority_and_dist_to_object = (
                                        priority,
                                        IThorEnvironment.position_dist(
                                            task.env.get_agent_location(), position, ignore_y=True, l1_dist=True
                                        )
                                    )
                                    if priority_and_dist_to_object < failed_places_and_min_dist:
                                        failed_places_and_min_dist = priority_and_dist_to_object
                                        target_obj_name = obj_name

                            self._last_to_interact_object_pose = None
                            if target_obj_name is not None:
                                # if target_obj_name in self.seen_obj_names_unshuffle:
                                #     target_obj = next(
                                #         (
                                #             cp for cp in cur_poses
                                #             if cp["name"] == target_obj_name
                                #         ),
                                #         None
                                #     )
                                #     assert target_obj is not None
                                # else:
                                #     target_obj = self.seen_obj_names_walkthrough[target_obj_name]
                                #     # objectId and positions should be replaced.
                                #     for k in ["position", "rotation"]:
                                #         target_obj[k] = next(
                                #             (
                                #                 obj[k] 
                                #                 for obj in task.env.last_event.metadata["objects"]
                                #                 if obj["name"] == target_obj["name"]
                                #             ),
                                #             None
                                #         )
                                                
                                #     target_obj["objectId"] = next(
                                #         (
                                #             obj["objectId"]
                                #             for obj in task.env.last_event.metadata["objects"]
                                #             if obj["name"] == target_obj["name"]
                                #         ),
                                #         None
                                #     )
                                #     if target_obj["objectId"] is None:
                                #         raise Exception

                                target_obj = next(
                                    (
                                        cp for cp in cur_poses
                                        if cp["name"] == target_obj_name
                                    ),
                                    None
                                )
                                assert target_obj is not None
                            
                                self._last_to_interact_object_pose = target_obj
                                self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                                obj_type = (
                                    self._last_to_interact_object_pose["type"] 
                                    if "type" in self._last_to_interact_object_pose 
                                    else self._last_to_interact_object_pose["objectType"]
                                )
                                self._last_subtask.set_subtask(
                                    subtask_type="Goto",
                                    obj_type=obj_type,
                                    target_map=self._last_to_interact_object_pose["target_map"],
                                )
                            else:
                                self._last_subtask.set_subtask("Explore", None, None)
                        else:
                            self._last_subtask.set_subtask("Explore", None, None)
                    else:
                        if was_nav_action:
                            if self.verbose:
                                get_logger().info(
                                    f'      update the navigation graph with failed action...'
                                )
                            shortest_path_navigator.update_graph_with_failed_action(
                                stringcase.pascalcase(action_str)
                            )
                        elif (
                            ("crouch" in action_str or "stand" in action_str)
                            # and held_object is not None
                        ):
                            if self.verbose:
                                get_logger().info(
                                    f'      invalidate the agent current pose to interact target object...'
                                )
                            agent_loc = task.unshuffle_env.get_agent_location()
                            agent_loc["standing"] = not agent_loc["standing"]
                            # self._invalidate_interactable_loc_for_pose(
                            #     env=task.unshuffle_env,
                            #     location=agent_loc,
                            #     obj_pose=task.unshuffle_env.obj_name_to_walkthrough_start_pose[
                            #         self._last_held_object_name
                            #     ]
                            # )
                            self._invalidate_interactable_loc_for_pose(
                                env=task.unshuffle_env,
                                location=agent_loc,
                                obj_pose=self._last_to_interact_object_pose,
                            )
            else:
                # when agent did not take the expert action
                if "pickup" in action_str and action_success:
                    # wrongly picked up something
                    # the agent should put object down original place.
                    if self.verbose:
                        get_logger().info(
                            f'      Wrongly picked up something...'
                        )
                    assert held_object is not None
                    self._last_to_interact_object_pose = copy.deepcopy(held_object)
                    self._last_to_interact_object_pose["target_map"] = "Walkthrough"
                    for k in ["position", "rotation"]:
                        self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                            held_object["name"]
                        ][k]
                    if self.verbose:
                        get_logger().info(
                            f'      Let the agent go to original place of holding object...'
                        )
                    obj_type = (
                        self._last_to_interact_object_pose["type"] 
                        if "type" in self._last_to_interact_object_pose 
                        else self._last_to_interact_object_pose["objectType"]
                    )
                    self._last_subtask.set_subtask(
                        subtask_type="Goto",
                        obj_type=obj_type,
                        target_map=self._last_to_interact_object_pose["target_map"],
                    )
                elif "drop_held_object_with_snap" in action_str and action_success:
                    # the agent just dropped the holding object unattendedly.
                    # it should go to the object to pickup.
                    if self.verbose:
                        get_logger().info(
                            f'      Wrongly dropped the holding object...'
                        )
                    assert held_object is None
                    # if self.verbose:
                    #     get_logger().info(
                    #         f'      Let the agent to go to the object in order to pick up again...'
                    #     )
                    # # assign position/rotation info about the dropped object
                    # self._last_to_interact_object_pose = next(
                    #     (
                    #         cp
                    #         for cp in cur_poses
                    #         if cp["name"] == self._last_to_interact_object_pose["name"]
                    #     ),
                    #     None
                    # )
                    # assert self._last_to_interact_object_pose is not None
                    # self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    # self._last_subtask.set_subtask(
                    #     subtask_type="Goto",
                    #     obj_type=self._last_to_interact_object_pose["type"],
                    #     target_map="Unshuffle"
                    # )
                    if self.verbose:
                        get_logger().info(
                            f'      explore...'
                        )
                    self._last_to_interact_object_pose = None
                    self._last_subtask.set_subtask("Explore", None, None)

        else:
            # the first update, assign Explore subtask.
            self._last_subtask.set_subtask(subtask_type="Explore", obj_type=None, target_map=None)
    
        assert self._last_subtask.subtask_type is not None

        if self.verbose:
            get_logger().info(
                f'      Planned Next Expert Subtask: {self._last_subtask}[{self._last_subtask.get_subtask_idx()}]'
            )

        self._record_expert_subtask_and_generate_and_record_action(task)

    def _record_expert_subtask_and_generate_and_record_action(self, task: UnshuffleTask):
        if task.num_steps_taken() == len(self.expert_subtask_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {task.num_steps_taken()}"
            )
            return

        if task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {task.num_steps_taken()}"
            )
            return
        
        assert task.num_steps_taken() == len(
            self.expert_subtask_list
        ), f"{task.num_steps_taken()} != {len(self.expert_subtask_list)}"

        assert task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{task.num_steps_taken()} != {len(self.expert_action_list)}"

        if self.verbose:
            get_logger().info(
                f'  ** IN _record_expert_subtask_and_generate_and_record_action() method'
            )
            get_logger().info(
                f'    Generate expert action dict based on the updated current subtask...'
            )

        expert_action_dict = self._generate_expert_action_dict(task)
        
        if self.verbose:
            get_logger().info(
                f'    Generated expert action: {expert_action_dict}'
            )
        subtask_id = self._last_subtask.get_subtask_idx()   # include (None, None, None)
        if subtask_id is not None:
            assert 0 <= subtask_id < NUM_SUBTASKS
        self.expert_subtask_list.append(subtask_id)

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in task.action_names():
            obj_type = stringcase.snakecase(
                expert_action_dict["objectId"].split("|")[0]
            )
            action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

    def _generate_expert_action_dict(self, task: UnshuffleTask):

        if task.env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {task.env.mode}"
            )
        if self.verbose:
            get_logger().info(
                f'  ** IN _generate_expert_action_dict() method'
            )
            get_logger().info(
                f'    *** self._last_subtask: {self._last_subtask}'
            )

        held_object = task.env.held_object
        if (
            held_object is not None
            and (
                (
                    self._last_to_interact_object_pose
                    and self._last_to_interact_object_pose["name"] != held_object["name"]
                )
                or self._last_to_interact_object_pose is None
            )
        ):
            if self.verbose:
                get_logger().info(
                    f'   The agent picked up different object...'
                )
            self._last_to_interact_object_pose = copy.deepcopy(held_object)
            self._last_to_interact_object_pose["target_map"] = "Walkthrough"
            for k in ["position", "rotation"]:
                self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                    held_object["name"]
                ][k]
            if self.verbose:
                get_logger().info(
                    f'      Re-plan the subtask to make agent go and put down the holding object...'
                )
            
            if self.object_name_to_priority[self._last_to_interact_object_pose["name"]] <= self.max_priority_per_object:
                if self.verbose:
                    get_logger().info(
                        f"    It should go to the original place of the holding object..."
                    )
                self._last_subtask.set_subtask("Goto", self._last_to_interact_object_pose["objectType"], self._last_to_interact_object_pose["target_map"])
            else:
                if self.verbose:
                    get_logger().info(
                        f"    It FAILED to go to the original place of the holding object..."
                    )
                    get_logger().info(
                        f"    Just drop the holding object and keep rearranging other objects..."
                    )
                # self._last_to_interact_object_pose = None
                self._last_subtask.set_subtask("PutObject", self._last_to_interact_object_pose["objectType"], None)
                return self._generate_expert_action_dict(task=task)
            if self.verbose:
                get_logger().info(
                    f'    Planned Next Expert Subtask: {self._last_subtask}[{self._last_subtask.get_subtask_idx()}]'
                )

        agent_loc = task.env.get_agent_location()
        
        if "Stop" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'     returning action {"Done"}'
                )
            return dict(action="Done")
        elif "Explore" in self._last_subtask.subtask_type:
            _, goal_poses, cur_poses = task.env.poses
            assert len(goal_poses) == len(cur_poses)
            obj_pose_to_go_to = None
            goal_obj_pos = None

            if self._last_to_interact_object_pose is None:
                if self.verbose:
                    get_logger().info(
                    f'  self._last_to_interact_object_pose is None.'
                )
                failed_places_and_min_dist = (float("inf"), float("inf"))
                for gp, cp in zip(goal_poses, cur_poses):
                    if self.verbose:
                        if (
                            (gp["broken"] == cp["broken"] == False)
                            and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                        ):
                            get_logger().info(
                                f"        expert knows that {cp['name']} should be rearranged..."
                            )
                            get_logger().info(
                                f"          self.object_name_to_priority[{cp['name']}] : {self.object_name_to_priority[cp['name']]}"
                            )
                    if (
                        (gp["broken"] == cp["broken"] == False)
                        and self.object_name_to_priority[gp["name"]]
                        <= self.max_priority_per_object
                        and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                    ):
                        priority = self.object_name_to_priority[gp["name"]]
                        priority_and_dist_to_object = (
                            priority,
                            IThorEnvironment.position_dist(
                                agent_loc, gp["position"], ignore_y=True, l1_dist=True
                            ),
                        )

                        if priority_and_dist_to_object < failed_places_and_min_dist:
                            failed_places_and_min_dist = priority_and_dist_to_object
                            obj_pose_to_go_to = cp
                            goal_obj_pos = gp

                self._last_to_interact_object_pose = obj_pose_to_go_to
                if obj_pose_to_go_to is not None:
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    if self.verbose:
                        get_logger().info(
                            f"    EXPERT let the agent move to the object {self._last_to_interact_object_pose['name']}"
                            f"    But the subtask is still {self._last_subtask}"
                        )
                else:
                    # should stop episode
                    if self.verbose:
                        get_logger().info(
                            f"    Should end this episode since there is no objects to handle..."
                        )
                    self._last_subtask.set_subtask("Stop", None, None)
                    return self._generate_expert_action_dict(task=task)
        
        # elif "Goto" in self._last_subtask.subtask_type:
        #     pass
        
        elif "PickupObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'     returning action {"Pickup"} | objectName {self._last_to_interact_object_pose["name"]} | objectId {self._last_to_interact_object_pose["objectId"]}'
                )
            return dict(
                action="Pickup",
                objectId=self._last_to_interact_object_pose["objectId"],
            )
        elif "OpenObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'     returning action {"OpenObject"} | objectName {self._last_to_interact_object_pose["name"]} | objectId {self._last_to_interact_object_pose["objectId"]}'
                )
            return dict(
                action="OpenByType",
                objectId=self._last_to_interact_object_pose["objectId"],
                openness=task.env.obj_name_to_walkthrough_start_pose[
                    self._last_to_interact_object_pose["name"]
                ]["openness"],
            )
        elif "PutObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'     returning action {"DropHeldObjectWithSnap"} | objectName {self._last_to_interact_object_pose["name"]} | objectId {self._last_to_interact_object_pose["objectId"]}'
                )
            return dict(action="DropHeldObjectWithSnap")

        assert (
            self._last_to_interact_object_pose is not None
            and 'target_map' in self._last_to_interact_object_pose
        ), f"self._last_subtask: {self._last_subtask} | self._last_to_interact_object_pose: {self._last_to_interact_object_pose}"
        
        if self.verbose:
            get_logger().info(
                f'    Generating expert navigation action to the object {self._last_to_interact_object_pose["name"]} in {self._last_to_interact_object_pose["target_map"]}, position: {self._last_to_interact_object_pose["position"]}, rotation: {self._last_to_interact_object_pose["rotation"]}'
            )
            if self._last_to_interact_object_pose["target_map"] == "Walkthrough":
                get_logger().info(
                    f'      The position and rotation of the target object place in the Walkthrough Environment: {task.env.obj_name_to_walkthrough_start_pose[self._last_to_interact_object_pose["name"]]}'
                )
            else:
                _, _, cur_poses = task.env.poses
                target = next(
                    (
                        cp for cp in cur_poses
                        if cp["name"] == self._last_to_interact_object_pose["name"]
                    ),
                    None
                )
                assert target is not None
                get_logger().info(
                    f'      The position and rotation of the target object place in the Unshuffle Environment: {target}'
                )
        expert_nav_action = self._expert_nav_action_to_obj(task=task, obj=self._last_to_interact_object_pose)
        if expert_nav_action is None:
            if self.verbose:
                get_logger().info(
                    f'        Generated navigation action is None...'
                )
            interactable_positions = task.env._interactable_positions_cache.get(
                scene_name=task.env.scene,
                obj=self._last_to_interact_object_pose,
                controller=task.env.controller,
            )
            if len(interactable_positions) != 0:
                # Could not find a path to the object, increment the place count of the object and
                # try generating a new action.
                if self.verbose:
                    get_logger().debug(
                        f"Could not find a path to {self._last_to_interact_object_pose}"
                        f" in scene {task.unshuffle_env.scene}"
                        f" when at position {task.unshuffle_env.get_agent_location()}."
                    )
            else:
                if self.verbose:
                    get_logger().debug(
                        f"Object {self._last_to_interact_object_pose} in scene {task.unshuffle_env.scene}"
                        f" has no interactable positions."
                    )
            if self.verbose:
                get_logger().info(
                    f'        interactable_positions: {interactable_positions}'
                )
                get_logger().info(
                    f'        Let the agent Explore...'
                )
            self.object_name_to_priority[self._last_to_interact_object_pose["name"]] += 1
            self._last_subtask.set_subtask("Explore", None, None)
            self._last_to_interact_object_pose = None
            return self._generate_expert_action_dict(task=task)
        
        elif expert_nav_action == "Pass":
            if self.verbose:
                get_logger().info(
                    f'        Generated navigation action is "Pass"'
                )
            # with include_object_data(task.env.controller):
            #     visible_objects = {
            #         o["name"]
            #         for o in task.env.last_event.metadata["objects"]
            #         if o["visible"]
            #     }
            visible_objects = self.get_object_names_from_current_view(
                task=task,
                env_type=self._last_to_interact_object_pose["target_map"],
                rearrange_targets=self._objects_to_rearrange,
            )

            if self._last_to_interact_object_pose["name"] not in visible_objects:
                if self.verbose:
                    get_logger().info(
                        f'        But the agent cannot find the object in the current view!!!'
                    )
                if self._invalidate_interactable_loc_for_pose(
                    env=task.env,
                    location=agent_loc, 
                    obj_pose=self._last_to_interact_object_pose
                ):
                    # self._last_subtask = None
                    # self._last_to_interact_object_pose = None
                    # return self._generate_expert_subtask_dict(task=task)
                    if self.verbose:
                        get_logger().info(
                            f'        Invalidate the current agent position and keep navigating to the object {self._last_to_interact_object_pose["name"]}'
                        )
                    return self._generate_expert_action_dict(task=task)

                raise RuntimeError("This should not be possible.")

            else:
                if self.verbose:
                    get_logger().info(
                        f'        The target object is visible'
                    )
                    get_logger().info(
                        f'        Do next subtask...'
                    )
                # The agent has arrived to the target object.
                if self._last_subtask.subtask_type == "Explore":
                    obj_type = (
                        self._last_to_interact_object_pose["type"] 
                        if "type" in self._last_to_interact_object_pose 
                        else self._last_to_interact_object_pose["objectType"]
                    )
                    self._last_subtask.next_subtask(
                        obj_type=obj_type,
                        target_map="Unshuffle"
                    )

                elif self._last_subtask.subtask_type == "Goto":
                    self._last_subtask.next_subtask(held_object=held_object)
                
                else:
                    raise NotImplementedError(f"what?")

                if self.verbose:
                    get_logger().info(
                        f'      Planned Next Expert Subtask: {self._last_subtask}[{self._last_subtask.get_subtask_idx()}]'
                    )
                    
                return self._generate_expert_action_dict(task=task)
        
        else:
            if self.verbose:
                get_logger().info(
                    f'     returning action {expert_nav_action} | objectName {self._last_to_interact_object_pose["name"]} | objectId {self._last_to_interact_object_pose["objectId"]}'
                )
            return dict(action=expert_nav_action)

    def _expert_nav_action_to_obj(self, task: UnshuffleTask, obj: Dict[str, Any]) -> Optional[str]:
        """Get the shortest path navigational action towards the object obj.

        The navigational action takes us to a position from which the
        object is interactable.
        """
        env: RearrangeTHOREnvironment = task.env
        agent_loc = env.get_agent_location()
        shortest_path_navigator = env.shortest_path_navigator

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj, controller=env.controller,
        )

        target_keys = [
            shortest_path_navigator.get_key(loc) for loc in interactable_positions
        ]
        if len(target_keys) == 0:
            return None

        source_state_key = shortest_path_navigator.get_key(env.get_agent_location())

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key, goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                # Could not find the expert actions
                return None

        if action != "Pass":
            return action
        else:
            agent_x = agent_loc["x"]
            agent_z = agent_loc["z"]
            for gdl in interactable_positions:
                d = round(abs(agent_x - gdl["x"]) + abs(agent_z - gdl["z"]), 2)
                if d <= 1e-2:
                    if _are_agent_locations_equal(agent_loc, gdl, ignore_standing=True):
                        if agent_loc["standing"] != gdl["standing"]:
                            return "Crouch" if agent_loc["standing"] else "Stand"
                        else:
                            # We are already at an interactable position
                            return "Pass"
            return None