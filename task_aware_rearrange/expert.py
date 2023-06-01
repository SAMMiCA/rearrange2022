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
from rearrange.utils import get_pose_info
from rearrange.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS, STEP_SIZE
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
        if self.verbose:
            get_logger().info(
                f'==== [STEP {task.num_steps_taken()}] ==== '
            )
            get_logger().info(
                f'  ** IN update() method'
            )
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

        _, goal_poses, cur_poses = task.env.poses
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
                    if self.verbose:
                        get_logger().info(
                            f'      {cp["name"]} is seen in unshuffle env.... '
                        )
                if gp["name"] in self.get_object_names_from_current_view(
                    task=task, env_type="Walkthrough", rearrange_targets=self._objects_to_rearrange
                ):
                    self.seen_obj_names_walkthrough[gp["name"]] = gp
                    if self.verbose:
                        get_logger().info(
                            f'      {gp["name"]} is seen in walkthrough env.... '
                        )
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

        if action_taken is not None:
            assert action_success is not None

            action_names = task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]
            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            assert self._last_subtask is not None
            last_expert_subtask = copy.deepcopy(self._last_subtask)

            if agent_took_expert_action:
                # assert last_expert_subtask.get_expert_action_str() == action_str
                if self.verbose:
                    get_logger().info(
                        f'      Agent took expert action!'
                    )

                if (
                    "pickup_" in action_str
                    and action_success
                ):
                    self._name_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose["name"]
                
                if "drop_held_object_with_snap" in action_str:
                    if self._name_of_object_we_wanted_to_pickup is not None:
                        self.object_name_to_priority[
                            self._name_of_object_we_wanted_to_pickup
                        ] += 1
                    else:
                        self.object_name_to_priority[self._last_held_object_name] += 1
                
                if "open_by_type" in action_str:
                    self.object_name_to_priority[
                        self._last_to_interact_object_pose["name"]
                    ] += 1
                
                if not action_success:
                    if (
                        "pickup_" in action_str
                        or "open_by_type_" in action_str
                    ):
                        assert self._last_to_interact_object_pose is not None
                        self._invalidate_interactable_loc_for_pose(
                            env=task.unshuffle_env,
                            location=task.unshuffle_env.get_agent_location(),
                            obj_pose=self._last_to_interact_object_pose,
                        )
                    elif (
                        "crouch" in action_str
                        or "stand" in action_str
                    ):
                        held_object_name = task.unshuffle_env.held_object["name"]
                        agent_loc = task.unshuffle_env.get_agent_location()
                        agent_loc["standing"] = not agent_loc["standing"]
                        self._invalidate_interactable_loc_for_pose(
                            env=task.unshuffle_env,
                            location=agent_loc,
                            obj_pose=task.unshuffle_env.obj_name_to_walkthrough_start_pose[
                                held_object_name
                            ],
                        )
                
            if not action_success:
                if was_nav_action:
                    shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
            else:
                if not was_nav_action:
                    self._last_to_interact_object_pose = None
        
        held_object = task.unshuffle_env.held_object
        if held_object is not None:
            self._last_held_object_name = held_object["name"]
    
        # self._record_expert_subtask_and_generate_and_record_action(task)
        self._generate_and_record_expert(task, action_taken, action_success)

    def _generate_and_record_expert(
        self,
        task: UnshuffleTask,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        if task.num_steps_taken() == len(self.expert_subtask_list) + 1:
            get_logger().warning(
                f"Already generated the expert subtask at step {task.num_steps_taken()}"
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

        expert_dict = self._generate_expert(task)

        subtask_id = self._last_subtask.get_subtask_idx()   # include (None, None, None)
        if subtask_id is not None:
            assert 0 <= subtask_id < NUM_SUBTASKS
        self.expert_subtask_list.append(subtask_id)

        action_str = stringcase.snakecase(expert_dict["action"])
        if action_str not in task.action_names():
            obj_type = stringcase.snakecase(
                expert_dict["objectId"].split("|")[0]
            )
            action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

    def _generate_expert(
        self,
        task: UnshuffleTask,
        replan_subtask: bool = False,
    ):
        # Generate a dictionary describing the next expert subtask and expert action.
        env = task.unshuffle_env
        if env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {task.env.mode}"
            )

        if self.verbose:
            get_logger().info(
                f'  ** IN _generate_expert() method'
            )
            get_logger().info(
                f'    replan_subtask: {replan_subtask}'
            )
        # Generate expert subtask...
        self._update_expert_subtask(task, replan_subtask)
        if self.verbose:
            get_logger().info(
                f'    planned_subtask: {self._last_subtask}'
            )

        # Generate expert action...
        expert_action_dict = self._generate_expert_action_dict(task)
        if self.verbose:
            get_logger().info(
                f'    generated expert action dict: {expert_action_dict}'
            )

        return expert_action_dict

    def _update_expert_subtask(
        self,
        task: UnshuffleTask,
        replan_subtask: bool,
    ):
        env = task.unshuffle_env
        last_expert_subtask = copy.deepcopy(self._last_subtask)
        was_interact_subtask = last_expert_subtask.is_interact_subtask()

        held_object = env.held_object
        agent_loc = env.get_agent_location()

        if self.verbose:
            get_logger().info(
                f'    ** IN _update_expert_subtask() method'
            )

        if not replan_subtask:
            # self._generate_expert() is not repeated
            if held_object is not None:
                # Should go to the goal pose of the held object
                if self.verbose:
                    get_logger().info(
                        f'      agent is holding object: {held_object["name"]}'
                    )
                self._last_to_interact_object_pose = copy.deepcopy(held_object)
                self._last_to_interact_object_pose["target_map"] = "Walkthrough"
                for k in ["position", "rotation"]:
                    self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                        held_object["name"]
                    ][k]
                self._last_subtask.set_subtask(
                    subtask_type="Goto",
                    obj_type=self._last_to_interact_object_pose["objectType"],
                    target_map=self._last_to_interact_object_pose["target_map"],
                )

            else:
                _, goal_poses, cur_poses = env.poses
                assert len(goal_poses) == len(cur_poses)

                failed_places_and_min_dist = (float("inf"), float("inf"))
                obj_pose_to_go_to = None
                goal_obj_pos = None
                # TODO: Explore task is ignored...
                for gp, cp in zip(goal_poses, cur_poses):
                    if (
                        (gp["broken"] == cp["broken"] == False)
                        and self.object_name_to_priority[gp["name"]]
                        <= self.max_priority_per_object
                        and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                        and cp["type"] in (OPENABLE_OBJECTS + PICKUPABLE_OBJECTS)
                    ):
                        priority = self.object_name_to_priority[gp["name"]]
                        priority_and_dist_to_object = (
                            priority,
                            IThorEnvironment.position_dist(
                                agent_loc, gp["position"], ignore_y=True, l1_dist=True
                            ),
                        )
                        if (
                            self._last_to_interact_object_pose is not None
                            and self._last_to_interact_object_pose["name"] == gp["name"]
                        ):
                            # Set distance to -1 for the currently targeted object
                            if self.verbose:
                                get_logger().info(
                                    f'      Set distance to -1 for the currently targeted object'
                                )
                            priority_and_dist_to_object = (
                                priority_and_dist_to_object[0],
                                -1,
                            )
                        elif (
                            gp["name"] in set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                        ):
                            if self.verbose:
                                get_logger().info(
                                    f'      set distance to -0.5 for the seen object {gp["name"]}'
                                )
                            priority_and_dist_to_object = (
                                priority_and_dist_to_object[0],
                                -0.5,
                            )

                        if priority_and_dist_to_object < failed_places_and_min_dist:
                            failed_places_and_min_dist = priority_and_dist_to_object
                            obj_pose_to_go_to = cp
                            goal_obj_pos = gp
                
                if self.verbose:
                    get_logger().info(
                        f'      obj_pose_to_go_to: {obj_pose_to_go_to}'
                    )
                if obj_pose_to_go_to is None:
                    if self.verbose:
                        get_logger().info(
                            f'      no items to rearrange...'
                        )
                    self._last_subtask.set_subtask("Stop", None, None)

                elif not (
                    set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                ):
                    if self.verbose:
                        get_logger().info(
                            f'      no seen items...'
                        )
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    self._last_subtask.set_subtask("Explore", None, None)

                else:
                    if self.verbose:
                        get_logger().info(
                            f'      agent is moving to target: {obj_pose_to_go_to["name"]}'
                        )
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    self._last_subtask.set_subtask(
                        subtask_type="Goto",
                        obj_type=self._last_to_interact_object_pose["type"],
                        target_map=self._last_to_interact_object_pose["target_map"],
                    )
        else:
            """
            self._generate_expert() is called by self._generate_expert_action_dict()
            since the Goto subtask is done.
            """
            if self.verbose:
                get_logger().info(
                    f'      Replanning Subtask...'
                )
            if held_object is not None:
                self._last_subtask.set_subtask(
                    subtask_type="PutObject",
                    obj_type=self._last_to_interact_object_pose["objectType"],
                    target_map=None,
                )

            else:
                _, goal_poses, cur_poses = task.env.poses
                assert len(goal_poses) == len(cur_poses)
                obj_pose_to_go_to = None
                goal_obj_pos = None
                for gp, cp in zip(goal_poses, cur_poses):
                    if (
                        self._last_to_interact_object_pose is not None
                        and self._last_to_interact_object_pose["name"] == gp["name"]
                    ):
                        obj_pose_to_go_to = cp
                        goal_obj_pos = gp
                
                assert obj_pose_to_go_to is not None
                if (
                    obj_pose_to_go_to["openness"] is not None
                    and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
                    and obj_pose_to_go_to["type"] in OPENABLE_OBJECTS
                ):
                    self._last_subtask.set_subtask(
                        subtask_type="OpenObject",
                        obj_type=obj_pose_to_go_to["type"],
                        target_map=None,
                    )
                elif obj_pose_to_go_to["pickupable"]:
                    self._last_subtask.set_subtask(
                        subtask_type="PickupObject",
                        obj_type=obj_pose_to_go_to["type"],
                        target_map=None,
                    )
                else:
                    self.object_name_to_priority[goal_obj_pos["name"]] = (
                        self.max_priority_per_object + 1
                    )
                    # TODO: Explore?
                    self._generate_expert(task=task)

    def _generate_expert_action_dict(self, task: UnshuffleTask):
        if self.verbose:
            get_logger().info(
                f'    ** IN _generate_expert_action_dict() method'
            )
        if task.env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {task.env.mode}"
            )

        if "Stop" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      All subtasks done'
                )
            return dict(action="Done")

        elif "Explore" in self._last_subtask.subtask_type:
            # TODO: to be updated
            if self.verbose:
                get_logger().info(
                    f'      Explore to find difference'
                )
            assert self._last_to_interact_object_pose is not None
        
        elif "Goto" in self._last_subtask.subtask_type:
            assert (
                self._last_to_interact_object_pose is not None
                and 'target_map' in self._last_to_interact_object_pose
            ), f"self._last_subtask: {self._last_subtask} | self._last_to_interact_object_pose: {self._last_to_interact_object_pose}"
            if self.verbose:
                get_logger().info(
                    f'      Goto {self._last_to_interact_object_pose["name"]}...'
                )

        elif "PickupObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      PickupObject Subtask...'
                )
            return dict(
                action="Pickup",
                objectId=self._last_to_interact_object_pose["objectId"],
            )
        elif "OpenObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      OpenObject Subtask...'
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
                    f'      PutObject Subtask...'
                )
            return dict(action="DropHeldObjectWithSnap")
        
        else:
            raise RuntimeError("??????????????????????")

        expert_nav_action = self._expert_nav_action_to_obj(
            task=task,
            obj=self._last_to_interact_object_pose
        )
        if self.verbose:
            get_logger().info(
                f'      Generated expert navigation action is {expert_nav_action}'
            )
        if expert_nav_action is None:
            interactable_positions = task.env._interactable_positions_cache.get(
                scene_name=task.env.scene,
                obj=self._last_to_interact_object_pose,
                controller=task.env.controller,
            )
            self.object_name_to_priority[self._last_to_interact_object_pose["name"]] += 1
            # self._last_subtask.set_subtask("Explore", None, None)
            # self._last_to_interact_object_pose = None
            if self.verbose:
                get_logger().info(
                    f'      Re-generate expert subtask and action...'
                )
            return self._generate_expert(task=task, replan_subtask=True)
        
        elif expert_nav_action == "Pass":
            with include_object_data(task.env.controller):
                visible_objects = {
                    o["name"]
                    for o in task.env.last_event.metadata["objects"]
                    if o["visible"]
                }
            
            if self._last_to_interact_object_pose["name"] not in visible_objects:
                if self._invalidate_interactable_loc_for_pose(
                    env=task.unshuffle_env,
                    location=task.env.get_agent_location(),
                    obj_pose=self._last_to_interact_object_pose
                ):
                    return self._generate_expert(task=task, replan_subtask=True)
                
                raise RuntimeError("This should not be possible.")

            if self.verbose:
                get_logger().info(
                    f'      Goto {self._last_to_interact_object_pose["name"]} is done.'
                )
                get_logger().info(
                    f'      Re-plan expert subtask to step forward'
                )
            return self._generate_expert(task=task, replan_subtask=True)

        else:
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
        
        
class SubtaskAndActionExpertSensor(AbstractExpertActionSensor):
    
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
        use_expert_agent_for_walkthrough: bool = True,
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
        self.use_expert_agent_for_walkthrough = use_expert_agent_for_walkthrough
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

    def reset(self, task: Union[UnshuffleTask, WalkthroughTask]) -> None:
        self.expert_action_list = []
        self.expert_subtask_list = []

        self._last_held_object_name = None
        self._last_to_interact_object_pose = None
        self._name_of_object_we_wanted_to_pickup = None
        self.object_name_to_priority = defaultdict(self.default_priority)
        
        self._objects_to_rearrange = None
        self.seen_obj_names_unshuffle = dict()
        if isinstance(task, UnshuffleTask):
            _, _, cps = task.env.poses
            self.seen_obj_names_walkthrough = {
                o["name"]: get_pose_info(o)
                for o in cps
                if o["name"] in task.object_names_seen_in_walkthrough
            } if task.object_names_seen_in_walkthrough is not None else dict()

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
    def pickupable_objects(
        task: Union[UnshuffleTask, WalkthroughTask],
        which_task_env: Optional[str] = None,
        visible_only: bool = True,
        objects_only: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    ):
        which_task_env = which_task_env.capitalize() if isinstance(which_task_env, str) else which_task_env
        env: RearrangeTHOREnvironment = (
            task.walkthrough_env 
            if isinstance(task, WalkthroughTask) 
            else (
                task.walkthrough_env
                if which_task_env == "Walkthrough"
                else task.unshuffle_env
            )
        )
        with include_object_data(env.controller):
            return [
                o
                for o in env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only) 
                    and (o["name"] in objects_only if objects_only is not None else True)
                    and o["pickupable"]
                )
            ]
            
    @staticmethod
    def openable_not_pickupable_objects(
        task: Union[UnshuffleTask, WalkthroughTask],
        which_task_env: Optional[str] = None,
        visible_only: bool = True,
        objects_only: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    ):
        which_task_env = which_task_env.capitalize() if isinstance(which_task_env, str) else which_task_env
        env: RearrangeTHOREnvironment = (
            task.walkthrough_env 
            if isinstance(task, WalkthroughTask) 
            else (
                task.walkthrough_env
                if which_task_env == "Walkthrough"
                else task.unshuffle_env
            )
        )
        with include_object_data(env.controller):
            return [
                o
                for o in env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only) 
                    and (o["name"] in objects_only if objects_only is not None else True)
                    and (o["openable"] and not o["pickupable"])
                )
            ]
            
    @staticmethod
    def pickupable_or_openable_objects(
        task: Union[UnshuffleTask, WalkthroughTask],
        which_task_env: Optional[str] = None,
        visible_only: bool = True,
        objects_only: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    ):
        which_task_env = which_task_env.capitalize() if isinstance(which_task_env, str) else which_task_env
        env: RearrangeTHOREnvironment = (
            task.walkthrough_env 
            if isinstance(task, WalkthroughTask) 
            else (
                task.walkthrough_env
                if which_task_env == "Walkthrough"
                else task.unshuffle_env
            )
        )
        with include_object_data(env.controller):
            return [
                o
                for o in env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only) 
                    and (o["name"] in objects_only if objects_only is not None else True)
                    and (o["pickupable"] or (o["openable"] and not o["pickupable"]))
                )
            ]
    
    @staticmethod
    def _invalidate_interactable_loc_for_pose(
        env: RearrangeTHOREnvironment,
        location: Dict[str, Any], 
        obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        
        assert hasattr(env, "shortest_path_navigator")
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
        task: Union[UnshuffleTask, WalkthroughTask], 
        *args: Any, 
        **kwargs: Any
    ) -> Union[OrderedDict, Tuple]:

        if (
            task.num_steps_taken() == 0
            and (
                len(task.actions_taken) == 0
                or (
                    len(self.expert_action_list) == 0
                    and len(self.expert_subtask_list) == 0
                )
            )
        ):
            if self.verbose:
                get_logger().info(
                    f"INITIALIZE the EXPERT SUBTASK&ACTION SENSOR!"
                )
            self.reset(task=task)
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
            if (
                isinstance(task, UnshuffleTask) 
                and task.object_names_seen_in_walkthrough is not None
            ):
                c = env.controller
                with include_object_data(c):
                    for o in c.last_event.metadata['objects']:
                        if o['name'] not in task.object_names_seen_in_walkthrough:
                            self.object_name_to_priority[o["name"]] = (
                                self.max_priority_per_object + 1
                            )
            self.update(
                task=task,
                shortest_path_navigator=env.shortest_path_navigator,
                action_taken=None,
                action_success=None,
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
                (
                    self.EXPERT_SUCCESS_LABEL, 
                    (
                        subtask is not None 
                        and (
                            self.use_expert_agent_for_walkthrough 
                            if isinstance(task, WalkthroughTask)
                            else True
                        )
                    )
                ),
            ]
        )
        actions_or_policies[self.ACTION_LABEL] = OrderedDict(
            [
                (self.ACTION_POLICY_LABEL, action),
                (
                    self.EXPERT_SUCCESS_LABEL,
                    (
                        action is not None
                        and (
                            self.use_expert_agent_for_walkthrough 
                            if isinstance(task, WalkthroughTask)
                            else True
                        )
                    )
                ),
            ]
        )

        return self.flatten_output(
            actions_or_policies
            if self.use_groups
            else actions_or_policies[self._NO_GROUPS_LABEL]
        )

    def query_expert(
        self, 
        task: Union[UnshuffleTask, WalkthroughTask], 
        expert_sensor_group_name: Optional[str] = None,
    ) -> Tuple[Any, bool]:
        env = task.env

        last_action = task.actions_taken[-1] if task.actions_taken else None    # string
        last_action_success = task.actions_taken_success[-1] if task.actions_taken_success else None
        last_action_ind = task.action_names().index(last_action) if last_action else None

        if (
            isinstance(task, WalkthroughTask) 
            and not self.use_expert_agent_for_walkthrough
        ):
            return 0, 0
        
        if (
            task.num_steps_taken() == len(self.expert_subtask_list) -1
            and task.num_steps_taken() == len(self.expert_action_list) -1
            and last_action is None
        ):
            if self.verbose:
                get_logger().warning(
                    f"\t\t\tAlready generated the expert subtask at step {task.num_steps_taken()}: {self.expert_subtask(task=task)}\n"
                    f"\t\t\tAlready generated the expert action at step {task.num_steps_taken()}: {self.expert_action(task=task)}\n"
                )
            
            return self.expert_subtask(task=task), self.expert_action(task=task)
        
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
        task: Union[UnshuffleTask, WalkthroughTask],
        shortest_path_navigator: ShortestPathNavigatorTHOR,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        """
        update and generate expert subtask
        """
        if self.verbose:
            get_logger().info(
                f'==== [STEP {task.num_steps_taken()}] ==== '
            )
            get_logger().info(
                f'\t** IN update() method'
            )
            get_logger().info(
                f'\t\tLast Expert Subtask: {self._last_subtask}[{self._last_subtask.get_subtask_idx()}]'
            )
            last_expert_action = None if len(self.expert_action_list) == 0 else self.expert_action_list[-1]
            last_expert_action_str = None if last_expert_action is None else task.action_names()[last_expert_action]
            get_logger().info(
                f'\t\tLast Expert Action: {last_expert_action_str}[{last_expert_action}]'
            )
            get_logger().info(
                f'\t\tLast Action Taken: {task.action_names()[action_taken] if action_taken else None}[{action_taken}] | Success: {action_success}'
            )
        
        if isinstance(task, WalkthroughTask):
            if task.num_steps_taken == 0:
                if self.verbose:
                    get_logger().info(
                        f"\t\tTotal objects in this environment: {task.pickupable_or_openable_objects(visible_only=False)}\n"
                    )
                if self.verbose:
                    get_logger().info(
                        f"\t\treachable positions in this environment: {len(task.reachable_positions)}\n"
                    )
            
            prop_visited_xz = len(task.visited_positions_xz) / len(task.reachable_positions)
            prop_visited_xzr = len({p[:3] for p in task.visited_positions_xzrsh}) / (4 * len(task.reachable_positions))
            n_obj_seen = len(task.seen_pickupable_objects) + len(task.seen_openable_objects)
            prop_obj_seen = n_obj_seen / task.total_pickupable_or_openable_objects
            if self.verbose:
                get_logger().info(
                    f"\t\tSeen objects during exploration ({n_obj_seen} objects / {task.total_pickupable_or_openable_objects} objects ({100 * prop_obj_seen:.2f}%)): \n"
                    f"\t\t\tPickupables: {task.seen_pickupable_objects}\n"
                    f"\t\t\tOpenables: {task.seen_openable_objects}\n"
                )
                get_logger().info(
                    f"\t\tCovered areas: \n"
                    f"\t\t\tXZ ({len(task.visited_positions_xz)} / {len(task.reachable_positions)}({100 * prop_visited_xz:.2f}%))\n"
                    f"\t\t\tXZR ({len(task.visited_positions_xzrsh)} / {4 * len(task.reachable_positions)}({100 * prop_visited_xzr}%))\n"
                )
        
        elif isinstance(task, UnshuffleTask):
            # TODO
            initial_poses, goal_poses, cur_poses = task.env.poses
            if task.num_steps_taken == 0:
                self._objects_to_rearrange == {
                    goal_poses[id]["name"]: goal_poses[id]
                    for id in task.start_energies.nonzero()[0]
                }
                if self.verbose:
                    get_logger().info(
                        f"\t\tTotal objects in this environment: "
                        f"{self.pickupable_or_openable_objects(task, visible_only=False)}\n"
                        f"\t\t\tobjects to be rearranged: {self._objects_to_rearrange.keys()}\n"
                    )
            if task.object_names_seen_in_walkthrough is None:
                # This means that the task is one-phase task(Not running a Walkthrough task)
                # Thus, the seen objects in both walkthrough and unshuffle tasks are considered
                
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
                        # Unshuffle Environment
                        if cp["name"] in self.pickupable_or_openable_objects(
                            task, which_task_env="Unshuffle", visible_only=True, objects_only=self._objects_to_rearrange
                        ):
                            self.seen_obj_names_unshuffle[cp["name"]] = cp
                            if self.verbose:
                                get_logger().info(
                                    f"\t\t\tUpdate seen object in unshuffle env: {cp['name']}."
                                )
                        # Walkthrough Environment
                        if gp["name"] in self.pickupable_or_openable_objects(
                            task, which_task_env="Walkthrough", visible_only=True, objects_only=self._objects_to_rearrange
                        ):
                            self.seen_obj_names_walkthrough[gp["name"]] = gp
                            if self.verbose:
                                get_logger().info(
                                    f"\t\t\tUpdate seen object in walkthrough env: {gp['name']}"
                                )
            else:
                # Two-phase Track
                # Unshuffle Task after shuffling the Walkthrough Task env
                # Only updates the objects seen in unshuffle environment
                for o in self.pickupable_or_openable_objects(
                    task, which_task_env="Unshuffle", visible_only=True, objects_only=self._objects_to_rearrange
                ):
                    op = get_pose_info(o)
                    wop = get_pose_info(task.env.obj_name_to_walkthrough_start_pose[o["name"]])
                    if (
                        not op["broken"]
                        and not RearrangeTHOREnvironment.are_poses_equal(wop, op)
                    ):
                        self.seen_obj_names_unshuffle[o["name"]] = op
                        if self.verbose:
                            get_logger().info(
                                f"\t\t\tUpdate seen object in unshuffle env: {o['name']}"
                            )
                            
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
        else:
            raise TypeError
        
        if action_taken is not None:
            assert action_success is not None

            action_names = task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]
            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            assert self._last_subtask is not None
            last_expert_subtask = copy.deepcopy(self._last_subtask)

            if agent_took_expert_action:
                # assert last_expert_subtask.get_expert_action_str() == action_str
                if self.verbose:
                    get_logger().info(
                        f'      Agent took expert action!'
                    )

                if (
                    "pickup_" in action_str
                    and action_success
                ):
                    self._name_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose["name"]
                
                if "drop_held_object_with_snap" in action_str:
                    if self._name_of_object_we_wanted_to_pickup is not None:
                        self.object_name_to_priority[
                            self._name_of_object_we_wanted_to_pickup
                        ] += 1
                    else:
                        self.object_name_to_priority[self._last_held_object_name] += 1
                
                if "open_by_type" in action_str:
                    self.object_name_to_priority[
                        self._last_to_interact_object_pose["name"]
                    ] += 1
                
                if not action_success:
                    if (
                        "pickup_" in action_str
                        or "open_by_type_" in action_str
                    ):
                        assert self._last_to_interact_object_pose is not None
                        self._invalidate_interactable_loc_for_pose(
                            env=task.env,
                            location=task.env.get_agent_location(),
                            obj_pose=self._last_to_interact_object_pose,
                        )
                    elif (
                        "crouch" in action_str
                        or "stand" in action_str
                    ):
                        held_object_name = task.env.held_object["name"]
                        agent_loc = task.env.get_agent_location()
                        agent_loc["standing"] = not agent_loc["standing"]
                        self._invalidate_interactable_loc_for_pose(
                            env=task.env,
                            location=agent_loc,
                            obj_pose=task.env.obj_name_to_walkthrough_start_pose[
                                held_object_name
                            ],
                        )
                
            if not action_success:
                if was_nav_action:
                    shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
            else:
                if not was_nav_action:
                    self._last_to_interact_object_pose = None
                    
        held_object = task.env.held_object
        if held_object is not None:
            self._last_held_object_name = held_object["name"]
    
        # self._record_expert_subtask_and_generate_and_record_action(task)
        self._generate_and_record_expert(task, action_taken, action_success)
        
    def _generate_and_record_expert(
        self,
        task: UnshuffleTask,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        if self.verbose:
            get_logger().info(
                f"\t\tGenerate and Record Expert!!"
            )
        if task.num_steps_taken() == len(self.expert_subtask_list) - 1:
            get_logger().warning(
                f"Already generated the expert subtask at step {task.num_steps_taken()}"
            )
            return

        if task.num_steps_taken() == len(self.expert_action_list) - 1:
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

        expert_dict = self._generate_expert(task)

        subtask_id = self._last_subtask.get_subtask_idx()   # include (None, None, None)
        if subtask_id is not None:
            assert 0 <= subtask_id < NUM_SUBTASKS
        self.expert_subtask_list.append(subtask_id)
        if self.verbose:
            get_logger().info(
                f"\t\t\tGenerated expert subtask: {subtask_id}"
                f" is stored at self.expert_subtask_list: {self.expert_subtask_list}"
            )
            
        action_str = stringcase.snakecase(expert_dict["action"])
        if action_str not in task.action_names():
            obj_type = stringcase.snakecase(
                expert_dict["objectId"].split("|")[0]
            )
            action_str = f"{action_str}_{obj_type}"

        try:
            action_id = task.action_names().index(action_str)
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            action_id = None
        self.expert_action_list.append(action_id)
        if self.verbose:
            get_logger().info(
                f"\t\t\tGenerated expert action: {action_id}"
                f" is stored at self.expert_action_list: {self.expert_action_list}"
            )

    def _generate_expert(
        self,
        task: Union[UnshuffleTask, WalkthroughTask],
        replan_subtask: bool = False,
    ):
        env: RearrangeTHOREnvironment = task.env
        if env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {task.env.mode}"
            )

        if self.verbose:
            get_logger().info(
                f'  ** IN _generate_expert() method'
            )
            get_logger().info(
                f'    replan_subtask: {replan_subtask}'
            )
        # Generate expert subtask...
        self._update_expert_subtask(task, replan_subtask)
        if self.verbose:
            get_logger().info(
                f'    planned_subtask: {self._last_subtask}'
            )

        # Generate expert action...
        expert_action_dict = self._generate_expert_action_dict(task)
        if self.verbose:
            get_logger().info(
                f'    generated expert action dict: {expert_action_dict}'
            )

        return expert_action_dict
    
    def _update_expert_subtask(
        self,
        task: Union[UnshuffleTask, WalkthroughTask],
        replan_subtask: bool,
    ):
        env: RearrangeTHOREnvironment = task.env
        last_expert_subtask = copy.deepcopy(self._last_subtask)
        was_interact_subtask = last_expert_subtask.is_interact_subtask()

        held_object = env.held_object
        agent_loc = env.get_agent_location()
        
        if self.verbose:
            get_logger().info(
                f'\t\t ** IN _update_expert_subtask() method'
            )
        
        if isinstance(task, WalkthroughTask):
            remaining_objects = [
                obj["name"]
                for obj in task.pickupable_or_openable_objects(visible_only=False)
                if (
                    obj["name"] not in (task.seen_pickupable_objects | task.seen_openable_objects)
                    and not obj["isBroken"]
                )
            ]
            if self.verbose:
                get_logger().info(
                    f"\t\t\tSeen objects: {task.seen_pickupable_objects | task.seen_openable_objects}"
                )
                get_logger().info(
                    f"\t\t\tRemaining objects have not been seen: {remaining_objects}"
                )
            
            failed_places_and_min_dist = (float("inf"), float("inf"))
            obj_pose_to_go_to = None
            for obj in remaining_objects:
                if (
                    self.object_name_to_priority[obj] <= self.max_priority_per_object
                    and obj in env.obj_name_to_walkthrough_start_pose
                ):
                    priority = self.object_name_to_priority[obj]
                    priority_and_dist_to_object = (
                        priority,
                        IThorEnvironment.position_dist(
                            agent_loc,
                            env.obj_name_to_walkthrough_start_pose[obj]["position"],
                            ignore_y=True,
                            l1_dist=True,
                        ),
                    )
                    if (
                        self._last_to_interact_object_pose is not None 
                        and self._last_to_interact_object_pose["name"] == obj
                    ):
                        if self.verbose:
                            get_logger().info(
                                f"\t\t\tSet distance to -1 for the currently targetted object"
                            )
                        priority_and_dist_to_object = (
                            priority_and_dist_to_object[0],
                            -1,
                        )
                    
                    if priority_and_dist_to_object < failed_places_and_min_dist:
                        failed_places_and_min_dist = priority_and_dist_to_object
                        obj_pose_to_go_to = env.obj_name_to_walkthrough_start_pose[obj]
                        
            if self.verbose:
                get_logger().info(
                    f'\t\t\tobj_pose_to_go_to: {obj_pose_to_go_to}'
                )
            if obj_pose_to_go_to is None:
                if self.verbose:
                    get_logger().info(
                        f"\t\t\tAll obejcts in the environment has been seen.\n"
                        f"\t\t\tStop the Walkthrough Task\n"
                    )
                
                self._last_subtask.set_subtask("Stop", None, None)
            else:
                if self.verbose:
                    get_logger().info(
                        f"\t\t\tStart to go to {obj_pose_to_go_to['name']}.\n"
                    )
                self._last_to_interact_object_pose = obj_pose_to_go_to
                self._last_subtask.set_subtask("Explore", None, None)
        
        elif isinstance(task, UnshuffleTask):
            if not replan_subtask:
            # self._generate_expert() is not repeated
                if held_object is not None:
                    # Should go to the goal pose of the held object
                    if self.verbose:
                        get_logger().info(
                            f'      agent is holding object: {held_object["name"]}'
                        )
                    self._last_to_interact_object_pose = copy.deepcopy(held_object)
                    self._last_to_interact_object_pose["target_map"] = "Walkthrough"
                    for k in ["position", "rotation"]:
                        self._last_to_interact_object_pose[k] = task.env.obj_name_to_walkthrough_start_pose[
                            held_object["name"]
                        ][k]
                    self._last_subtask.set_subtask(
                        subtask_type="Goto",
                        obj_type=self._last_to_interact_object_pose["objectType"],
                        target_map=self._last_to_interact_object_pose["target_map"],
                    )

                else:
                    _, goal_poses, cur_poses = env.poses
                    assert len(goal_poses) == len(cur_poses)

                    failed_places_and_min_dist = (float("inf"), float("inf"))
                    obj_pose_to_go_to = None
                    goal_obj_pos = None
                    # TODO: Explore task is ignored...
                    for gp, cp in zip(goal_poses, cur_poses):
                        if (
                            (gp["broken"] == cp["broken"] == False)
                            and self.object_name_to_priority[gp["name"]]
                            <= self.max_priority_per_object
                            and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                            and cp["type"] in (OPENABLE_OBJECTS + PICKUPABLE_OBJECTS)
                        ):
                            priority = self.object_name_to_priority[gp["name"]]
                            priority_and_dist_to_object = (
                                priority,
                                IThorEnvironment.position_dist(
                                    agent_loc, gp["position"], ignore_y=True, l1_dist=True
                                ),
                            )
                            if (
                                self._last_to_interact_object_pose is not None
                                and self._last_to_interact_object_pose["name"] == gp["name"]
                            ):
                                # Set distance to -1 for the currently targeted object
                                if self.verbose:
                                    get_logger().info(
                                        f'      Set distance to -1 for the currently targeted object'
                                    )
                                priority_and_dist_to_object = (
                                    priority_and_dist_to_object[0],
                                    -1,
                                )
                            elif (
                                gp["name"] in set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                            ):
                                if self.verbose:
                                    get_logger().info(
                                        f'      set distance to -0.5 for the seen object {gp["name"]}'
                                    )
                                priority_and_dist_to_object = (
                                    priority_and_dist_to_object[0],
                                    -0.5,
                                )

                            if priority_and_dist_to_object < failed_places_and_min_dist:
                                failed_places_and_min_dist = priority_and_dist_to_object
                                obj_pose_to_go_to = cp
                                goal_obj_pos = gp
                    
                    if self.verbose:
                        get_logger().info(
                            f'      obj_pose_to_go_to: {obj_pose_to_go_to}'
                        )
                    if obj_pose_to_go_to is None:
                        if self.verbose:
                            get_logger().info(
                                f'      no items to rearrange...'
                            )
                        self._last_subtask.set_subtask("Stop", None, None)

                    elif not (
                        set(self.seen_obj_names_unshuffle) | set(self.seen_obj_names_walkthrough)
                    ):
                        if self.verbose:
                            get_logger().info(
                                f'      no seen items...'
                            )
                        self._last_to_interact_object_pose = obj_pose_to_go_to
                        self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                        self._last_subtask.set_subtask("Explore", None, None)

                    else:
                        if self.verbose:
                            get_logger().info(
                                f'      agent is moving to target: {obj_pose_to_go_to["name"]}'
                            )
                        self._last_to_interact_object_pose = obj_pose_to_go_to
                        self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                        self._last_subtask.set_subtask(
                            subtask_type="Goto",
                            obj_type=self._last_to_interact_object_pose["type"],
                            target_map=self._last_to_interact_object_pose["target_map"],
                        )
            else:
                """
                self._generate_expert() is called by self._generate_expert_action_dict()
                since the Goto subtask is done.
                """
                if self.verbose:
                    get_logger().info(
                        f'      Replanning Subtask...'
                    )
                if held_object is not None:
                    self._last_subtask.set_subtask(
                        subtask_type="PutObject",
                        obj_type=self._last_to_interact_object_pose["objectType"],
                        target_map=None,
                    )

                else:
                    _, goal_poses, cur_poses = task.env.poses
                    assert len(goal_poses) == len(cur_poses)
                    obj_pose_to_go_to = None
                    goal_obj_pos = None
                    for gp, cp in zip(goal_poses, cur_poses):
                        if (
                            self._last_to_interact_object_pose is not None
                            and self._last_to_interact_object_pose["name"] == gp["name"]
                        ):
                            obj_pose_to_go_to = cp
                            goal_obj_pos = gp
                    
                    assert obj_pose_to_go_to is not None
                    if (
                        obj_pose_to_go_to["openness"] is not None
                        and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
                        and obj_pose_to_go_to["type"] in OPENABLE_OBJECTS
                    ):
                        self._last_subtask.set_subtask(
                            subtask_type="OpenObject",
                            obj_type=obj_pose_to_go_to["type"],
                            target_map=None,
                        )
                    elif obj_pose_to_go_to["pickupable"]:
                        self._last_subtask.set_subtask(
                            subtask_type="PickupObject",
                            obj_type=obj_pose_to_go_to["type"],
                            target_map=None,
                        )
                    else:
                        self.object_name_to_priority[goal_obj_pos["name"]] = (
                            self.max_priority_per_object + 1
                        )
                        # TODO: Explore?
                        self._generate_expert(task=task)
        
        else:
            raise TypeError
        
    def _generate_expert_action_dict(self, task: Union[WalkthroughTask, UnshuffleTask]):
        if self.verbose:
            get_logger().info(
                f'    ** IN _generate_expert_action_dict() method'
            )
        if task.env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {task.env.mode}"
            )

        if "Stop" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      All subtasks done'
                )
            return dict(action="Done")

        elif "Explore" in self._last_subtask.subtask_type:
            # TODO: to be updated
            if self.verbose:
                get_logger().info(
                    f'      Explore to find difference'
                    f'      Oracle target object: {self._last_to_interact_object_pose["name"]}'
                )
            assert self._last_to_interact_object_pose is not None
        
        elif "Goto" in self._last_subtask.subtask_type:
            assert (
                self._last_to_interact_object_pose is not None
                and 'target_map' in self._last_to_interact_object_pose
            ), f"self._last_subtask: {self._last_subtask} | self._last_to_interact_object_pose: {self._last_to_interact_object_pose}"
            if self.verbose:
                get_logger().info(
                    f'      Goto {self._last_to_interact_object_pose["name"]}...'
                )

        elif "PickupObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      PickupObject Subtask...'
                )
            return dict(
                action="Pickup",
                objectId=self._last_to_interact_object_pose["objectId"],
            )
        elif "OpenObject" in self._last_subtask.subtask_type:
            if self.verbose:
                get_logger().info(
                    f'      OpenObject Subtask...'
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
                    f'      PutObject Subtask...'
                )
            return dict(action="DropHeldObjectWithSnap")
        
        else:
            raise RuntimeError("??????????????????????")

        expert_nav_action = self._expert_nav_action_to_obj(
            task=task,
            obj=self._last_to_interact_object_pose
        )
        if self.verbose:
            get_logger().info(
                f'      Generated expert navigation action is {expert_nav_action}'
            )
        if expert_nav_action is None:
            interactable_positions = task.env._interactable_positions_cache.get(
                scene_name=task.env.scene,
                obj=self._last_to_interact_object_pose,
                controller=task.env.controller,
            )
            self.object_name_to_priority[self._last_to_interact_object_pose["name"]] += 1
            # self._last_subtask.set_subtask("Explore", None, None)
            # self._last_to_interact_object_pose = None
            if self.verbose:
                get_logger().info(
                    f'      Re-generate expert subtask and action...'
                )
            return self._generate_expert(task=task, replan_subtask=True)
        
        elif expert_nav_action == "Pass":
            visible_objects = [o["name"] for o in self.pickupable_or_openable_objects(task)]
            # with include_object_data(task.env.controller):
            #     visible_objects = {
            #         o["name"]
            #         for o in task.env.last_event.metadata["objects"]
            #         if o["visible"]
            #     }
            
            if self.verbose:
                get_logger().info(
                    f"The agent has reached to the interactable pose to object {self._last_to_interact_object_pose['name']}"
                    f" at location {task.env.get_agent_location()}\n"
                    f"The visible objects at current view: {visible_objects}"
                )
            
            if self._last_to_interact_object_pose["name"] not in visible_objects:
                if self.verbose:
                    get_logger().info(
                        f" but the object is not still visible.\n"
                    )
                if self._invalidate_interactable_loc_for_pose(
                    env=task.env,
                    location=task.env.get_agent_location(),
                    obj_pose=self._last_to_interact_object_pose
                ):
                    return self._generate_expert(task=task, replan_subtask=True)
                
                raise RuntimeError("This should not be possible.")

            if self.verbose:
                get_logger().info(
                    f'      Goto {self._last_to_interact_object_pose["name"]} is done.'
                )
                get_logger().info(
                    f'      Re-plan expert subtask to step forward'
                )
            return self._generate_expert(task=task, replan_subtask=True)

        else:
            return dict(action=expert_nav_action)

    def _expert_nav_action_to_obj(
        self,
        task: Union[WalkthroughTask, UnshuffleTask],
        obj: Dict[str, Any]
    ) -> Optional[str]:
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
        