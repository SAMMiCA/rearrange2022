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
from custom.constants import IDX_TO_OBJECT_TYPE, NUM_OBJECT_TYPES, OBJECT_TYPES_TO_IDX, UNKNOWN_OBJECT_STR
from custom.subtask import IDX_TO_SUBTASK_TYPE, INTERACT_SUBTASK_TYPES, MAP_TYPES, MAP_TYPES_TO_IDX, SUBTASK_TYPES, SUBTASK_TYPES_TO_IDX
from rearrange.environment import RearrangeMode, RearrangeTHOREnvironment
from rearrange.expert import ShortestPathNavigatorTHOR
from rearrange.tasks import AbstractRearrangeTask, UnshuffleTask, WalkthroughTask
from rearrange.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS, STEP_SIZE
from example_utils import ForkedPdb


if TYPE_CHECKING:
    from allenact.base_abstractions.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")

SpaceDict = gyms.Dict


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

        self._last_subtask: Optional[str] = None

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

        self._last_subtask: Optional[str] = None        

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

    def update(
        self,
        task: UnshuffleTask, 
        shortest_path_navigator: ShortestPathNavigatorTHOR,
        action_taken: Optional[int],
        action_success: Optional[bool],
    ):
        if action_taken is not None:
            assert action_success is not None

            action_names = task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            last_expert_subtask = self._last_subtask
            was_interact_subtask = last_expert_subtask in INTERACT_SUBTASK_TYPES
            
            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if (
                "pickup_" in action_str
                and action_taken == last_expert_action
                and action_success
            ):
                self._name_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose[
                    "name"
                ]

            if "drop_held_object_with_snap" in action_str and agent_took_expert_action:
                if self._name_of_object_we_wanted_to_pickup is not None:
                    self.object_name_to_priority[
                        self._name_of_object_we_wanted_to_pickup
                    ] += 1
                else:
                    self.object_name_to_priority[self._last_held_object_name] += 1

            if "open_by_type" in action_str and agent_took_expert_action:
                self.object_name_to_priority[
                    self._last_to_interact_object_pose["name"]
                ] += 1

            if not action_success:
                if was_nav_action:
                    shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                elif (
                    ("pickup_" in action_str or "open_by_type_" in action_str)
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        env=task.unshuffle_env,
                        location=task.unshuffle_env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str)
                    and task.unshuffle_env.held_object is not None
                ) and action_taken == last_expert_action:
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
            else:
                # If the action succeeded and was not a move action then let's force an update
                # of our currently targeted object
                if "Explore" in last_expert_subtask:
                    if self._last_to_interact_object_pose is None:
                        vis_objs = self.get_object_names_from_current_view(
                            task=task, env_type="Unshuffle", rearrange_targets=self._objects_to_rearrange
                        )
                        w_vis_objs = self.get_object_names_from_current_view(
                            task=task, env_type="Walkthrough", rearrange_targets=self._objects_to_rearrange
                        )
                        if (set(vis_objs) | set(w_vis_objs)):
                            self._last_subtask = None
                    
                    else:
                        assert (
                            self._last_to_interact_object_pose is not None
                            and "target_map" in self._last_to_interact_object_pose
                        )
                        vis_objs = self.get_object_names_from_current_view(
                            task=task, env_type=self._last_to_interact_object_pose["target_map"], rearrange_targets=self._objects_to_rearrange
                        )
                        if self._last_to_interact_object_pose["name"] in vis_objs:
                            # self.object_name_to_priority[
                            #     self._last_to_interact_object_pose["name"]
                            # ] += 1
                            self._last_subtask = None
                            self._last_to_interact_object_pose = None
                
                elif was_interact_subtask:
                    assert (
                        self._last_to_interact_object_pose is not None
                        and "target_map" in self._last_to_interact_object_pose
                    )
                    arg_type = next(
                        (
                            o['objectType'] for o in task.unshuffle_env.last_event.metadata['objects']
                            if o['name'] == self._last_to_interact_object_pose["name"]
                        ),
                        None
                    )
                    if "PickupObject" in last_expert_subtask:
                        target_action = f'pickup_{stringcase.snakecase(arg_type)}'
                    elif "OpenObject" in last_expert_subtask:
                        target_action = f'open_by_type_{stringcase.snakecase(arg_type)}'
                    else:
                        target_action = f'drop_held_object_with_snap'

                    if action_str == target_action and action_success:
                        self._last_subtask = None
                        self._last_to_interact_object_pose = None
                
                elif "Stop" in last_expert_subtask:
                    self._last_subtask = None
                    self._last_to_interact_object_pose = None

        held_object = task.env.held_object
        if held_object:
            self._last_held_object_name = held_object['name']

        self._generate_and_record_expert_subtask_and_action(task)

    def _generate_and_record_expert_subtask_and_action(self, task: UnshuffleTask):
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

        expert_subtask_and_action_dict = self._generate_expert_subtask_dict(task)
        type_id = SUBTASK_TYPES_TO_IDX[expert_subtask_and_action_dict["subtask"]]
        arg_id = (
            OBJECT_TYPES_TO_IDX[expert_subtask_and_action_dict["arg"]["objectId"].split("|")[0]] 
            if expert_subtask_and_action_dict["arg"] is not None 
            else -1
        )
        target_map_id = (
            MAP_TYPES_TO_IDX[expert_subtask_and_action_dict["arg"]["target_map"]]
            if (
                expert_subtask_and_action_dict["arg"] is not None 
                and "target_map" in expert_subtask_and_action_dict["arg"]
            )
            else 0
        )

        subtask = (
            type_id * NUM_OBJECT_TYPES * len(MAP_TYPES)
            + (arg_id + 1) * len(MAP_TYPES)
            + target_map_id
        )
        assert 0 <= subtask <= (len(SUBTASK_TYPES) - 1) * NUM_OBJECT_TYPES * len(MAP_TYPES)
        self.expert_subtask_list.append(subtask)

        action_str = stringcase.snakecase(expert_subtask_and_action_dict["action"])
        if action_str not in task.action_names():
            obj_type = stringcase.snakecase(
                expert_subtask_and_action_dict["objectId"].split("|")[0]
            )
            action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)

        if self.verbose:
            get_logger().info(
                f'    Expert Subtask: {expert_subtask_and_action_dict["subtask"]}[{type_id}] | Argument: {expert_subtask_and_action_dict["arg"]["name"] if expert_subtask_and_action_dict["arg"] else "NIL"}[{arg_id}] | Map: {expert_subtask_and_action_dict["arg"]["target_map"] if expert_subtask_and_action_dict["arg"] else "Unshuffle"}[{target_map_id}]'
            )
            get_logger().info(
                f'    Expert Action: {action_str}[{task.action_names().index(action_str)}] | ObjectID: {expert_subtask_and_action_dict["objectId"] if "objectId" in expert_subtask_and_action_dict else None}'
            )

    def _generate_expert_subtask_dict(self, task: UnshuffleTask):
        env = task.unshuffle_env
        w_env = task.walkthrough_env

        if env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {env.mode}"
            )
        
        held_object = env.held_object
        agent_loc = env.get_agent_location()
        
        _, goal_poses, cur_poses = env.poses
        assert len(goal_poses) == len(cur_poses)
        if task.num_steps_taken() == 0:
            self._objects_to_rearrange = {
                goal_poses[id]["name"]: goal_poses[id]
                for id in task.start_energies.nonzero()[0]
            }
        
        # update objects has been seen
        for gp, cp in zip(goal_poses, cur_poses):
            if (
                (gp["broken"] == cp["broken"] == False)
                and self.object_name_to_priority[gp["name"]]
                <= self.max_priority_per_object
                and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
            ):
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
                and self.object_name_to_priority[gp["name"]]
                <= self.max_priority_per_object
                and RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                and (
                    gp["name"] in self.seen_obj_names_unshuffle
                    or gp["name"] in self.seen_obj_names_walkthrough
                )
            ):
                if gp["name"] in self.seen_obj_names_unshuffle:
                    del self.seen_obj_names_unshuffle[gp["name"]]
                if gp["name"] in self.seen_obj_names_walkthrough:
                    del self.seen_obj_names_walkthrough[gp["name"]]
        
        if held_object is not None:
            self._last_to_interact_object_pose = copy.deepcopy(held_object)
            if (
                held_object["name"] in self.seen_obj_names_walkthrough
                or self.object_name_to_priority[held_object["name"]] > self.max_priority_per_object
                or held_object["name"] != self._name_of_object_we_wanted_to_pickup
            ):
                self._last_subtask = "PutObject"
            else:
                self._last_subtask = "Explore"
            self._last_to_interact_object_pose["target_map"] = "Walkthrough"      
            # Replace position and rotation for held object with walkthrough start pose
            for k in ["position", "rotation"]:
                self._last_to_interact_object_pose[k] = env.obj_name_to_walkthrough_start_pose[
                    held_object["name"]
                ][k]

        elif self._last_subtask is None:
            
            min_priorities = (0, float("inf"))
            obj_pose_to_go_to = None
            goal_obj_pos = None
            type_id = 3
            
            for gp, cp in zip(goal_poses, cur_poses):
                if (
                    (gp["broken"] == cp["broken"] == False)
                    and self.object_name_to_priority[gp["name"]]
                    <= self.max_priority_per_object
                    and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                ):
                    if (
                        self.seen_obj_names_unshuffle is not None
                        and gp["name"] in self.seen_obj_names_unshuffle
                    ):
                        type_priority = -2
                    elif (
                        self.seen_obj_names_walkthrough is not None
                        and gp["name"] in self.seen_obj_names_walkthrough
                    ):
                        type_priority = -1
                    else:
                        type_priority = 0

                    priority = self.object_name_to_priority[gp["name"]]
                    priorities_to_object = (
                        type_priority, priority
                    )

                    if priorities_to_object < min_priorities:
                        min_priorities = priorities_to_object
                        obj_pose_to_go_to = cp
                        goal_obj_pos = gp
                        type_id = -type_priority

            if type_id == 3:
                self._last_subtask = "Stop"
                self._last_to_interact_object_pose = None

            elif type_id == 2:
                if (
                    obj_pose_to_go_to['openness'] is not None
                    and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
                ):
                    self._last_subtask = "OpenObject"
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    
                elif obj_pose_to_go_to["pickupable"]:
                    self._last_subtask = "PickupObject"
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    
                else:
                    self.object_name_to_priority[goal_obj_pos["name"]] = (
                        self.max_priority_per_object + 1
                    )
                    return self._generate_expert_subtask_dict(task)

            elif type_id == 1:
                # The object difference found in walkthrough environment
                if (
                    obj_pose_to_go_to['openness'] is not None
                    and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
                ):
                    self._last_subtask = "OpenObject"
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Walkthrough"
                    
                    # replace position and rotation with goal state
                    for k in ["position", "rotation"]:
                        self._last_to_interact_object_pose[k] = goal_obj_pos[k]
                    
                elif obj_pose_to_go_to["pickupable"]:
                    self._last_subtask = "Explore"
                    self._last_to_interact_object_pose = obj_pose_to_go_to
                    self._last_to_interact_object_pose["target_map"] = "Unshuffle"
                    
                else:
                    self.object_name_to_priority[goal_obj_pos["name"]] = (
                        self.max_priority_per_object + 1
                    )
                    return self._generate_expert_subtask_dict(task)
            else:
                self._last_subtask = "Explore"
                self._last_to_interact_object_pose = None
        
        assert self._last_subtask is not None

        expert_action_dict = self._generate_expert_action_dict(task=task)
        # if self.verbose:
        #     arg_name = "NIL"
        #     arg_id = -1
        #     if self._last_to_interact_object_pose is not None:
        #         arg_name = self._last_to_interact_object_pose["name"]
        #         arg_id = self._last_to_interact_object_pose["objectId"]
            
        #     act = expert_action_dict["action"]

        #     get_logger().info(
        #         f'  Estimated next subtask: {self._last_subtask} | subtask_argument: {arg_name}({arg_id})'
        #     )
        # return dict(subtask=self._last_subtask, arg=self._last_to_interact_object_pose, **expert_action_dict)
        return {
            **{
                'subtask': self._last_subtask,
                'arg': self._last_to_interact_object_pose,
            },
            **expert_action_dict
        }

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

    def _generate_expert_action_dict(self, task: UnshuffleTask):
        env = task.unshuffle_env
        w_env = task.walkthrough_env

        if env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {env.mode}"
            )
        
        held_object = env.held_object
        agent_loc = env.get_agent_location()

        if "Stop" in self._last_subtask:
            return dict(action="Done")

        elif "Explore" in self._last_subtask:
            _, goal_poses, cur_poses = env.poses
            assert len(goal_poses) == len(cur_poses)
            obj_pose_to_go_to = None
            goal_obj_pos = None

            if self._last_to_interact_object_pose is None:

                failed_places_and_min_dist = (float("inf"), float("inf"))
                for gp, cp in zip(goal_poses, cur_poses):
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
        
        elif "PutObject" in self._last_subtask and not held_object:
            # PutObject without holding object
            # self._last_to_interact_object_pose saves put object info
            self._invalidate_interactable_loc_for_pose(
                env=task.unshuffle_env,
                location=agent_loc,
                obj_pose=task.unshuffle_env.obj_name_to_walkthrough_start_pose[
                    self._last_to_interact_object_pose["name"]
                ],
            )
            self._last_subtask = None
            self._last_to_interact_object_pose = None
            return self._generate_expert_subtask_dict(task=task)
                
        if self._last_to_interact_object_pose is None:
            return dict(action="Done")

        assert (
            self._last_to_interact_object_pose is not None
            and 'target_map' in self._last_to_interact_object_pose
        )

        expert_nav_action = self._expert_nav_action_to_obj(task=task, obj=self._last_to_interact_object_pose)
        if expert_nav_action is None:
            if "PutObject" in self._last_subtask:
                return dict(action="DropHeldObjectWithSnap")

            interactable_positions = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=self._last_to_interact_object_pose,
                controller=env.controller,
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
            self.object_name_to_priority[self._last_to_interact_object_pose["name"]] += 1
            self._last_subtask = None
            self._last_to_interact_object_pose = None
            return self._generate_expert_subtask_dict(task=task)

        elif expert_nav_action == "Pass":
            if "PutObject" in self._last_subtask:
                return dict(action="DropHeldObjectWithSnap")
                
            with include_object_data(env.controller):
                visible_objects = {
                    o["name"]
                    for o in env.last_event.metadata["objects"]
                    if o["visible"]
                }

            if self._last_to_interact_object_pose["name"] not in visible_objects:
                if self._invalidate_interactable_loc_for_pose(
                    env=env,
                    location=agent_loc, 
                    obj_pose=self._last_to_interact_object_pose
                ):
                    self._last_subtask = None
                    self._last_to_interact_object_pose = None
                    return self._generate_expert_subtask_dict(task=task)

                raise RuntimeError("This should not be possible.")

            if "OpenObject" in self._last_subtask:
                return dict(
                    action="OpenByType",
                    objectId=self._last_to_interact_object_pose["objectId"],
                    openness=env.obj_name_to_walkthrough_start_pose[
                        self._last_to_interact_object_pose["name"]
                    ]["openness"],
                )
            elif "PickupObject" in self._last_subtask:
                return dict(
                    action="Pickup",
                    objectId=self._last_to_interact_object_pose["objectId"],
                )
            elif "Explore" in self._last_subtask:
                # agent has reached to the object but is not seen in walkthrough environment.
                if self._invalidate_interactable_loc_for_pose(
                    env=env,
                    location=agent_loc, 
                    obj_pose=self._last_to_interact_object_pose
                ):
                    self._last_subtask = None
                    self._last_to_interact_object_pose = None
                    return self._generate_expert_subtask_dict(task=task)

                raise RuntimeError("This should not be possible.")
            else:
                get_logger().warning(
                    f"{self._last_to_interact_object_pose['name']} has moved but is not pickupable."
                )
                self.object_name_to_priority[self._last_to_interact_object_pose["name"]] = (
                    self.max_priority_per_object + 1
                )
                self._last_subtask = None
                self._last_to_interact_object_pose = None
                return self._generate_expert_subtask_dict(task=task)
        else:
            return dict(action=expert_nav_action)

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

        if self.verbose:
            get_logger().info(
                f'==== [STEP {task.num_steps_taken()}] ==== '
            )
            get_logger().info(
                f'  Last Action Taken: {last_action}[{last_action_ind}] | Last Action Taken Success: {last_action_success}'
            )

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