import copy
from typing import (
    Any,
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
from custom.subtask import INTERACT_SUBTASK_TYPES, MAP_TYPES, MAP_TYPES_TO_IDX, SUBTASK_TYPES, SUBTASK_TYPES_TO_IDX
from example_utils import ForkedPdb
from rearrange.environment import RearrangeTHOREnvironment
from rearrange.tasks import AbstractRearrangeTask, UnshuffleTask, WalkthroughTask
from rearrange.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS


if TYPE_CHECKING:
    from allenact.base_abstractions.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")

SpaceDict = gyms.Dict


class PoseSensor(Sensor[IThorEnvironment, Task[IThorEnvironment]]):
    
    def __init__(
        self, 
        uuid: str = "pose", 
        base: int = 90, 
        reference_pose: bool = False,
        **kwargs: Any,
    ) -> None:
        observation_space = self._get_observation_space()
        self.base = base
        self.reference_pose = reference_pose
        self.T_u2w = None
        self.latest_pose = None

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:

        return gym.spaces.Dict(
            {
                'cam_horizon_deg': gym.spaces.Box(
                    low=-30, high=60, dtype=np.float32, shape=(1,),
                ),
                "cam_pos_enu": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                ),
                "rot_3d_enu_deg": gym.spaces.Box(
                    low=0, high=360, dtype=np.float32, shape=(3,),
                ),
                "T_world_to_cam": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(4, 4),
                ),
                "T_unity_to_world": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(4, 4),
                ),
                "agent_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                ),
                "agent_pos_unity": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                ),
            }
        )

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[Task[IThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Dict:

        if self.reference_pose:
            if self.T_u2w is None:
                self.T_u2w = self.create_new_transformation_matrix()
            self.from_environment(env)

        else:
            if task.num_steps_taken() == 0:
                self.latest_pose = self.create_new_initial()
                self.T_u2w = self.create_new_transformation_matrix()

                init_pos = self.get_agent_position_from_environment(env)
                init_pos = np.array(
                    [
                        init_pos['x'],
                        0,
                        init_pos['z']
                    ]
                )
                init_rot = self.get_agent_rotation_from_environment(env)
                init_rot = np.array(
                    [
                        init_rot['x'],  # 0
                        init_rot['y'],
                        init_rot['z']   # 0
                    ]
                )

                # world frame is temporary frame to transform into (relative) world frame
                T_u2t = self.T_u2w.copy()

                # Rotation temp pts to world frame
                # p^w = T^w_temp @ p^temp
                T_t2w = np.eye(4, dtype=np.float32)
                R_t2w = euler.euler2mat(
                    *np.radians(
                        T_u2t[:3, :3] @ init_rot
                    )
                )

                # world frame origin w.r.t. unity frame: (o_w)^u = self.latest_pose['init_pos']
                # world frame origin w.r.t. temp frame: (o_w)^temp = T^temp_u @ (o_w)^u
                # temp frame origin w.r.t. world frame: (o_temp)^w = -(R^temp_w)^(-1) @ (o_w)^temp 
                #                                                  = -R^w_temp @ (o_w)^temp
                o_t_wrt_w = -R_t2w @ T_u2t[:3, :3] @ init_pos

                # T^w_temp = [R^w_temp | (o_temp)^w]
                #            [    0    |      1    ]
                T_t2w[:3, :3] = R_t2w
                T_t2w[:3, 3] = o_t_wrt_w
                
                # T^w_u = T^w_temp @ T^temp_u
                self.T_u2w = T_t2w @ T_u2t

            elif task.actions_taken_success[-1]:
                self.simulate_successful_nav_action(task.actions_taken[-1])

        T_world_to_cam = self.get_pose_mat()
        agent_pos = self.get_agent_pos()
        agent_pos_unity = self.get_agent_pos_unity()

        return {
            **{k: np.asarray(v) for k, v in self.latest_pose.items()},
            "T_world_to_cam": T_world_to_cam,
            "T_unity_to_world": self.T_u2w,
            "agent_pos": agent_pos,
            "agent_pos_unity": agent_pos_unity,
        }

    def get_agent_position_from_environment(
        self,
        env: IThorEnvironment
    ):
        return env.last_event.metadata['agent']['position']

    def get_agent_rotation_from_environment(
        self,
        env: IThorEnvironment
    ):
        return env.last_event.metadata['agent']['rotation']

    def from_environment(
        self,
        env: IThorEnvironment
    ):
        event = env.last_event
        cam_horizon_deg = event.metadata["agent"]["cameraHorizon"]

        cam_pos_dict_3d_unity = event.metadata["cameraPosition"]
        cam_pos_unity = np.array(
            [
                cam_pos_dict_3d_unity['x'],
                cam_pos_dict_3d_unity['y'],
                cam_pos_dict_3d_unity['z']
            ]
        )
        cam_pos_enu = self.T_u2w[:3, :3] @ cam_pos_unity

        rot_dict_3d_unity = event.metadata["agent"]["rotation"]
        rot_3d_unity_deg = np.array(
            [
                rot_dict_3d_unity['x'], 
                rot_dict_3d_unity['y'], 
                rot_dict_3d_unity['z']
            ]
        )
        rot_3d_enu_deg = -self.T_u2w[:3, :3] @ rot_3d_unity_deg
        
        self.latest_pose = {
            "cam_pos_enu": cam_pos_enu,
            "rot_3d_enu_deg": rot_3d_enu_deg,
            "cam_horizon_deg": cam_horizon_deg,
        }
    
    @staticmethod
    def create_new_initial():
        cam_horizon_deg = 0.0
        cam_pos_enu = np.array([0.0, 0.0, 1.576])
        rot_3d_enu_deg = np.array([0.0, 0.0, 0.0])

        return {
            "cam_horizon_deg": cam_horizon_deg,
            "cam_pos_enu": cam_pos_enu,
            "rot_3d_enu_deg": rot_3d_enu_deg,
        }

    @staticmethod
    def create_new_transformation_matrix():
        return np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ],
            dtype=np.float32,
        )

    def simulate_successful_nav_action(
        self,
        action: str,
    ):
        MOVE_STEP = 0.25
        PITCH_STEP = 30
        YAW_STEP = 90

        latest_pose = copy.deepcopy(self.latest_pose)
        action_type = stringcase.pascalcase(action)
        
        if action_type == "RotateLeft":
            self.latest_pose["rot_3d_enu_deg"][2] = round_to_factor(latest_pose["rot_3d_enu_deg"][2] + YAW_STEP, self.base)
        elif action_type == "RotateRight":
            self.latest_pose["rot_3d_enu_deg"][2] = round_to_factor(latest_pose["rot_3d_enu_deg"][2] - YAW_STEP, self.base)
        elif action_type == "LookDown":
            self.latest_pose["cam_horizon_deg"] = latest_pose["cam_horizon_deg"] + PITCH_STEP
        elif action_type == "LookUp":
            self.latest_pose["cam_horizon_deg"] = latest_pose["cam_horizon_deg"] - PITCH_STEP
        elif action_type in ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack"):
            if action_type == "MoveAhead":
                step = np.array([MOVE_STEP, 0.0])
            elif action_type == "MoveLeft":
                step = np.array([0.0, MOVE_STEP])
            elif action_type == "MoveBack":
                step = np.array([-MOVE_STEP, 0.0])
            elif action_type == "MoveRight":
                step = np.array([0.0, -MOVE_STEP])
            else:
                raise ValueError("Wrong Navigation Action")
            
            theta = latest_pose["rot_3d_enu_deg"][2] / 180.0 * np.pi # [rad]
            ct = np.cos(theta)
            st = np.sin(theta)
            rot_mat = np.array(
                [
                    [ct, -st],
                    [st, ct]
                ]
            )
            self.latest_pose["cam_pos_enu"][0:2] += (rot_mat @ step.reshape(-1, 1)).reshape(-1)

    def get_pose_mat(self):
        # Translation from world origin to camera/agent position, T^ap_w
        T_world_to_agent_pos = np.eye(4)
        T_world_to_agent_pos[:3, 3] = -self.latest_pose["cam_pos_enu"]

        # ... rotation to agent frame (x-forward, y-left, z-up): T^a_ap
        T_agent_pos_to_agent = np.eye(4, dtype=np.float32)
        T_agent_pos_to_agent[:3, :3] = euler.euler2mat(*np.radians(-self.latest_pose["rot_3d_enu_deg"]))

        # ... transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch: T^cf_a
        T_agent_to_camflat = np.eye(4, dtype=np.float32)
        T_agent_to_camflat[:3, :3] = euler.euler2mat(*np.radians([90, -90, 0]))

        # ... transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch: T^c_cf
        T_camflat_to_cam = np.eye(4, dtype=np.float32)
        T_camflat_to_cam[:3, :3] = euler.euler2mat(*np.radians([self.latest_pose["cam_horizon_deg"], 0, 0]))

        # compose into a transform from world to camera (camera extrinsic matrix)
        # T^cam_world = T^cam_camflat @ T^camflat_agent @ T^agent_agentpos @ T^agentpos_world
        T_world_to_cam = T_camflat_to_cam @ T_agent_to_camflat @ T_agent_pos_to_agent @ T_world_to_agent_pos

        return T_world_to_cam

    def get_agent_pos(self):
        agent_pos = np.array(
            [
                self.latest_pose["cam_pos_enu"][0],
                self.latest_pose["cam_pos_enu"][1],
                self.latest_pose["cam_pos_enu"][2]
            ]
        )

        return agent_pos

    def get_agent_pos_unity(self):
        return np.linalg.inv(self.T_u2w)[:3, :3] @ self.get_agent_pos()


class UnshuffledPoseSensor(PoseSensor):

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[Task[IThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Dict:

        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
    
        if self.reference_pose:
            if self.T_u2w is None:
                self.T_u2w = self.create_new_transformation_matrix()
            self.from_environment(walkthrough_env)

        else:
            if task.num_steps_taken() == 0:
                self.latest_pose = self.create_new_initial()
                self.T_u2w = self.create_new_transformation_matrix()

                init_pos = self.get_agent_position_from_environment(walkthrough_env)
                init_pos = np.array(
                    [
                        init_pos['x'],
                        0,
                        init_pos['z']
                    ]
                )
                init_rot = self.get_agent_rotation_from_environment(walkthrough_env)
                init_rot = np.array(
                    [
                        init_rot['x'],  # 0
                        init_rot['y'],
                        init_rot['z']   # 0
                    ]
                )

                # world frame is temporary frame to transform into (relative) world frame
                T_u2t = self.T_u2w.copy()

                # Rotation temp pts to world frame
                # p^w = T^w_temp @ p^temp
                T_t2w = np.eye(4, dtype=np.float32)
                R_t2w = euler.euler2mat(
                    *np.radians(
                        T_u2t[:3, :3] @ init_rot
                    )
                )

                # world frame origin w.r.t. unity frame: (o_w)^u = self.latest_pose['init_pos']
                # world frame origin w.r.t. temp frame: (o_w)^temp = T^temp_u @ (o_w)^u
                # temp frame origin w.r.t. world frame: (o_temp)^w = -(R^temp_w)^(-1) @ (o_w)^temp 
                #                                                  = -R^w_temp @ (o_w)^temp
                o_t_wrt_w = -R_t2w @ T_u2t[:3, :3] @ init_pos

                # T^w_temp = [R^w_temp | (o_temp)^w]
                #            [    0    |      1    ]
                T_t2w[:3, :3] = R_t2w
                T_t2w[:3, 3] = o_t_wrt_w
                
                # T^w_u = T^w_temp @ T^temp_u
                self.T_u2w = T_t2w @ T_u2t

            elif task.actions_taken_success[-1]:
                self.simulate_successful_nav_action(task.actions_taken[-1])

        T_world_to_cam = self.get_pose_mat()
        agent_pos = self.get_agent_pos()
        agent_pos_unity = self.get_agent_pos_unity()

        return {
            **{k: np.asarray(v) for k, v in self.latest_pose.items()},
            "T_world_to_cam": T_world_to_cam,
            "T_unity_to_world": self.T_u2w,
            "agent_pos": agent_pos,
            "agent_pos_unity": agent_pos_unity,
        }

class InventoryObjectSensor(Sensor[IThorEnvironment, Task[IThorEnvironment]]):
    
    def __init__(
        self, 
        ordered_object_types: Sequence[str],
        uuid: str = "inventory", 
        reference_inventory: bool = False,
        **kwargs: Any,
    ) -> None:
        self.reference_inventory = reference_inventory
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        )
        self.inventory_object_ids = []
        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        self.object_type_to_idx[UNKNOWN_OBJECT_STR] = len(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types) + 1
        
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0, high=1, dtype=bool, shape=(self.num_objects, )
        )

    def get_observation(
        self, 
        env: IThorEnvironment, 
        task: Optional[Task[IThorEnvironment]], 
        *args: Any, 
        **kwargs: Any
    ) -> np.ndarray:
        
        if self.reference_inventory:
            self.from_environment(env)
        else:
            if task.num_steps_taken() == 0:
                self.empty_inventory()
            elif task.actions_taken_success[-1]:
                self.simulate_successful_inventory_action(task.actions_taken[-1])

        inventory_vector = np.zeros((self.num_objects, ), dtype=bool)
        for object_id in self.inventory_object_ids:
            inventory_vector[object_id] = 1
        
        return inventory_vector

    def from_environment(
        self, 
        env: IThorEnvironment,
    ):
        event = env.last_event

        for obj in event.metadata['inventoryObjects']:
            object_str = obj['objectType'].split("_")[0]
            object_id = self.object_type_to_idx[object_str]
            self.inventory_object_ids.append(object_id)
        
    def empty_inventory(self):
        self.inventory_object_ids = []

    def simulate_successful_inventory_action(
        self, 
        action: str,
    ):
        split_ = action.split('_')

        if split_[0] == "pickup":
            object_type = stringcase.pascalcase('_'.join(split_[1:]))
            object_id = self.object_type_to_idx[object_type]
            if len(self.inventory_object_ids) == 0:
                self.inventory_object_ids.append(object_id)
        elif split_[0] == "drop":
            self.inventory_object_ids = []


class SemanticSegmentationSensor(
    Sensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]
    ],
):
    def __init__(
        self,
        ordered_object_types: Sequence[str],
        height: Optional[int] = 224,
        width: Optional[int] = 224,
        uuid: str = "semseg",
        **kwargs: Any
    ):
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        )
        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        self.object_type_to_idx[UNKNOWN_OBJECT_STR] = len(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types) + 1
        self.height = height
        self.width = width
        
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0, high=1, shape=(self.num_objects, self.height, self.width), dtype=np.bool,
        )

    def get_observation(
        self,
        env: Union[IThorEnvironment, RoboThorEnvironment],
        task: Optional[Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        e = env.last_event
        semantic_masks = np.zeros((self.num_objects, self.height, self.width), dtype=np.bool)

        for obj_str, class_mask in e.class_masks.items():
            obj_type_str = obj_str if obj_str in self.ordered_object_types else UNKNOWN_OBJECT_STR
            obj_type_idx = self.object_type_to_idx[obj_type_str]
            semantic_masks[obj_type_idx] += class_mask

        return semantic_masks


class InstanceSegmentationSensor(
    Sensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]
    ],
):
    def __init__(
        self,
        ordered_object_types: Sequence[str],
        height: Optional[int] = 224,
        width: Optional[int] = 224,
        uuid: str = "instseg",
        **kwargs: Any
    ):
        self.ordered_object_types = list(ordered_object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        )
        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        # self.object_type_to_idx[UNKNOWN_OBJECT_STR] = len(self.ordered_object_types)
        self.num_objects = len(self.ordered_object_types)
        self.height = height
        self.width = width
        
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                'inst_masks': gym.spaces.Box(
                    low=0, high=np.inf, dtype=np.int32, shape=(self.num_objects, self.height, self.width)
                ),
                'inst_detected': gym.spaces.Box(
                    low=0, high=np.inf, dtype=np.int32, shape=(self.num_objects,)
                ), 
            }
        )

    def get_observation(
        self,
        env: Union[IThorEnvironment, RoboThorEnvironment],
        task: Optional[Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        e = env.last_event

        detected_objs = [obj for obj in e.instance_masks if obj.split("|")[0] in self.ordered_object_types]
        inst_masks = np.zeros((self.num_objects, self.height, self.width), dtype=np.int32)
        inst_detected = np.zeros(self.num_objects, dtype=np.int32)

        for obj in detected_objs:
            obj_type = obj.split("|")[0]
            obj_type_idx = self.object_type_to_idx[obj_type]
            inst_mask = e.instance_masks[obj]
            inst_masks[obj_type_idx] += inst_mask * (2 ** inst_detected[obj_type_idx])

            inst_detected[obj_type_idx] += 1

        return {
            'inst_masks': inst_masks,
            'inst_detected': inst_detected,
        }


class UnshuffledDepthRearrangeSensor(DepthSensorThor):

    def frame_from_env(
        self, 
        env: RearrangeTHOREnvironment, 
        task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
        return walkthrough_env.last_event.depth_frame.copy()


class UnshuffledSemanticSegmentationSensor(SemanticSegmentationSensor):

    def get_observation(
        self, 
        env: Union[IThorEnvironment, RoboThorEnvironment], 
        task: Optional[Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]], 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
        
        e = walkthrough_env.last_event
        semantic_masks = np.zeros((self.num_objects, self.height, self.width), dtype=np.bool)

        for obj_str, class_mask in e.class_masks.items():
            obj_type_str = obj_str if obj_str in self.ordered_object_types else UNKNOWN_OBJECT_STR
            obj_type_idx = self.object_type_to_idx[obj_type_str]
            semantic_masks[obj_type_idx] += class_mask

        return semantic_masks


class UnshuffledInstanceSegmentationSensor(InstanceSegmentationSensor):

    def get_observation(
        self,
        env: Union[IThorEnvironment, RoboThorEnvironment],
        task: Optional[Union[Task[IThorEnvironment], Task[RoboThorEnvironment]]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
    
        e = env.last_event

        detected_objs = [obj for obj in e.instance_masks if obj.split("|")[0] in self.ordered_object_types]
        inst_masks = np.zeros((self.num_objects, self.height, self.width), dtype=np.int32)
        inst_detected = np.zeros(self.num_objects, dtype=np.int32)

        for obj in detected_objs:
            obj_type = obj.split("|")[0]
            obj_type_idx = self.object_type_to_idx[obj_type]
            inst_mask = e.instance_masks[obj]
            inst_masks[obj_type_idx] += inst_mask * (2 ** inst_detected[obj_type_idx])

            inst_detected[obj_type_idx] += 1

        return {
            'inst_masks': inst_masks,
            'inst_detected': inst_detected,
        }
