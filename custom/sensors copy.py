import copy
from typing import (
    Any,
    Optional,
    Sequence,
    Union,
    Dict
)
import numpy as np
import torch
import math
from transforms3d import euler
import gym.spaces
from allenact.base_abstractions.task import Task
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment


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
        self.latest_pose = None
        self.latest_pose_rel = None
        self.initial_pose_rel = None

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
                # "cam_pos_unity": gym.spaces.Box(
                #     low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                # ),
                "rot_3d_enu_deg": gym.spaces.Box(
                    low=0, high=360, dtype=np.float32, shape=(3,),
                ),
                # "rot_3d_unity_deg": gym.spaces.Box(
                #     low=0, high=360, dtype=np.float32, shape=(3,),
                # ),
                "T_world_to_cam": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(4, 4),
                ),
                "T_unity_to_world": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(4, 4),
                ),
                "agent_pos_enu": gym.spaces.Box(
                    low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                ),
                # "agent_pos_unity": gym.spaces.Box(
                #     low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
                # ),
            }
        )

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[Task[IThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Dict:

        # if self.reference_pose:
        #     self.from_environment(env)

        # else:
        #     if task.num_steps_taken() == 0:
        #         self.latest_pose = self.create_new_initial()
        #     if env.last_event.metadata["lastActionSuccess"]:
        #         self.simulate_successful_nav_action(env)
        
        # relative pose
        if task.num_steps_taken() == 0:
            self.latest_pose_rel = self.create_new_initial()
            self.initial_pose_rel = dict()

            init_pos = env.last_event.metadata['agent']['position']
            self.initial_pose_rel['position'] = np.array(
                [
                    init_pos['x'],
                    0,
                    init_pos['z']
                ]
            )

            init_rot = env.last_event.metadata['agent']['rotation']
            self.initial_pose_rel['rotation'] = np.array(
                [
                    init_rot['x'],
                    init_rot['y'],
                    init_rot['z']
                ]
            )
        if env.last_event.metadata['lastActionSuccess']:
            self.simulate_successful_nav_action(env)

        # absolute pose from ai2thor environment
        self.from_environment(env)

        T_world_to_cam, T_unity_to_world = self.get_pose_mat()
        T_world_rel_to_cam, T_unity_to_world_rel = self.get_rel_pose_mat()
        agent_pos = self.get_agent_pos()
        agent_pos_rel = self.get_agent_pos_rel()

        return {
            **{k: np.asarray(v) for k, v in self.latest_pose.items()},
            **{f'{k}_rel': np.asarray(v) for k, v in self.latest_pose_rel.items()},
            "T_world_to_cam": T_world_to_cam,
            "T_unity_to_world": T_unity_to_world,
            "T_world_rel_to_cam": T_world_rel_to_cam,
            "T_unity_to_world_rel": T_unity_to_world_rel,
            "agent_pos": agent_pos,
            "agent_pos_rel": agent_pos_rel,
        }

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

        cam_pos_enu = np.array(
            [
                cam_pos_dict_3d_unity['z'],
                -cam_pos_dict_3d_unity['x'],
                cam_pos_dict_3d_unity['y']
            ]
        )

        rot_dict_3d_unity = event.metadata["agent"]["rotation"]
        rot_3d_unity_deg = np.array(
            [
                rot_dict_3d_unity['x'], 
                rot_dict_3d_unity['y'], 
                rot_dict_3d_unity['z']]
        )

        rot_3d_enu_deg = np.array(
            [
                -rot_dict_3d_unity['z'], 
                rot_dict_3d_unity['x'], 
                -rot_dict_3d_unity['y']]
        )
        
        self.latest_pose = {
            # "cam_pos_unity": cam_pos_unity,
            "cam_pos_enu": cam_pos_enu,
            # "rot_3d_unity_deg": rot_3d_unity_deg,
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

    def simulate_successful_nav_action(
        self,
        env: IThorEnvironment,
    ):
        MOVE_STEP = 0.25
        PITCH_STEP = 30
        YAW_STEP = 90

        event = env.last_event
        # latest_pose = self.latest_pose.copy()
        latest_pose = copy.deepcopy(self.latest_pose_rel)
        
        if event.metadata["lastAction"] == "RotateLeft":
            self.latest_pose_rel["rot_3d_enu_deg"][2] = round_to_factor(latest_pose["rot_3d_enu_deg"][2] + YAW_STEP, self.base)
        elif event.metadata["lastAction"] == "RotateRight":
            self.latest_pose_rel["rot_3d_enu_deg"][2] = round_to_factor(latest_pose["rot_3d_enu_deg"][2] - YAW_STEP, self.base)
        elif event.metadata["lastAction"] == "LookDown":
            self.latest_pose_rel["cam_horizon_deg"] = latest_pose["cam_horizon_deg"] + PITCH_STEP
        elif event.metadata["lastAction"] == "LookUp":
            self.latest_pose_rel["cam_horizon_deg"] = latest_pose["cam_horizon_deg"] - PITCH_STEP
        elif event.metadata["lastAction"] in ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack"):
            if event.metadata["lastAction"] == "MoveAhead":
                step = np.array([MOVE_STEP, 0.0])
            elif event.metadata["lastAction"] == "MoveLeft":
                step = np.array([0.0, MOVE_STEP])
            elif event.metadata["lastAction"] == "MoveBack":
                step = np.array([-MOVE_STEP, 0.0])
            elif event.metadata["lastAction"] == "MoveRight":
                step = np.array([0.0, -MOVE_STEP])
            else:
                raise ValueError("Wrong Action")
            
            theta = latest_pose["rot_3d_enu_deg"][2] / 180.0 * np.pi # [rad]
            ct = np.cos(theta)
            st = np.sin(theta)
            rot_mat = np.array(
                [
                    [ct, -st],
                    [st, ct]
                ]
            )
            self.latest_pose_rel["cam_pos_enu"][0:2] += (rot_mat @ step.reshape(-1, 1)).reshape(-1)

    def get_pose_mat(self):
        # Rotation from unity to world frame
        T_unity_to_world = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ]
        )

        # Translation from world origin to camera/agent position
        T_world_to_agent_pos = np.eye(4)
        T_world_to_agent_pos[:3, 3] = self.latest_pose["cam_pos_enu"]

        # ... rotation to agent frame (x-forward, y-left, z-up)
        T_agent_pos_to_agent = np.eye(4)
        T_agent_pos_to_agent[:3, :3] = euler.euler2mat(*np.radians(self.latest_pose["rot_3d_enu_deg"]))

        # ... transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch
        T_agent_to_camflat = np.eye(4)
        T_agent_to_camflat[:3, :3] = euler.euler2mat(*np.radians([-90, 0, -90]))

        # ... transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch
        T_camflat_to_cam = np.eye(4)
        T_camflat_to_cam[:3, :3] = euler.euler2mat(*np.radians([-self.latest_pose["cam_horizon_deg"], 0, 0]))

        # compose into a transform from world to camera
        T_world_to_cam = T_world_to_agent_pos @ T_agent_pos_to_agent @ T_agent_to_camflat @ T_camflat_to_cam

        return T_world_to_cam, T_unity_to_world

    def get_agent_pos(self):
        agent_pos = np.array(
            [
                self.latest_pose["cam_pos_enu"][0],
                self.latest_pose["cam_pos_enu"][1],
                self.latest_pose["cam_pos_enu"][2]
            ]
        )

        return agent_pos

    def get_rel_pose_mat(self):
        # Rotation from unity to world frame
        T_unity_to_world = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ]
        )
        T_unity_to_world_rel = np.eye(4)
        T_unity_to_world_rel[:3, :3] = euler.euler2mat(*np.radians(self.initial_pose_rel['rotation'])) @ T_unity_to_world[:3, :3]
        T_unity_to_world_rel[:3, 3] = self.initial_pose_rel['position']

        # Translation from world origin to camera/agent position
        T_world_to_agent_pos = np.eye(4)
        T_world_to_agent_pos[:3, 3] = self.latest_pose_rel["cam_pos_enu"]

        # ... rotation to agent frame (x-forward, y-left, z-up)
        T_agent_pos_to_agent = np.eye(4)
        T_agent_pos_to_agent[:3, :3] = euler.euler2mat(*np.radians(self.latest_pose_rel["rot_3d_enu_deg"]))

        # ... transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch
        T_agent_to_camflat = np.eye(4)
        T_agent_to_camflat[:3, :3] = euler.euler2mat(*np.radians([-90, 0, -90]))

        # ... transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch
        T_camflat_to_cam = np.eye(4)
        T_camflat_to_cam[:3, :3] = euler.euler2mat(*np.radians([-self.latest_pose_rel["cam_horizon_deg"], 0, 0]))

        # compose into a transform from world to camera
        T_world_to_cam = T_world_to_agent_pos @ T_agent_pos_to_agent @ T_agent_to_camflat @ T_camflat_to_cam

        return T_world_to_cam, T_unity_to_world_rel

    def get_agent_pos_rel(self):
        agent_pos_rel = np.array(
            [
                self.latest_pose_rel["cam_pos_enu"][0],
                self.latest_pose_rel["cam_pos_enu"][1],
                self.latest_pose_rel["cam_pos_enu"][2]
            ]
        )

        return agent_pos_rel