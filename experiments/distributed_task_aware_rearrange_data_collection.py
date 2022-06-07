from typing import Tuple, Sequence, Optional, Dict, Any, List, Union
import gym
import numpy as np

import torch
from torch import nn, cuda, optim
import ai2thor
import copy

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
    Builder
)
from custom.models import TaskAwareRearrangeDataCollectionModel
from custom.preprocessors import SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor
from datagen import datagen_utils
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, TrainingPipeline
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from rearrange.sensors import RGBRearrangeSensor, UnshuffledRGBRearrangeSensor, DepthRearrangeSensor
from custom.sensors import PoseSensor, UnshuffledInstanceSegmentationSensor, UnshuffledPoseSensor, UnshuffledDepthRearrangeSensor, InventoryObjectSensor, SemanticSegmentationSensor, UnshuffledSemanticSegmentationSensor, InstanceSegmentationSensor
from custom.expert import OnePhaseSubtaskAndActionExpertSensor
from custom.constants import NUM_SUBTASK_TYPES, NUM_OBJECT_TYPES, NUM_MAP_TYPES ,MAP_TYPES, SUBTASK_TYPES
from experiments.distributed_one_phase_task_aware_rearrange_base import DistributedOnePhaseTaskAwareRearrangeBaseExperimentConfig


class DistributedTaskAwareRearrangeDataCollectionExperimentConfig(DistributedOnePhaseTaskAwareRearrangeBaseExperimentConfig):

    NUM_DISTRIBUTED_NODES: int = 4
    NUM_DEVICES: Sequence[int] = [1, 2, 2, 2]
    NUM_PROCESSES: Union[int, Sequence[int]] = [8, 6, 6, 6]
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = None

    EXPERT_SUBTASK_ACTION_UUID = "expert_subtask_action"
    EXPERT_ACTION_UUID = "expert_action"
    EXPERT_SUBTASK_UUID = "expert_subtask"

    INSTANCE_SEGMENTATION_UUID = "instseg"
    UNSHUFFLED_INSTANCE_SEGMENTATION_UUID = "unshuffled_instseg"
    
    @classmethod
    def sensors(cls):
        sensors = [
            RGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=False,
                uuid=cls.EGOCENTRIC_RGB_UUID,
            ),
            UnshuffledRGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=False,
                uuid=cls.UNSHUFFLED_RGB_UUID,
            ),
            PoseSensor(
                reference_pose=cls.REFERENCE_POSE,
                uuid=cls.POSE_UUID,
            ),
            UnshuffledPoseSensor(
                reference_pose=cls.REFERENCE_POSE,
                uuid=cls.UNSHUFFLED_POSE_UUID,
            ),
            InventoryObjectSensor(
                reference_inventory=cls.REFERENCE_INVENTORY, 
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                uuid=cls.INVENTORY_UUID,
            ),
            DepthRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.DEPTH_UUID,
                use_normalization=cls.DEPTH_NORMALIZATION,
            ),
            UnshuffledDepthRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.UNSHUFFLED_DEPTH_UUID,
                use_normalization=cls.DEPTH_NORMALIZATION,
            ),
            SemanticSegmentationSensor(
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.SEMANTIC_SEGMENTATION_UUID,
            ),
            UnshuffledSemanticSegmentationSensor(
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID,
            ),
            InstanceSegmentationSensor(
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.INSTANCE_SEGMENTATION_UUID,
            ),
            UnshuffledInstanceSegmentationSensor(
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.UNSHUFFLED_INSTANCE_SEGMENTATION_UUID,
            ),
            OnePhaseSubtaskAndActionExpertSensor(
                action_space=(
                    (len(SUBTASK_TYPES) - 1) * NUM_OBJECT_TYPES * len(MAP_TYPES) + 1,
                    len(cls.actions()),
                ),
                uuid=cls.EXPERT_SUBTASK_ACTION_UUID,
                # verbose=True,
            ),
        ]

        return sensors

    @classmethod
    def create_subtask_action_expert_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SubtaskActionExpertPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            device=cls.DEVICE,            
        )

    @classmethod
    def create_subtask_expert_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SubtaskExpertPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            device=cls.DEVICE,
        )
    
    @classmethod
    def create_preprocessor_graph(cls, mode: str) -> SensorPreprocessorGraph:
        preprocessors = [
            cls.create_semantic_map_preprocessor_builder(
                in_uuids=[
                    cls.SEMANTIC_SEGMENTATION_UUID, 
                    cls.DEPTH_UUID, 
                    cls.POSE_UUID
                ],
                out_uuid=cls.SEMANTIC_MAP_UUID,
            ),
            cls.create_semantic_map_preprocessor_builder(
                in_uuids=[
                    cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID, 
                    cls.UNSHUFFLED_DEPTH_UUID,
                    cls.UNSHUFFLED_POSE_UUID
                ],
                out_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID
            ),
            cls.create_subtask_action_expert_preprocessor_builder(
                in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                out_uuid=cls.EXPERT_ACTION_UUID,
            ),
            cls.create_subtask_expert_preprocessor_builder(
                in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                out_uuid=cls.EXPERT_SUBTASK_UUID,
            ),
        ]
        additional_output_uuids = [
            "semseg", "depth", "pose", 
            "unshuffled_semseg", "unshuffled_depth", "unshuffled_pose"
        ]

        if not cls.REFERENCE_DEPTH:
            # TODO: Implement Depth Inference Model
            pass

        if not cls.REFERENCE_SEGMENTATION:
            # TODO: Implement Segmentation Inference Model
            preprocessors.append(
                cls.create_hlsm_segmentation_builder(
                    in_uuid=cls.EGOCENTRIC_RAW_RGB_UUID,
                    out_uuid=cls.SEMANTIC_SEGMENTATION_UUID
                )
            )
            preprocessors.append(
                cls.create_hlsm_segmentation_builder(
                    in_uuid=cls.UNSHUFFLED_RAW_RGB_UUID,
                    out_uuid=cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID
                )
            )

        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            preprocessors.append(
                cls.create_resnet_bulder(
                    in_uuid=cls.EGOCENTRIC_RGB_UUID,
                    out_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                )
            )
            preprocessors.append(
                cls.create_resnet_bulder(
                    in_uuid=cls.UNSHUFFLED_RGB_UUID,
                    out_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
                )
            )

        return (
            None
            if len(preprocessors) == 0
            else Builder(
                SensorPreprocessorGraph,
                {
                    "source_observation_spaces": SensorSuite(cls.sensors()).observation_spaces,
                    "preprocessors": preprocessors,
                    "additional_output_uuids": additional_output_uuids,
                }
            )
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> MachineParams:
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        nprocesses = 1
        devices = (
            list(range(min(nprocesses, num_gpus)))
            if has_gpu
            else [torch.device("cpu")]
        )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )
        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=cls.create_preprocessor_graph(mode=mode),
            # visualizer=self.create_visualizer(mode=self.mode),
        )
        
        if isinstance(cls.NUM_PROCESSES, int):
            num_processes = [[cls.NUM_PROCESSES] for _ in range(cls.NUM_DISTRIBUTED_NODES)]
        else:
            num_processes = cls.NUM_PROCESSES

        devices = sum(
            [
                list(range(min(num_processes[idx], cls.NUM_DEVICES[idx])))
                if cls.NUM_DEVICES[idx] > 0 and cuda.is_available()
                else torch.device("cpu")
                for idx in range(cls.NUM_DISTRIBUTED_NODES)
            ], []
        )
        params.devices = tuple(
            torch.device("cpu") if d == -1 else torch.device(d) for d in devices
        )
        params.nprocesses = sum(
            [
                split_processes_onto_devices(
                    num_processes[idx] if cuda.is_available() and cls.NUM_DEVICES[idx] > 0 else 1,
                    cls.NUM_DEVICES[idx] if cls.NUM_DEVICES[idx] > 0 else 1
                )
                for idx in range(cls.NUM_DISTRIBUTED_NODES)
            ], []
        )

        if "machine_id" in kwargs:
            machine_id = kwargs["machine_id"]
            assert (
                0 <= machine_id < cls.NUM_DISTRIBUTED_NODES
            ), f"machine_id {machine_id} out of range [0, {cls.NUM_DISTRIBUTED_NODES - 1}."
            machine_num_gpus = cuda.device_count()
            machine_has_gpu = machine_num_gpus != 0
            assert (
                0 <= cls.NUM_DEVICES[machine_id] <= machine_num_gpus
            ), f"Number of devices for machine_id {machine_id} exceeds the number of gpus."

            local_worker_ids = list(
                range(
                    sum(cls.NUM_DEVICES[:machine_id]),
                    sum(cls.NUM_DEVICES[:machine_id + 1])
                )
            )
        
            params.set_local_worker_ids(local_worker_ids)

        return params

    @classmethod
    def training_pipeline(cls) -> TrainingPipeline:
        """Define how the model trains."""
        return None

    @classmethod
    def stagewise_task_sampler_args(
        self,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            # Split scenes more evenly as the train scenes will have more episodes
            train_scenes = datagen_utils.get_scenes("train")
            other_scenes = datagen_utils.get_scenes("val") + datagen_utils.get_scenes(
                "test"
            )
            assert len(train_scenes) == 2 * len(other_scenes)
            scenes = []
            while len(train_scenes) != 0:
                scenes.append(train_scenes.pop())
                scenes.append(train_scenes.pop())
                scenes.append(other_scenes.pop())
            assert len(train_scenes) == len(other_scenes)
        else:
            scenes = datagen_utils.get_scenes(stage)

        if total_processes > len(scenes):
            assert stage == "train" and total_processes % len(scenes) == 0, (
                f"stage {stage} should be equal to 'train' and total_processes {total_processes} should be multiple of "
                f"len(scenes) {len(scenes)}: total_processes % len(scenes) = {total_processes % len(scenes)}"
            )
            scenes = scenes * (total_processes // len(scenes))

        allowed_scenes = list(
            sorted(partition_sequence(seq=scenes, parts=total_processes,)[process_ind])
        )

        scene_to_allowed_rearrange_inds = None
        if allowed_rearrange_inds_subset is not None:
            allowed_rearrange_inds_subset = tuple(allowed_rearrange_inds_subset)
            assert stage in ["valid", "train_unseen"]
            scene_to_allowed_rearrange_inds = {
                scene: allowed_rearrange_inds_subset for scene in allowed_scenes
            }
        seed = md5_hash_str_as_int(str(allowed_scenes))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            # else torch.device("cpu")
            else None
        )
        x_display: Optional[str] = None
        gpu_device: Optional[int] = device
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if self.HEADLESS:
            thor_platform = ai2thor.platform.CloudRendering

        kwargs = {
            "stage": stage,
            "allowed_scenes": allowed_scenes,
            "scene_to_allowed_rearrange_inds": scene_to_allowed_rearrange_inds,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }

        sensors = kwargs.get("sensors", copy.deepcopy(self.sensors()))
        kwargs["sensors"] = sensors
        kwargs["epochs"] = 1
        kwargs["force_cache_reset"] = True

        return kwargs
        
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        return TaskAwareRearrangeDataCollectionModel(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID 
            if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None 
            else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID 
            if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None 
            else cls.UNSHUFFLED_RGB_RESNET_UUID,
            inventory_uuid=cls.INVENTORY_UUID,
            expert_action_uuid=cls.EXPERT_ACTION_UUID,
            expert_subtask_uuid=cls.EXPERT_SUBTASK_UUID,
            sem_map_uuid=cls.SEMANTIC_MAP_UUID,
            unshuffled_sem_map_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID,
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
        )
    @classmethod
    def tag(cls) -> str:
        return f"DistributedTaskAwareDataCollection"