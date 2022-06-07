from typing import Tuple, Sequence, Optional, Dict, Any, Type
import gym
import numpy as np

import torch
from torch import nn, cuda, optim
import torchvision

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.experiment_config import (
    MachineParams,
    split_processes_onto_devices,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from baseline_configs.one_phase.one_phase_rgb_base import OnePhaseRGBBaseExperimentConfig
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, UnshuffledRGBRearrangeSensor
from custom.sensors import InventoryObjectSensor, PoseSensor, SemanticSegmentationSensor, UnshuffledDepthRearrangeSensor, UnshuffledPoseSensor, UnshuffledSemanticSegmentationSensor
from custom.preprocessors import HLSMSegmentationPreprocessor, Semantic3DMapPreprocessor
from custom.hlsm.voxel_grid import GridParameters
from custom.models import TaskAwareOnePhaseRearrangeBaseNetwork
from custom.losses import SubtaskPredictionLoss


class TaskAwareRearrangeBaseExperimentConfig(RearrangeBaseExperimentConfig):
    # CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = None
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))

    # Sensor Info
    REFERENCE_DEPTH = True
    REFERENCE_SEGMENTATION = True
    REFERENCE_POSE = False
    REFERENCE_INVENTORY = False

    FOV = 90
    GRID_PARAMETERS = GridParameters()

    EGOCENTRIC_RAW_RGB_UUID = "rgb_raw"
    UNSHUFFLED_RAW_RGB_UUID = "unshuffled_rgb_raw"
    DEPTH_UUID = "depth"
    UNSHUFFLED_DEPTH_UUID = "unshuffled_depth"
    SEMANTIC_SEGMENTATION_UUID = "semseg"
    UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID = "unshuffled_semseg"
    POSE_UUID = "pose"
    UNSHUFFLED_POSE_UUID = "unshuffled_pose"
    INVENTORY_UUID = "inventory"
    SEMANTIC_MAP_UUID = "semmap"
    UNSHUFFLED_SEMANTIC_MAP_UUID = "unshuffled_semmap"
    SUBTASK_EXPERT_UUID = "subtask_expert"

    # Model parameters
    IS_WALKTHROUGH_PHASE_EMBEDING_DIM: int = 32
    RNN_TYPE: str = "LSTM"

    RGB_NORMALIZATION = False
    DEPTH_NORMALIZATION = False

    # Environment parameters
    THOR_CONTROLLER_KWARGS = {
        **RearrangeBaseExperimentConfig.THOR_CONTROLLER_KWARGS,
        "renderDepthImage": REFERENCE_DEPTH,
        "renderSemanticSegmentation": REFERENCE_SEGMENTATION,
        "renderInstanceSegmentation": REFERENCE_SEGMENTATION,
    }
    HEADLESS = True

    # Training parameters
    DEVICE = torch.device('cuda')

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        mean, stdev = None, None
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
            if pretraining_type.strip().lower() == "clip":
                from allenact_plugins.clip_plugin.clip_preprocessors import (
                    ClipResNetPreprocessor,
                )

                mean = ClipResNetPreprocessor.CLIP_RGB_MEANS
                stdev = ClipResNetPreprocessor.CLIP_RGB_STDS
            else:
                mean = IMAGENET_RGB_MEANS
                stdev = IMAGENET_RGB_STDS

        sensors = [
            RGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=cls.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            UnshuffledRGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=cls.UNSHUFFLED_RGB_UUID,
                mean=mean,
                stdev=stdev,
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
        ]

        if cls.REFERENCE_DEPTH:
            sensors.append(
                DepthRearrangeSensor(
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    uuid=cls.DEPTH_UUID,
                    use_normalization=cls.DEPTH_NORMALIZATION,
                )
            )
            sensors.append(
                UnshuffledDepthRearrangeSensor(
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    uuid=cls.UNSHUFFLED_DEPTH_UUID,
                    use_normalization=cls.DEPTH_NORMALIZATION,
                )
            )

        if cls.REFERENCE_SEGMENTATION:
            sensors.append(
                SemanticSegmentationSensor(
                    ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    uuid=cls.SEMANTIC_SEGMENTATION_UUID,
                )
            )
            sensors.append(
                UnshuffledSemanticSegmentationSensor(
                    ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    uuid=cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID,
                )
            )
        else:
            # add raw rgb sensors to infer semantic segmentation masks
            sensors.append(
                RGBRearrangeSensor(
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    use_resnet_normalization=False,
                    uuid=cls.EGOCENTRIC_RAW_RGB_UUID,
                )
            )
            sensors.append(
                UnshuffledRGBRearrangeSensor(
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    use_resnet_normalization=False,
                    uuid=cls.UNSHUFFLED_RAW_RGB_UUID,
                )
            )
        
        return sensors
    
    @classmethod
    def create_hlsm_segmentation_builder(
        cls,
        in_uuid: str,
        out_uuid: str,
    ):
        return HLSMSegmentationPreprocessor(
            input_uuids=[in_uuid],
            output_uuid=out_uuid,
            input_height=cls.SCREEN_SIZE,
            input_width=cls.SCREEN_SIZE,
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
            device=cls.DEVICE,
        )
    
    @classmethod
    def create_semantic_map_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return Semantic3DMapPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            fov=cls.FOV,
            grid_parameters=cls.GRID_PARAMETERS,
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
            device=cls.DEVICE,            
        )

    @classmethod
    def create_resnet_bulder(
        cls,
        in_uuid: str,
        out_uuid: str,
    ):
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            raise NotImplementedError
        
        cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
        if pretraining_type == "imagenet":
            assert cnn_type in (
                "RN18",
                "RN50",
            ), "Only allow using RN18/RN50 with 'imagenet' pretrained weights."

            return ResNetPreprocessor(
                input_height=cls.THOR_CONTROLLER_KWARGS["height"],
                input_width=cls.THOR_CONTROLLER_KWARGS["width"],
                output_width=7,
                output_height=7,
                output_dims=512 if "18" in cnn_type else 2048,
                pool=False,
                torchvision_resnet_model=getattr(
                    torchvision.models, f"resnet{cnn_type.replace('RN', '')}"
                ),
                input_uuids=[in_uuid],
                output_uuid=out_uuid,
                device=cls.DEVICE,
            )
        elif pretraining_type == "clip":
            from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
            import clip

            clip.load(cnn_type, "cpu")

            return ClipResNetPreprocessor(
                rgb_input_uuid=in_uuid,
                clip_model_type=cnn_type,
                pool=False,
                output_uuid=out_uuid,
                device=cls.DEVICE,
            )
        else:
            raise NotImplementedError
        

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
        ]
        additional_output_uuids = []

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
        """Return the number of processes and gpu_ids to use with training."""
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = cls.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = cls.num_valid_processes() if has_gpu else 0
        else:
            nprocesses = cls.num_test_processes() if has_gpu else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=cls.create_preprocessor_graph(mode=mode),
        )
