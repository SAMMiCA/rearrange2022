from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
import gym
import numpy as np
import copy
import platform

import torch
from torch import nn, cuda, optim
import torchvision

import ai2thor
import ai2thor.platform
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.base_abstractions.experiment_config import (
    MachineParams,
    split_processes_onto_devices,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.system import get_logger
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from baseline_configs.one_phase.one_phase_rgb_base import OnePhaseRGBBaseExperimentConfig
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, UnshuffledRGBRearrangeSensor

import datagen.datagen_utils as datagen_utils
from rearrange.tasks import RearrangeTaskSampler

from task_aware_rearrange.expert import OnePhaseSubtaskAndActionExpertSensor
from task_aware_rearrange.preprocessors import SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor
from task_aware_rearrange.sensors import InventoryObjectSensor, PoseSensor, SemanticSegmentationSensor, UnshuffledDepthRearrangeSensor, UnshuffledPoseSensor, UnshuffledSemanticSegmentationSensor
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from task_aware_rearrange.utils import get_open_x_displays
from task_aware_rearrange.voxel_utils import GridParameters
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor
from task_aware_rearrange.models import (
    OnePhaseSemanticMappingWithInventorySubtaskHistoryActorCriticRNN,
    OnePhaseTaskAwareActorCriticRNN,
)


class ExpertTestExpConfig(RearrangeBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    NUM_PROCESSES: int = 4
    ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))

    # Sensor Info
    REQUIRE_SEMANTIC_3D_MAP = True
    REQUIRE_SEMANTIC_SEGMENTATION = True
    REQUIRE_EXPERTS = True
    REQUIRE_INVENTORY = True

    REFERENCE_DEPTH = True
    REFERENCE_SEGMENTATION = True
    REFERENCE_POSE = False
    REFERENCE_INVENTORY = False

    EXPERT_SUBTASK_ACTION_UUID = "expert_subtask_action"
    EXPERT_ACTION_UUID = "expert_action"
    EXPERT_SUBTASK_UUID = "expert_subtask"

    EGOCENTRIC_RAW_RGB_UUID = "raw_rgb"
    UNSHUFFLED_RAW_RGB_UUID = "w_raw_rgb"

    DEPTH_UUID = "u_depth"
    UNSHUFFLED_DEPTH_UUID = "w_depth"

    SEMANTIC_SEGMENTATION_UUID = "u_semseg"
    UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID = "w_semseg"

    POSE_UUID = "u_pose"
    UNSHUFFLED_POSE_UUID = "w_pose"

    INVENTORY_UUID = "inventory"

    SEMANTIC_MAP_UUID = "semmap"
    UNSHUFFLED_SEMANTIC_MAP_UUID = "unshuffled_semmap"

    # Model parameters
    PREV_ACTION_EMBEDDING_DIM: int = 32
    RNN_TYPE: str = "LSTM"
    NUM_RNN_LAYERS: int = 1
    HIDDEN_SIZE: int = 512

    DEPTH_NORMALIZATION = False

    # Environment parameters
    THOR_CONTROLLER_KWARGS = {
        **RearrangeBaseExperimentConfig.THOR_CONTROLLER_KWARGS,
        "renderDepthImage": REFERENCE_DEPTH,
        "renderSemanticSegmentation": (REQUIRE_SEMANTIC_SEGMENTATION and REFERENCE_SEGMENTATION),
        "renderInstanceSegmentation": (REQUIRE_SEMANTIC_SEGMENTATION and REFERENCE_SEGMENTATION),
    }
    HEADLESS = True

    FOV = 90
    GRID_PARAMETERS = GridParameters()

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
            PoseSensor(
                reference_pose=cls.REFERENCE_POSE,
                uuid=cls.POSE_UUID,
            ),
            UnshuffledPoseSensor(
                reference_pose=cls.REFERENCE_POSE,
                uuid=cls.UNSHUFFLED_POSE_UUID,
            ),
        ]

        if cls.REQUIRE_EXPERTS:
            sensors.append(
                OnePhaseSubtaskAndActionExpertSensor(
                    action_space=(
                        NUM_SUBTASKS,
                        len(cls.actions()),
                    ),
                    uuid=cls.EXPERT_SUBTASK_ACTION_UUID,
                    verbose=False,
                )
            )            

        if cls.REQUIRE_SEMANTIC_SEGMENTATION:
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

        if cls.REQUIRE_INVENTORY:
            sensors.append(
                InventoryObjectSensor(
                    reference_inventory=cls.REFERENCE_INVENTORY, 
                    ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                    uuid=cls.INVENTORY_UUID,
                )
            )
        return sensors

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
    def preprocessors(cls) -> Sequence[Preprocessor]:
        preprocessors = []
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
        
        if cls.REQUIRE_EXPERTS:
            preprocessors.append(
                cls.create_subtask_action_expert_preprocessor_builder(
                    in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                    out_uuid=cls.EXPERT_ACTION_UUID,
                )
            )
            preprocessors.append(
                cls.create_subtask_expert_preprocessor_builder(
                    in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                    out_uuid=cls.EXPERT_SUBTASK_UUID,
                )
            )

        if cls.REQUIRE_SEMANTIC_3D_MAP:
            preprocessors.append(
                cls.create_semantic_map_preprocessor_builder(
                    in_uuids=[
                        cls.SEMANTIC_SEGMENTATION_UUID, 
                        cls.DEPTH_UUID, 
                        cls.POSE_UUID
                    ],
                    out_uuid=cls.SEMANTIC_MAP_UUID,
                )
            )
            preprocessors.append(
                cls.create_semantic_map_preprocessor_builder(
                    in_uuids=[
                        cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID, 
                        cls.UNSHUFFLED_DEPTH_UUID,
                        cls.UNSHUFFLED_POSE_UUID
                    ],
                    out_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID
                )
            )
        
        return preprocessors

    @classmethod
    def create_preprocessor_graph(cls, mode: str) -> SensorPreprocessorGraph:
        additional_output_uuids = []

        return (
            None
            if len(cls.preprocessors()) == 0
            else Builder(
                SensorPreprocessorGraph,
                {
                    "source_observation_spaces": SensorSuite(cls.sensors()).observation_spaces,
                    "preprocessors": cls.preprocessors(),
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
        nprocesses = cls.NUM_PROCESSES if torch.cuda.is_available() else 1
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

    @classmethod
    def stagewise_task_sampler_args(
        cls,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False
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
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if cls.HEADLESS:
            gpu_device = device
            thor_platform = ai2thor.platform.CloudRendering

        elif platform.system() == "Linux":
            try:
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if devices is not None and len(
                    [d for d in devices if d != torch.device("cpu")]
                ) > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]
            except IOError:
                # Could not find an open `x_display`, use CloudRendering instead.
                assert all(
                    [d != torch.device("cpu") and d >= 0 for d in devices]
                ), "Cannot use CPU devices when there are no open x-displays as CloudRendering requires specifying a GPU."
                gpu_device = device
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

        sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors()))
        kwargs["sensors"] = sensors

        return kwargs

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            **kwargs,
        )

    @classmethod
    def num_train_processes(cls) -> int:
        return 2

    @classmethod
    def num_valid_processes(cls) -> int:
        return 0

    @classmethod
    def num_test_processes(cls) -> int:
        return 1

    @classmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        training_steps = cls.TRAINING_STEPS

        return dict(
            named_losses=dict(
                ppo_loss=PPO(clip_decay=LinearDecay(training_steps), **PPOConfig),
            ),
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=training_steps,)
            ],
            num_steps=64,
            num_mini_batch=1,
            update_repeats=3,
            use_lr_decay=True,
            lr=3e-4,
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseSemanticMappingWithInventorySubtaskHistoryActorCriticRNN(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.UNSHUFFLED_RGB_RESNET_UUID,
            prev_action_embedding_dim=cls.PREV_ACTION_EMBEDDING_DIM,
            hidden_size=cls.HIDDEN_SIZE,
            num_rnn_layers=cls.NUM_RNN_LAYERS,
            rnn_type=cls.RNN_TYPE,
            sem_map_uuid=cls.SEMANTIC_MAP_UUID,
            unshuffled_sem_map_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID,
            inventory_uuid=cls.INVENTORY_UUID,
            num_repeats=cls.training_pipeline().training_settings.update_repeats,
            expert_subtask_uuid=cls.EXPERT_SUBTASK_UUID
        )