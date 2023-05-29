from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym
import os

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from baseline_configs.one_phase.one_phase_rgb_il_base import StepwiseLinearDecay

from experiments.one_phase.one_phase_ta_il_base import OnePhaseTaskAwareRearrangeILBaseExperimentConfig
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, UnshuffledRGBRearrangeSensor
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from task_aware_rearrange.models import OnePhaseSubtaskAwarePolicy
from task_aware_rearrange.sensors import PoseSensor, UnshuffledDepthRearrangeSensor, UnshuffledPoseSensor
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor
from task_aware_rearrange.losses import SubtaskPredictionLoss
from task_aware_rearrange.expert import OnePhaseSubtaskAndActionExpertSensor
from task_aware_rearrange.constants import NUM_OBJECT_TYPES, ADDITIONAL_MAP_CHANNELS
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from task_aware_rearrange.voxel_utils import GridParameters
from semseg.semseg_sensors import SemanticRearrangeSensor
from semseg.semseg_preprocessors import SemanticPreprocessor
from semseg.semseg_constants import CLASS_TO_COLOR_ORIGINAL, ORDERED_CLASS_TO_COLOR, ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE


class OnePhaseAblationExerimentConfig(OnePhaseTaskAwareRearrangeILBaseExperimentConfig):
    
    TRAINING_STEPS = int(25e6)
    SAVE_INTERVAL = int(5e5)
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "10proc"
    
    SAP_SUBTASK_HISTORY: bool = False
    SAP_SEMANTIC_MAP: bool = False
    
    REQUIRE_SEMANTIC_SEGMENTATION: bool = True if SAP_SEMANTIC_MAP else False
    REFERENCE_SEGMENTATION = False
    
    SEMANTIC_SEGMENTATION_USE_MASS = True
    if SEMANTIC_SEGMENTATION_USE_MASS:
        CLASS_TO_COLOR = CLASS_TO_COLOR_ORIGINAL
        ORDERED_OBJECT_TYPES = list(CLASS_TO_COLOR.keys())
        CLASS_MAPPING = ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE
    else:
        CLASS_TO_COLOR = ORDERED_CLASS_TO_COLOR
        CLASS_MAPPING = list(range(len(CLASS_TO_COLOR)))
    DETECTION_THRESHOLD = 0.8
    SEMANTIC_SEGMENTATION_BASE = "mask-rcnn"
    SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
        "semseg",
        "checkpoints",
        "model_final.pth"
    )
    
    RGB_NORMALIZATION: bool = True
    DEPTH_NORMALIZATION: bool = False
    
    SEMANTIC_SEGMENTATION_UUID: str = "semantic"
    UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID: str = "unshuffled_semantic"
    NUM_SEMANTIC_CLASSES: int = NUM_OBJECT_TYPES
    NUM_ADDITIONAL_MAP_CHANNELS: int = ADDITIONAL_MAP_CHANNELS
    NUM_MAP_CHANNELS: int = NUM_SEMANTIC_CLASSES + NUM_ADDITIONAL_MAP_CHANNELS
    
    FOV = 90
    GRID_PARAMETERS = GridParameters()
    
    ONLINE_SUBTASK_PREDICTION: bool = False
    ONLINE_SUBTASK_PREDICTION_USE_EGOVIEW: bool = False
    ONLINE_SUBTASK_PREDICTION_USE_PREV_ACTION: bool = False
    ONLINE_SUBTASK_PREDICTION_USE_SUBTASK_HISTORY: bool = False
    ONLINE_SUBTASK_PREDICTION_USE_SEMANTIC_MAP: bool = False
    
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
                use_resnet_normalization=cls.RGB_NORMALIZATION,
                uuid=cls.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            UnshuffledRGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=cls.RGB_NORMALIZATION,
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
            OnePhaseSubtaskAndActionExpertSensor(
                action_space=(
                    NUM_SUBTASKS,
                    len(cls.actions()),
                ),
                uuid=cls.EXPERT_SUBTASK_ACTION_UUID,
                verbose=False,
            ),
        ]

        if cls.REQUIRE_SEMANTIC_SEGMENTATION:
            if cls.REFERENCE_SEGMENTATION:
                sensors.append(
                    SemanticRearrangeSensor(
                        ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                        class_to_color=cls.CLASS_TO_COLOR,
                        height=cls.SCREEN_SIZE,
                        width=cls.SCREEN_SIZE,
                        uuid=cls.SEMANTIC_SEGMENTATION_UUID,
                    )
                )
                sensors.append(
                    SemanticRearrangeSensor(
                        ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                        class_to_color=cls.CLASS_TO_COLOR,
                        height=cls.SCREEN_SIZE,
                        width=cls.SCREEN_SIZE,
                        uuid=cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID,
                        which_task_env="walkthrough",
                    )
                )
            elif cls.RGB_NORMALIZATION:
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
    def create_maskrcnn_semseg_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SemanticPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
            class_to_color=cls.CLASS_TO_COLOR,
            class_mapping=cls.CLASS_MAPPING,
            detection_threshold=cls.DETECTION_THRESHOLD,
            model_weight_path=cls.SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH,
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
            num_semantic_classes=cls.NUM_SEMANTIC_CLASSES,
            num_additional_channels=cls.NUM_ADDITIONAL_MAP_CHANNELS,
            device=cls.DEVICE,
        )
    
    @classmethod
    def preprocessors(cls) -> Sequence[Preprocessor]:
        preprocessors: List = super().preprocessors()
        
        if cls.SAP_SEMANTIC_MAP:
            preprocessors.append(
                cls.create_semantic_map_preprocessor_builder(
                    in_uuids=[
                        cls.SEMANTIC_SEGMENTATION_UUID,
                        cls.DEPTH_UUID,
                        cls.POSE_UUID,
                    ],
                    out_uuid=cls.SEMANTIC_MAP_UUID,
                )
            )
            preprocessors.append(
                cls.create_semantic_map_preprocessor_builder(
                    in_uuids=[
                        cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID,
                        cls.UNSHUFFLED_DEPTH_UUID,
                        cls.UNSHUFFLED_POSE_UUID,
                    ],
                    out_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID,
                )
            )

        if (
            cls.REQUIRE_SEMANTIC_SEGMENTATION
            and not cls.REFERENCE_SEGMENTATION
        ):
            if cls.RGB_NORMALIZATION:
                u_in_uuids = [cls.EGOCENTRIC_RAW_RGB_UUID]
                w_in_uuids = [cls.UNSHUFFLED_RAW_RGB_UUID]
            else:
                u_in_uuids = [cls.EGOCENTRIC_RGB_UUID]
                w_in_uuids = [cls.UNSHUFFLED_RGB_UUID]
            u_out_uuid = cls.SEMANTIC_SEGMENTATION_UUID
            w_out_uuid = cls.UNSHUFFLED_SEMANTIC_SEGMENTATION_UUID

            if cls.SEMANTIC_SEGMENTATION_BASE == "mask-rcnn":
                preprocessors.append(
                    cls.create_maskrcnn_semseg_preprocessor_builder(
                        in_uuids=u_in_uuids,
                        out_uuid=u_out_uuid,
                    )
                )
                preprocessors.append(
                    cls.create_maskrcnn_semseg_preprocessor_builder(
                        in_uuids=w_in_uuids,
                        out_uuid=w_out_uuid,
                    )
                )
            elif cls.SEMANTIC_SEGMENTATION_BASE == "solq":
                pass
        
        return preprocessors
    
    @classmethod
    def create_preprocessor_graph(cls, mode: str, additional_output_uuids: Sequence[str] = []) -> SensorPreprocessorGraph:
        sensor_preprocessor_graph = (
            None
            if len(cls.preprocessors()) == 0
            else SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(cls.sensors()).observation_spaces,
                preprocessors=cls.preprocessors(),
                additional_output_uuids=additional_output_uuids,
            )
        )
        sensor_preprocessor_graph.compute_order = list(reversed(sensor_preprocessor_graph.compute_order))
        
        return sensor_preprocessor_graph
    
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseSubtaskAwarePolicy(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID 
            if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None 
            else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID 
            if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None
            else cls.UNSHUFFLED_RGB_RESNET_UUID,
            expert_subtask_uuid=cls.EXPERT_SUBTASK_UUID,
            prev_action_embedding_dim=cls.PREV_ACTION_EMBEDDING_DIM,
            sap_subtask_history=cls.SAP_SUBTASK_HISTORY,
            sap_semantic_map=cls.SAP_SEMANTIC_MAP,
            semantic_map_uuid=cls.SEMANTIC_MAP_UUID
            if cls.SAP_SEMANTIC_MAP else None,
            unshuffled_semantic_map_uuid=cls.UNSHUFFLED_SEMANTIC_MAP_UUID
            if cls.SAP_SEMANTIC_MAP else None,
            num_map_channels=cls.NUM_MAP_CHANNELS
            if cls.SAP_SEMANTIC_MAP else None,
            online_subtask_prediction=cls.ONLINE_SUBTASK_PREDICTION,
            osp_egoview=cls.ONLINE_SUBTASK_PREDICTION_USE_EGOVIEW,
            osp_prev_action=cls.ONLINE_SUBTASK_PREDICTION_USE_PREV_ACTION,
            osp_subtask_history=cls.ONLINE_SUBTASK_PREDICTION_USE_SUBTASK_HISTORY,
            osp_semantic_map=cls.ONLINE_SUBTASK_PREDICTION_USE_SEMANTIC_MAP,
            num_repeats=cls.training_pipeline().training_settings.update_repeats,
            hidden_size=cls.HIDDEN_SIZE,
            num_rnn_layers=cls.NUM_RNN_LAYERS,
            rnn_type=cls.RNN_TYPE,
        )
        
    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""
        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]
        
        named_losses = dict(
            imitation_loss=Imitation(),
        )
        if cls.ONLINE_SUBTASK_PREDICTION:
            named_losses["subtask_loss"] = SubtaskPredictionLoss(
                subtask_expert_uuid=cls.EXPERT_SUBTASK_UUID,
                subtask_logits_uuid="subtask_logits",
            )

        return dict(
            named_losses=named_losses,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    loss_weights=[1.0,] * len(named_losses),
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **params
        )