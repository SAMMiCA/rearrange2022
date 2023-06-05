from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
import os
import torch.nn as nn
import gym

from allenact.utils.misc_utils import all_unique
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig, PPO
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, UnshuffledRGBRearrangeSensor
from rearrange.tasks import RearrangeTaskSampler
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from experiments.ta_base import TaskAwareBaseExperimentConfig
from task_aware_rearrange.expert import OnePhaseSubtaskAndActionExpertSensor
from task_aware_rearrange.losses import SubtaskPredictionLoss
from task_aware_rearrange.models import OnePhaseSubtaskAwarePolicy
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor, SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor
from task_aware_rearrange.sensors import PoseSensor, SemanticSegmentationSensor, UnshuffledDepthRearrangeSensor, UnshuffledPoseSensor, UnshuffledSemanticSegmentationSensor
from task_aware_rearrange.constants import NUM_OBJECT_TYPES, ADDITIONAL_MAP_CHANNELS
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from task_aware_rearrange.voxel_utils import GridParameters
from semseg.semseg_sensors import SemanticRearrangeSensor
from semseg.semseg_preprocessors import SemanticPreprocessor
from semseg.semseg_constants import CLASS_TO_COLOR_ORIGINAL, ORDERED_CLASS_TO_COLOR, ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE


class StepwiseLinearDecay:
    def __init__(self, cumm_steps_and_values: Sequence[Tuple[int, float]]):
        assert len(cumm_steps_and_values) >= 1

        self.steps_and_values = list(sorted(cumm_steps_and_values))
        self.steps = [steps for steps, _ in cumm_steps_and_values]
        self.values = [value for _, value in cumm_steps_and_values]

        assert all_unique(self.steps)
        assert all(0 <= v <= 1 for v in self.values)

    def __call__(self, epoch: int) -> float:
        """Get the value for the input number of steps."""
        if epoch <= self.steps[0]:
            return self.values[0]
        elif epoch >= self.steps[-1]:
            return self.values[-1]
        else:
            # TODO: Binary search would be more efficient but seems overkill
            for i, (s0, s1) in enumerate(zip(self.steps[:-1], self.steps[1:])):
                if epoch < s1:
                    p = (epoch - s0) / (s1 - s0)
                    v0 = self.values[i]
                    v1 = self.values[i + 1]
                    return p * v1 + (1 - p) * v0


def training_params(label: str, training_steps: int):
    """
    label: Nproc(_longtf)(-il)
    """
    use_lr_decay = False

    lr = 3e-4
    num_train_processes = int(label.split('-')[0].split('_')[0][:-4])
    isIL = True if 'il' in label.split('-')[1].split('_') else False
    longtf = True if ('longtf' in label.split('-')[0].split('_') and isIL) else False
    num_steps = 64
    dagger_steps = min(int(5e6), training_steps // 10) if longtf else min(int(1e6), training_steps // 10)
    bc_tf1_steps = min(int(5e5), training_steps // 10) if longtf else min(int(1e5), training_steps // 10)
    update_repeats = 3
    num_mini_batch = 1

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=use_lr_decay,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )


class OnePhaseTaskAwareRearrangeBaseExperimentConfig(TaskAwareBaseExperimentConfig):
    
    TRAINING_STEPS = int(25e6)
    SAVE_INTERVAL = int(5e5)
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    PIPELINE_TYPE: Optional[str] = None     # Nproc(-longtf)_(training)
    IL_LOSS_WEIGHT: Optional[float] = None  # IL loss
    RL_LOSS_WEIGHT: Optional[float] = None  # RL loss
    
    EXPERT_SUBTASK_ACTION_UUID = "expert_subtask_action"
    EXPERT_ACTION_UUID = "expert_action"
    EXPERT_SUBTASK_UUID = "expert_subtask"
    EXPERT_VERBOSE: bool = False
    
    SAP_SUBTASK_HISTORY: bool = False
    SAP_SEMANTIC_MAP: bool = False
    
    REQUIRE_SEMANTIC_SEGMENTATION: bool = False
    REFERENCE_SEGMENTATION = False
    
    SEMANTIC_SEGMENTATION_USE_MASS = True
    if SEMANTIC_SEGMENTATION_USE_MASS:
        CLASS_TO_COLOR = CLASS_TO_COLOR_ORIGINAL
        ORDERED_OBJECT_TYPES = list(CLASS_TO_COLOR.keys())
        CLASS_MAPPING = ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE
    else:
        CLASS_TO_COLOR = ORDERED_CLASS_TO_COLOR
        CLASS_MAPPING = list(range(len(CLASS_TO_COLOR)))
        
    DETECTION_THRESHOLD: float = 0.8
    SEMANTIC_SEGMENTATION_BASE = "mask-rcnn"
    SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
        "semseg",
        "checkpoints",
        "model_final.pth"
    )
    
    RGB_NORMALIZATION: bool = True
    DEPTH_NORMALIZATION: bool = False
    
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
    ONLINE_SUBTASK_LOSS_WEIGHT: Optional[float] = None    
    
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
        ]
        
        if "il" in cls.PIPELINE_TYPE.lower().split('-')[1].split('_'):
            sensors.append(
                OnePhaseSubtaskAndActionExpertSensor(
                    action_space=(
                        NUM_SUBTASKS,
                        len(cls.actions()),
                    ),
                    uuid=cls.EXPERT_SUBTASK_ACTION_UUID,
                    verbose=False,
                ),
            )

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
        
        if "il" in cls.PIPELINE_TYPE.lower().split('-')[1].split('_'):
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
        if cls.PIPELINE_TYPE is not None:
            return cls._use_label_to_get_training_params()["num_train_processes"]
        else:
            raise NotImplementedError

    @classmethod
    def _use_label_to_get_training_params(cls):
        return training_params(
            label=cls.PIPELINE_TYPE.lower(), training_steps=cls.TRAINING_STEPS
        )
        
    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""
        assert cls.PIPELINE_TYPE is not None
        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        isIL = ('il' in cls.PIPELINE_TYPE.lower().split('-')[1].split('_'))
        isRL = ('rl' in cls.PIPELINE_TYPE.lower().split('-')[1].split('_'))
        bc_tf1_steps = params["bc_tf1_steps"] if isIL else 0
        dagger_steps = params["dagger_steps"] if isIL else 0
        
        named_losses = {}
        pipeline_stages: List[PipelineStage] = []
        if isIL:
            assert cls.IL_LOSS_WEIGHT is not None
            named_losses["imitation_loss"] = Imitation()
        
        if isRL:
            assert cls.RL_LOSS_WEIGHT is not None
            named_losses["ppo_loss"] = PPO(
                clip_decay=LinearDecay(
                    training_steps if not isIL
                    else training_steps - (bc_tf1_steps + dagger_steps / 2)
                ),
                **PPOConfig,
            )
        
        if cls.ONLINE_SUBTASK_PREDICTION:
            assert cls.ONLINE_SUBTASK_LOSS_WEIGHT is not None
            named_losses["subtask_loss"] = SubtaskPredictionLoss(
                subtask_expert_uuid=cls.EXPERT_SUBTASK_UUID,
                subtask_logits_uuid="subtask_logits",
            )
            
        if isIL:
            loss_names = ["imitation_loss"] + (
                ["subtask_loss"] if cls.ONLINE_SUBTASK_PREDICTION else []
            )
            loss_weights = [cls.IL_LOSS_WEIGHT] + (
                [cls.ONLINE_SUBTASK_LOSS_WEIGHT] if cls.ONLINE_SUBTASK_PREDICTION else []
            )
            max_stage_steps = training_steps if not isRL else (bc_tf1_steps + dagger_steps / 2)
            cumm_steps_and_values = [
                (bc_tf1_steps, 1.0), (bc_tf1_steps + dagger_steps, 0.0)
            ] if not isRL else [
                (bc_tf1_steps, 1.0), (bc_tf1_steps + dagger_steps / 2, 0.5)
            ]
            pipeline_stages.append(
                PipelineStage(
                    loss_names=loss_names,
                    loss_weights=loss_weights,
                    max_stage_steps=max_stage_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=cumm_steps_and_values
                    ),
                )
            )
            if isRL:
                loss_names = ["imitation_loss", "ppo_loss"] + (
                    ["subtask_loss"] if cls.ONLINE_SUBTASK_PREDICTION else []
                )
                loss_weights = [cls.IL_LOSS_WEIGHT, cls.RL_LOSS_WEIGHT] + (
                    [cls.ONLINE_SUBTASK_LOSS_WEIGHT] if cls.ONLINE_SUBTASK_PREDICTION else []
                )
                pipeline_stages.append(
                    PipelineStage(
                        loss_names=loss_names,
                        loss_weights=loss_weights,
                        max_stage_steps=(dagger_steps / 2),
                        teacher_forcing=StepwiseLinearDecay(
                            cumm_steps_and_values=[(0, 0.5), (dagger_steps / 2, 0.0)]
                        ),
                    )
                )
                
        if isRL:
            loss_names = ["ppo_loss"] + (
                ["subtask_loss"] if cls.ONLINE_SUBTASK_PREDICTION else []
            )
            loss_weights = [cls.RL_LOSS_WEIGHT] + (
                [cls.ONLINE_SUBTASK_LOSS_WEIGHT] if cls.ONLINE_SUBTASK_PREDICTION else []
            )
            max_stage_steps = (training_steps - (bc_tf1_steps + dagger_steps))
            pipeline_stages.append(
                PipelineStage(
                    loss_names=loss_names,
                    loss_weights=loss_weights,
                    max_stage_steps=max_stage_steps,
                )
            )
        
        return dict(
            named_losses=named_losses,
            pipeline_stages=pipeline_stages,
            **params
        )