from typing import Tuple, Sequence, Optional, Dict, Any, Type
import gym
from regex import R

import torch
from torch import nn
import torchvision

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.utils.experiment_utils import PipelineStage, Builder
from allenact.utils.misc_utils import all_unique
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from custom.expert import OnePhaseSubtaskAndActionExpertSensor
from custom.hlsm.voxel_grid import GridParameters
from custom.models import TaskAwareOnePhaseRearrangeSubtaskModelTrainingNetwork
from custom.preprocessors import SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor
from experiments.one_phase_task_aware_rearrange_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig
from custom.losses import SubtaskActionImitationLoss, SubtaskPredictionLoss
from custom.subtask import IDX_TO_SUBTASK_TYPE, IDX_TO_MAP_TYPE, MAP_TYPES, SUBTASK_TYPES
from custom.constants import NUM_OBJECT_TYPES, IDX_TO_OBJECT_TYPE, OBJECT_TYPES_TO_IDX


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


def il_training_params(label: str, training_steps: int):
    use_lr_decay = False

    if label == "80proc":
        lr = 3e-4
        num_train_processes = 80
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 2 if torch.cuda.is_available() else 1

    elif label == "40proc":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    elif label == "40proc-longtf":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(5e6), training_steps // 10)
        bc_tf1_steps = min(int(5e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    else:
        lr = 3e-4
        num_train_processes = int(label.split('-')[0][:-4])
        longtf = True if len(label.split('-')) == 2 and label.split('-')[1] == "longtf" else False
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


class OnePhaseSubtaskModelTrainingExperimentConfig(OnePhaseTaskAwareRearrangeBaseExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")

    # Sensor Info
    REFERENCE_DEPTH = True
    REFERENCE_SEGMENTATION = True
    REFERENCE_POSE = False
    REFERENCE_INVENTORY = False

    # Environment parameters
    THOR_CONTROLLER_KWARGS = {
        **RearrangeBaseExperimentConfig.THOR_CONTROLLER_KWARGS,
        "renderDepthImage": REFERENCE_DEPTH,
        "renderSemanticSegmentation": REFERENCE_SEGMENTATION,
        "renderInstanceSegmentation": REFERENCE_SEGMENTATION,
    }
    HEADLESS = True

    FOV = 90
    GRID_PARAMETERS = GridParameters()

    EXPERT_SUBTASK_ACTION_UUID = "expert_subtask_action"
    EXPERT_ACTION_UUID = "expert_action"
    EXPERT_SUBTASK_UUID = "expert_subtask"

    # Training parameters
    DEVICE = torch.device('cuda')
    IL_PIPELINE_TYPE = "8proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        sensors = [
            *super().sensors(),
            OnePhaseSubtaskAndActionExpertSensor(
                action_space=(
                    (len(SUBTASK_TYPES) - 1) * NUM_OBJECT_TYPES * len(MAP_TYPES) + 1,
                    len(RearrangeBaseExperimentConfig.actions()),
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
            # "rgb", "semseg", "depth", "pose", 
            # "unshuffled_rgb", "unshuffled_semseg", "unshuffled_depth", "unshuffled_pose"
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
    def create_model(cls, **kwargs) -> nn.Module:
        def get_sensor_uuid(stype: Type[Sensor]) -> Optional[str]:
            s = next((s for s in cls.sensors() if isinstance(s, stype)), None,)
            return None if s is None else s.uuid

        walkthrougher_should_ignore_action_mask = [
            any(k in a for k in ["drop", "open", "pickup"]) for a in cls.actions()
        ]

        return TaskAwareOnePhaseRearrangeSubtaskModelTrainingNetwork(
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
            rnn_type=cls.RNN_TYPE,
        )

    @classmethod
    def num_valid_processes(cls) -> int:
        return 0

    @classmethod
    def num_test_processes(cls) -> int:
        return max(1, torch.cuda.device_count() * 2)

    @classmethod
    def num_train_processes(cls) -> int:
        return cls._use_label_to_get_training_params()["num_train_processes"]

    @classmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        # bc_tf1_steps = params["bc_tf1_steps"]
        # dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(
                # ppo_loss=PPO(clip_decay=LinearDecay(training_steps), **PPOConfig),
                # subtask_action_imitation=SubtaskActionImitationLoss(
                #     subtask_expert_uuid=cls.SUBTASK_EXPERT_UUID,
                # ),
                subtask_loss=SubtaskPredictionLoss(
                    subtask_expert_uuid=cls.EXPERT_SUBTASK_UUID, 
                    subtask_logits_uuid="subtask_logits"
                ),
                # imitation_loss=Imitation(),
            ),
            pipeline_stages=[
                # PipelineStage(
                #     loss_names=["imitation_loss", "subtask_loss"], 
                #     loss_weights=[1.0, 10.0],
                #     max_stage_steps=training_steps,
                #     teacher_forcing=StepwiseLinearDecay(
                #         cumm_steps_and_values=[
                #             (bc_tf1_steps, 1.0),
                #             (bc_tf1_steps + dagger_steps, 0.0),
                #         ]
                #     ),
                # )
                PipelineStage(
                    loss_names=["subtask_loss"],
                    max_stage_steps=training_steps,
                )
            ],
            num_steps=64,
            num_mini_batch=1,
            update_repeats=1,
            use_lr_decay=True,
            lr=3e-4,
        )

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseSubtaskModelTrainingExperiment_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def _use_label_to_get_training_params(cls):
        return il_training_params(
            label=cls.IL_PIPELINE_TYPE.lower(), training_steps=cls.TRAINING_STEPS
        )
