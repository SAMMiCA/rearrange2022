from typing import Dict, Any
from torch import nn, cuda, optim
import gym

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import LinearDecay, PipelineStage

from experiments.one_phase.one_phase_ta_il_base import OnePhaseTaskAwareRearrangeILBaseExperimentConfig
from task_aware_rearrange.losses import SubtaskPredictionLoss
from task_aware_rearrange.models import OnePhaseSubtaskResNet


class OnePhaseSubtaskBaseExperimentConfig(OnePhaseTaskAwareRearrangeILBaseExperimentConfig):

    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    REFERENCE_SEGMENTATION = True
    HEADLESS = True

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseSubtaskResNet(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.UNSHUFFLED_RGB_RESNET_UUID,
            prev_action_embedding_dim=cls.PREV_ACTION_EMBEDDING_DIM,
            hidden_size=cls.HIDDEN_SIZE,
        )

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseSubtaskTest"

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
                subtask_loss=SubtaskPredictionLoss(
                    subtask_expert_uuid=cls.EXPERT_SUBTASK_UUID,
                    subtask_logits_uuid="subtask_logits",
                ),
            ),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["subtask_loss"],
                    max_stage_steps=training_steps,
                )
            ],
            num_steps=64,
            num_mini_batch=1,
            update_repeats=3,
            use_lr_decay=True,
            lr=3e-4,
        )