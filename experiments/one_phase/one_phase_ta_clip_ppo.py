from typing import Dict, Any
from torch import nn, cuda, optim
import gym

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import LinearDecay, PipelineStage

from experiments.one_phase.one_phase_ta_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig
from task_aware_rearrange.models import OnePhaseResNetActorCriticRNN


class OnePhaseTaskAwareRearrangeClipPPOExperimentConfig(OnePhaseTaskAwareRearrangeBaseExperimentConfig):

    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    REFERENCE_SEGMENTATION = False
    HEADLESS = True

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseResNetActorCriticRNN(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None else cls.UNSHUFFLED_RGB_RESNET_UUID,
            prev_action_embedding_dim=cls.PREV_ACTION_EMBEDDING_DIM,
            hidden_size=cls.HIDDEN_SIZE,
            num_rnn_layers=cls.NUM_RNN_LAYERS,
            rnn_type=cls.RNN_TYPE,
        )

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseTaskAwareRearrangeClipPPO"

    @classmethod
    def num_train_processes(cls) -> int:
        return 10

    @classmethod
    def num_valid_processes(cls) -> int:
        return 2

    @classmethod
    def num_test_processes(cls) -> int:
        return 4

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