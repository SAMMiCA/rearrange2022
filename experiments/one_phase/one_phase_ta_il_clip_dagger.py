from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym

from experiments.one_phase.one_phase_ta_il_base import OnePhaseTaskAwareRearrangeILBaseExperimentConfig
from task_aware_rearrange.models import OnePhaseResNetActorCriticRNN


class OnePhaseTaskAwareRearrangeILClipDaggerExperimentConfig(OnePhaseTaskAwareRearrangeILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "10proc"

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
        return "OnePhaseTaskAwareRearrangeILClipDagger"

    @classmethod
    def num_valid_processes(cls) -> int:
        return 2

    @classmethod
    def num_test_processes(cls) -> int:
        return 4