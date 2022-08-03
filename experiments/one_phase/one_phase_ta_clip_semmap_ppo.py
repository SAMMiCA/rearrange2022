from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
import gym

from torch import nn, cuda, optim
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph, Preprocessor
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder

from experiments.one_phase.one_phase_ta_clip_ppo import OnePhaseTaskAwareRearrangeClipPPOExperimentConfig
from task_aware_rearrange.models import OnePhaseSemanticMappingActorCriticRNN
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor
from task_aware_rearrange.voxel_utils import GridParameters


class OnePhaseTaskAwareRearrangeClipSemmapPPOExperimentConfig(OnePhaseTaskAwareRearrangeClipPPOExperimentConfig):

    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    REFERENCE_SEGMENTATION = True
    HEADLESS = True
    
    FOV = 90
    GRID_PARAMETERS = GridParameters()

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseTaskAwareRearrangeClipSemmapPPO"

    @classmethod
    def num_train_processes(cls) -> int:
        return 4

    @classmethod
    def num_valid_processes(cls) -> int:
        return 1

    @classmethod
    def num_test_processes(cls) -> int:
        return 4

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
        return [
            *super().preprocessors(),
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

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseSemanticMappingActorCriticRNN(
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
        )