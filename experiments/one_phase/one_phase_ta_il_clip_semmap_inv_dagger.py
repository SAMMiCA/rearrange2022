from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor, SensorSuite

from experiments.one_phase.one_phase_ta_il_clip_semmap_dagger import OnePhaseTaskAwareRearrangeILClipSemmapDaggerExperimentConfig
from task_aware_rearrange.models import OnePhaseSemanticMappingWithInventoryActorCriticRNN
from task_aware_rearrange.sensors import InventoryObjectSensor
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor
from task_aware_rearrange.voxel_utils import GridParameters


class OnePhaseTaskAwareRearrangeILClipSemmapWithInvDaggerExperimentConfig(OnePhaseTaskAwareRearrangeILClipSemmapDaggerExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "4proc"
    REFERENCE_SEGMENTATION = True
    HEADLESS = True

    FOV = 90
    GRID_PARAMETERS = GridParameters()

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [
            *super().sensors(),
            InventoryObjectSensor(
                reference_inventory=cls.REFERENCE_INVENTORY, 
                ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                uuid=cls.INVENTORY_UUID,
            )
        ]
    

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseSemanticMappingWithInventoryActorCriticRNN(
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
        )

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseTaskAwareRearrangeILClipSemmapWithInventoryDagger"

    @classmethod
    def num_valid_processes(cls) -> int:
        return 1

    @classmethod
    def num_test_processes(cls) -> int:
        return 4