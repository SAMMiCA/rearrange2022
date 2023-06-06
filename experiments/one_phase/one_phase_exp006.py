from typing import Sequence, Union
import torch

from allenact.base_abstractions.experiment_config import MachineParams
from experiments.one_phase.one_phase_ta_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig
from allenact.base_abstractions.experiment_config import (
    MachineParams,
    split_processes_onto_devices,
)


class OnePhaseExp006Config(OnePhaseTaskAwareRearrangeBaseExperimentConfig):
    
    # NUM_DISTRIBUTED_NODES: int = 3
    # NUM_DEVICES: Union[int, Sequence] = 4
    
    PIPELINE_TYPE = "8proc-rl"
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    
    SAVE_INTERVAL = int(1e6)
    # IL_LOSS_WEIGHT = 1.0
    RL_LOSS_WEIGHT = 10.0
    
    RGB_NORMALIZATION = True
    EXPERT_VERBOSE = False
    
    REQUIRE_SEMANTIC_SEGMENTATION = True
    SAP_SEMANTIC_MAP = True
    SAP_SUBTASK_HISTORY = True
    
    ONLINE_SUBTASK_PREDICTION = True
    ONLINE_SUBTASK_PREDICTION_USE_EGOVIEW = False
    ONLINE_SUBTASK_PREDICTION_USE_PREV_ACTION = False
    ONLINE_SUBTASK_PREDICTION_USE_SEMANTIC_MAP = True
    ONLINE_SUBTASK_PREDICTION_USE_SUBTASK_HISTORY = True
    ONLINE_SUBTASK_LOSS_WEIGHT = 1.0
    
    @classmethod
    def tag(cls) -> str:
        return "OnePhaseExp006"
    
    @classmethod
    def num_valid_processes(cls) -> int:
        return 0
    
    @classmethod
    def num_test_processes(cls) -> int:
        return 1
    