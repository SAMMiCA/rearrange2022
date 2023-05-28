from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym


from experiments.one_phase.ablation_base import OnePhaseAblationExerimentConfig
from task_aware_rearrange.losses import SubtaskPredictionLoss


class OnePhaseAblation007ExerimentConfig(OnePhaseAblationExerimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "8proc"
    
    REQUIRE_SEMANTIC_SEGMENTATION = True
    SAP_SEMANTIC_MAP = True
    SAP_SUBTASK_HISTORY = True

    ONLINE_SUBTASK_PREDICTION = True
    ONLINE_SUBTASK_PREDICTION_USE_EGOVIEW = False
    ONLINE_SUBTASK_PREDICTION_USE_PREV_ACTION = False
    ONLINE_SUBTASK_PREDICTION_USE_SUBTASK_HISTORY = True
    
    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseAblation007"
    
    @classmethod
    def num_valid_processes(cls) -> int:
        return 1
    
    @classmethod
    def num_test_processes(cls) -> int:
        return 4
    
