from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym


from experiments.one_phase.ablation_base import OnePhaseAblationExerimentConfig
from task_aware_rearrange.losses import SubtaskPredictionLoss


class OnePhaseAblation001ExerimentConfig(OnePhaseAblationExerimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "8proc"
    
    REQUIRE_SEMANTIC_SEGMENTATION = False
    SAP_SEMANTIC_MAP = False
    SAP_SUBTASK_HISTORY = False
    
    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseAblation001"
    
    @classmethod
    def num_valid_processes(cls) -> int:
        return 1
    
    @classmethod
    def num_test_processes(cls) -> int:
        return 4
    
