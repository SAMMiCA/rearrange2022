from experiments.two_phase.two_phase_ta_base_temp import TwoPhaseTaskAwareRearrangeExperimentConfig
from allenact_plugins.ithor_plugin.ithor_sensors import RelativePositionChangeTHORSensor


class TwoPhaseTestConfig(TwoPhaseTaskAwareRearrangeExperimentConfig):
    PIPELINE_TYPE = "1proc-il-il_rl"
    
    IL_LOSS_WEIGHT = 1.0
    RL_LOSS_WEIGHT = 1.0
    
    WALKTHROUGH_TRAINING_PPO = True
    WALKTHROUGH_PPO_LOSS_WEIGHT = 1.0
    # HEADLESS = False
    RGB_NORMALIZATION = True
    EXPERT_VERBOSE = False
    
    SAP_SUBTASK_HISTORY = True
    SAP_SEMANTIC_MAP = True
    REQUIRE_SEMANTIC_SEGMENTATION = True
    
    ONLINE_SUBTASK_PREDICTION = True
    ONLINE_SUBTASK_PREDICTION_USE_EGOVIEW = False
    ONLINE_SUBTASK_PREDICTION_USE_PREV_ACTION = False
    ONLINE_SUBTASK_PREDICTION_USE_IS_WALKTHROUGH_PHASE = True
    ONLINE_SUBTASK_PREDICTION_USE_SEMANTIC_MAP = True
    ONLINE_SUBTASK_PREDICTION_USE_SUBTASK_HISTORY = True
    ONLINE_SUBTASK_LOSS_WEIGHT = 1.0
    
    @classmethod
    def tag(cls) -> str:
        return "TwoPhaseTest"
    
    @classmethod
    def num_valid_processes(cls) -> int:
        return 1
    
    @classmethod
    def num_test_processes(cls) -> int:
        return 1
    