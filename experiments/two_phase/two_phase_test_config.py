from experiments.two_phase.two_phase_ta_base import TwoPhaseTaskAwareRearrangeExperimentConfig
from allenact_plugins.ithor_plugin.ithor_sensors import RelativePositionChangeTHORSensor


class TwoPhaseTestConfig(TwoPhaseTaskAwareRearrangeExperimentConfig):
    IL_PIPELINE_TYPE = "1proc"
    WALKTHROUGH_TRAINING_PPO = False
    # HEADLESS = False
    RGB_NORMALIZATION = False
    EXPERT_VERBOSE = False
    
    @classmethod
    def sensors(cls):
        sensors = super().sensors()
        sensors.append(
            RelativePositionChangeTHORSensor(
                uuid="test_pos",
            )
        )
        
        return sensors