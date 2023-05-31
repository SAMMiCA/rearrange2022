from experiments.two_phase.two_phase_ta_base import TwoPhaseTaskAwareRearrangeExperimentConfig


class TwoPhaseTestConfig(TwoPhaseTaskAwareRearrangeExperimentConfig):
    IL_PIPELINE_TYPE = "1proc"
    WALKTHROUGH_TRAINING_PPO = False
    # HEADLESS = False
    RGB_NORMALIZATION = False