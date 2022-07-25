from typing import Dict, Any

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import LinearDecay, PipelineStage

from experiments.one_phase.one_phase_ta_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig


class OnePhaseTaskAwareRearrangeClipPPOExperimentConfig(OnePhaseTaskAwareRearrangeBaseExperimentConfig):

    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    HEADLESS = False

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseTaskAwareRearrangeClipPPO"

    @classmethod
    def num_train_processes(cls) -> int:
        return 4

    @classmethod
    def num_valid_processes(cls) -> int:
        return 2

    @classmethod
    def num_test_processes(cls) -> int:
        return 20

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