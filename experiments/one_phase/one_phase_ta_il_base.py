from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
import torch

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.utils.misc_utils import all_unique
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph, Preprocessor

from rearrange.tasks import RearrangeTaskSampler
from experiments.one_phase.one_phase_ta_base import OnePhaseTaskAwareRearrangeBaseExperimentConfig
from task_aware_rearrange.expert import OnePhaseSubtaskAndActionExpertSensor
from task_aware_rearrange.preprocessors import SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor
from task_aware_rearrange.subtasks import NUM_SUBTASKS


class StepwiseLinearDecay:
    def __init__(self, cumm_steps_and_values: Sequence[Tuple[int, float]]):
        assert len(cumm_steps_and_values) >= 1

        self.steps_and_values = list(sorted(cumm_steps_and_values))
        self.steps = [steps for steps, _ in cumm_steps_and_values]
        self.values = [value for _, value in cumm_steps_and_values]

        assert all_unique(self.steps)
        assert all(0 <= v <= 1 for v in self.values)

    def __call__(self, epoch: int) -> float:
        """Get the value for the input number of steps."""
        if epoch <= self.steps[0]:
            return self.values[0]
        elif epoch >= self.steps[-1]:
            return self.values[-1]
        else:
            # TODO: Binary search would be more efficient but seems overkill
            for i, (s0, s1) in enumerate(zip(self.steps[:-1], self.steps[1:])):
                if epoch < s1:
                    p = (epoch - s0) / (s1 - s0)
                    v0 = self.values[i]
                    v1 = self.values[i + 1]
                    return p * v1 + (1 - p) * v0


def il_training_params(label: str, training_steps: int):
    use_lr_decay = False

    if label == "80proc":
        lr = 3e-4
        num_train_processes = 80
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 2 if torch.cuda.is_available() else 1

    elif label == "40proc":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    elif label == "40proc-longtf":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(5e6), training_steps // 10)
        bc_tf1_steps = min(int(5e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    else:
        lr = 3e-4
        num_train_processes = int(label.split('-')[0][:-4])
        longtf = True if 'longtf' in label.split('-') else False
        num_steps = 64
        dagger_steps = min(int(5e6), training_steps // 10) if longtf else min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(5e5), training_steps // 10) if longtf else min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=use_lr_decay,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )


class OnePhaseTaskAwareRearrangeILBaseExperimentConfig(OnePhaseTaskAwareRearrangeBaseExperimentConfig):
    
    IL_PIPELINE_TYPE: Optional[str] = None
    EXPERT_SUBTASK_ACTION_UUID = "expert_subtask_action"
    EXPERT_ACTION_UUID = "expert_action"
    EXPERT_SUBTASK_UUID = "expert_subtask"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [
            *super(OnePhaseTaskAwareRearrangeILBaseExperimentConfig, cls).sensors(),
            OnePhaseSubtaskAndActionExpertSensor(
                action_space=(
                    NUM_SUBTASKS,
                    len(cls.actions()),
                ),
                uuid=cls.EXPERT_SUBTASK_ACTION_UUID,
                verbose=False,
            ),
        ]

    @classmethod
    def create_subtask_action_expert_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SubtaskActionExpertPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            device=cls.DEVICE,            
        )

    @classmethod
    def create_subtask_expert_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SubtaskExpertPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            device=cls.DEVICE,
        )

    @classmethod
    def preprocessors(cls) -> Sequence[Preprocessor]:
        return [
            *super().preprocessors(),
            cls.create_subtask_action_expert_preprocessor_builder(
                in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                out_uuid=cls.EXPERT_ACTION_UUID,
            ),
            cls.create_subtask_expert_preprocessor_builder(
                in_uuids=[cls.EXPERT_SUBTASK_ACTION_UUID],
                out_uuid=cls.EXPERT_SUBTASK_UUID,
            ),
        ]

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(imitation_loss=Imitation()),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **params
        )

    @classmethod
    def num_train_processes(cls) -> int:
        return cls._use_label_to_get_training_params()["num_train_processes"]

    @classmethod
    def _use_label_to_get_training_params(cls):
        return il_training_params(
            label=cls.IL_PIPELINE_TYPE.lower(), training_steps=cls.TRAINING_STEPS
        )