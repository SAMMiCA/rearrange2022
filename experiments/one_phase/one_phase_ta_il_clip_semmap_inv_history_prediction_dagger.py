from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
from torch import nn, cuda, optim
import gym

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.utils.experiment_utils import LinearDecay, PipelineStage, Builder

from baseline_configs.one_phase.one_phase_rgb_il_base import StepwiseLinearDecay
from experiments.one_phase.one_phase_ta_il_clip_semmap_inv_dagger import OnePhaseTaskAwareRearrangeILClipSemmapWithInvDaggerExperimentConfig
from task_aware_rearrange.models import OnePhaseTaskAwareActorCriticRNN
from task_aware_rearrange.sensors import InventoryObjectSensor
from task_aware_rearrange.preprocessors import Semantic3DMapPreprocessor
from task_aware_rearrange.voxel_utils import GridParameters
from task_aware_rearrange.losses import SubtaskPredictionLoss


class OnePhaseTaskAwareRearrangeILClipSemmapWithInvSubtasHistorySubtaskPredictionDaggerExperimentConfig(OnePhaseTaskAwareRearrangeILClipSemmapWithInvDaggerExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE: Optional[str] = "4proc"
    REFERENCE_SEGMENTATION = True
    HEADLESS = True

    FOV = 90
    GRID_PARAMETERS = GridParameters()

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return OnePhaseTaskAwareActorCriticRNN(
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
            num_repeats=cls.training_pipeline().training_settings.update_repeats,
        )

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseTaskAwareRearrangeILClipSemmapWithInventorySubtaskHistorySubtaskPredictionDagger"

    @classmethod
    def num_valid_processes(cls) -> int:
        return 1

    @classmethod
    def num_test_processes(cls) -> int:
        return 4

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""
        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(
                imitation_loss=Imitation(),
                subtask_loss=SubtaskPredictionLoss(
                    subtask_expert_uuid=cls.EXPERT_SUBTASK_UUID,
                    subtask_logits_uuid="subtask_logits",
                )
            ),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss", "subtask_loss"],
                    loss_weights=[1.0, 1.0],
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