from collections import OrderedDict
from typing import Union, cast, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr, ConditionalDistr
from allenact.base_abstractions.sensor import AbstractExpertSensor
from allenact.base_abstractions.misc import ActorCriticOutput

from custom.constants import IDX_TO_OBJECT_TYPE, NUM_OBJECT_TYPES, OBJECT_TYPES_TO_IDX, UNKNOWN_OBJECT_STR
from custom.subtask import IDX_TO_SUBTASK_TYPE, INTERACT_SUBTASK_TYPES, MAP_TYPES, MAP_TYPES_TO_IDX, SUBTASK_TYPES, SUBTASK_TYPES_TO_IDX
from example_utils import ForkedPdb


class SubtaskPredictionLoss(AbstractActorCriticLoss):

    def __init__(
        self,
        subtask_expert_uuid: str,
        subtask_logits_uuid: str,
    ):
        super().__init__()
        self.subtask_expert_uuid = subtask_expert_uuid
        self.subtask_logits_uuid = subtask_logits_uuid
        self.nllloss = nn.NLLLoss()

    def loss(
        self, 
        step_count: int, 
        batch: ObservationType, 
        actor_critic_output: ActorCriticOutput[CategoricalDistr], 
        *args, 
        **kwargs
    ):
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        
        subtask_gt = observations[self.subtask_expert_uuid][:, :, 0].long().view(-1)

        subtask_logits = actor_critic_output.extras[self.subtask_logits_uuid]
        bs, fs = subtask_logits.shape[:2], subtask_logits.shape[2:]
        subtask_logits = subtask_logits.view(-1, *fs)

        loss = self.nllloss(input=subtask_logits, target=subtask_gt)
        return (
            loss,
            {"subtask_ce": loss.item()},
        )
