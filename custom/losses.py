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
        
        subtask_gt = observations[self.subtask_expert_uuid][:, :, 0]
        subtask_type_gt = (subtask_gt / (NUM_OBJECT_TYPES * len(MAP_TYPES))).long().view(-1)
        subtask_arg_gt = (subtask_gt % (NUM_OBJECT_TYPES * len(MAP_TYPES)) / len(MAP_TYPES)).long().view(-1)
        subtask_target_map_type_gt = (subtask_gt % len(MAP_TYPES)).long().view(-1)

        subtask_logits = actor_critic_output.extras[self.subtask_logits_uuid]
        subtask_type_logits = subtask_logits['type'].view(-1, subtask_logits['type'].shape[-1])
        subtask_arg_logits = subtask_logits['arg'].view(-1, subtask_logits['arg'].shape[-1])
        subtask_target_map_type_logits = subtask_logits['target_map'].view(-1, subtask_logits['target_map'].shape[-1])

        type_loss = self.nllloss(input=subtask_type_logits, target=subtask_type_gt)
        arg_loss = self.nllloss(input=subtask_arg_logits, target=subtask_arg_gt)
        target_map_loss = self.nllloss(input=subtask_target_map_type_logits, target=subtask_target_map_type_gt)

        total_loss = type_loss + arg_loss + target_map_loss
        return (
            total_loss,
            {"subtask_ce": total_loss.item()},
        )


class SubtaskActionImitationLoss(AbstractActorCriticLoss):
    
    def __init__(
        self, 
        subtask_expert_uuid: str, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.subtask_expert_uuid = subtask_expert_uuid

    @staticmethod
    def group_loss(
        distribution: Union[CategoricalDistr, ConditionalDistr],
        expert_actions: torch.Tensor,
        expert_actions_masks: torch.Tensor,
    ):
        assert isinstance(distribution, CategoricalDistr) or (
            isinstance(distribution, ConditionalDistr)
            and isinstance(distribution.distr, CategoricalDistr)
        ), "This implementation only supports (groups of) `CategoricalDistr`"

        expert_successes = expert_actions_masks.sum()

        log_probs = distribution.log_prob(cast(torch.LongTensor, expert_actions))
        assert (
            log_probs.shape[: len(expert_actions_masks.shape)]
            == expert_actions_masks.shape
        )

        # Add dimensions to `expert_actions_masks` on the right to allow for masking
        # if necessary.
        len_diff = len(log_probs.shape) - len(expert_actions_masks.shape)
        assert len_diff >= 0
        expert_actions_masks = expert_actions_masks.view(
            *expert_actions_masks.shape, *((1,) * len_diff)
        )

        group_loss = -(expert_actions_masks * log_probs).sum() / torch.clamp(
            expert_successes, min=1
        )

        return group_loss, expert_successes

    def loss(
        self, 
        step_count: int, 
        batch: ObservationType, 
        actor_critic_output: ActorCriticOutput[CategoricalDistr], 
        *args, 
        **kwargs
    ) -> Union[
        Tuple[torch.FloatTensor, Dict[str, float]], 
        Tuple[torch.FloatTensor, Dict[str, float], 
        Dict[str, float]],
    ]:
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        losses = OrderedDict()

        should_report_loss = False

        expert_actions_and_mask = observations[self.subtask_expert_uuid][..., -2:]
        assert expert_actions_and_mask.shape[-1] == 2

        expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)
        expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
            *expert_actions_and_mask.shape[:-1], 1
        )

        expert_actions_masks = (
            expert_actions_and_mask_reshaped[:, 1]
            .float()
            .view(*expert_actions_and_mask.shape[:-1], 1)
        )

        total_loss, expert_successes = self.group_loss(
            cast(CategoricalDistr, actor_critic_output.distributions),
            expert_actions,
            expert_actions_masks,
        )

        should_report_loss = expert_successes.item() != 0

        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(), **losses}
            if should_report_loss
            else {},
        )
