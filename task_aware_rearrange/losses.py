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
from rearrange.losses import MaskedPPO


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


class TaskAwareMaskedPPO(MaskedPPO):

    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        mask = batch["observations"][self.mask_uuid].float()
        denominator = mask.sum().item()

        losses_per_step, _ = self._ppo_loss.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: ((loss * mask).sum() / max(denominator, 1), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        if denominator == 0:
            losses_to_record = {}
        else:
            losses_to_record = {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()}
            }

        return (
            total_loss,
            losses_to_record,
        )