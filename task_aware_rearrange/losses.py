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
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
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
        
        
class TaskAwareReverselyMaskedPPO(MaskedPPO):

    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        mask = (~batch["observations"][self.mask_uuid]).float()
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
        
        
class MaskedImitation(Imitation):
    
    def __init__(
        self, mask_uuid, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_uuid = mask_uuid
    
    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output,
        *args,
        **kwargs,
    ):
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        
        losses = OrderedDict()
        
        should_report_loss = False
        
        if "expert_action" in observations:
            if self.expert_sensor is None or not self.expert_sensor.use_groups:
                expert_actions_and_mask = observations["expert_action"]
                mask = observations[self.mask_uuid].float()
                
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
                mask = mask.view(*expert_actions_masks.shape)
                
                total_loss, expert_success = self.group_loss(
                    cast(CategoricalDistr, actor_critic_output.distributions),
                    expert_actions,
                    expert_actions_masks * mask,
                )
                
                should_report_loss = expert_success.item() != 0
            
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(), **losses}
            if should_report_loss
            else {},
        )
        

class ReverselyMaskedImitation(Imitation):
    
    def __init__(
        self, mask_uuid, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_uuid = mask_uuid
    
    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output,
        *args,
        **kwargs,
    ):
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        
        losses = OrderedDict()
        
        should_report_loss = False
        
        if "expert_action" in observations:
            if self.expert_sensor is None or not self.expert_sensor.use_groups:
                expert_actions_and_mask = observations["expert_action"]
                mask = (~observations[self.mask_uuid]).float()
                
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
                mask = mask.view(*expert_actions_masks.shape)
                
                total_loss, expert_success = self.group_loss(
                    cast(CategoricalDistr, actor_critic_output.distributions),
                    expert_actions,
                    expert_actions_masks * mask,
                )
                
                should_report_loss = expert_success.item() != 0
            
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(), **losses}
            if should_report_loss
            else {},
        )