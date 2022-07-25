from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)
from regex import M
import stringcase
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    FullMemorySpecType,
    ActionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from allenact.utils.system import get_logger
from custom.constants import ADDITIONAL_MAP_CHANNELS, NUM_OBJECT_TYPES

from task_aware_rearrange.layers import EgocentricViewEncoderPooled, SemanticMap2DEncoderPooled
from task_aware_rearrange.subtasks import MAP_TYPE_TO_IDX
from task_aware_rearrange.mapping_utils import update_semantic_map


class OnePhaseResNetActorCriticRNN(ActorCriticModel):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.num_rnn_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers

        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        self.prev_action_embedder = nn.Embedding(
            action_space.n + 1, embedding_dim=prev_action_embedding_dim
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size
                + prev_action_embedding_dim  
            ),
            hidden_size=self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.device = device if device else torch.device("cpu")

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_layers),
                    ("sampler", None),
                    ("hidden", self._hidden_size),
                ),
                torch.float32,
            ),
        )

    def forward(
        self, 
        observations: ObservationType, 
        memory: Memory, 
        prev_actions: ActionType, 
        masks: torch.FloatTensor
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """
        observations: [steps, samplers, (agents), ...]
        memory: [sampler, ...] 
        prev_actions: [steps, samplers, ...]
        masks: [steps, samplers, agents, 1], zero indicates the steps where a new episode/task starts
        """
        nsteps, nsamplers = masks.shape[:2]
        
        # Egocentric images
        ego_img = observations[self.rgb_uuid]
        w_ego_img = observations[self.unshuffled_rgb_uuid]
        ego_img_embeddings = self.visual_encoder(
            u_img_emb=ego_img,
            w_img_emb=w_ego_img
        )   # [steps, samplers, vis_feature_embedding_dim]

        # Previous actions (low-level actions)
        prev_action_embeddings = self.prev_action_embedder(
            (masks.long() * (prev_actions.unsqueeze(-1) + 1))
        ).squeeze(-2)   # [steps, samplers, prev_action_embedding_dim]

        to_cat = [
            ego_img_embeddings,
            prev_action_embeddings
        ]
        obs_for_rnn = torch.cat(to_cat, dim=-1)

        rnn_outs, rnn_hidden_states = self.state_encoder(
            obs_for_rnn,
            memory.tensor("rnn"),
            masks
        )                
        extras = {}

        return (
            ActorCriticOutput(
                distributions=self.actor(rnn_outs), values=self.critic(rnn_outs), extras=extras
            ), 
            memory.set_tensor("rnn", rnn_hidden_states)
        )


class OnePhaseSemanticMappingActorCriticRNN(OnePhaseResNetActorCriticRNN):

    def __init__(
        self,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.n_map_channels = NUM_OBJECT_TYPES + ADDITIONAL_MAP_CHANNELS
        self.sem_map_encoder = SemanticMap2DEncoderPooled(
            n_map_channels=self.n_map_channels,
            hidden_size=self.hidden_size,
        )
        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 2
                + self.prev_action_embedding_dim  
            ),
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.device = device if device else torch.device("cpu")

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_layers),
                    ("sampler", None),
                    ("hidden", self._hidden_size),
                ),
                torch.float32,
            ),
            sem_map=(
                (
                    ("sampler", None),
                    ("map_type", 2),
                    ("channels", self.map_channel),
                    ("width", self.map_width),
                    ("length", self.map_length),
                    ("height", self.map_height),
                ),
                torch.bool,
            ),
        )

    def forward(
        self, 
        observations: ObservationType, 
        memory: Memory, 
        prev_actions: ActionType, 
        masks: torch.FloatTensor
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """
        observations: [steps, samplers, (agents), ...]
        memory: [sampler, ...] 
        prev_actions: [steps, samplers, ...]
        masks: [steps, samplers, agents, 1], zero indicates the steps where a new episode/task starts
        """
        nsteps, nsamplers = masks.shape[:2]
        
        # Egocentric images
        ego_img = observations[self.rgb_uuid]
        w_ego_img = observations[self.unshuffled_rgb_uuid]
        ego_img_embeddings = self.visual_encoder(
            u_img_emb=ego_img,
            w_img_emb=w_ego_img
        )   # [steps, samplers, vis_feature_embedding_dim]

        # Previous actions (low-level actions)
        prev_action_embeddings = self.prev_action_embedder(
            (masks.long() * (prev_actions.unsqueeze(-1) + 1))
        ).squeeze(-2)   # [steps, samplers, prev_action_embedding_dim]

        to_cat = [
            ego_img_embeddings,
            prev_action_embeddings
        ]
        rnn_hidden_states = memory.tensor("rnn")
        obs_for_rnn = torch.cat(to_cat, dim=-1)

        rnn_outs = []

        # Semantic maps: (sampler, channels, width, length, height)
        sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPE_TO_IDX["Unshuffle"]]
        w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPE_TO_IDX["Walkthrough"]]

        map_masks = masks.view(*masks.shape[:2], 1, 1, 1, 1)
        sem_maps = observations[self.sem_map_uuid]
        w_sem_maps = observations[self.unshuffled_sem_map_uuid]

        for step in range(nsteps):
            sem_map_prev = update_semantic_map(
                sem_map=sem_maps[step],
                sem_map_prev=sem_map_prev,
                map_mask=map_masks[step],
            )
            w_sem_map_prev = update_semantic_map(
                sem_map=w_sem_maps[step],
                sem_map_prev=w_sem_map_prev,
                map_mask=map_masks[step],
            )
            sem_maps_prev = torch.stack(
                (sem_map_prev, w_sem_map_prev), dim=1
            )

            sem_map_embed = self.sem_map_encoder(
                unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
                walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
            )

            rnn_input = torch.cat(
                (
                    obs_for_rnn[step],
                    sem_map_embed,
                ),
                dim=-1
            ).unsqueeze(0)

            rnn_out, rnn_hidden_states = self.state_encoder(
                rnn_input,
                rnn_hidden_states,
                masks[step:step+1]
            )
            
            rnn_outs.append(rnn_out)

        rnn_outs = torch.cat(rnn_outs, dim=0)
                
        extras = {}

        memory = memory.set_tensor(
            key="sem_map",
            tensor=sem_maps_prev.type(torch.bool)
        )
        memory = memory.set_tensor(
            key="rnn",
            tensor=rnn_hidden_states,
        )

        return (
            ActorCriticOutput(
                distributions=self.actor(rnn_outs), values=self.critic(rnn_outs), extras=extras
            ), 
            memory
        )