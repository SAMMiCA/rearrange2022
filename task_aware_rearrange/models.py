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
from torch import Tensor, is_grad_enabled

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

from task_aware_rearrange.layers import EgocentricViewEncoderPooled, Semantic2DMapWithInventoryEncoderPooled, SemanticMap2DEncoderPooled, SubtaskHistoryEncoder, SubtaskPredictor
from task_aware_rearrange.subtasks import MAP_TYPE_TO_IDX, NUM_SUBTASK_TARGET_OBJECTS, NUM_SUBTASK_TYPES, NUM_SUBTASKS
from task_aware_rearrange.mapping_utils import update_semantic_map
from task_aware_rearrange.utils import ForkedPdb


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
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        self.prev_action_embedding_dim = prev_action_embedding_dim

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_recurrent_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers

        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        self.prev_action_embedder = nn.Embedding(
            self.action_space.n + 1, embedding_dim=self.prev_action_embedding_dim
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size
                + self.prev_action_embedding_dim  
            ),
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_recurrent_layers),
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
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.n_map_channels = NUM_OBJECT_TYPES + ADDITIONAL_MAP_CHANNELS
        self.sem_map_encoder = SemanticMap2DEncoderPooled(
            n_map_channels=self.n_map_channels,
            hidden_size=self._hidden_size,
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

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_recurrent_layers),
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

    @property
    def map_size(self):
        self._map_size = self.observation_space[self.sem_map_uuid].shape[-4:]
        return self._map_size

    @property
    def map_channel(self):
        return self.map_size[0]

    @property
    def map_width(self):
        return self.map_size[1]

    @property
    def map_length(self):
        return self.map_size[2]

    @property
    def map_height(self):
        return self.map_size[3]


class OnePhaseSemanticMappingWithInventoryActorCriticRNN(OnePhaseResNetActorCriticRNN):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        inventory_uuid: str,
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid
        self.inventory_uuid = inventory_uuid

        self.n_map_channels = NUM_OBJECT_TYPES + ADDITIONAL_MAP_CHANNELS
        self.sem_map_inv_encoder = Semantic2DMapWithInventoryEncoderPooled(
            n_map_channels=self.n_map_channels,
            hidden_size=self._hidden_size,
            additional_map_channels=ADDITIONAL_MAP_CHANNELS,
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

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_recurrent_layers),
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
        inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]

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

            sem_map_inv_embedding = self.sem_map_inv_encoder(
                unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
                walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
                inventory_vector=inventory_vectors[step],
            )

            rnn_input = torch.cat(
                (
                    obs_for_rnn[step],
                    sem_map_inv_embedding,
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

    @property
    def map_size(self):
        self._map_size = self.observation_space[self.sem_map_uuid].shape[-4:]
        return self._map_size

    @property
    def map_channel(self):
        return self.map_size[0]

    @property
    def map_width(self):
        return self.map_size[1]

    @property
    def map_length(self):
        return self.map_size[2]

    @property
    def map_height(self):
        return self.map_size[3]


class OnePhaseSemanticMappingWithInventorySubtaskHistoryActorCriticRNN(OnePhaseSemanticMappingWithInventoryActorCriticRNN):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        inventory_uuid: str,
        # expert_subtask_uuid: str,
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            sem_map_uuid=sem_map_uuid,
            unshuffled_sem_map_uuid=unshuffled_sem_map_uuid,
            inventory_uuid=inventory_uuid,
        )

        # self.expert_subtask_uuid = expert_subtask_uuid
        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            num_subtask_types=num_subtask_types,
            num_subtask_arguments=num_subtask_arguments,
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 3
                + self.prev_action_embedding_dim  
            ),
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
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
        inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]
        subtask_history = observations[self.expert_subtask_uuid]    # [nsteps, nsamplers, 2]

        subtask_history_embeddings = self.subtask_history_encoder(
            subtask_index_history=subtask_history[..., 0].permute(1, 0).reshape(-1).contiguous(),
            seq_masks=masks.permute(1, 0, 2).reshape(-1).contiguous(),
        )
        subtask_history_embeddings = subtask_history_embeddings[:-1]
        subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(nsteps, nsamplers, -1)

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

            sem_map_inv_embedding = self.sem_map_inv_encoder(
                unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
                walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
                inventory_vector=inventory_vectors[step],
            )

            rnn_input = torch.cat(
                (
                    obs_for_rnn[step],
                    sem_map_inv_embedding,
                    subtask_history_embeddings[step],
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


class OnePhaseTaskAwareActorCriticRNN(OnePhaseSemanticMappingWithInventoryActorCriticRNN):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        inventory_uuid: str,
        # expert_subtask_uuid: str,
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
        num_subtasks: int = NUM_SUBTASKS,
        num_repeats: int = 1,
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            sem_map_uuid=sem_map_uuid,
            unshuffled_sem_map_uuid=unshuffled_sem_map_uuid,
            inventory_uuid=inventory_uuid,
        )

        self.num_repeats = num_repeats

        # self.expert_subtask_uuid = expert_subtask_uuid
        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            # num_subtask_types=num_subtask_types,
            # num_subtask_arguments=num_subtask_arguments,
            num_subtasks=num_subtasks,
        )

        self.subtask_predictor = SubtaskPredictor(
            hidden_size=hidden_size,
            # num_subtask_types=num_subtask_types,
            # num_subtask_arguments=num_subtask_arguments,
            # joint_prob=False,
            num_subtasks=num_subtasks,
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 3
                + self.prev_action_embedding_dim  
            ),
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )

        self.subtask_history = []
        self.action_history = []
        self.repeat_count = 0

    def _reset_history(self):
        self.subtask_history = []
        self.action_history = []
        self.repeat_count = 0

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
            rnn=(
                (
                    ("layer", self.num_rnn_recurrent_layers),
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
            agent_history=(
                (
                    ("sampler", None),
                    ("history", 2),     # 0 for subtask, 1 for prev_action
                    ("length", 512),    # history vector length (max_steps // num_mini_batch + 1)
                ),
                torch.long,
            )
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
        inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]
        # subtask_history = observations[self.expert_subtask_uuid]    # [nsteps, nsamplers, 2]

        # histories: (sampler, type, length)
        history = memory.tensor('agent_history')        # [nsamplers, 2, length]
        history_masks = masks.view(*masks.shape[:2])

        if torch.is_grad_enabled():
            # when the loss is updated
            # Generate sampler-wise subtask_history_embeddings
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                assert (
                    len(self.subtask_history[sampler]) == nsteps + 1
                    and len(self.action_history[sampler]) == nsteps + 1
                ), f"????"
                subtask_index_history = masks.new_tensor(self.subtask_history[sampler][:-1], dtype=torch.long)
                action_index_history = masks.new_tensor(self.action_history[sampler][:-1], dtype=torch.long)
                subtask_history_embedding = self.subtask_history_encoder(subtask_index_history=subtask_index_history, seq_masks=history_masks[:, sampler],)   # [nsteps + 1, emb_feat_dims]
                subtask_history_embeddings.append(subtask_history_embedding[:-1])
            
            subtask_history_embeddings = torch.stack(subtask_history_embeddings, dim=1) # [nsteps, nsamplers, emb_feat_dims]

        else:
            # nsteps == 1
            assert nsteps == 1
            # reset history memory when environment reset
            history_ = (history * masks.squeeze(0).unsqueeze(-1).repeat((1, 2, 1))).long()

            nonzero_idxs = []
            for i in range(nsamplers):
                idxs = []
                for j in range(2):
                    idxs.append((history_[i, j] == 0).nonzero().min())
                idxs = torch.stack(idxs)
                nonzero_idxs.append(idxs)
            nonzero_idxs = torch.stack(nonzero_idxs)

            if (
                len(self.action_history) == 0
                or len(self.action_history) != nsamplers
            ):
                self.action_history = [[] for _ in range(nsamplers)]

            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                subtask_index_history = history_[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
                history_[sampler, 1, nonzero_idxs[sampler, 1]] = prev_actions[0][sampler] + 1 # +1 !!
                self.action_history[sampler].append(prev_actions[0][sampler].item())
                memory = memory.set_tensor(
                    key="agent_history",
                    tensor=history_,
                )
                seq_masks = torch.zeros_like(subtask_index_history)
                subtask_history_embedding = self.subtask_history_encoder(subtask_index_history=subtask_index_history, seq_masks=seq_masks,)   # [N + 1, emb_feat_dims]
                subtask_history_embeddings.append(subtask_history_embedding[-1:])
            
            subtask_history_embeddings = torch.stack(subtask_history_embeddings, dim=1) # [1, nsamplers, emb_feat_dims]

        subtask_logits = []
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

            sem_map_inv_embedding = self.sem_map_inv_encoder(
                unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
                walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
                inventory_vector=inventory_vectors[step],
            )   # [nsamplers, emb_feat_dims]

            subtask_logprob = self.subtask_predictor.forward_embedding(
                sem_map_inv_embeddings=sem_map_inv_embedding, 
                subtask_history_embeddings=subtask_history_embeddings[step],
            )   # [nsamplers, NUM_SUBTASKS]
            subtask_index = torch.max(subtask_logprob, dim=-1).indices  # [nsamplers, ]

            if torch.is_grad_enabled():
                subtask_logits.append(subtask_logprob)
            else:
                assert nsteps == 1
                history = memory.tensor('agent_history')
                history_ = torch.clone(history)
                if (
                    len(self.subtask_history) == 0
                    or len(self.subtask_history) != nsamplers
                ):
                    self.subtask_history = [[] for _ in range(nsamplers)]
                for sampler in range(nsamplers):
                    history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_index[sampler] + 1 # +1 !!
                    # history_[sampler, 1, nonzero_idxs[sampler, 1] + 1] = prev_actions[step][sampler]
                    self.subtask_history[sampler].append(subtask_index[sampler].item())
                
                memory = memory.set_tensor(
                    key="agent_history",
                    tensor=history_,
                )

            rnn_input = torch.cat(
                (
                    obs_for_rnn[step],
                    sem_map_inv_embedding,
                    subtask_history_embeddings[step],
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
        if torch.is_grad_enabled():
            extras["subtask_logits"] = torch.stack(subtask_logits)  # [nsteps, nsamplers, NUM_SUBTASKS]
            self.repeat_count += 1

        memory = memory.set_tensor(
            key="sem_map",
            tensor=sem_maps_prev.type(torch.bool)
        )
        memory = memory.set_tensor(
            key="rnn",
            tensor=rnn_hidden_states,
        )

        if torch.is_grad_enabled() and self.repeat_count == self.num_repeats:
            self._reset_history()

        return (
            ActorCriticOutput(
                distributions=self.actor(rnn_outs), values=self.critic(rnn_outs), extras=extras
            ), 
            memory
        )