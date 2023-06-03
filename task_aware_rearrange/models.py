from typing import (
    Mapping,
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

from task_aware_rearrange.constants import ADDITIONAL_MAP_CHANNELS, NUM_OBJECT_TYPES
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


class OnePhaseResNetWithInventoryActorCriticRNN(OnePhaseResNetActorCriticRNN):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        prev_action_embedding_dim: int = 32,
        inventory_embedding_dim: int = 32,
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

        self.inventory_uuid = inventory_uuid
        self.inventory_embedding_dim = inventory_embedding_dim
        self.inventory_embedder = nn.Embedding(
            NUM_OBJECT_TYPES, embedding_dim=self.inventory_embedding_dim
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size
                + self.prev_action_embedding_dim
                + self.inventory_embedding_dim
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

        # Inventory vectors
        # inventory one-hot vector - 0 ~ 71: objects, 72: unknown object
        # Let the index 0 becomes True when the agent is holding an unknown object or not holding
        inventory = observations[self.inventory_uuid]                           # [nsteps, nsamplers, num_objects]
        inventory_index = (inventory.max(-1).indices + 1) % NUM_OBJECT_TYPES    # [nsteps, nsamplers]
        inventory_embeddings = self.inventory_embedder(inventory_index)

        to_cat = [
            ego_img_embeddings,
            prev_action_embeddings,
            inventory_embeddings,
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


class OnePhaseResNetWithInventorySubtaskHistoryActorCriticRNN(OnePhaseResNetWithInventoryActorCriticRNN):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        expert_subtask_uuid: str,
        prev_action_embedding_dim: int = 32,
        inventory_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        num_subtasks: int = NUM_SUBTASKS,
        num_repeats: int = 1,
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            inventory_uuid=inventory_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            inventory_embedding_dim=inventory_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        self.num_repeats = num_repeats

        self.expert_subtask_uuid = expert_subtask_uuid
        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            num_subtasks=num_subtasks,
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 2
                + self.prev_action_embedding_dim 
                + self.inventory_embedding_dim
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

        # Inventory vectors
        # inventory one-hot vector - 0 ~ 71: objects, 72: unknown object
        # Let the index 0 becomes True when the agent is holding an unknown object or not holding
        inventory = observations[self.inventory_uuid]                           # [nsteps, nsamplers, num_objects]
        inventory_index = (inventory.max(-1).indices + 1) % NUM_OBJECT_TYPES    # [nsteps, nsamplers]
        inventory_embeddings = self.inventory_embedder(inventory_index)

        # Histories: (sampler, type, length) - from memory
        history = memory.tensor('agent_history')
        history_masks = masks.view(*masks.shape[:2])
        subtask_history = observations[self.expert_subtask_uuid][..., 0]        # [nsteps, nsamplers]

        if torch.is_grad_enabled():
            # when updating loss
            # generate the sampler-wise subtask_history_embeddings
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                assert (
                    len(self.subtask_history[sampler]) == nsteps + 1
                    and len(self.action_history[sampler]) == nsteps + 1
                )
                subtask_index_history = masks.new_tensor(
                    self.subtask_history[sampler][:-1], dtype=torch.long,
                )
                action_index_history = masks.new_tensor(
                    self.action_history[sampler][:-1], dtype=torch.long
                )
                subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_index_history,
                    seq_masks=history_masks[:, sampler],
                )
                subtask_history_embeddings.append(subtask_history_embedding[:-1])
            
            subtask_history_embeddings = torch.stack(
                subtask_history_embeddings, dim=1
            )   # [nsteps, nsamplers, emb_feat_dims]

        else:
            # When Collecting the step-results (Inference)
            assert nsteps ==1

            # Reset the history memory when the environment is reset
            history_ = (history * masks.squeeze(0).unsqueeze(-1).repeat((1, 2, 1))).long()

            # To identify the valid portion of memory tensor
            nonzero_idxs = []
            for sampler in range(nsamplers):
                idxs = []
                for type in range(2):
                    idxs.append(
                        (history_[sampler, type] == 0).nonzero().min()
                    )
                idxs = torch.stack(idxs)
                nonzero_idxs.append(idxs)
            nonzero_idxs = torch.stack(nonzero_idxs)    # [nsamplers, 2]

            # Initialize the history list
            if (
                len(self.action_history) == 0
                or len(self.action_history) != nsamplers
            ):
                self.action_history = [[] for _ in range(nsamplers)]

            if (
                len(self.subtask_history) == 0
                or len(self.subtask_history) != nsamplers
            ):
                self.subtask_history = [[] for _ in range(nsamplers)]

            # Generate the subtask history embeddins using stored subtask history
            # for `CURRENT` episode from each sampler
            # To distinguish the start of new episode, history memory stores the (index + 1)
            # instead of just storing `index` of subtask/actions
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                subtask_index_history = history_[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
                
                history_[sampler, 1, nonzero_idxs[sampler, 1]] = prev_actions[0][sampler] + 1
                self.action_history[sampler].append(prev_actions[0][sampler].item())

                seq_masks = torch.zeros_like(subtask_index_history)
                subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_index_history,
                    seq_masks=seq_masks
                )   # [episode_len + 1, emb_feat_dims]
                subtask_history_embeddings.append(subtask_history_embedding[-1:])

                history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_history[0, sampler] + 1
                self.subtask_history[sampler].append(subtask_history[0, sampler].item())
            
            memory = memory.set_tensor(
                key="agent_history",
                tensor=history_,
            )
            subtask_history_embeddings = torch.stack(
                subtask_history_embeddings, dim=1
            )   # [1, nsamplers, emb_feat_dims]

        assert subtask_history_embeddings.shape[:2] == masks.shape[:2]

        to_cat = [
            ego_img_embeddings,
            prev_action_embeddings,
            inventory_embeddings,
            subtask_history_embeddings,
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


class OnePhaseResNetWithInventorySubtaskHistoryPredictionActorCriticRNN(OnePhaseResNetWithInventoryActorCriticRNN):
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        prev_action_embedding_dim: int = 32,
        inventory_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        num_subtasks: int = NUM_SUBTASKS,
        num_repeats: int = 1,
        num_losses: int = 2,
    ):
        super().__init__(
            action_space=action_space, 
            observation_space=observation_space,
            rgb_uuid=rgb_uuid,
            unshuffled_rgb_uuid=unshuffled_rgb_uuid,
            inventory_uuid=inventory_uuid,
            prev_action_embedding_dim=prev_action_embedding_dim,
            inventory_embedding_dim=inventory_embedding_dim,
            hidden_size=hidden_size,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        self.num_repeats = num_repeats
        self.num_losses = num_losses

        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            num_subtasks=num_subtasks,
        )

        self.subtask_predictor = SubtaskPredictor(
            input_size=(
                self._hidden_size * 2
                + self.prev_action_embedding_dim 
                + self.inventory_embedding_dim
            ),
            hidden_size=hidden_size,
            num_subtasks=num_subtasks,
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 2
                + self.prev_action_embedding_dim 
                + self.inventory_embedding_dim
            ),
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )

        self.subtask_history = []
        self.action_history = []
        self.repeat_count = 0

    def _reset_history(self, nsamplers: int):
        self.subtask_history = [[] for _ in range(nsamplers)]
        self.action_history = [[] for _ in range(nsamplers)]
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

        # Inventory vectors
        # inventory one-hot vector - 0 ~ 71: objects, 72: unknown object
        # Let the index 0 becomes True when the agent is holding an unknown object or not holding
        inventory = observations[self.inventory_uuid]                           # [nsteps, nsamplers, num_objects]
        inventory_index = (inventory.max(-1).indices + 1) % NUM_OBJECT_TYPES    # [nsteps, nsamplers]
        inventory_embeddings = self.inventory_embedder(inventory_index)

        # Histories: (sampler, type, length) - from memory
        history = memory.tensor('agent_history')
        history_masks = masks.view(*masks.shape[:2])

        if torch.is_grad_enabled():
            # when updating loss
            # generate the sampler-wise subtask_history_embeddings
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                assert (
                    len(self.subtask_history[sampler]) == nsteps + 1
                    and len(self.action_history[sampler]) == nsteps + 1
                ), f"""
                len(self.subtask_history[{sampler}]) = {len(self.subtask_history[sampler])}, 
                len(self.action_history[{sampler}]) = {len(self.action_history[sampler])},
                nsteps + 1 = {nsteps + 1}
                """
                subtask_index_history = masks.new_tensor(
                    self.subtask_history[sampler][:-1], dtype=torch.long,
                )
                action_index_history = masks.new_tensor(
                    self.action_history[sampler][:-1], dtype=torch.long
                )
                subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_index_history,
                    seq_masks=history_masks[:, sampler],
                )
                subtask_history_embeddings.append(subtask_history_embedding[:-1])
            
            subtask_history_embeddings = torch.stack(
                subtask_history_embeddings, dim=1
            )   # [nsteps, nsamplers, emb_feat_dims]

        else:
            # When Collecting the step-results (Inference)
            assert nsteps == 1

            # Reset the history memory when the environment is reset
            history_ = (history * masks.squeeze(0).unsqueeze(-1).repeat((1, 2, 1))).long()

            # To identify the valid portion of memory tensor
            nonzero_idxs = []
            for sampler in range(nsamplers):
                idxs = []
                for type in range(2):
                    idxs.append(
                        (history_[sampler, type] == 0).nonzero().min()
                    )
                idxs = torch.stack(idxs)
                nonzero_idxs.append(idxs)
            nonzero_idxs = torch.stack(nonzero_idxs)    # [nsamplers, 2]

            # Initialize the history list
            if (
                len(self.action_history) == 0
                or len(self.action_history) != nsamplers
            ):
                self.action_history = [[] for _ in range(nsamplers)]

            if (
                len(self.subtask_history) == 0
                or len(self.subtask_history) != nsamplers
            ):
                self.subtask_history = [[] for _ in range(nsamplers)]

            # Generate the subtask history embeddins using stored subtask history
            # for `CURRENT` episode from each sampler
            # To distinguish the start of new episode, history memory stores the (index + 1)
            # instead of just storing `index` of subtask/actions
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                subtask_index_history = history_[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
                
                history_[sampler, 1, nonzero_idxs[sampler, 1]] = prev_actions[0][sampler] + 1
                self.action_history[sampler].append(prev_actions[0][sampler].item())
                memory = memory.set_tensor(
                    key="agent_history",
                    tensor=history_,
                )

                seq_masks = torch.zeros_like(subtask_index_history)
                subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_index_history,
                    seq_masks=seq_masks
                )   # [episode_len + 1, emb_feat_dims]
                subtask_history_embeddings.append(subtask_history_embedding[-1:])

            subtask_history_embeddings = torch.stack(
                subtask_history_embeddings, dim=1
            )   # [1, nsamplers, emb_feat_dims]

        assert subtask_history_embeddings.shape[:2] == masks.shape[:2]

        to_cat = [
            ego_img_embeddings,
            prev_action_embeddings,
            inventory_embeddings,
            subtask_history_embeddings,
        ]
        obs_for_rnn = torch.cat(to_cat, dim=-1)

        rnn_outs, rnn_hidden_states = self.state_encoder(
            obs_for_rnn,
            memory.tensor("rnn"),
            masks
        )

        extras = {}
        subtask_logits = self.subtask_predictor(obs_for_rnn)    # [nsteps, nsamplers, num_subtasks]
        if torch.is_grad_enabled():
            extras["subtask_logits"] = subtask_logits
            self.repeat_count += 1
        else:
            assert nsteps == 1
            history = memory.tensor("agent_history")
            history_ = torch.clone(history)
            if (
                len(self.subtask_history) == 0
                or len(self.subtask_history) != nsamplers
            ):
                self.subtask_history = [[] for _ in range(nsamplers)]

            subtask_index = torch.max(subtask_logits, dim=-1).indices   # [nsteps, nsamplers]
            for sampler in range(nsamplers):
                history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_index[0, sampler] + 1
                self.subtask_history[sampler].append(subtask_index[0, sampler].item())
            
            memory = memory.set_tensor(
                key="agent_history",
                tensor=history_,
            )

        if torch.is_grad_enabled() and self.repeat_count == self.num_repeats:
            self._reset_history(nsamplers=nsamplers)

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
        expert_subtask_uuid: str,
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

        self.expert_subtask_uuid = expert_subtask_uuid
        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            # num_subtask_types=num_subtask_types,
            # num_subtask_arguments=num_subtask_arguments,
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
        subtask_history = observations[self.expert_subtask_uuid]    # [nsteps, nsamplers, 2]

        # histories: (sampler, type, length)
        history = memory.tensor('agent_history')
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
            # if (masks == 0).nonzero().shape[0] > 0:
            #     import pdb; pdb.set_trace()

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

            if (
                len(self.subtask_history) == 0
                or len(self.subtask_history) != nsamplers
            ):
                self.subtask_history = [[] for _ in range(nsamplers)]

            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                subtask_index_history = history_[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
                history_[sampler, 1, nonzero_idxs[sampler, 1]] = prev_actions[0][sampler] + 1 # +1 !!
                self.action_history[sampler].append(prev_actions[0][sampler].item())
                
                seq_masks = torch.zeros_like(subtask_index_history)
                subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_index_history,
                    seq_masks=seq_masks,
                )   # [N + 1, emb_feat_dims]
                subtask_history_embeddings.append(subtask_history_embedding[-1:])

                history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_history[0, sampler, 0] + 1
                self.subtask_history[sampler].append(subtask_history[0, sampler, 0].item())

            memory = memory.set_tensor(
                key="agent_history",
                tensor=history_,
            )

            subtask_history_embeddings = torch.stack(subtask_history_embeddings, dim=1) # [1, nsamplers, emb_feat_dims]

        # subtask_history_embeddings = self.subtask_history_encoder(
        #     subtask_index_history=subtask_history[..., 0].permute(1, 0).reshape(-1).contiguous(),
        #     seq_masks=masks.permute(1, 0, 2).reshape(-1).contiguous(),
        # )
        # subtask_history_embeddings = subtask_history_embeddings[:-1]
        # subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(nsteps, nsamplers, -1)

        # subtask_history_embeddings = self.subtask_history_encoder(
        #     subtask_index_history=subtask_history[..., 0].permute(1, 0).contiguous(),
        #     seq_masks=masks.permute(1, 0, 2).squeeze(-1).contiguous(),
        # )
        # subtask_history_embeddings = subtask_history_embeddings[:-1]
        # subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(nsteps, nsamplers, -1)

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
        if torch.is_grad_enabled() and self.repeat_count == self.num_repeats:
            self._reset_history()

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
                env_embeddings=sem_map_inv_embedding, 
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


# class OnePhaseSubtaskResNet(ActorCriticModel):

#     def __init__(
#         self,
#         action_space: gym.Space,
#         observation_space: gym.spaces.Dict,
#         rgb_uuid: str,
#         unshuffled_rgb_uuid: str,
#         prev_action_embedding_dim: int = 32,
#         hidden_size: int = 512,
#         num_subtasks: int = NUM_SUBTASKS,
#         num_repeats: int = 1,
#     ):
#         super().__init__(action_space=action_space, observation_space=observation_space)
#         self._hidden_size = hidden_size
#         self.prev_action_embedding_dim = prev_action_embedding_dim

#         self.rgb_uuid = rgb_uuid
#         self.unshuffled_rgb_uuid = unshuffled_rgb_uuid

#         self.visual_encoder = EgocentricViewEncoderPooled(
#             img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
#             hidden_dim=self._hidden_size,
#         )
#         self.prev_action_embedder = nn.Embedding(
#             self.action_space.n + 1, embedding_dim=self.prev_action_embedding_dim
#         )

#         self.num_repeats = num_repeats

#         self.subtask_history_encoder = SubtaskHistoryEncoder(
#             hidden_size=hidden_size,
#             num_subtasks=num_subtasks,
#         )
#         self.subtask_predictor = SubtaskPredictor(
#             hidden_size=hidden_size,
#             num_subtasks=num_subtasks,
#         )

#         self.subtask_history = []
#         self.action_history = []
#         self.repeat_count = 0

#     def _reset_history(self):
#         self.subtask_history = []
#         self.action_history = []
#         self.repeat_count = 0

#     def _recurrent_memory_specification(self):
#         return dict(
#             agent_history=(
#                 (
#                     ("sampler", None),
#                     ("history", 2),     # 0 for subtask, 1 for prev_action
#                     ("length", 512),    # history vector length (max_steps // num_mini_batch + 1)
#                 ),
#                 torch.long,
#             )
#         )

#     def forward(
#         self,
#         observations: ObservationType, 
#         memory: Memory, 
#         prev_actions: ActionType, 
#         masks: torch.FloatTensor
#     ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
#         """
#         observations: [steps, samplers, (agents), ...]
#         memory: [sampler, ...] 
#         prev_actions: [steps, samplers, ...]
#         masks: [steps, samplers, agents, 1], zero indicates the steps where a new episode/task starts
#         """
#         nsteps, nsamplers = masks.shape[:2]

#         # Egocentric images
#         ego_img = observations[self.rgb_uuid]
#         w_ego_img = observations[self.unshuffled_rgb_uuid]
#         ego_img_embeddings = self.visual_encoder(
#             u_img_emb=ego_img,
#             w_img_emb=w_ego_img
#         )   # [steps, samplers, vis_feature_embedding_dim]

#         # Previous actions (low-level actions)
#         prev_action_embeddings = self.prev_action_embedder(
#             (masks.long() * (prev_actions.unsqueeze(-1) + 1))
#         ).squeeze(-2)   # [steps, samplers, prev_action_embedding_dim]

#         # Histories (sampler, type, length)
#         history = memory.tensor('agent_history')        # [nsamplers, 2, length]
#         history_masks = masks.view(*masks.shape[:2])    # [nsteps, nsamplers]

#         if torch.is_grad_enabled():
#             # Updating Loss
#             # Generate Sampler-wise Subtask History Embeddings
#             subtask_history_embeddings = []
#             for sampler in range(nsamplers):
#                 assert (
#                     len(self.subtask_history[sampler]) == nsteps + 1
#                     and len(self.action_history[sampler]) == nsteps + 1
#                 )
#                 subtask_index_history = masks.new_tensor(
#                     self.subtask_history[sampler][:-1],
#                     dtype=torch.long
#                 )
#                 subtask_index_history = masks.new_tensor(
#                     self.action_history[sampler][:-1],
#                     dtype=torch.long
#                 )
#                 subtask_history_embedding = self.subtask_history_encoder(
#                     subtask_index_history=subtask_index_history,
#                     seq_masks=history_masks[:, sampler],
#                 )   # [nsteps + 1, emb_feat_dims]
#                 subtask_history_embeddings.append(subtask_history_embedding[:-1])
            
#             subtask_history_embeddings = torch.stack(
#                 subtask_history_embeddings,
#                 dim=1
#             )       # [nsteps, nsamplers, emb_feat_dims]
#             # ForkedPdb().set_trace()
        
#         else:
#             # Stepwise Agent Action Inference
#             # nsteps == 1
#             assert nsteps == 1
#             history_ = (history * masks.squeeze(0).unsqueeze(-1).repeat((1, 2, 1))).long()

#             nonzero_idxs = []
#             for i in range(nsamplers):
#                 idxs = []
#                 for j in range(2):
#                     idxs.append((history_[i, j] == 0).nonzero().min())

#                 idxs = torch.stack(idxs)
#                 nonzero_idxs.append(idxs)

#             nonzero_idxs = torch.stack(nonzero_idxs)

#             if (
#                 len(self.action_history) == 0
#                 or len(self.action_history) != nsamplers
#             ):
#                 self.action_history = [[] for _ in range(nsamplers)]
            
#             subtask_history_embeddings = []
#             for sampler in range(nsamplers):
#                 subtask_index_history = history_[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
#                 history_[sampler, 1, nonzero_idxs[sampler, 1]] = prev_actions[0][sampler] + 1
#                 self.action_history[sampler].append(prev_actions[0][sampler].item())
#                 memory = memory.set_tensor(
#                     key="agent_history",
#                     tensor=history_,
#                 )
#                 seq_masks = torch.zeros_like(subtask_index_history)
#                 subtask_history_embedding = self.subtask_history_encoder(
#                     subtask_index_history=subtask_index_history,
#                     seq_masks=seq_masks,
#                 )       # [N + 1, emb_feat_dims]
#                 subtask_history_embeddings.append(subtask_history_embedding[-1:])
            
#             subtask_history_embeddings = torch.stack(
#                 subtask_history_embeddings,
#                 dim=1
#             )           # [1, nsamplers, emb_feat_dims]

#         # Logits for subtask prediction
#         subtask_logits = []
        
#         # Stepwise calculation especially for Map Generation and Subtask Prediction
#         for step in range(nsteps):
#             # Define the environment feature embedding for current step
#             env_embedding = ego_img_embeddings[step]    # [nsamplers, vis_feature_embedding_dim]

#             # Subtask Prediction
#             subtask_logprob = self.subtask_predictor.forward_embedding(
#                 env_embeddings=env_embedding,
#                 subtask_history_embeddings=subtask_history_embeddings[step],    # [nsamplers, emb_feat_dims]
#             )
#             subtask_index = torch.max(subtask_logprob, dim=-1).indices      # [nsamplers,]
            
#             # # Subtasks from Expert
#             # subtask_index = observations["expert_subtask"][step, :, 0]
#             if torch.is_grad_enabled():
#                 subtask_logits.append(subtask_logprob)
#                 # pass
#             else:
#                 assert nsteps == 1
#                 history = memory.tensor('agent_history')
#                 history_ = torch.clone(history)
#                 if (
#                     len(self.subtask_history) == 0
#                     or len(self.subtask_history) != nsamplers
#                 ):
#                     self.subtask_history = [[] for _ in range(nsamplers)]
#                 for sampler in range(nsamplers):
#                     history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_index[sampler] + 1
#                     self.subtask_history[sampler].append(subtask_index[sampler].item())
                
#                 memory = memory.set_tensor(
#                     key="agent_history",
#                     tensor=history_,
#                 )
        
#         extras = {}
#         if torch.is_grad_enabled():
#             extras["subtask_logits"] = torch.stack(subtask_logits)      # [nsteps, nsamplers, NUM_SUBTASKS]
#             self.repeat_count += 1

#         if torch.is_grad_enabled() and self.repeat_count == self.num_repeats:
#             # Reset the history attributes after finishing the loss update
#             self._reset_history()

#         return (
#             ActorCriticOutput(
#                 distributions=CategoricalDistr(
#                     probs=F.one_hot(observations["expert_action"][..., 0], num_classes=self.action_space.n)
#                 ),  # This agent acts like an expert
#                 values=masks.new_zeros((*masks.shape[:2], 1)),  # Value is meaningless
#                 extras=extras,
#             ),
#             memory,
#         )


class OnePhaseSubtaskAwarePolicy(ActorCriticModel):
    
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        expert_subtask_uuid: str,
        prev_action_embedding_dim: int = 32,
        sap_subtask_history: bool = False,
        sap_semantic_map: bool = False,
        semantic_map_uuid: Optional[str] = None,
        unshuffled_semantic_map_uuid: Optional[str] = None,
        num_map_channels: Optional[int] = None,
        online_subtask_prediction: bool = False,
        osp_egoview: bool = False,
        osp_prev_action: bool = False,
        osp_subtask_history: bool = False,
        osp_semantic_map: bool = False,
        num_repeats: Optional[int] = None,
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
        num_subtasks: int = NUM_SUBTASKS,
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
        
        self.expert_subtask_uuid = expert_subtask_uuid
        
        self.online_subtask_prediction = online_subtask_prediction
        self.osp_egoview = osp_egoview if self.online_subtask_prediction else False
        self.osp_prev_action = osp_prev_action if self.online_subtask_prediction else False
        self.osp_subtask_history = osp_subtask_history if self.online_subtask_prediction else False
        self.osp_semantic_map = osp_semantic_map if self.online_subtask_prediction else False
        
        sap_input_size = 0
        osp_input_size = 0
        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=(3 * self.observation_space[self.rgb_uuid].shape[0]),
            hidden_dim=self._hidden_size,
        )
        sap_input_size += hidden_size
        if self.online_subtask_prediction and self.osp_egoview:
            osp_input_size += hidden_size
        
        self.prev_action_embedder = nn.Embedding(
            self.action_space.n + 1, embedding_dim=self.prev_action_embedding_dim
        )
        sap_input_size += self.prev_action_embedding_dim
        if self.online_subtask_prediction and self.osp_prev_action:
            osp_input_size += self.prev_action_embedding_dim
        
        self.sap_subtask_history = sap_subtask_history if self.online_subtask_prediction else False
        self.sap_semantic_map = sap_semantic_map
        
        if self.sap_semantic_map:
            assert (
                semantic_map_uuid is not None
                and unshuffled_semantic_map_uuid is not None
                and num_map_channels is not None
            ), \
                f"semantic_map_uuid: {semantic_map_uuid} or unshuffled_semantic_map_uuid: {unshuffled_semantic_map_uuid}"\
                f" num_map_channels: {num_map_channels}"
            self.semantic_map_uuid = semantic_map_uuid
            self.unshuffled_semantic_map_uuid = unshuffled_semantic_map_uuid
            self.num_map_channels = num_map_channels
            self.semantic_map_encoder = SemanticMap2DEncoderPooled(
                n_map_channels=self.num_map_channels,
                hidden_size=self._hidden_size,
            )
            sap_input_size += hidden_size
            if self.online_subtask_prediction and self.osp_semantic_map:
                osp_input_size += hidden_size
            
        if self.sap_subtask_history:
            self.subtask_history_encoder = SubtaskHistoryEncoder(
                hidden_size=hidden_size,
                num_subtasks=num_subtasks,
            )
            sap_input_size += hidden_size
            if self.online_subtask_prediction and self.osp_subtask_history:
                osp_input_size += hidden_size
        
        if self.online_subtask_prediction:
            self.online_subtask_predictor = SubtaskPredictor(
                input_size=osp_input_size,
                hidden_size=hidden_size,
                num_subtasks=num_subtasks
            )
        
        self.state_encoder = RNNStateEncoder(
            input_size=sap_input_size,
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )
        
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        
        self.subtask_history = []
        self.action_history = []
        self.repeat_count = 0
        self.num_repeats = num_repeats

    def _reset_history(self):
        self.subtask_history = []
        self.action_history = []
        self.repeat_count = 0
        
    @property
    def map_size(self):
        self._map_size = self.observation_space[self.semantic_map_uuid].shape[-4:]
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
        
    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        memory_spec_dict = dict(
            rnn=(
                (
                    ("layer", self.num_rnn_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self._hidden_size),
                ),
                torch.float32,
            )
        )
        if self.sap_semantic_map:
            memory_spec_dict["sem_map"] = (
                (
                    ("sampler", None),
                    ("map_type", 2),
                    ("channels", self.map_channel),
                    ("width", self.map_width),
                    ("length", self.map_length),
                    ("height", self.map_height),
                ),
                torch.bool,
            )
        if self.sap_subtask_history:
            memory_spec_dict["agent_history"] = (
                (
                    ("sampler", None),
                    ("history", 2),     # 0 for subtask, 1 for prev_action
                    ("length", 512),    # history vector length (max_steps // num_mini_batch + 1)
                ),
                torch.long,
            )
        return memory_spec_dict
        
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
        
        if self.sap_semantic_map:
            # Semantic maps: (sampler, channels, width, length, height)
            sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPE_TO_IDX["Unshuffle"]]
            w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPE_TO_IDX["Walkthrough"]]
            
            map_masks = masks.view(*masks.shape[:2], 1, 1, 1, 1)
            sem_maps = observations[self.semantic_map_uuid]
            w_sem_maps = observations[self.unshuffled_semantic_map_uuid]
            
        
        if self.sap_subtask_history:
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
                
        rnn_hidden_states = memory.tensor("rnn")
        rnn_outs = []
        
        subtask_logits = []
        for step in range(nsteps):
            if self.sap_semantic_map:
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

                semantic_map_embedding = self.semantic_map_encoder(
                    unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
                    walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
                )   # [nsamplers, emb_feat_dims]

            if self.online_subtask_prediction:
                online_subtask_prediction_inputs = []
                if self.osp_egoview:
                    online_subtask_prediction_inputs.append(ego_img_embeddings[step])
                if self.osp_prev_action:
                    online_subtask_prediction_inputs.append(prev_action_embeddings[step])
                if self.sap_semantic_map and self.osp_semantic_map:
                    online_subtask_prediction_inputs.append(semantic_map_embedding)
                if self.sap_subtask_history and self.osp_subtask_history:
                    online_subtask_prediction_inputs.append(subtask_history_embeddings[step])
                    
                osp_inputs = torch.cat(online_subtask_prediction_inputs, dim=-1)
                subtask_logprob = self.online_subtask_predictor(osp_inputs) # [nsamplers, NUM_SUBTASKS]
                subtask_index = torch.max(subtask_logprob, dim=-1).indices  # [nsamplers, ]
                
                if torch.is_grad_enabled():
                    subtask_logits.append(subtask_logprob)
                else:
                    assert nsteps == 1
                    if self.sap_subtask_history:
                        history = memory.tensor('agent_history')
                        history_ = torch.clone(history)
                        
                        if (
                            len(self.subtask_history) == 0
                            or len(self.subtask_history)!= nsamplers
                        ):
                            self.subtask_history = [[] for _ in range(nsamplers)]
                        
                        for sampler in range(nsamplers):
                            history_[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_index[sampler] + 1
                            self.subtask_history[sampler].append(subtask_index[sampler].item())
                            
                        memory = memory.set_tensor(
                            key="agent_history",
                            tensor=history_,
                        )
            
            rnn_input = [
                ego_img_embeddings[step],
                prev_action_embeddings[step],
            ]
            if self.sap_semantic_map:
                rnn_input.append(semantic_map_embedding)
            if self.sap_subtask_history:
                rnn_input.append(subtask_history_embeddings[step])
                
            rnn_input = torch.cat(rnn_input, dim=-1).unsqueeze(0)
            
            rnn_out, rnn_hidden_states = self.state_encoder(
                rnn_input,
                rnn_hidden_states,
                masks[step:step + 1]
            )
            
            rnn_outs.append(rnn_out)
        
        rnn_outs = torch.cat(rnn_outs, dim=0)
        
        extras = {}
        
        if torch.is_grad_enabled():
            if len(subtask_logits) > 0:
                extras["subtask_logits"] = torch.stack(subtask_logits)  # [nsteps, nsamplers, NUM_SUBTASKS]
            self.repeat_count += 1
        
        if self.sap_semantic_map:
            memory = memory.set_tensor(
                key="sem_map",
                tensor=sem_maps_prev.type(torch.bool),
            )
        memory = memory.set_tensor(
            key="rnn",
            tensor=rnn_hidden_states,
        )
        
        if torch.is_grad_enabled() and self.repeat_count == self.num_repeats:
            self._reset_history()
            
        return (
            ActorCriticOutput(
                distributions=self.actor(rnn_outs), values=self.critic(rnn_outs), extras=extras,
            ),
            memory
        )


class TwoPhaseSubtaskAwarePolicy(ActorCriticModel):
    
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        expert_subtask_uuid: str,
        done_action_index: int,
        in_walkthrough_phase_uuid: str,
        num_steps: int,
        is_walkthrough_phase_embedding_dim: int = 32,
        prev_action_embedding_dim: int = 32,
        sap_subtask_history: bool = False,
        sap_semantic_map: bool = False,
        semantic_map_uuid: Optional[str] = None,
        num_map_channels: Optional[int] = None,
        online_subtask_prediction: bool = False,
        osp_egoview: bool = False,
        osp_prev_action: bool = False,
        osp_walkthrough_phase: bool = False,
        osp_subtask_history: bool = False,
        osp_semantic_map: bool = False,
        num_repeats: Optional[int] = None,
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
        num_subtasks: int = NUM_SUBTASKS,
        walkthrougher_should_ignore_action_mask: Optional[Sequence[float]] = None,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size
        self.prev_action_embedding_dim = prev_action_embedding_dim
        
        self.done_action_index = done_action_index
        self.in_walkthrough_phase_uuid = in_walkthrough_phase_uuid
        self.is_walkthrough_phase_embedding_dim = is_walkthrough_phase_embedding_dim
        self.walkthrougher_should_ignore_action_mask = walkthrougher_should_ignore_action_mask
        
        self.rgb_uuid = rgb_uuid
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.num_steps = num_steps
        
        self.expert_subtask_uuid = expert_subtask_uuid
        
        self.online_subtask_prediction = online_subtask_prediction
        self.osp_egoview = osp_egoview if self.online_subtask_prediction else False
        self.osp_prev_action = osp_prev_action if self.online_subtask_prediction else False
        self.osp_subtask_history = osp_subtask_history if self.online_subtask_prediction else False
        self.osp_semantic_map = osp_semantic_map if self.online_subtask_prediction else False
        self.osp_walkthrough_phase = osp_walkthrough_phase if self.online_subtask_prediction else False
        
        sap_input_size = 0
        osp_input_size = 0
        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        sap_input_size += hidden_size
        if self.online_subtask_prediction and self.osp_egoview:
            osp_input_size += hidden_size
        
        self.prev_action_embedder = nn.Embedding(
            self.action_space.n + 1, embedding_dim=self.prev_action_embedding_dim
        )
        sap_input_size += self.prev_action_embedding_dim
        if self.online_subtask_prediction and self.osp_prev_action:
            osp_input_size += self.prev_action_embedding_dim
            
        self.is_walkthrough_phase_embedder = nn.Embedding(
            num_embeddings=2, embedding_dim=is_walkthrough_phase_embedding_dim
        )
        sap_input_size += is_walkthrough_phase_embedding_dim
        sap_input_size += hidden_size
        if self.online_subtask_prediction and self.osp_walkthrough_phase:
            osp_input_size += is_walkthrough_phase_embedding_dim
            osp_input_size += hidden_size
        
        self.walkthrough_good_action_logits: Optional[torch.Tensor]
        if walkthrougher_should_ignore_action_mask is not None:
            self.register_buffer(
                "walkthrough_good_action_logits",
                -1000 * torch.FloatTensor(walkthrougher_should_ignore_action_mask),
                persistent=False,
            )
        else:
            self.walkthrough_good_action_logits = None
        
        self.sap_subtask_history = sap_subtask_history if self.online_subtask_prediction else False
        self.sap_semantic_map = sap_semantic_map
        
        if self.sap_semantic_map:
            assert (
                semantic_map_uuid is not None
                and num_map_channels is not None
            ), \
                f"semantic_map_uuid: {semantic_map_uuid} num_map_channels: {num_map_channels}"
            self.semantic_map_uuid = semantic_map_uuid
            self.num_map_channels = num_map_channels
            self.semantic_map_encoder = SemanticMap2DEncoderPooled(
                n_map_channels=self.num_map_channels,
                hidden_size=self._hidden_size,
            )
            sap_input_size += hidden_size
            if self.online_subtask_prediction and self.osp_semantic_map:
                osp_input_size += hidden_size
            
        if self.sap_subtask_history:
            self.subtask_history_encoder = SubtaskHistoryEncoder(
                hidden_size=hidden_size,
                num_subtasks=num_subtasks,
            )
            sap_input_size += hidden_size
            if self.online_subtask_prediction and self.osp_subtask_history:
                osp_input_size += hidden_size
        
        if self.online_subtask_prediction:
            self.online_subtask_predictor = SubtaskPredictor(
                input_size=osp_input_size,
                hidden_size=hidden_size,
                num_subtasks=num_subtasks
            )
        
        self.state_encoder = RNNStateEncoder(
            input_size=sap_input_size,
            hidden_size=self._hidden_size,
            num_layers=self.num_rnn_layers,
            rnn_type=self.rnn_type,
        )
        
        self.walkthrough_encoder = RNNStateEncoder(
            self._hidden_size, self._hidden_size, num_layers=1, rnn_type="GRU",
        )
        
        self.walkthrough_actor = LinearActorHead(self._hidden_size, action_space.n)
        self.walkthrough_actor.linear.bias.data[self.done_action_index] -= 3    # give a bias to the done action
        self.walkthrough_critic = LinearCriticHead(self._hidden_size)
        self.unshuffle_actor = LinearActorHead(self._hidden_size, action_space.n)
        self.unshuffle_critic = LinearCriticHead(self._hidden_size)
        
        self.subtask_history = [[]]
        self.repeat_count = 0
        self.num_repeats = num_repeats
        
        self.train()
        
    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        for key in list(state_dict.keys()):
            if "explore" in key:
                new_key = key.replace("explore", "walkthrough")
                assert new_key not in state_dict
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        if "walkthrough_good_action_logits" in state_dict:
            del state_dict["walkthrough_good_action_logits"]
            
        return super(TwoPhaseSubtaskAwarePolicy, self).load_state_dict(state_dict, strict)
    
    def _reset_history(self, nsamplers: int):
        self.subtask_history = [[] for _ in range(nsamplers)]
        self.num_steps = 0
        self.repeat_count = 0

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers
    
    @property
    def recurrent_hidden_state_size(self) -> int:
        return self._hidden_size
    
    @property
    def map_size(self):
        if self.semantic_map_uuid is not None:
            self._map_size = self.observation_space[self.semantic_map_uuid].shape[-4:]
            return self._map_size
        else:
            return None

    @property
    def map_channel(self):
        if self.map_size is not None:
            return self.map_size[0]
        return None

    @property
    def map_width(self):
        if self.map_size is not None:
            return self.map_size[1]
        return None

    @property
    def map_length(self):
        if self.map_size is not None:
            return self.map_size[2]
        return None

    @property
    def map_height(self):
        if self.map_size is not None:
            return self.map_size[3]
        return None
    
    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        memory_spec_dict = dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            walkthrough_encoding=(
                (
                    ("layer", self.walkthrough_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )
        if self.sap_semantic_map:
            assert (
                self.map_channel is not None
                and self.map_width is not None
                and self.map_length is not None
                and self.map_height is not None
            )
            memory_spec_dict["semmap"] = (
                (
                    ("sampler", None),
                    ("map_type", 2),
                    ("channels", self.map_channel),
                    ("width", self.map_width),
                    ("length", self.map_length),
                    ("height", self.map_height),
                ),
                torch.bool,
            )
        if self.sap_subtask_history:
            memory_spec_dict["agent_history"] = (
                (
                    ("sampler", None),
                    ("history", 2),     # 0 for subtask, 1 for prev_action
                    ("length", 500),    # history vector length (max_steps // num_mini_batch + 1)
                ),
                torch.long,
            )
        return memory_spec_dict
        
    def forward(
        self,
        observations: ObservationType, 
        memory: Memory, 
        prev_actions: ActionType, 
        masks: torch.FloatTensor
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """
        observations: [steps, samplers, (agents), ...]
            self.rgb_uuid ("rgb_resnet" or "rgb")
                torch.float32,  [nsteps, nsamplers, egoview_feat_dim, height / 32, width / 32]
            self.depth_uuid ("depth")
                torch.float32,  [nsteps, nsamplers, height, width, 1]
            self.in_walkthrough_phase_uuid ("in_walkthrough_phase")
                torch.bool,     [nsteps, nsamplers, 1]
            self.expert_subtask_uuid ("expert_subtask")
                torch.int64,    [nsteps, nsamplers, 2]
            self.expert_action_uuid ("expert_action")
                torch.int64,    [nsteps, nsamplers, 2]
            self.semantic_map_uuid ("semmap")
                torch.bool,     [nsteps, nsamplers, map_channels, map_width, map_length, map_height]
                
        memory: [sampler, ...] 
            "rnn"       for hidden_states of state_encoder
                torch.float32,  [nlayers, nsamplers, _hidden_size]
            "walkthrough_encoding"      for hidden_states of walkthrough_encoder
                torch.float32,  [nlayers, nsamplers, _hidden_size]
            "sem_map"   for storing semantic map in walkthrough env (0) and unshuffle env (1)
                torch.bool,     [nsamplers, nenvs, map_channels, map_width, map_length, map_height]
            "agent_history"     for storing agent's history of inferred subtasks (0) & took actions (1)
                torch.int64,    [nsamplers, 2, buffer_length]
        prev_actions:       action_index
            torch.int64,    [steps, samplers]
            
        masks: [steps, samplers, 1], zero indicates the steps where a new episode/task starts
            torch.float32,  [nsteps, nsamplers, 1]
        """
        nsteps, nsamplers = masks.shape[:2]
        in_walkthrough_phase_mask = observations[self.in_walkthrough_phase_uuid]
        in_unshuffle_phase_mask = ~in_walkthrough_phase_mask
        in_walkthrough_float = in_walkthrough_phase_mask.float()
        in_unshuffle_float = in_unshuffle_phase_mask.float()
        
        # Don't reset hidden state at the start of the unshuffle task
        masks_no_unshuffle_reset = (masks.bool() | in_unshuffle_phase_mask).float()

            
        # masks.bool() is True during episode except the start of the episode
        # in_unshuffle_phase_mask is True when the episode is in the unshuffle phase
        # Thus, masks_no_unshuffle_reset is only False at the start of the walkthrough phase
        masks_with_unshuffle_reset = masks.float()
        del masks
        
        # Egocentric Image
        ego_image = observations[self.rgb_uuid]
        ego_img_embeddings = self.visual_encoder(ego_image)     # [steps, samplers, _hidden_size]
        
        # Previous actions (low-level actions)
        masked_prev_actions = (
            (masks_with_unshuffle_reset.bool()).long()
            * (prev_actions.unsqueeze(-1) + 1)
        )   # [steps, samplers, 1]
        prev_action_embeddings = self.prev_action_embedder(
            masked_prev_actions
        ).squeeze(-2)   # [steps, samplers, prev_action_embedding_dim]
        
        # Is Walkthrough Phase
        is_walkthrough_phase_embeddings = self.is_walkthrough_phase_embedder(
            in_walkthrough_phase_mask.long()
        ).squeeze(-2)   # [steps, samplers, is_walkthrough_phase_embedding_dim]
        walkthrough_encoding = memory.tensor("walkthrough_encoding")
        
        # RNN hidden states
        rnn_hidden_states = memory.tensor("rnn")
        
        rnn_output_list = []
        
        if self.sap_semantic_map:
            # Semantic maps: (sampler, channels, width, length, height)
            w_semmap_prev = memory.tensor('semmap')[:, MAP_TYPE_TO_IDX["Walkthrough"]]
            u_semmap_prev = memory.tensor('semmap')[:, MAP_TYPE_TO_IDX["Unshuffle"]]
            
            map_masks = masks_no_unshuffle_reset.view(nsteps, nsamplers, 1, 1, 1, 1)
            in_walkthrough_map_masks = in_walkthrough_float.view(nsteps, nsamplers, 1, 1, 1, 1)
            in_unshuffle_map_masks = in_unshuffle_float.view(nsteps, nsamplers, 1, 1, 1, 1)
            
            semmaps = observations[self.semantic_map_uuid]
            w_semmaps = (semmaps * in_walkthrough_map_masks).bool()
            u_semmaps = (semmaps * in_unshuffle_map_masks).bool()
                
        if self.sap_subtask_history:
            history = memory.tensor('agent_history')        # (nsamplers, 2, buffer_length)
            
            history_masks = masks_with_unshuffle_reset.view(nsteps, nsamplers, 1, 1)
            in_walkthrough_history_masks = in_walkthrough_float.view(nsteps, nsamplers, 1, 1)
            in_unshuffle_history_masks = in_unshuffle_float.view(nsteps, nsamplers, 1, 1)
            
            # Reset the agent history if the task is started at step 0
            history = (history * history_masks[0].bool()).long()
            
            nonzero_idxs = []
            for i in range(nsamplers):
                idxs = []
                for j in range(2):
                    idxs.append((history[i, j] == 0).nonzero().min())
                idxs = torch.stack(idxs)
                nonzero_idxs.append(idxs)
            nonzero_idxs = torch.stack(nonzero_idxs)
            
            subtask_history_embeddings = []
            for sampler in range(nsamplers):
                subtask_history = history[sampler, 0, :(nonzero_idxs[sampler, 0])] - 1
                if not torch.is_grad_enabled():
                    # Update action history based on the prev_actions
                    history[
                        sampler,
                        1,
                        nonzero_idxs[sampler, 1] : nonzero_idxs[sampler, 1] + nsteps
                    ] = masked_prev_actions[:, sampler]
                    memory = memory.set_tensor("agent_history", history)
                    
                    seq_masks = torch.zeros_like(subtask_history)
                else:
                    # When the loss is updated
                    # append agent history with rollout history
                    assert (
                        len(self.subtask_history[sampler]) == nsteps + 1 == self.num_steps + 1
                    )
                    seq_masks = torch.zeros_like(subtask_history)
                    subtask_history = [
                        subtask_history,
                        subtask_history.new_tensor(self.subtask_history[sampler][:-1], dtype=torch.long)
                    ]
                    subtask_history = torch.cat(subtask_history, dim=-1)
                    seq_masks = [
                        seq_masks,
                        history_masks[:, sampler].view(nsteps).type(seq_masks.dtype)
                    ]
                    seq_masks = torch.cat(seq_masks, dim=-1)
                
                sampler_subtask_history_embedding = self.subtask_history_encoder(
                    subtask_index_history=subtask_history,
                    seq_masks=seq_masks,
                )       # [len(subtask_history) + 1, _hidden_size]
                subtask_history_embeddings.append(
                    sampler_subtask_history_embedding[-nsteps:]
                )       # [nsteps, _hidden_size]
            subtask_history_embeddings = torch.stack(
                subtask_history_embeddings, dim=1
            )   # [nsteps, nsamplers, _hidden_size]
            
        subtask_logits = [] if self.online_subtask_prediction else None
        for step in range(nsteps):
            if self.sap_semantic_map:
                w_semmap_prev = update_semantic_map(
                    sem_map=w_semmaps[step],
                    sem_map_prev=w_semmap_prev,
                    map_mask=map_masks[step],
                )
                u_semmap_prev = update_semantic_map(
                    sem_map=u_semmaps[step],
                    sem_map_prev=u_semmap_prev,
                    map_mask=map_masks[step],
                )
                
                # Update map for agent position
                current_agent_map = semmaps[step, :, 0:1]
                w_semmap_prev[:, 0:1] = current_agent_map
                u_semmap_prev[:, 0:1] = current_agent_map
                
                semmaps_prev = torch.stack(
                    (w_semmap_prev, u_semmap_prev), dim=1
                )
                
                semantic_map_embedding = self.semantic_map_encoder(
                    unshuffle_sem_map_data=u_semmap_prev.max(-1).values,
                    walkthrough_sem_map_data=w_semmap_prev.max(-1).values,
                )   # [nsampler, emb_feat_dims]
            
            if self.online_subtask_prediction:
                online_subtask_prediction_inputs = []
                if self.osp_egoview:
                    online_subtask_prediction_inputs.append(ego_img_embeddings[step])
                if self.osp_prev_action:
                    online_subtask_prediction_inputs.append(prev_action_embeddings[step])
                if self.osp_walkthrough_phase:
                    online_subtask_prediction_inputs.append(is_walkthrough_phase_embeddings[step])
                    online_subtask_prediction_inputs.append(
                        walkthrough_encoding[0] * masks_no_unshuffle_reset[step],
                    )
                if self.sap_semantic_map and self.osp_semantic_map:
                    online_subtask_prediction_inputs.append(semantic_map_embedding)
                if self.sap_subtask_history and self.osp_subtask_history:
                    online_subtask_prediction_inputs.append(subtask_history_embeddings[step])
                
                osp_inputs = torch.cat(online_subtask_prediction_inputs, dim=-1)
                subtask_logprob = self.online_subtask_predictor(osp_inputs) # [nsamplers, NUM_SUBTASKS]
                subtask_index = torch.max(subtask_logprob, dim=-1).indices  # [nsamplers, ]
                # during the walkthrough task, we enforce the subtask to explore (0)
                subtask_index = (
                    subtask_index * in_unshuffle_phase_mask[step].squeeze(-1)
                )
                
                if not torch.is_grad_enabled():
                    if len(self.subtask_history) != nsamplers:
                        self._reset_history(nsamplers=nsamplers)
                    for sampler in range(nsamplers):
                        history[sampler, 0, nonzero_idxs[sampler, 0]] = subtask_index[sampler] + 1
                        self.subtask_history[sampler].append(
                            subtask_index[sampler].item()
                        )
                    memory = memory.set_tensor("agent_history", history)
                subtask_logits.append(subtask_logprob)

            rnn_input = [
                ego_img_embeddings[step],
                prev_action_embeddings[step],
                is_walkthrough_phase_embeddings[step],
                walkthrough_encoding[0] * masks_no_unshuffle_reset[step],
            ]
            if self.sap_semantic_map:
                rnn_input.append(semantic_map_embedding)
            if self.sap_subtask_history:
                rnn_input.append(subtask_history_embeddings[step])
            
            rnn_input = torch.cat(rnn_input, dim=-1).unsqueeze(0)
            
            rnn_out, rnn_hidden_states = self.state_encoder(
                rnn_input,
                rnn_hidden_states,
                masks_no_unshuffle_reset[step : step + 1]
            )
            rnn_output_list.append(rnn_out)
            
            new_walkthrough_encoding, _ = self.walkthrough_encoder(
                rnn_out,
                walkthrough_encoding,
                masks_no_unshuffle_reset[step : step + 1],
            )
            walkthrough_encoding = (
                walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + new_walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )
            
        memory = memory.set_tensor("walkthrough_encoding", walkthrough_encoding)
        memory = memory.set_tensor("rnn", rnn_hidden_states)
        if self.sap_semantic_map:
            memory = memory.set_tensor("semmap", semmaps_prev.type(torch.bool))
        
        rnn_out = torch.cat(rnn_output_list, dim=0)

        extras = {}
        if self.online_subtask_prediction:
            if torch.is_grad_enabled():
                assert len(subtask_logits) > 0
                extras["subtask_logits"] = torch.stack(subtask_logits)  # [nsteps, nsamplers, NUM_SUBTASKS]
                self.repeat_count += 1
                if self.repeat_count == self.num_repeats:
                    self._reset_history(nsamplers=nsamplers)

        walkthrough_dist = self.walkthrough_actor(rnn_out)
        walkthrough_vals = self.walkthrough_critic(rnn_out)
        unshuffle_dist = self.unshuffle_actor(rnn_out)
        unshuffle_vals = self.unshuffle_critic(rnn_out)
        
        assert len(in_walkthrough_float.shape) == len(walkthrough_dist.logits.shape)
        
        if self.walkthrough_good_action_logits is not None:
            walkthrough_logits = (
                walkthrough_dist.logits
                + self.walkthrough_good_action_logits.view(
                    *((1, ) * (len(walkthrough_dist.logits.shape) - 1)), -1
                )
            )
        else:
            walkthrough_logits = walkthrough_dist.logits
            
        actor = CategoricalDistr(
            logits=(
                in_walkthrough_float * walkthrough_logits
                + in_unshuffle_float * unshuffle_dist.logits
            )
        )
        values = (
            in_walkthrough_float * walkthrough_vals
            + in_unshuffle_float * unshuffle_vals
        )
        
        return (
            ActorCriticOutput(distributions=actor, values=values, extras=extras),
            memory,
        )
        # test_logits = torch.rand_like(walkthrough_dist.logits)
        # test_logits[..., 0] -= 3
        # return (
        #     ActorCriticOutput(
        #         distributions=CategoricalDistr(logits=test_logits),
        #         values=torch.rand_like(in_walkthrough_float),
        #         extras=extras,
        #     ),
        #     memory
        # )