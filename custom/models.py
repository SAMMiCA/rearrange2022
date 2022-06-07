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
from custom.utils import batch_ids_to_ranges, build_attention_masks, index_to_onehot, masks_to_batch_ids, positional_encoding, subtask_index_to_type_arg_target_map
from rearrange.constants import OBJECT_TYPES_WITH_PROPERTIES, OPENABLE_OBJECTS, PICKUPABLE_OBJECTS
from custom.constants import ADDITIONAL_MAP_CHANNELS, IDX_TO_OBJECT_TYPE, NUM_MAP_TYPES, NUM_OBJECT_TYPES, NUM_SUBTASK_TYPES, ORDERED_OBJECT_TYPES
from custom.hlsm.image_to_voxel import ImageToVoxels
from custom.hlsm.voxel_3d_observability import Voxel3DObservability
from custom.hlsm.voxel_grid import GridParameters, VoxelGrid, DefaultGridParameters
from custom.subtask import IDX_TO_MAP_TYPE, IDX_TO_SUBTASK_TYPE, MAP_TYPES_TO_IDX, SUBTASK_TYPES, Subtask
from custom.voxel_utils import create_empty_voxel_data, image_to_semantic_maps, update_semantic_map
from example_utils import ForkedPdb


class EgocentricViewEncoderPooled(nn.Module):

    def __init__(
        self,
        img_embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * img_embedding_dim, hidden_dim, 1, ),
            nn.ReLU(inplace=True),
        )

        attention_dim = int(hidden_dim / 4)
        self.attention = nn.Sequential(
            nn.Conv2d(3 * img_embedding_dim, attention_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1, ),
        )

        self.img_embedding_dim = img_embedding_dim
        self.hidden_dim = hidden_dim

    def forward(
        self, 
        u_img_emb: torch.Tensor,
        w_img_emb: torch.Tensor,
    ):
        concat_img = torch.cat(
            (
                u_img_emb,
                w_img_emb,
                u_img_emb * w_img_emb,
            ),
            dim=-3,
        )
        bs, fs = concat_img.shape[:-3], concat_img.shape[-3:]
        concat_img_reshaped = concat_img.view(-1, *fs)
        attention_logits = self.attention(concat_img_reshaped)
        attention_probs = torch.softmax(
            attention_logits.view(concat_img_reshaped.shape[0], -1),
            dim=-1,
        ).view(concat_img_reshaped.shape[0], 1, *concat_img_reshaped.shape[-2:])

        ego_img_pooled = (self.encoder(concat_img_reshaped) * attention_probs).mean(-1).mean(-1)

        return ego_img_pooled.view(*bs, -1)


class Semantic2DMapWithInventoryEncoderPooled(nn.Module):

    def __init__(
        self,
        n_map_channels: int,
        hidden_size: int,
        # num_head: int = 8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - ADDITIONAL_MAP_CHANNELS, hidden_size, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 1, ),
            nn.ReLU(inplace=True)
        )

        attention_dim = int(hidden_size / 4)
        self.attention = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - ADDITIONAL_MAP_CHANNELS, attention_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1, ),
        )

        self.n_map_channels = n_map_channels
        self.hidden_size = hidden_size
        # self.num_head = num_head

    def forward(
        self, 
        unshuffle_sem_map_data: torch.Tensor,
        walkthrough_sem_map_data: torch.Tensor,
        inventory_vector: torch.Tensor,
    ):
        """
        unshuffle_sem_map_data: [batch_size, n_map_channels, width, height]
        walkthrough_sem_map_data: [batch_size, n_map_channels, width, height]
        inventory_vector: [batch_size, n_channels(=n_map_channels-2)]
        """
        inventory = inventory_vector[:, :, None, None].repeat(
            [1, 1, *walkthrough_sem_map_data.shape[-2:]]
        ).type(walkthrough_sem_map_data.dtype)
        concat_sem_map_inv = torch.cat(
            (
                unshuffle_sem_map_data,
                walkthrough_sem_map_data,
                unshuffle_sem_map_data * walkthrough_sem_map_data,
                inventory,
            ),
            dim=-3,
        )
        batch_shape, features_shape = concat_sem_map_inv.shape[:-3], concat_sem_map_inv.shape[-3:]
        concat_sem_map_inv_reshaped = concat_sem_map_inv.view(-1, *features_shape)

        attention_logits = self.attention(concat_sem_map_inv_reshaped)
        attention_probs = torch.softmax(
            attention_logits.view(concat_sem_map_inv_reshaped.shape[0], -1),
            dim=-1,
        ).view(concat_sem_map_inv_reshaped.shape[0], 1, *concat_sem_map_inv_reshaped.shape[-2:])

        sem_map_inv_pooled = (
            self.encoder(concat_sem_map_inv_reshaped) * attention_probs
        ).mean(-1).mean(-1)

        return sem_map_inv_pooled.view(*batch_shape, -1)


# From https://github.com/valtsblukis/hlsm
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dim_ff, dropout=0.1, kvdim=None):
        super().__init__()
        kvdim = kvdim if kvdim is not None else d_model
        self.mh_attention = nn.MultiheadAttention(d_model, n_head, dropout, kdim=kvdim, vdim=kvdim)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = nn.LeakyReLU()

    def forward(self, src_labels: torch.Tensor, attn_mask: torch.Tensor, inputs_are_labels=True, return_attn=False):
        src_labels = src_labels[:, None, :]
        # Create an extra "batch" dimension, and treat the current batch dimension as a pos dimension
        seq_mask = attn_mask
        x, attn_w = self.mh_attention(src_labels, src_labels, src_labels, attn_mask=attn_mask)

        # inputs contain information about ground truth of the outputs for each element
        # and thus cannot be added with a residual connection.
        # The attn_mask is responsible for preventing label leakage.
        if inputs_are_labels:
            x = self.dropout(x)
        else:
            x = src_labels + self.dropout(x)

        x = self.norm1(x)
        y = self.linear2(self.dropout(self.act(self.linear1(x))))
        y = x + self.dropout2(y)
        y = self.norm2(y)
        if return_attn:
            return y[:, 0, :], attn_w
        else:
            return y[:, 0, :]


class SubtaskHistoryEncoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        n_head: int = 8,
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_OBJECT_TYPES,
        num_subtask_target_map_types: int = NUM_MAP_TYPES,
        ablate_no_subtask_hist: bool = False,
        ablate_no_pos_emb: torch.Tensor = False,
    ):
        super().__init__()
        self.num_subtask_types = num_subtask_types
        self.num_subtask_arguments = num_subtask_arguments
        self.num_subtask_target_map_types = num_subtask_target_map_types
        self.hidden_size = hidden_size
        self.n_head = n_head

        self.ablate_no_subtask_hist = ablate_no_subtask_hist
        self.ablate_no_pos_emb = ablate_no_pos_emb

        self.type_linear = nn.Linear(num_subtask_types, hidden_size)
        self.arg_linear = nn.Linear(num_subtask_arguments, hidden_size)
        self.target_map_type_linear = nn.Linear(num_subtask_target_map_types, hidden_size)
        self.sos_token_emb = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        self.transformer_layer_a = TransformerEncoderLayer(
            d_model=hidden_size,
            n_head=n_head,
            dim_ff=hidden_size
        )
        self.transformer_layer_b = TransformerEncoderLayer(
            d_model=hidden_size,
            n_head=n_head,
            dim_ff=hidden_size
        )

    def forward(
        self,
        subtask_index_history: torch.Tensor,
        seq_masks: torch.Tensor,
    ):
        """
        subtask_index_history: [batch_size, ]
        seq_masks: [batch_size, ]
        """
        # subtask_index_history = subtask_history[..., 0].permute(1, 0).reshape(-1).contiguous()
        batch_ids = masks_to_batch_ids(seq_masks)
        idxs = subtask_index_to_type_arg_target_map(subtask_index_history)
        type_oh = index_to_onehot(idxs[..., 0], self.num_subtask_types)
        arg_oh = index_to_onehot(idxs[..., 1], self.num_subtask_arguments)
        tmap_oh = index_to_onehot(idxs[..., 2], self.num_subtask_target_map_types)
        
        if self.ablate_no_subtask_hist:
            type_oh = torch.zeros_like(type_oh)
            arg_oh = torch.zeros_like(arg_oh)
            tmap_oh = torch.zeros_like(tmap_oh)

        type_emb = self.type_linear(type_oh)
        arg_emb = self.arg_linear(arg_oh)
        tmap_emb = self.target_map_type_linear(tmap_oh)

        pos_enc = positional_encoding(type_emb, batch_ids)
        if self.ablate_no_pos_emb:
            pos_enc = torch.zeros_like(pos_enc)

        subtask_emb = type_emb + arg_emb + tmap_emb
        start_and_subtask_emb = torch.cat([self.sos_token_emb[None, :], subtask_emb], dim=0)

        self_attention_masks_a = build_attention_masks(
            batch_ids=batch_ids,
            add_sos_token=True,
            include_self=False,
        )
        self_attention_masks_b = build_attention_masks(
            batch_ids=batch_ids,
            add_sos_token=True,
            include_self=False,
        )

        # self_attn_masks indicates for each column, whether the corresponding
        # row maps to a previous action in the same rollout.
        # Rows and columns are both over the sequence of actions.
        #       sos, s_0, s_1, s_2, s_3, s_4
        # sos   1    0    0    0    0    0
        # s_0   1    0    0    0    0    0
        # s_1   1    1    0    0    0    0
        # s_2   1    1    1    0    0    0
        # s_3   1    0    0    0    0    0
        # s_4   1    0    0    0    1    0
        
        # Two transformer layers
        enc_seq_a, aw_a = self.transformer_layer_a(
            src_labels=start_and_subtask_emb, 
            attn_mask=self_attention_masks_a,
            inputs_are_labels=True,
            return_attn=True,
        )
        enc_seq_b, aw_b = self.transformer_layer_a(
            src_labels=enc_seq_a, 
            attn_mask=self_attention_masks_b,
            inputs_are_labels=True,
            return_attn=True,
        )
        return enc_seq_b


class SubtaskPredictionModel(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_subtask_types: int = NUM_SUBTASK_TYPES,
        num_subtask_arguments: int = NUM_OBJECT_TYPES,
        num_subtask_target_map_types: int = NUM_MAP_TYPES,
        joint_prob: bool = False,
        device: Union[torch.device, int, str] = "cpu",
    ) -> None:
        super().__init__()
        self.num_subtask_types = num_subtask_types
        self.num_subtask_arguments = num_subtask_arguments
        self.num_subtask_target_map_types = num_subtask_target_map_types
        self.hidden_size = hidden_size
        self.joint_prob = joint_prob

        self.sem_map_inv_encoder = Semantic2DMapWithInventoryEncoderPooled(
            n_map_channels=num_subtask_arguments + ADDITIONAL_MAP_CHANNELS,
            hidden_size=hidden_size,
        )

        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            num_subtask_types=num_subtask_types,
            num_subtask_arguments=num_subtask_arguments,
            num_subtask_target_map_types=num_subtask_target_map_types,
        )

        # self.linear_a = nn.Linear(hidden_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_a1 = nn.Linear(hidden_size, hidden_size)
        self.linear_a2 = nn.Linear(hidden_size * 2, hidden_size)

        if self.joint_prob:
            self.linear_b = nn.Linear(
                hidden_size * 3, 
                (
                    num_subtask_types 
                    + num_subtask_arguments * num_subtask_types 
                    + num_subtask_target_map_types
                ),
            )
        else:
            self.linear_b = nn.Linear(
                hidden_size * 3, 
                num_subtask_types + num_subtask_arguments + num_subtask_target_map_types
            )
        
        self.act = nn.LeakyReLU()

    def forward(
        self,
        semantic_maps: torch.Tensor,
        inventory_vectors: torch.Tensor,
        subtask_index_history: torch.Tensor,
        seq_masks: torch.FloatTensor,
        nsteps: int,
        nsamplers: int,
    ):
        """
        semantic_maps: [batch_size, 2, n_map_channels, width, length, height]
        inventory_vectors: [batch_size, n_channels],
        subtask_history: [batch_size, ]
        seq_masks: [batch_size, ]
        *** subtask_history and seq_masks is reshaped after transposing axis for steps and samplers...
        """

        # Maxpooling 3D Semantic maps at axis for height
        batch_size = semantic_maps.shape[0]
        assert (
            batch_size == (nsteps * nsamplers)
            and all(
                [
                    batch_size == input_tensor.shape[0]
                    for input_tensor in (semantic_maps, inventory_vectors, subtask_index_history, seq_masks)
                ]
            )
        )

        sem_map_inv_embeddings = self.sem_map_inv_encoder(
            unshuffle_sem_map_data=semantic_maps[:, MAP_TYPES_TO_IDX["Unshuffle"]].max(-1).values,
            walkthrough_sem_map_data=semantic_maps[:, MAP_TYPES_TO_IDX["Walkthrough"]].max(-1).values,
            inventory_vector=inventory_vectors,
        )

        subtask_history_embeddings = self.subtask_history_encoder(
            subtask_index_history=subtask_index_history,
            seq_masks=seq_masks,
        )
        # Drop the last subtask history embedding
        subtask_history_embeddings = subtask_history_embeddings[:-1]
        # Since the batch order of subtask history embedding is different from sem_map_inv_embeddings,
        # we should re-order the order of the output embedding
        subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(batch_size, -1)

        return self.forward_embedding(
            sem_map_inv_embeddings=sem_map_inv_embeddings,
            subtask_history_embeddings=subtask_history_embeddings,
        )

    def forward_embedding(
        self,
        sem_map_inv_embeddings: torch.Tensor,
        subtask_history_embeddings: torch.Tensor,
    ):
        combined_embeddings = torch.cat([sem_map_inv_embeddings, subtask_history_embeddings], dim=1)

        x1 = self.act(self.linear_a(combined_embeddings))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)

        subtask_type_logits = x[:, :self.num_subtask_types]
        subtask_arg_logits = x[:, self.num_subtask_types:-self.num_subtask_target_map_types]
        subtask_target_map_type_logits = x[:, -self.num_subtask_target_map_types:]

        b = subtask_arg_logits.shape[0]
        subtask_type_logprob = F.log_softmax(subtask_type_logits, dim=1)
        if self.joint_prob:
            subtask_arg_logits = subtask_arg_logits.view([b, self.num_subtask_types, self.num_subtask_arguments])
            subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=2)
        else:
            subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
        # subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
        subtask_target_map_type_logprob = F.log_softmax(subtask_target_map_type_logits, dim=1)

        return subtask_type_logprob, subtask_arg_logprob, subtask_target_map_type_logprob

    # def forward(
    #     self,
    #     # observation: ObservationType,
    #     # prev_action: ActionType,
    #     semantic_maps: torch.Tensor,
    #     inventory_vectors: torch.Tensor,
    #     subtask_index_history: torch.Tensor,
    #     seq_masks: torch.FloatTensor,
    #     nsteps: int,
    #     nsamplers: int,
    # ):
    #     """
    #     semantic_maps: [batch_size, 2, n_map_channels, width, length, height]
    #     inventory_vectors: [batch_size, n_channels],
    #     subtask_history: [batch_size, ]
    #     seq_masks: [batch_size, ]
    #     *** subtask_history and seq_masks is reshaped after transposing axis for steps and samplers...
    #     """

    #     # Maxpooling 3D Semantic maps at axis for height
    #     batch_size = semantic_maps.shape[0]
    #     assert (
    #         batch_size == (nsteps * nsamplers)
    #         and all(
    #             [
    #                 batch_size == input_tensor.shape[0]
    #                 for input_tensor in (semantic_maps, inventory_vectors, subtask_index_history, seq_masks)
    #             ]
    #         )
    #     )

    #     sem_map_inv_embeddings = self.sem_map_inv_encoder(
    #         unshuffle_sem_map_data=semantic_maps[:, MAP_TYPES_TO_IDX["Unshuffle"]].max(-1).values,
    #         walkthrough_sem_map_data=semantic_maps[:, MAP_TYPES_TO_IDX["Walkthrough"]].max(-1).values,
    #         inventory_vector=inventory_vectors,
    #     )

    #     subtask_history_embeddings = self.subtask_history_encoder(
    #         subtask_index_history=subtask_index_history,
    #         seq_masks=seq_masks,
    #     )
    #     # Drop the last subtask history embedding
    #     subtask_history_embeddings = subtask_history_embeddings[:-1]
    #     # Since the batch order of subtask history embedding is different from sem_map_inv_embeddings,
    #     # we should re-order the order of the output embedding
    #     subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(batch_size, -1)

    #     combined_embeddings = torch.cat([sem_map_inv_embeddings, subtask_history_embeddings], dim=1)
    #     # combined_embeddings = torch.cat([map_embeddings, map_embeddings], dim=1)

    #     x1 = self.act(self.linear_a(combined_embeddings))
    #     x2 = self.act(self.linear_a1(x1))
    #     x12 = torch.cat([x1, x2], dim=1)
    #     x3 = self.act(self.linear_a2(x12))

    #     x123 = torch.cat([x1, x2, x3], dim=1)
    #     x = self.linear_b(x123)

    #     subtask_type_logits = x[:, :self.num_subtask_types]
    #     subtask_arg_logits = x[:, self.num_subtask_types:-self.num_subtask_target_map_types]
    #     subtask_target_map_type_logits = x[:, -self.num_subtask_target_map_types:]

    #     b = subtask_arg_logits.shape[0]
    #     subtask_type_logprob = F.log_softmax(subtask_type_logits, dim=1)
    #     if self.joint_prob:
    #         subtask_arg_logits = subtask_arg_logits.view([b, self.num_subtask_types, self.num_subtask_arguments])
    #         subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=2)
    #     else:
    #         subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
    #     # subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
    #     subtask_target_map_type_logprob = F.log_softmax(subtask_target_map_type_logits, dim=1)

    #     return subtask_type_logprob, subtask_arg_logprob, subtask_target_map_type_logprob, sem_map_inv_embeddings


class TaskAwareOnePhaseRearrangeBaseNetwork(ActorCriticModel):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        expert_subtask_uuid: str,
        expert_action_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        ordered_object_types: Sequence[str],
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        fov: int = 90,
        grid_parameters: GridParameters = GridParameters(),
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.inventory_uuid = inventory_uuid
        self.expert_subtask_uuid = expert_subtask_uuid
        self.expert_action_uuid = expert_action_uuid
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.ordered_object_types = ordered_object_types
        self.num_rnn_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers

        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        self.subtask_model = SubtaskPredictionModel(
            hidden_size=self._hidden_size,
        )
        self.prev_action_embedder = nn.Embedding(
            action_space.n + 1, embedding_dim=prev_action_embedding_dim
        )

        self.fov = fov
        self.grid_parameters = grid_parameters
        self._map_size = None
        
        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 2
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
        rnn_outs = []
        obs_for_rnn = torch.cat(to_cat, dim=-1)

        # Semantic maps: (sampler, channels, width, length, height)
        sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Unshuffle"]]
        w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Walkthrough"]]

        map_masks = masks.view(*masks.shape[:2], 1, 1, 1, 1)
        sem_maps = observations[self.sem_map_uuid]
        w_sem_maps = observations[self.unshuffled_sem_map_uuid]
        subtask_history = observations[self.expert_subtask_uuid]    # [nsteps, nsamplers, 2]
        inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]

        updated_sem_maps = []
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

            (
                subtask_type_logprob, 
                subtask_arg_lobprob, 
                subtask_target_map_type_logprob,
                sem_map_inv_embedding
            ) = self.subtask_model(
                semantic_maps=sem_maps_prev,
                inventory_vectors=inventory_vectors[step],
                subtask_index_history=subtask_history[step, 0].reshape(-1).contiguous(),
                seq_masks=masks[step].reshape(-1).contiguous(),
                nsteps=nsteps,
                nsamplers=nsamplers,
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
            updated_sem_maps.append(sem_maps_prev)

        # stack updated semantic maps along with new axis (timesteps)
        # [nsteps, nsamplers, n_map_types, nchannels, width, length, height]
        updated_sem_maps = torch.stack(updated_sem_maps, dim=0)
        rnn_outs = torch.cat(rnn_outs, dim=0)
                
        extras = {}

        memory = memory.set_tensor(
            key="sem_map",
            tensor=sem_maps_prev.type(torch.bool)
        )

        # test_x = torch.rand(nsteps, nsamplers, self._hidden_size).to(masks.device)
        return (
            ActorCriticOutput(
                distributions=self.actor(rnn_outs), values=self.critic(rnn_outs), extras=extras
            ), 
            memory
        )

    @property
    def num_objects(self):
        return len(self.ordered_object_types) + 1

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


class TaskAwareOnePhaseRearrangeSubtaskModelTrainingNetwork(ActorCriticModel):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        expert_subtask_uuid: str,
        expert_action_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        ordered_object_types: Sequence[str],
        prev_action_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        fov: int = 90,
        grid_parameters: GridParameters = GridParameters(),
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.inventory_uuid = inventory_uuid
        self.expert_subtask_uuid = expert_subtask_uuid
        self.expert_action_uuid = expert_action_uuid
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.ordered_object_types = ordered_object_types
        self.num_rnn_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers

        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        self.subtask_model = SubtaskPredictionModel(
            hidden_size=self._hidden_size,
        )
        self.prev_action_embedder = nn.Embedding(
            action_space.n + 1, embedding_dim=prev_action_embedding_dim
        )

        self.fov = fov
        self.grid_parameters = grid_parameters
        self._map_size = None

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

        # # Egocentric images
        # ego_img = observations[self.rgb_uuid]
        # w_ego_img = observations[self.unshuffled_rgb_uuid]
        # ego_img_embeddings = self.visual_encoder(
        #     u_img_emb=ego_img,
        #     w_img_emb=w_ego_img
        # )   # [steps, samplers, vis_feature_embedding_dim]

        # # Previous actions (low-level actions)
        # prev_action_embeddings = self.prev_action_embedder(
        #     (masks.long() * (prev_actions.unsqueeze(-1) + 1))
        # ).squeeze(-2)   # [steps, samplers, prev_action_embedding_dim]
        
        # During the training of SubtaskModel, we use the result of subtask_expert as subtask_history.
        # with indicators that distinguishes the different episodes/tasks
        subtask_history = observations[self.expert_subtask_uuid]    # [nsteps, nsamplers, 2]
        inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]

        subtask_history_embeddings = self.subtask_model.subtask_history_encoder(
            subtask_index_history=subtask_history[..., 0].permute(1, 0).reshape(-1).contiguous(),
            seq_masks=masks.permute(1, 0, 2).reshape(-1).contiguous(),
        )
        subtask_history_embeddings = subtask_history_embeddings[:-1]
        subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(nsteps * nsamplers, -1)

        # Semantic maps: (sampler, channels, width, length, height)
        sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Unshuffle"]]
        w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Walkthrough"]]

        map_masks = masks.view(*masks.shape[:2], 1, 1, 1, 1)
        sem_maps = observations[self.sem_map_uuid]
        w_sem_maps = observations[self.unshuffled_sem_map_uuid]

        sem_map_inv_embeddings = []
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

            sem_map_inv_embedding = self.subtask_model.sem_map_inv_encoder(
                unshuffle_sem_map_data=sem_maps_prev[:, MAP_TYPES_TO_IDX["Unshuffle"]].max(-1).values,
                walkthrough_sem_map_data=sem_maps_prev[:, MAP_TYPES_TO_IDX["Walkthrough"]].max(-1).values,
                inventory_vector=inventory_vectors[step],
            )
            sem_map_inv_embeddings.append(sem_map_inv_embedding)

        sem_map_inv_embeddings = torch.stack(sem_map_inv_embeddings, dim=0)
        (
            subtask_type_logprobs, 
            subtask_arg_lobprobs, 
            subtask_target_map_type_logprobs,
        ) = self.subtask_model.forward_embedding(
            sem_map_inv_embeddings=sem_map_inv_embeddings.view(-1, *sem_map_inv_embeddings.shape[2:]),
            subtask_history_embeddings=subtask_history_embeddings,
        )
        subtask_type_logprobs = subtask_type_logprobs.view(
            nsteps, nsamplers, *subtask_type_logprobs.shape[1:]
        )   # [nsteps, nsamplers, NUM_SUBTASK_TYPES]
        subtask_arg_lobprobs = subtask_arg_lobprobs.view(
            nsteps, nsamplers, *subtask_arg_lobprobs.shape[1:]
        )   # [nsteps, nsamplers, NUM_OBJECTS_TYPES]
        subtask_target_map_type_logprobs = subtask_target_map_type_logprobs.view(
            nsteps, nsamplers, *subtask_target_map_type_logprobs.shape[1:]
        )   # [nsteps, nsamplers, NUM_MAP_TYPES]

        extras = dict(
            subtask_logits=dict(
                type=subtask_type_logprobs,
                arg=subtask_arg_lobprobs,
                target_map=subtask_target_map_type_logprobs,
            ),
        )
        
        memory = memory.set_tensor(
            key="sem_map",
            tensor=sem_maps_prev.type(torch.bool)
        )

        # rnn_hidden_states = memory.tensor("rnn")
        # rnn_inputs = torch.cat(
        #     (
        #         ego_img_embeddings,
        #         prev_action_embeddings,
        #     ),
        #     dim=-1
        # )
        # rnn_outs, rnn_hidden_states = self.state_encoder(
        #     rnn_inputs,
        #     rnn_hidden_states,
        #     masks,
        # )

        # generate action distribution based on expert action
        expert_actions = observations[self.expert_action_uuid][..., 0]              # [nsteps, nsamplers]
        expert_actions_oh = index_to_onehot(expert_actions, self.action_space.n)    # [nsteps, nsamplers, NUM_ACTIONS]

        # set prob of expert action as p => add (1-p)/(pn-1) to all elements
        p = 0.90
        action_probs = expert_actions_oh + (1 - p) / (p * self.action_space.n - 1)
        action_probs = action_probs / action_probs.sum(-1, keepdim=True)
        action_distr = CategoricalDistr(probs=action_probs)

        empty_values = torch.zeros((nsteps, nsamplers, 1), device=expert_actions.device, dtype=torch.float32)

        # test_x = torch.rand(nsteps, nsamplers, self._hidden_size).to(masks.device)
        return (
            ActorCriticOutput(
                distributions=action_distr, values=empty_values, extras=extras
            ), 
            memory
        )

    @property
    def num_objects(self):
        return len(self.ordered_object_types) + 1

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

    
class TaskAwareRearrangeDataCollectionModel(ActorCriticModel):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        inventory_uuid: str,
        expert_subtask_uuid: str,
        expert_action_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        ordered_object_types: Sequence[str],
        hidden_size: int = 512,
        fov: int = 90,
        grid_parameters: GridParameters = GridParameters(),
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.inventory_uuid = inventory_uuid
        self.expert_subtask_uuid = expert_subtask_uuid
        self.expert_action_uuid = expert_action_uuid
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.ordered_object_types = ordered_object_types

        self.fov = fov
        self.grid_parameters = grid_parameters
        self._map_size = None
      
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.device = device if device else torch.device("cpu")

    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        return dict(
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
        # Semantic maps: (sampler, channels, width, length, height)
        sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Unshuffle"]]
        w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Walkthrough"]]
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
        
        memory = memory.set_tensor(
            key="sem_map",
            tensor=sem_maps_prev.type(torch.bool)
        )
        # generate action distribution based on expert action
        expert_actions = observations[self.expert_action_uuid][..., 0]              # [nsteps, nsamplers]
        expert_actions_oh = index_to_onehot(expert_actions, self.action_space.n)    # [nsteps, nsamplers, NUM_ACTIONS]

        # set prob of expert action as p => add (1-p)/(pn-1) to all elements
        p = 0.90
        action_probs = expert_actions_oh + (1 - p) / (p * self.action_space.n - 1)
        action_probs = action_probs / action_probs.sum(-1, keepdim=True)
        action_distr = CategoricalDistr(probs=action_probs)

        empty_values = torch.zeros((nsteps, nsamplers, 1), device=expert_actions.device, dtype=torch.float32)

        extras = {
            'sem_map': sem_maps_prev
        }
        return (
            ActorCriticOutput(
                distributions=action_distr, values=empty_values, extras=extras
            ), 
            memory
        )

    @property
    def num_objects(self):
        return len(self.ordered_object_types) + 1

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