from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS

from task_aware_rearrange.constants import ADDITIONAL_MAP_CHANNELS, NUM_OBJECT_TYPES
from task_aware_rearrange.subtasks import NUM_MAP_TYPES, NUM_SUBTASK_TARGET_OBJECTS, NUM_SUBTASK_TYPES, MAP_TYPE_TO_IDX, NUM_SUBTASKS
from task_aware_rearrange.layer_utils import batch_ids_to_ranges, build_attention_masks, index_to_onehot, masks_to_batch_ids, positional_encoding, subtask_index_to_type_arg


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


class SemanticMap2DEncoderPooled(nn.Module):

    def __init__(
        self,
        n_map_channels: int,
        hidden_size: int,
        # num_head: int = 8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * n_map_channels, hidden_size, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 1, ),
            nn.ReLU(inplace=True)
        )

        attention_dim = int(hidden_size / 4)
        self.attention = nn.Sequential(
            nn.Conv2d(3 * n_map_channels, attention_dim, 1, ),
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
    ):
        """
        unshuffle_sem_map_data: [batch_size, n_map_channels, width, height]
        walkthrough_sem_map_data: [batch_size, n_map_channels, width, height]
        """
        concat_sem_map = torch.cat(
            (
                unshuffle_sem_map_data,
                walkthrough_sem_map_data,
                unshuffle_sem_map_data * walkthrough_sem_map_data,
            ),
            dim=-3,
        )
        batch_shape, features_shape = concat_sem_map.shape[:-3], concat_sem_map.shape[-3:]
        concat_sem_map_reshaped = concat_sem_map.view(-1, *features_shape)

        attention_logits = self.attention(concat_sem_map_reshaped)
        attention_probs = torch.softmax(
            attention_logits.view(concat_sem_map_reshaped.shape[0], -1),
            dim=-1,
        ).view(concat_sem_map_reshaped.shape[0], 1, *concat_sem_map_reshaped.shape[-2:])

        sem_map_pooled = (
            self.encoder(concat_sem_map_reshaped) * attention_probs
        ).mean(-1).mean(-1)

        return sem_map_pooled.view(*batch_shape, -1)


class Semantic2DMapWithInventoryEncoderPooled(nn.Module):

    def __init__(
        self,
        n_map_channels: int,
        hidden_size: int,
        additional_map_channels: int = ADDITIONAL_MAP_CHANNELS,
        # num_head: int = 8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - additional_map_channels, hidden_size, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 1, ),
            nn.ReLU(inplace=True)
        )

        attention_dim = int(hidden_size / 4)
        self.attention = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - additional_map_channels, attention_dim, 1, ),
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
        inventory_vector: [batch_size, n_channels(=n_map_channels-3)]
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
        # num_subtask_types: int = NUM_SUBTASK_TYPES,
        # num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
        num_subtasks: int = NUM_SUBTASKS,
        ablate_no_subtask_hist: bool = False,
        ablate_no_pos_emb: torch.Tensor = False,
    ):
        super().__init__()
        # self.num_subtask_types = num_subtask_types
        # self.num_subtask_arguments = num_subtask_arguments
        self.num_subtasks = num_subtasks
        self.hidden_size = hidden_size
        self.n_head = n_head

        self.ablate_no_subtask_hist = ablate_no_subtask_hist
        self.ablate_no_pos_emb = ablate_no_pos_emb

        # self.type_linear = nn.Linear(num_subtask_types, hidden_size)
        # self.arg_linear = nn.Linear(num_subtask_arguments, hidden_size)
        self.linear = nn.Linear(num_subtasks, hidden_size)
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
        # idxs = subtask_index_to_type_arg(subtask_index_history)
        idxs = subtask_index_history
        oh = index_to_onehot(idxs, self.num_subtasks)
        # type_oh = index_to_onehot(idxs[..., 0], self.num_subtask_types)
        # arg_oh = index_to_onehot(idxs[..., 1], self.num_subtask_arguments)
        
        if self.ablate_no_subtask_hist:
            # type_oh = torch.zeros_like(type_oh)
            # arg_oh = torch.zeros_like(arg_oh)
            oh = torch.zeros_like(oh)

        # type_emb = self.type_linear(type_oh)
        # arg_emb = self.arg_linear(arg_oh)
        subtask_emb = self.linear(oh)

        # pos_enc = positional_encoding(type_emb, batch_ids)
        pos_enc = positional_encoding(subtask_emb, batch_ids)
        if self.ablate_no_pos_emb:
            pos_enc = torch.zeros_like(pos_enc)

        # subtask_emb = type_emb + arg_emb
        subtask_emb += pos_enc
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


# class SubtaskPredictionModel(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         num_subtask_types: int = NUM_SUBTASK_TYPES,
#         num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
#         joint_prob: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_subtask_types = num_subtask_types
#         self.num_subtask_arguments = num_subtask_arguments
#         self.hidden_size = hidden_size
#         self.joint_prob = joint_prob

#         self.sem_map_inv_encoder = Semantic2DMapWithInventoryEncoderPooled(
#             n_map_channels=num_subtask_arguments + ADDITIONAL_MAP_CHANNELS,
#             hidden_size=hidden_size,
#         )

#         self.subtask_history_encoder = SubtaskHistoryEncoder(
#             hidden_size=hidden_size,
#             num_subtask_types=num_subtask_types,
#             num_subtask_arguments=num_subtask_arguments,
#         )

#         # self.linear_a = nn.Linear(hidden_size, hidden_size)
#         self.linear_a = nn.Linear(hidden_size * 2, hidden_size)
#         self.linear_a1 = nn.Linear(hidden_size, hidden_size)
#         self.linear_a2 = nn.Linear(hidden_size * 2, hidden_size)

#         if self.joint_prob:
#             self.linear_b = nn.Linear(
#                 hidden_size * 3, 
#                 (
#                     num_subtask_types 
#                     + num_subtask_arguments * num_subtask_types
#                 ),
#             )
#         else:
#             self.linear_b = nn.Linear(
#                 hidden_size * 3, 
#                 num_subtask_types + num_subtask_arguments
#             )
        
#         self.act = nn.LeakyReLU()

#     def forward(
#         self,
#         semantic_maps: torch.Tensor,
#         inventory_vectors: torch.Tensor,
#         subtask_index_history: torch.Tensor,
#         seq_masks: torch.FloatTensor,
#         nsteps: int,
#         nsamplers: int,
#     ):
#         """
#         semantic_maps: [batch_size, 2, n_map_channels, width, length, height]
#         inventory_vectors: [batch_size, n_channels],
#         subtask_history: [batch_size, ]
#         seq_masks: [batch_size, ]
#         *** subtask_history and seq_masks is reshaped after transposing axis for steps and samplers...
#         """

#         # Maxpooling 3D Semantic maps at axis for height
#         batch_size = semantic_maps.shape[0]
#         assert (
#             batch_size == (nsteps * nsamplers)
#             and all(
#                 [
#                     batch_size == input_tensor.shape[0]
#                     for input_tensor in (semantic_maps, inventory_vectors, subtask_index_history, seq_masks)
#                 ]
#             )
#         )

#         sem_map_inv_embeddings = self.sem_map_inv_encoder(
#             unshuffle_sem_map_data=semantic_maps[:, MAP_TYPE_TO_IDX["Unshuffle"]].max(-1).values,
#             walkthrough_sem_map_data=semantic_maps[:, MAP_TYPE_TO_IDX["Walkthrough"]].max(-1).values,
#             inventory_vector=inventory_vectors,
#         )

#         subtask_history_embeddings = self.subtask_history_encoder(
#             subtask_index_history=subtask_index_history,
#             seq_masks=seq_masks,
#         )
#         # Drop the last subtask history embedding
#         subtask_history_embeddings = subtask_history_embeddings[:-1]
#         # Since the batch order of subtask history embedding is different from sem_map_inv_embeddings,
#         # we should re-order the order of the output embedding
#         subtask_history_embeddings = subtask_history_embeddings.view(nsamplers, nsteps, -1).permute(1, 0, 2).contiguous().reshape(batch_size, -1)

#         return self.forward_embedding(
#             sem_map_inv_embeddings=sem_map_inv_embeddings,
#             subtask_history_embeddings=subtask_history_embeddings,
#         )

#     def forward_embedding(
#         self,
#         sem_map_inv_embeddings: torch.Tensor,
#         subtask_history_embeddings: torch.Tensor,
#     ):
#         combined_embeddings = torch.cat([sem_map_inv_embeddings, subtask_history_embeddings], dim=1)

#         x1 = self.act(self.linear_a(combined_embeddings))
#         x2 = self.act(self.linear_a1(x1))
#         x12 = torch.cat([x1, x2], dim=1)
#         x3 = self.act(self.linear_a2(x12))

#         x123 = torch.cat([x1, x2, x3], dim=1)
#         x = self.linear_b(x123)

#         subtask_type_logits = x[:, :self.num_subtask_types]
#         subtask_arg_logits = x[:, self.num_subtask_types:]

#         b = subtask_arg_logits.shape[0]
#         subtask_type_logprob = F.log_softmax(subtask_type_logits, dim=1)
#         if self.joint_prob:
#             subtask_arg_logits = subtask_arg_logits.view([b, self.num_subtask_types, self.num_subtask_arguments])
#             subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=2)
#         else:
#             subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
#         # subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)

#         return subtask_type_logprob, subtask_arg_logprob


# class SubtaskPredictor(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         # num_subtask_types: int = NUM_SUBTASK_TYPES,
#         # num_subtask_arguments: int = NUM_SUBTASK_TARGET_OBJECTS,
#         num_subtasks: int = NUM_SUBTASKS,
#         # joint_prob: bool = False,
#     ) -> None:
#         super().__init__()
#         # self.num_subtask_types = num_subtask_types
#         # self.num_subtask_arguments = num_subtask_arguments
#         self.num_subtasks = num_subtasks
#         self.hidden_size = hidden_size
#         # self.joint_prob = joint_prob

#         self.linear_a = nn.Linear(hidden_size * 2, hidden_size)
#         self.linear_a1 = nn.Linear(hidden_size, hidden_size)
#         self.linear_a2 = nn.Linear(hidden_size * 2, hidden_size)

#         # if self.joint_prob:
#         #     self.linear_b = nn.Linear(
#         #         hidden_size * 3, 
#         #         (
#         #             num_subtask_types 
#         #             + num_subtask_arguments * num_subtask_types
#         #         ),
#         #     )
#         # else:
#         #     self.linear_b = nn.Linear(
#         #         hidden_size * 3, 
#         #         num_subtask_types + num_subtask_arguments
#         #     )
        
#         self.linear_b = nn.Linear(hidden_size * 3, num_subtasks)
#         self.act = nn.LeakyReLU()

#     def forward_embedding(
#         self,
#         sem_map_inv_embeddings: torch.Tensor,
#         subtask_history_embeddings: torch.Tensor,
#     ):
#         combined_embeddings = torch.cat([sem_map_inv_embeddings, subtask_history_embeddings], dim=1)

#         x1 = self.act(self.linear_a(combined_embeddings))
#         x2 = self.act(self.linear_a1(x1))
#         x12 = torch.cat([x1, x2], dim=1)
#         x3 = self.act(self.linear_a2(x12))

#         x123 = torch.cat([x1, x2, x3], dim=1)
#         x = self.linear_b(x123)

#         return F.log_softmax(x, dim=1)
#         # subtask_type_logits = x[:, :self.num_subtask_types]
#         # subtask_arg_logits = x[:, self.num_subtask_types:]

#         # b = subtask_arg_logits.shape[0]
#         # subtask_type_logprob = F.log_softmax(subtask_type_logits, dim=1)
#         # if self.joint_prob:
#         #     subtask_arg_logits = subtask_arg_logits.view([b, self.num_subtask_types, self.num_subtask_arguments])
#         #     subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=2)
#         # else:
#         #     subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
#         # # subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)

#         # return subtask_type_logprob, subtask_arg_logprob


class SubtaskPredictor(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        input_size: Optional[int] = None,
        num_subtasks: int = NUM_SUBTASKS,
    ) -> None:
        super().__init__()
        self.num_subtasks = num_subtasks
        self.hidden_size = hidden_size

        if input_size is None:
            input_size = hidden_size * 2

        self.linear_a = nn.Linear(input_size, hidden_size)
        self.linear_a1 = nn.Linear(hidden_size, hidden_size)
        self.linear_a2 = nn.Linear(hidden_size * 2, hidden_size)
        
        self.linear_b = nn.Linear(hidden_size * 3, num_subtasks)
        self.act = nn.LeakyReLU()

    def forward(
        self,
        x: torch.Tensor,
    ):
        x1 = self.act(self.linear_a(x))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)

        return F.log_softmax(x, dim=1)

    def forward_embedding(
        self,
        env_embeddings: torch.Tensor,
        subtask_history_embeddings: torch.Tensor,
    ):
        combined_embeddings = torch.cat([env_embeddings, subtask_history_embeddings], dim=1)

        x1 = self.act(self.linear_a(combined_embeddings))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)

        return F.log_softmax(x, dim=1)
