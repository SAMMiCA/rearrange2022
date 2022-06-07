from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)
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
from custom.constants import IDX_TO_OBJECT_TYPE, NUM_OBJECT_TYPES, ORDERED_OBJECT_TYPES
from custom.hlsm.voxel_grid import VoxelGrid
from custom.subtask import IDX_TO_SUBTASK_TYPE, Subtask
from example_utils import ForkedPdb
from rearrange.constants import OBJECT_TYPES_WITH_PROPERTIES, OPENABLE_OBJECTS, PICKUPABLE_OBJECTS


class Semantic3DMapEncoderPooled(nn.Module):

    def __init__(
        self,
        n_map_channels: int,
        hidden_dim: int,
        num_head: int = 8,
    ):
        # self.task_layer = nn.Linear(hidden_dim, hidden_dim * num_head)
        # self.state_layer = nn.Linear(num_subtask_arguments * 2, hidden_dim)
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4 * n_map_channels - 2, hidden_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, 1, ),
            nn.ReLU(inplace=True)
        )

        attention_dim = int(hidden_dim / 4)
        self.attention = nn.Sequential(
            nn.Conv3d(4 * n_map_channels - 2, attention_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv3d(attention_dim, 1, 1, ),
        )

        self.n_map_channels = n_map_channels
        self.hidden_dim = hidden_dim
        self.num_head = num_head

    def forward(
        self, 
        unshuffle_sem_map_data: torch.tensor,
        walkthrough_sem_map_data: torch.tensor,
        inventory_vector: torch.tensor,
    ):
        inventory = inventory_vector[:, :, None, None, None].repeat(
            [1, 1, *walkthrough_sem_map_data.shape[-3:]]
        ).type(walkthrough_sem_map_data.dtype)
        concat_sem_map_inv = torch.cat(
            (
                unshuffle_sem_map_data,
                walkthrough_sem_map_data,
                unshuffle_sem_map_data * walkthrough_sem_map_data,
                inventory,
            ),
            dim=-4,
        )
        batch_shape, features_shape = concat_sem_map_inv.shape[:-4], concat_sem_map_inv.shape[-4:]
        concat_sem_map_inv_reshaped = concat_sem_map_inv.view(-1, *features_shape)

        attention_logits = self.attention(concat_sem_map_inv_reshaped)
        attention_probs = torch.softmax(
            attention_logits.view(concat_sem_map_inv_reshaped.shape[0], -1),
            dim=-1,
        ).view(concat_sem_map_inv_reshaped.shape[0], 1, *concat_sem_map_inv_reshaped.shape[-3:])

        sem_map_inv_pooled = (
            self.encoder(concat_sem_map_inv_reshaped) * attention_probs
        ).mean(-1).mean(-1).mean(-1)

        return sem_map_inv_pooled.view(*batch_shape, -1)


class Semantic2DMapEncoderPooled(nn.Module):

    def __init__(
        self,
        n_map_channels: int,
        hidden_dim: int,
        num_head: int = 8,
    ):
        # self.task_layer = nn.Linear(hidden_dim, hidden_dim * num_head)
        # self.state_layer = nn.Linear(num_subtask_arguments * 2, hidden_dim)
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - 2, hidden_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, ),
            nn.ReLU(inplace=True)
        )

        attention_dim = int(hidden_dim / 4)
        self.attention = nn.Sequential(
            nn.Conv2d(4 * n_map_channels - 2, attention_dim, 1, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1, ),
        )

        self.n_map_channels = n_map_channels
        self.hidden_dim = hidden_dim
        self.num_head = num_head

    def forward(
        self, 
        unshuffle_sem_map_data: torch.tensor,
        walkthrough_sem_map_data: torch.tensor,
        inventory_vector: torch.tensor,
    ):
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
        u_img_emb: torch.tensor,
        w_img_emb: torch.tensor,
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


class SubtaskPredictionModel(nn.Module):

    def __init__(
        self,
        num_subtask_types: int,
        num_subtask_arguments: int,
        hidden_dim: int,
        joint_prob: bool = False,
    ):
        super().__init__()
        self.num_subtask_types = num_subtask_types
        self.num_subtask_arguments = num_subtask_arguments
        self.hidden_dim = hidden_dim
        self.joint_prob = joint_prob

        self.linear_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear_a1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_a2 = nn.Linear(hidden_dim * 2, hidden_dim)

        if self.joint_prob:
            self.linear_b = nn.Linear(hidden_dim * 3, num_subtask_types + num_subtask_arguments * num_subtask_types + 2)
        else:
            self.linear_b = nn.Linear(hidden_dim * 3, num_subtask_types + num_subtask_arguments + 2)
        
        self.act = nn.LeakyReLU()

    def forward(
        self,
        map_inv_embeddings: torch.Tensor,
        # subtask_hist_embeddings
    ):
        """

        state_embeddings: nsamplers x hidden_dim
        action_hist_embeddings:
        """

        # combined_embeddings = torch.cat([state_embeddings, subtask_hist_embeddings], dim=1)
        # combined_embeddings = torch.cat([map_embeddings, map_embeddings], dim=1)

        x1 = self.act(self.linear_a(map_inv_embeddings))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)

        subtask_type_logits = x[:, :self.num_subtask_types]
        subtask_arg_logits = x[:, self.num_subtask_types:-2]
        subtask_target_map_type_logits = x[:, -2:]

        b = subtask_arg_logits.shape[0]
        subtask_type_logprob = F.log_softmax(subtask_type_logits, dim=1)
        # if self.joint_prob:
        #     subtask_arg_logits = subtask_arg_logits.view([b, self.num_subtask_types, self.num_subtask_arguments])
        #     subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=2)
        # else:
        #     subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
        subtask_arg_logprob = F.log_softmax(subtask_arg_logits, dim=1)
        subtask_target_map_type_logprob = F.log_softmax(subtask_target_map_type_logits, dim=1)

        return subtask_type_logprob, subtask_arg_logprob, subtask_target_map_type_logprob

    # def predict(
    #     self,
    #     unshuffle_comb_map: torch.Tensor,
    #     walkthrough_comb_map: torch.Tensor,
    #     inventory_vector: torch.Tensor,
    #     # subtask_hist_embeddings
    # ):
    #     subtask_type_logprob, subtask_arg_logprob, subtask_tmap_logprob = self.forward(
    #         unshuffle_comb_map=unshuffle_comb_map,
    #         walkthrough_comb_map=walkthrough_comb_map,
    #         inventory_vector=inventory_vector,
    #         # subtask_hist_embeddings=subtask_hist_embeddings,
    #     )

    #     subtask_type_distr = torch.exp(subtask_type_logprob)
    #     subtask_arg_distr = torch.exp(subtask_arg_logprob)
    #     if self.joint_prob:
    #         subtask_arg_distr = subtask_arg_distr / subtask_arg_distr.sum(dim=2, keepdim=True)  # Re-normalize
    #     subtask_tmap_distr = torch.exp(subtask_tmap_logprob)

    #     subtasks = self.sample_subtasks(
    #         type_distr=subtask_type_distr,
    #         arg_vectors=subtask_arg_distr,
    #         tmap_distr=subtask_tmap_distr,
    #     )

    #     # mask generation for subtasks
    #     arg_mask_3d = self.masking(
    #         unshuffle_comb_map=unshuffle_comb_map,
    #         walkthrough_comb_map=walkthrough_comb_map,
    #         subtask=subtasks,
    #     )
    #     arg_mask_voxelgrid = VoxelGrid.create_from_mask(arg_mask_3d)
    #     subtasks.argument_mask = arg_mask_voxelgrid

    #     return subtasks

    def sample_subtasks(
        self,
        type_distr: torch.Tensor,
        arg_vectors: torch.Tensor,
        tmap_distr: torch.Tensor,
    ):
        ns, _ = type_distr.shape

        subtasks = []
        for i in range(ns):
            subtask_type_id = torch.distributions.Categorical(type_distr[i]).sample().item()
            subtask_type_str = Subtask.subtask_type_intid_to_str(subtask_type_id)
            if self.joint_prob:
                arg_vector = arg_vectors[i:i+1, subtask_type_id, :].clone()
            else:
                arg_vector = arg_vectors[i:i+1].clone()

            # for debugging
            # print out top-k subtasks

            pass_objects = arg_vector > 0.04
            if pass_objects.any():
                arg_vector = arg_vector * pass_objects
            arg_vector /= (arg_vector.sum() + 1e-10)

            subtask_arg_id = torch.distributions.Categorical(arg_vector).sample().item()
            arg_vector_out = torch.zeros_like(arg_vector)
            arg_vector_out[0, subtask_arg_id] = 1.0

            subtask_tmap_id = torch.distributions.Categorical(tmap_distr[i]).sample().item()
            subtask_tmap_str = Subtask.target_map_type_intid_to_str(subtask_tmap_id)

            subtasks.append(
                Subtask.from_type_str_arg_vector_and_target_map_str(
                    type_str=subtask_type_str,
                    arg_vec=arg_vector_out,
                    tmap_str=subtask_tmap_str,
                ).to(type_distr.device)
            )
        
        return Subtask.collate(subtasks)

    # def masking(
    #     self,
    #     unshuffle_comb_map: torch.Tensor,
    #     walkthrough_comb_map: torch.Tensor,
    #     subtask: Subtask,
        
    # ):
    #     # TODO: more tasks on masking via NN
    #     proposed_subtask_masks = subtask.build_spatial_arg_proposal(
    #         unshuffle_comb_map=unshuffle_comb_map,
    #         walkthrough_comb_map=walkthrough_comb_map,
    #     )
    #     # proposed_subtask_masks_2d = proposed_subtask_masks.max(dim=-1)
    #     # proposed_subtask_types = subtask.type_oh()
    #     # proposed_typed_masks_2d = proposed_subtask_types[:, :, None, None] * proposed_subtask_masks_2d

    #     return proposed_subtask_masks


class SubtaskPlanningModel(nn.Module):

    def __init__(
        self,
        num_subtask_types: int,
        num_subtask_arguments: int,
        joint_prob: bool = False,
    ):
        super().__init__()
        self.num_subtask_types = num_subtask_types
        self.num_subtask_arguments = num_subtask_arguments
        self.joint_prob = joint_prob

    @staticmethod
    def difference(
        unshuffle_comb_map: torch.Tensor,
        walkthrough_comb_map: torch.Tensor,
    ):

        """
        comb_map: [samplers, channels, width, length, height], 
                  channels: num_object_class + 1 (others) + 1 (occupancy) + 1 (observability)
        """

        # only care about the grids that observed in unshuffle environment
        u_sem_map = unshuffle_comb_map[:, :-2]
        u_occu_map = unshuffle_comb_map[:, -2:-1]
        u_obs_map = unshuffle_comb_map[:, -1:]
        w_sem_map = walkthrough_comb_map[:, :-2]
        w_occu_map = walkthrough_comb_map[:, -2:-1]
        w_obs_map = walkthrough_comb_map[:, -1:]


        diff_sem_map = u_sem_map * u_obs_map - w_sem_map * u_obs_map
        max_diff_sem_map = diff_sem_map.max(-1).values.max(-1).values.max(-1).values
        min_diff_sem_map = diff_sem_map.min(-1).values.min(-1).values.min(-1).values

        type_diff_sem_map = 2 * max_diff_sem_map - min_diff_sem_map
        
        # [sampler]
        val_diff, ind_diff = type_diff_sem_map.max(-1).values, type_diff_sem_map.max(-1).indices

        return val_diff, ind_diff

    @staticmethod
    def plan(
        unshuffle_comb_map: torch.Tensor,
        walkthrough_comb_map: torch.Tensor,
        inventory_vector: torch.Tensor,
        val_diff: torch.Tensor = None,
        ind_diff: torch.Tensor = None,
    ):
        """
        comb_map: [sampler, map_channels, width, length, height]
        val_diff: [sampler], 0 - 3,
                       0: (nU / nW) To explore 
                       1: (nU / W) To explore to find object in UNSHUFFLE ENV
                       2: (U / nW) Goto & Pickup object and explore to find object in WALKTHROUGH ENV
                       3: (U / W) Goto object[UNSHUFFLE] & Pickup object and then Goto object[WALKTHROUGH] & Put object
        ind_diff: [sampelr], 0 - (num_object_class), 
                       0 - (num_object_class - 1): object ids, 
                       (num_object_class): id for unknown object class
        inventory_vector: [sampler, num_object_class], dtype: bool
        """
        subtasks = []
        device = inventory_vector.device
        inventory = torch.any(inventory_vector, dim=-1)
        if val_diff is None and ind_diff is None:
            val_diff, ind_diff = SubtaskPlanningModel.difference(
                unshuffle_comb_map=unshuffle_comb_map,
                walkthrough_comb_map=walkthrough_comb_map,
            )
        comb_map_stack = torch.stack((unshuffle_comb_map, walkthrough_comb_map), dim=0)

        for i in range(len(val_diff)):
            val = val_diff[i].item()
            ind = ind_diff[i].item()
            hand = inventory[i].item()

            if ind == len(ORDERED_OBJECT_TYPES):
                ind = -1
            
            assert val in range(4)
            if ind == -1:
                type_str = "Explore"
                arg_id = -1     # NIL
                target_map_type_str = "Unshuffle"
            else:
                if val == 0:
                    # ind should be 0
                    assert ind == 0
                    type_str = "Explore"
                    arg_id = -1     # NIL
                    target_map_type_str = "Unshuffle"
                    
                elif val == 1:
                    # ind should be the known object class id
                    assert ind < len(ORDERED_OBJECT_TYPES) and ind >= 0
                    
                    if not hand:
                        type_str = "Explore"
                        target_map_type_str = "Unshuffle"
                    else:
                        if IDX_TO_OBJECT_TYPE[ind] in PICKUPABLE_OBJECTS:
                            type_str = "PutObject"
                        # elif IDX_TO_OBJECT_TYPE[ind] in OPENABLE_OBJECTS:
                        #     type_str = "OpenObject"
                        else:
                            raise NotImplementedError
                        target_map_type_str = "Walkthrough"
                    arg_id = ind    # explore UNSHUFFLE ENV to find OBJECT[ind]

                elif val == 2:
                    # ind should be the known object class id
                    assert ind < len(ORDERED_OBJECT_TYPES) and ind >= 0
                    if not hand:
                        if IDX_TO_OBJECT_TYPE[ind] in PICKUPABLE_OBJECTS:
                            type_str = "PickupObject"
                        # elif IDX_TO_OBJECT_TYPE[ind] in OPENABLE_OBJECTS:
                        #     type_str = "OpenObject"
                        else:
                            raise NotImplementedError
                        target_map_type_str = "Unshuffle"
                    else:
                        type_str = "Explore"
                        target_map_type_str = "Walkthrough"
                    arg_id = ind

                elif val == 3:
                    # ind should be the known object class id
                    assert ind < len(ORDERED_OBJECT_TYPES) and ind >= 0
                    if not hand:
                        if IDX_TO_OBJECT_TYPE[ind] in PICKUPABLE_OBJECTS:
                            type_str = "PickupObject"
                        # elif IDX_TO_OBJECT_TYPE[ind] in OPENABLE_OBJECTS:
                        #     type_str = "OpenObject"
                        else:
                            raise NotImplementedError
                        target_map_type_str = "Unshuffle"
                    else:
                        if IDX_TO_OBJECT_TYPE[ind] in PICKUPABLE_OBJECTS:
                            type_str = "PutObject"
                        # elif IDX_TO_OBJECT_TYPE[ind] in OPENABLE_OBJECTS:
                        #     type_str = "OpenObject"
                        else:
                            raise NotImplementedError
                        target_map_type_str = "Walkthrough"
                    arg_id = ind
                else:
                    raise NotImplementedError

            target_map_type_intid = Subtask.target_map_type_str_to_intid(target_map_type_str)
            if arg_id != -1:
                subtasks.append(
                    Subtask.from_type_str_arg_target_map_id_with_mask(
                        type_str=type_str,
                        arg_id=arg_id,
                        tmap_id=target_map_type_intid,
                        mask=VoxelGrid.create_from_mask(comb_map_stack[target_map_type_intid, i:i+1, arg_id:arg_id+1]),
                    ).to(device)
                )
            else:
                subtasks.append(
                    Subtask.from_type_str_arg_target_map_id(
                        type_str=type_str,
                        arg_id=arg_id,
                        tmap_id=target_map_type_intid,
                    ).to(device)
                )

        return Subtask.collate(subtasks)


class TaskAwareOnePhaseRearrangeNetwork(ActorCriticModel):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        in_walkthrough_phase_uuid: str,
        depth_uuid: str,
        unshuffled_depth_uuid: str,
        sem_seg_uuid: str,
        unshuffled_sem_seg_uuid: str,
        pose_uuid: str,
        inventory_uuid: str,
        sem_map_uuid: str,
        unshuffled_sem_map_uuid: str,
        is_walkthrough_phase_embedding_dim: int,
        done_action_index: int,
        ordered_object_types: Sequence[str],
        prev_action_embedding_dim: int = 32,
        subtask_type_embedding_dim: int = 32,
        sutask_arg_embedding_dim: int = 32,
        subtask_target_map_type_embedding_dim: int = 32,
        hidden_size: int = 512,
        num_rnn_layers: int = 1,
        rnn_type: str = "LSTM",
        heuristics: bool = False,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.in_walkthrough_phase_uuid = in_walkthrough_phase_uuid
        self.depth_uuid = depth_uuid
        self.unshuffled_depth_uuid = unshuffled_depth_uuid
        self.sem_seg_uuid = sem_seg_uuid
        self.unshuffled_sem_seg_uuid = unshuffled_sem_seg_uuid
        self.pose_uuid = pose_uuid
        self.inventory_uuid = inventory_uuid
        self.sem_map_uuid = sem_map_uuid
        self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

        self.done_action_index = done_action_index
        self.ordered_object_types = ordered_object_types
        self.num_rnn_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers
        self.heuristics = heuristics

        self.prev_action_embedder = nn.Embedding(
            action_space.n + 1, embedding_dim=prev_action_embedding_dim
        )
        self.subtask_type_embedder = nn.Embedding(
            num_embeddings=len(IDX_TO_SUBTASK_TYPE) + 1,    # to indicate Null subtask type
            embedding_dim=subtask_type_embedding_dim,
        )
        self.subtask_arg_embedder = nn.Embedding(
            num_embeddings=self.num_objects,
            embedding_dim=sutask_arg_embedding_dim,
        )
        self.subtask_target_map_type_embedder = nn.Embedding(
            num_embeddings=2,
            embedding_dim=subtask_target_map_type_embedding_dim,
        )
        # self.is_walkthrough_phase_embedder = nn.Embedding(
        #     num_embeddings=2, embedding_dim=is_walkthrough_phase_embedding_dim
        # )

        # Separate visual encoder? (Navigation / Interaction)
        assert (
            self.observation_space[self.rgb_uuid].shape[0] 
            == self.observation_space[self.unshuffled_rgb_uuid].shape[0]
        )
        self.visual_encoder = EgocentricViewEncoderPooled(
            img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
            hidden_dim=self._hidden_size,
        )
        self.sem_map_encoder = Semantic2DMapEncoderPooled(
            n_map_channels=self.map_channel,
            hidden_dim=self._hidden_size,
        )
        self.sem_diff_predictor = nn.Linear(self._hidden_size, self.map_channel - 3)

        # if heuristics:
        #     self.subtask_model = SubtaskPlanningModel(
        #         num_subtask_types=len(IDX_TO_SUBTASK_TYPE),
        #         num_subtask_arguments=self.num_objects,
        #     )
        # else:
        #     self.subtask_model = SubtaskPredictionModel(
        #         num_subtask_types=len(IDX_TO_SUBTASK_TYPE),
        #         num_subtask_arguments=self.num_objects,
        #         hidden_dim=self._hidden_size,
        #     )
        # self.test_planner = SubtaskPlanningModel(
        #     num_subtask_types=len(IDX_TO_SUBTASK_TYPE),
        #     num_subtask_arguments=self.num_objects,
        # )
        self.test_predictor = SubtaskPredictionModel(
            num_subtask_types=len(IDX_TO_SUBTASK_TYPE),
            num_subtask_arguments=self.num_objects,
            hidden_dim=self._hidden_size,
        )

        # State encoder for navigation and interaction
        self.state_encoder = RNNStateEncoder(
            input_size=(
                self._hidden_size * 2
                + prev_action_embedding_dim
                + subtask_type_embedding_dim
                + sutask_arg_embedding_dim
                + subtask_target_map_type_embedding_dim                
            ),
            hidden_size=self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        
        for v in IDX_TO_SUBTASK_TYPE.values():
            setattr(
                self,
                f'{stringcase.snakecase(v)}_ac',
                LinearActorCriticHead(self._hidden_size, self.action_space.n),
            )
        # self.actor = LinearActorHead(self._hidden_size, action_space.n)
        # self.critic = LinearCriticHead(self._hidden_size)
        # ActorCriticHead for every subtask types

        self.device = device if device else torch.device("cpu")

    def _create_visual_encoder(self) -> Optional[nn.Module]:
        """Create the visual encoder for the model."""
        return None

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
            walkthrough_sem_map=(
                (
                    ("sampler", None),
                    ("channels", self.map_channel),
                    ("width", self.map_width),
                    ("length", self.map_length),
                    ("hegith", self.map_height),
                ),
                torch.bool,
            ),
            sem_map=(
                (
                    ("sampler", None),
                    ("channels", self.map_channel),
                    ("width", self.map_width),
                    ("length", self.map_length),
                    ("hegith", self.map_height),
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
        memory: [steps/sampler, ...] #TODO: findout 
        prev_actions: [steps, samplers, ...]
        masks: [steps, samplers, agents, 1], zero indicates the steps where a new episode/task starts
        """
        nsteps, nsamplers = masks.shape[:2]
        # Egocentric images
        u_img = observations[self.rgb_uuid]
        w_img = observations[self.unshuffled_rgb_uuid]

        ego_img_embeddings = self.visual_encoder(
            u_img_emb=u_img,
            w_img_emb=w_img
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

        # Semantic maps
        w_sem_map_prev = memory.tensor('walkthrough_sem_map')
        sem_map_prev = memory.tensor('sem_map')

        map_mask = masks.view(*masks.shape[:2], 1, 1, 1, 1)
        sem_map = observations[self.sem_map_uuid]
        w_sem_map = observations[self.unshuffled_sem_map_uuid]
        
        map_embeddings = []
        subtasks = []
        subtask_type_logprobs = []
        subtask_arg_logprobs = []
        subtask_target_map_type_logprobs = []
        # subtasks_planned = []
        for step in range(nsteps):
            # Map update for walkthrough environment
            w_sem_map_prev = w_sem_map_prev * map_mask[step]
            w_sem_map_prev[:, :-2] = (
                w_sem_map_prev[:, :-2] * ~w_sem_map[step, :, -1:]
                + w_sem_map[step, :, :-2] * w_sem_map[step, :, -1:]
            )
            w_sem_map_prev[:, -2:-1] = (
                w_sem_map_prev[:, -2:-1] * ~w_sem_map[step, :, -1:]
                + w_sem_map[step, :, -2:-1] * w_sem_map[step, :, -1:]
            )
            w_sem_map_prev[:, -1:] = torch.max(
                w_sem_map[step, :, -1:], w_sem_map_prev[:, -1:]
            )

            # Map update for unshuffle environment
            sem_map_prev = sem_map_prev * map_mask[step]
            sem_map_prev[:, :-2] = (
                sem_map_prev[:, :-2] * ~sem_map[step, :, -1:]
                + sem_map[step, :, :-2] * sem_map[step, :, -1:]
            )
            sem_map_prev[:, -2:-1] = (
                sem_map_prev[:, -2:-1] * ~sem_map[step, :, -1:]
                + sem_map[step, :, -2:-1] * sem_map[step, :, -1:]
            )
            sem_map_prev[:, -1:] = torch.max(
                sem_map[step, :, -1:], sem_map_prev[:, -1:]
            )

            # Memory Issue :(
            # Changed semantic map encoder from Conv3D to Conv2D
            sem_map_embedding = self.sem_map_encoder(
                walkthrough_sem_map_data=w_sem_map_prev.max(-1).values,
                unshuffle_sem_map_data=sem_map_prev.max(-1).values,
                inventory_vector=observations[self.inventory_uuid][step]
            )
            # # Heuristic subtask planning
            # subtask_planned = self.test_planner.plan(
            #     unshuffle_comb_map=sem_map_prev,
            #     walkthrough_comb_map=w_sem_map_prev,
            #     inventory_vector=observations[self.inventory_uuid][step],
            #     # val_diff=sem_map_diff_val,
            #     # ind_diff=sem_map_diff_ind,
            # ).to_tensor(device=sem_map_embedding.device)

            # subtask prediction based on sem_map_inv_embedding
            # subtask_type_logprob: [sampler, num_subtask_types]
            # subtask_arg_logprob: [sampler, num_arguments (num_obj_classes + 1)]
            # subtask_target_map_type_logprob: [sampler, 2 (unshuffle / walkthrough)]
            subtask_type_logprob, subtask_arg_logprob, subtask_target_map_type_logprob = self.test_predictor(
                map_inv_embeddings=sem_map_embedding,
            )
            subtask = self.test_predictor.sample_subtasks(
                type_distr=subtask_type_logprob,
                arg_vectors=subtask_arg_logprob,
                tmap_distr=subtask_target_map_type_logprob,
            ).to_tensor(device=sem_map_embedding.device)
            # [sampler, 3 (subtask_type, subtask_argument, subtask_target_map)] 
            subtask_type_embedding = self.subtask_type_embedder(
                (masks[step].long() * (subtask[:, 0].unsqueeze(-1) + 1))
            ).squeeze(-2)   # [sampler, subtask_type_embedding_dim]
            subtask_arg_embedding = self.subtask_arg_embedder(
                (masks[step].long() * (subtask[:, 1].unsqueeze(-1) + 1))
            ).squeeze(-2)   # [sampler, subtask_arg_embedding_dim]
            subtask_target_map_type_embedding = self.subtask_target_map_type_embedder(
                subtask[:, 2].unsqueeze(-1)
            ).squeeze(-2)   # [sampler, subtask_target_map_type_embedding_dim]

            rnn_input = torch.cat(
                (
                    obs_for_rnn[step],
                    sem_map_embedding,
                    subtask_type_embedding,
                    subtask_arg_embedding,
                    subtask_target_map_type_embedding,
                ),
                dim=-1,
            ).unsqueeze(0)

            rnn_out, rnn_hidden_states = self.state_encoder(
                rnn_input, 
                rnn_hidden_states, 
                masks[step:step+1],
            )
            
            rnn_outs.append(rnn_out)
            map_embeddings.append(sem_map_embedding)
            subtasks.append(subtask)
            subtask_type_logprobs.append(subtask_type_logprob)
            subtask_arg_logprobs.append(subtask_arg_logprob)
            subtask_target_map_type_logprobs.append(subtask_target_map_type_logprob)
            # subtasks_planned.append(subtask_planned)

        rnn_outs = torch.cat(rnn_outs, dim=0)   # nsteps, snamplers, hdims
        map_embeddings = torch.stack(map_embeddings, dim=0)     # nsteps x nsamplers x hdims
        subtasks = torch.stack(subtasks, dim=0)     # nsteps x nsamplers x 3
        subtask_type_logprobs = torch.stack(subtask_type_logprobs, dim=0)
        subtask_arg_logprobs = torch.stack(subtask_arg_logprobs, dim=0)
        subtask_target_map_type_logprobs = torch.stack(subtask_target_map_type_logprobs, dim=0)
        # subtasks_planned = torch.stack(subtasks_planned, dim=0)     # nsteps x nsamplers x 3

        memory = memory.set_tensor(key="walkthrough_sem_map", tensor=w_sem_map_prev.bool())
        memory = memory.set_tensor(key="sem_map", tensor=sem_map_prev.bool())

        subtask_logits = []
        subtask_vals = []
        for v in IDX_TO_SUBTASK_TYPE.values():
            snake_v = stringcase.snakecase(v)
            ac = getattr(self, f"{snake_v}_ac")
            exec(f"{snake_v}_distr, {snake_v}_vals = ac(rnn_outs)")
            exec(f"{snake_v}_logits = {snake_v}_distr.logits")
            exec(f"subtask_logits.append({snake_v}_logits)")
            exec(f"subtask_vals.append({snake_v}_vals)")

        subtask_logits = torch.stack(subtask_logits, dim=0)
        subtask_vals = torch.stack(subtask_vals, dim=0)

        subtask_oh = F.one_hot(subtasks[:, :, 0:1], len(IDX_TO_SUBTASK_TYPE)).permute(-1, 0, 1, 2)
        actor_logits = torch.einsum('abcd,abcd->bcd', subtask_logits, subtask_oh.float())
        actor = CategoricalDistr(logits=actor_logits,)
        values = torch.einsum('abcd,abcd->bcd', subtask_vals, subtask_oh.float())

        memory.set_tensor(key="rnn", tensor=rnn_hidden_states)
        # test_x = torch.rand(nsteps, nsamplers, self._hidden_size).to(masks.device)
        return (
            ActorCriticOutput(
                distributions=actor, values=values, extras={
                    "subtask_logits": {
                        "type": subtask_type_logprobs,
                        "arg": subtask_arg_logprobs,
                        "target_map": subtask_target_map_type_logprobs,
                    },
                },
            ),
            memory,
        )

    @property
    def num_objects(self):
        return len(self.ordered_object_types) + 1

    @property
    def map_size(self):
        return self.observation_space[self.sem_map_uuid].shape

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



###########################################################################
# class TaskAwareOnePhaseRearrangeNetwork(ActorCriticModel):

#     def __init__(
#         self,
#         action_space: gym.Space,
#         observation_space: gym.spaces.Dict,
#         rgb_uuid: str,
#         unshuffled_rgb_uuid: str,
#         in_walkthrough_phase_uuid: str,
#         depth_uuid: str,
#         unshuffled_depth_uuid: str,
#         sem_seg_uuid: str,
#         unshuffled_sem_seg_uuid: str,
#         pose_uuid: str,
#         unshuffled_pose_uuid: str,
#         inventory_uuid: str,
#         subtask_expert_uuid: str,
#         sem_map_uuid: str,
#         unshuffled_sem_map_uuid: str,
#         is_walkthrough_phase_embedding_dim: int,
#         done_action_index: int,
#         ordered_object_types: Sequence[str],
#         prev_action_embedding_dim: int = 32,
#         subtask_type_embedding_dim: int = 32,
#         sutask_arg_embedding_dim: int = 32,
#         subtask_target_map_type_embedding_dim: int = 32,
#         hidden_size: int = 512,
#         num_rnn_layers: int = 1,
#         rnn_type: str = "LSTM",
#         fov: int = 90,
#         grid_parameters: GridParameters = GridParameters(),
#         episode_max_length: int = 500,
#         device: torch.device = None,
#         **kwargs,
#     ):
#         super().__init__(action_space=action_space, observation_space=observation_space)
#         self._hidden_size = hidden_size

#         self.rgb_uuid = rgb_uuid
#         self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
#         self.in_walkthrough_phase_uuid = in_walkthrough_phase_uuid
#         self.depth_uuid = depth_uuid
#         self.unshuffled_depth_uuid = unshuffled_depth_uuid
#         self.sem_seg_uuid = sem_seg_uuid
#         self.unshuffled_sem_seg_uuid = unshuffled_sem_seg_uuid
#         self.pose_uuid = pose_uuid
#         self.unshuffled_pose_uuid = unshuffled_pose_uuid
#         self.inventory_uuid = inventory_uuid
#         self.subtask_expert_uuid = subtask_expert_uuid
#         self.sem_map_uuid = sem_map_uuid
#         self.unshuffled_sem_map_uuid = unshuffled_sem_map_uuid

#         self.ordered_object_types = ordered_object_types
#         self.num_rnn_layers = 2 * num_rnn_layers if "LSTM" in rnn_type else num_rnn_layers
#         self.episode_max_length = episode_max_length

#         # for v in IDX_TO_SUBTASK_TYPE.values():
#         #     setattr(
#         #         self,
#         #         f'{stringcase.snakecase(v)}_ac',
#         #         LinearActorCriticHead(self._hidden_size, self.action_space.n),
#         #     )
#         # self._lowlevel_actor_critics = {
#         #     k: getattr(self, f"{stringcase.snakecase(v)}_ac")
#         #     for k, v in IDX_TO_SUBTASK_TYPE.items()
#         # }

#         self.visual_encoder = EgocentricViewEncoderPooled(
#             img_embedding_dim=self.observation_space[self.rgb_uuid].shape[0],
#             hidden_dim=self._hidden_size,
#         )

#         self.subtask_model = SubtaskPredictionModel(
#             hidden_size=128,
#         )

#         self.fov = fov
#         self.grid_parameters = grid_parameters
#         self._map_size = None
        

#         self.actor = LinearActorHead(self._hidden_size, action_space.n)
#         self.critic = LinearCriticHead(self._hidden_size)


#         self.device = device if device else torch.device("cpu")

#     def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
#         return dict(
#             rnn=(
#                 (
#                     ("layer", self.num_rnn_layers),
#                     ("sampler", None),
#                     ("hidden", self._hidden_size),
#                 ),
#                 torch.float32,
#             ),
#             sem_map=(
#                 (
#                     ("sampler", None),
#                     ("map_type", 2),
#                     ("channels", self.map_channel),
#                     ("width", self.map_width),
#                     ("length", self.map_length),
#                     ("height", self.map_height),
#                 ),
#                 torch.bool,
#             ),
#             # Subtask Histories -> for inference, not required for training subtask model!!
#             # subtask_history=(
#             #     (
#             #         ("max_length", self.episode_max_length),
#             #         ("sampler", None),
#             #         ("channels", 4),
#             #     ),
#             #     torch.int,
#             # ),
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
        
#         # Subtask Histories -> for inference, not required for training subtask model!!
#         # subtask_history = memory.tensor('subtask_history')
#         # for sampler_id in range(nsamplers):
#         #     nz_idxs = subtask_history[:, sampler_id, -1].nonzero()
#         #     max_idx = nz_idxs.max().item() + 1
#         # # find out maximum index
#         # inv_idx = torch.arange(subtask_history.size(0) - 1, -1, -1, device=subtask_history.device)
#         # max_indices = subtask_history.size(0) - 1 - subtask_history.index_select(0, inv_idx)[:, :, -1:].max(0).indices
#         # sampler_subtask_histories = []
#         # for sampler_id in range(nsamplers):
#         #     inds = (
#         #         list(range(max_indices[sampler_id].item() + 1)),
#         #         [inds] * len(range(max_indices[sampler_id].item() + 1))
#         #     )
#         #     sampler_subtask_histories.append(
#         #         subtask_history[inds]
#         #     )

#         # Semantic maps: (sampler, channels, width, length, height)
#         sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Unshuffle"]]
#         w_sem_map_prev = memory.tensor('sem_map')[:, MAP_TYPES_TO_IDX["Walkthrough"]]

#         map_masks = masks.view(*masks.shape[:2], 1, 1, 1, 1)
#         sem_maps = observations[self.sem_map_uuid]
#         w_sem_maps = observations[self.unshuffled_sem_map_uuid]

#         updated_sem_maps = []
#         for step in range(nsteps):
#             sem_map_prev = update_semantic_map(
#                 sem_map=sem_maps[step],
#                 sem_map_prev=sem_map_prev,
#                 map_mask=map_masks[step],
#             )
#             w_sem_map_prev = update_semantic_map(
#                 sem_map=w_sem_maps[step],
#                 sem_map_prev=w_sem_map_prev,
#                 map_mask=map_masks[step],
#             )
#             sem_maps_prev = torch.stack(
#                 (sem_map_prev, w_sem_map_prev), dim=1
#             )
#             updated_sem_maps.append(sem_maps_prev)

#         # stack updated semantic maps along with new axis (timesteps)
#         # [nsteps, nsamplers, n_map_types, nchannels, width, length, height]
#         updated_sem_maps = torch.stack(updated_sem_maps, dim=0)
        
#         # During the training of SubtaskModel, we use the result of subtask_expert as subtask_history.
#         # with indicators that distinguishes the different episodes/tasks
#         subtask_history = observations[self.subtask_expert_uuid]    # [nsteps, nsamplers, 4]
#         inventory_vectors = observations[self.inventory_uuid]       # [nsteps, nsamplers, num_objects]
        
#         extras = {}
#         if nsteps > 1:

#             subtask_type_logprobs, subtask_arg_lobprobs, subtask_target_map_type_logprobs = self.subtask_model(
#                 semantic_maps=updated_sem_maps.view(-1, *updated_sem_maps.shape[2:]),
#                 inventory_vectors=inventory_vectors.view(-1, *inventory_vectors.shape[2:]),
#                 subtask_index_history=subtask_history[..., 0].permute(1, 0).reshape(-1).contiguous(),
#                 seq_masks=masks.permute(1, 0, 2).reshape(-1).contiguous(),
#                 nsteps=nsteps,
#                 nsamplers=nsamplers,
#             )

#             subtask_type_logprobs = subtask_type_logprobs.view(
#                 nsteps, nsamplers, *subtask_type_logprobs.shape[1:]
#             )   # [nsteps, nsamplers, NUM_SUBTASK_TYPES]
#             subtask_arg_lobprobs = subtask_arg_lobprobs.view(
#                 nsteps, nsamplers, *subtask_arg_lobprobs.shape[1:]
#             )   # [nsteps, nsamplers, NUM_OBJECTS_TYPES]
#             subtask_target_map_type_logprobs = subtask_target_map_type_logprobs.view(
#                 nsteps, nsamplers, *subtask_target_map_type_logprobs.shape[1:]
#             )   # [nsteps, nsamplers, NUM_MAP_TYPES]
#             extras["subtask_logits"] = {
#                 "type": subtask_type_logprobs,
#                 "arg": subtask_arg_lobprobs,
#                 "target_map": subtask_target_map_type_logprobs,
#             }

#         memory = memory.set_tensor(
#             key="sem_map",
#             tensor=sem_maps_prev.type(torch.bool)
#         )

#         test_x = torch.rand(nsteps, nsamplers, self._hidden_size).to(masks.device)
#         return (
#             ActorCriticOutput(
#                 distributions=self.actor(test_x), values=self.critic(test_x), extras=extras
#             ), 
#             memory
#         )

#     @property
#     def num_objects(self):
#         return len(self.ordered_object_types) + 1

#     @property
#     def map_size(self):
#         if self._map_size is None:
#             empty_voxel, *_ = create_empty_voxel_data()
#             self._map_size = empty_voxel.shape[-3:]
#         return self._map_size

#     @property
#     def map_channel(self):
#         return self.num_objects + 2

#     @property
#     def map_width(self):
#         return self.map_size[0]

#     @property
#     def map_length(self):
#         return self.map_size[1]

#     @property
#     def map_height(self):
#         return self.map_size[2]