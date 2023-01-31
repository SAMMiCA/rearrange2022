from typing import Dict

import torch
import torch.nn as nn

from task_aware_rearrange.layers import (
    EgocentricViewEncoderPooled,
    Semantic2DMapWithInventoryEncoderPooled,
    SemanticMap2DEncoderPooled,
    SubtaskHistoryEncoder,
    SubtaskPredictor,
)
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from task_aware_rearrange.constants import NUM_OBJECT_TYPES
from experiments.test_exp import ExpertTestExpConfig


class SubtaskPredictionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        prev_action_embedding_dim: int = 32,
        egoview_embedding_dim: int = 2048,
        resnet_embed: bool = True,
        semantic_map_embed: bool = False,
        inventory: bool = True,
        num_actions: int = len(ExpertTestExpConfig.actions()),
        num_subtasks: int = NUM_SUBTASKS,
        num_objects: int = NUM_OBJECT_TYPES,
    ) -> None:
        super().__init__()

        subtask_predictor_input_size = 0
        self.egoview_encoder = EgocentricViewEncoderPooled(egoview_embedding_dim, hidden_size)
        subtask_predictor_input_size += hidden_size

        self.prev_action_embedder = nn.Embedding(num_actions + 1, prev_action_embedding_dim)
        subtask_predictor_input_size += prev_action_embedding_dim
        
        self.resnet_embed = resnet_embed
        self.semantic_map_embed = semantic_map_embed
        self.inventory = inventory

        self.semmap_encoder = None
        if semantic_map_embed:
            if inventory:
                self.semmap_encoder = Semantic2DMapWithInventoryEncoderPooled(
                    n_map_channels=NUM_OBJECT_TYPES + 3,
                    hidden_size=hidden_size,
                    additional_map_channels=3,
                )
            else:
                self.semmap_encoder = SemanticMap2DEncoderPooled(
                    n_map_channels=NUM_OBJECT_TYPES + 3,
                    hidden_size=hidden_size,
                )
            subtask_predictor_input_size += hidden_size

        self.subtask_history_encoder = SubtaskHistoryEncoder(
            hidden_size=hidden_size,
            num_subtasks=NUM_SUBTASKS,
        )
        subtask_predictor_input_size += hidden_size

        self.subtask_predictor = SubtaskPredictor(
            input_size=subtask_predictor_input_size,
            hidden_size=hidden_size,
            num_subtasks=NUM_SUBTASKS
        )

        self.hidden_size = hidden_size
        self.input_size = subtask_predictor_input_size
        self.prev_action_embedding_dim = prev_action_embedding_dim
        self.egoview_embedding_dim = egoview_embedding_dim

    def forward(
        self,
        x: Dict[str, torch.Tensor],
    ):
        subtask_predictor_inputs = []
        u_rgb_resnet = x["unshuffle_rgb_resnet"].float()
        w_rgb_resnet = x["walkthrough_rgb_resnet"].float()
        egoview_embeddings = self.egoview_encoder(
            u_img_emb=u_rgb_resnet,
            w_img_emb=w_rgb_resnet,
        )   # (bsize, hidden_size)
        subtask_predictor_inputs.append(egoview_embeddings)

        prev_action_embeddings = self.prev_action_embedder(
            (x["masks"].long() * (x["prev_actions"] + 1))
        )   # (bsize, prev_action_embedding_dim)
        subtask_predictor_inputs.append(prev_action_embeddings)

        semmap_embeddings = None
        semmap_encoder_inputs = {}
        if self.semantic_map_embed:
            u_semmap = x["unshuffle_semmap"].max(-1).values
            w_semmap = x["walkthrough_semmap"].max(-1).values
            semmap_encoder_inputs["unshuffle_sem_map_data"] = u_semmap.float()
            semmap_encoder_inputs["walkthrough_sem_map_data"] = w_semmap.float()
            if self.inventory:
                semmap_encoder_inputs["inventory_vector"] = x["inventory"]

            semmap_embeddings = self.semmap_encoder(
                **semmap_encoder_inputs
            )   # (bsize, hidden_size)

        if semmap_embeddings is not None:
            subtask_predictor_inputs.append(semmap_embeddings)

        subtask_history_embeddings = self.subtask_history_encoder(
            subtask_index_history=x["subtask_history"],
            seq_masks=x["masks"].long(),
        )[:-1]  # (bsize, hidden_size)
        subtask_predictor_inputs.append(subtask_history_embeddings)

        subtask_logprob = self.subtask_predictor(torch.cat(subtask_predictor_inputs, dim=-1))

        return subtask_logprob
