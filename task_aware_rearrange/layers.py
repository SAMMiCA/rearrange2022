import torch
import torch.nn as nn
import torch.nn.functional as F


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