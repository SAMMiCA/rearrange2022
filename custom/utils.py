import torch

from custom.constants import NUM_MAP_TYPES, NUM_OBJECT_TYPES, NUM_SUBTASK_TYPES


def index_to_onehot(
    index_tensor: torch.Tensor,
    num_classes: int,
):
    # index_tensor = index_tensor.unsqueeze(-1)
    device = index_tensor.device
    
    ones = torch.ones(*index_tensor.shape, device=device)
    oh = torch.zeros((*index_tensor.shape, num_classes), device=device)
    oh = oh.scatter_add(dim=-1, index=index_tensor.unsqueeze(-1), src=ones.unsqueeze(-1))

    return oh


def subtask_index_to_type_arg_target_map(
    index_tensor: torch.Tensor,
    num_subtask_types: int = NUM_SUBTASK_TYPES,
    num_subtask_argument: int = NUM_OBJECT_TYPES,
    num_subtask_target_map_types: int = NUM_MAP_TYPES,
):
    subtask_type_index = subtask_index_to_type_index(
        index_tensor, num_subtask_types, num_subtask_argument, num_subtask_target_map_types
    )

    subtask_argument_index = subtask_index_to_argument_index(
        index_tensor, num_subtask_types, num_subtask_argument, num_subtask_target_map_types
    )

    subtask_target_map_type_index = subtask_index_to_target_map_type_index(
        index_tensor, num_subtask_types, num_subtask_argument, num_subtask_target_map_types
    )

    return torch.stack(
        (
            subtask_type_index,
            subtask_argument_index,
            subtask_target_map_type_index,
        ),
        dim=-1
    ).long()



def subtask_index_to_type_index(
    index_tensor: torch.Tensor,
    num_subtask_types: int = NUM_SUBTASK_TYPES,
    num_subtask_argument: int = NUM_OBJECT_TYPES,
    num_subtask_target_map_types: int = NUM_MAP_TYPES,
):
    """
    subtask_index = (
        subtask_type_index * num_subtask_argument * num_subtask_target_map_types 
        + subtask_argument_index * num_subtask_target_map_types
        + subtask_target_map_type_index
    )
    [0 ~ (num_subtask_types - 1) * num_subtask_argument * num_subtask_target_map_types]
    subtask["Stop"] does not requires any argument or target map
    subtask_type_index = int(subtask_index / (num_subtask_argument * num_target_map_types))
    """

    return (
        index_tensor / (num_subtask_argument * num_subtask_target_map_types)
    ).long()



def subtask_index_to_argument_index(
    index_tensor: torch.Tensor,
    num_subtask_types: int = NUM_SUBTASK_TYPES,
    num_subtask_argument: int = NUM_OBJECT_TYPES,
    num_subtask_target_map_types: int = NUM_MAP_TYPES,
):
    """
    subtask_index = (
        subtask_type_index * num_subtask_argument * num_subtask_target_map_types 
        + subtask_argument_index * num_subtask_target_map_types
        + subtask_target_map_type_index
    )
    [0 ~ (num_subtask_types - 1) * num_subtask_argument * num_subtask_target_map_types]
    subtask["Stop"] does not requires any argument or target map
    subtask_argument_index = int((subtask_index % (num_subtask_argument * num_subtask_target_map_types)) / num_subtask_target_map_types)
    """

    return (
        (index_tensor % (num_subtask_argument * num_subtask_target_map_types)) / num_subtask_target_map_types
    ).long()


def subtask_index_to_target_map_type_index(
    index_tensor: torch.Tensor,
    num_subtask_types: int = NUM_SUBTASK_TYPES,
    num_subtask_argument: int = NUM_OBJECT_TYPES,
    num_subtask_target_map_types: int = NUM_MAP_TYPES,
):
    """
    subtask_index = (
        subtask_type_index * num_subtask_argument * num_subtask_target_map_types 
        + subtask_argument_index * num_subtask_target_map_types
        + subtask_target_map_type_index
    )
    [0 ~ (num_subtask_types - 1) * num_subtask_argument * num_subtask_target_map_types]
    subtask["Stop"] does not requires any argument or target map
    subtask_target_map_type_index = subtask_index % num_subtask_target_map_types
    """

    return (
        index_tensor % num_subtask_target_map_types
    ).long()


def masks_to_batch_ids(
    seq_masks: torch.Tensor,
):
    """
    seq_masks: [batch_size, ]
    """
    # seq_masks = masks.permute(1, 0, 2).reshape(-1).contiguous()
    batch_ids = torch.zeros_like(seq_masks)
    for it, mask in enumerate(seq_masks):
        if mask == 0 and it > 0:
            batch_ids[it:] += 1

    return batch_ids


def batch_ids_to_ranges(
    batch_ids: torch.Tensor
):
    batch_ranges = torch.zeros_like(batch_ids)
    ord = 0
    prev_bid = -1
    for it, b_id in enumerate(batch_ids):
        if b_id != prev_bid:
            ord = 0
        batch_ranges[it] = ord
        ord += 1
        prev_bid = b_id
    
    return batch_ranges
        

def positional_encoding(
    x: torch.Tensor,
    batch_ids: torch.Tensor,
):
    b, c = x.shape
    assert c % 2 == 0

    inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=x.device).float() / c))
    pos_b = batch_ids_to_ranges(batch_ids)
    sin_inp_h = torch.einsum("i,j->ij", pos_b, inv_freq)
    emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)

    return emb_h


def build_attention_mask(
    batch_ids: torch.Tensor,
    batch_id: int,
    include_self:bool = False,
):
    is_b = (batch_ids == batch_id)
    n = torch.sum(is_b)
    mask = torch.ones((n, n), device=batch_ids.device, dtype=torch.float32)

    if include_self:
        mask = torch.triu(mask, diagonal=0)
    else:
        mask = torch.triu(mask, diagonal=1)

    return mask


def build_attention_masks(
    batch_ids: torch.Tensor,
    add_sos_token=False,
    include_self=False,
):
    batch_size = len(batch_ids)
    num_episodes = int(max(batch_ids).item() + 1 if len(batch_ids) > 0 else 0)

    # Build attention masks
    attention_mask = torch.zeros((batch_size, batch_size), device=batch_ids.device, dtype=torch.float32)
    s = 0
    for epi in range(num_episodes):
        mask = build_attention_mask(batch_ids, epi, include_self)
        n = mask.shape[0]
        attention_mask[s:s+n, s:s+n] = mask
        s += n

    # Rows correspond to elements in output sequence. Columns to elements in in put sequence
    attention_mask = attention_mask.T

    # Every output element may attend to the first element in the input sequence
    # The first element represents the "SOS" token
    if add_sos_token:
        attention_mask = torch.cat(
            [
                torch.zeros([1, batch_size], device=batch_ids.device, dtype=torch.float32),
                attention_mask
            ],
            dim=0
        )

        attention_mask = torch.cat(
            [
                torch.ones([batch_size+1, 1], device=batch_ids.device, dtype=torch.float32),
                attention_mask
            ],
            dim=1
        )

    attention_mask = 1 - attention_mask
    return attention_mask.bool()