import torch
from task_aware_rearrange.constants import IDX_TO_OBJECT_TYPE, NUM_OBJECT_TYPES, UNKNOWN_OBJECT_STR
from task_aware_rearrange.subtasks import IDX_TO_SUBTASK, MAP_TYPE_TO_IDX, NUM_SUBTASK_TYPES, NUM_MAP_TYPES, SUBTASK_TARGET_OBJECT_TO_IDX, SUBTASK_TO_IDX, SUBTASK_TYPE_TO_IDX


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


def subtask_index_to_type_arg(
    index_tensor: torch.Tensor,
):
    outs = []
    for index in index_tensor:
        subtask_type, target_obj, target_map = IDX_TO_SUBTASK[index.item()]
        type_idx = SUBTASK_TYPE_TO_IDX[subtask_type]
        if target_map is None:
            target_map = "Unshuffle"
        if target_obj is None:
            target_obj = UNKNOWN_OBJECT_STR
        else:
            target_obj = '_'.join(target_obj, target_map)
        obj_idx = SUBTASK_TARGET_OBJECT_TO_IDX[target_obj]
        outs.append([type_idx, obj_idx])

    return torch.tensor(outs, device=index_tensor.device)

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