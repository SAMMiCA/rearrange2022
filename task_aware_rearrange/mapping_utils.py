import torch

def update_semantic_map(
    sem_map: torch.Tensor,
    sem_map_prev: torch.Tensor,
    map_mask: torch.Tensor,
):
    """
    sem_map [nsampler, nchannels, width, length, height]: sem_maps[step]
    sem_map_prev [nsampler, nchannels, width, length, height]
    map_mask  [nsampler, 1, 1, 1, 1]: map_masks[step]
    """
    if len(sem_map.shape) == 4:
        assert len(sem_map_prev.shape) == 4, f"len(sem_map_prev.shape): {len(sem_map_prev.shape)}"
        assert len(map_mask.shape) == 4, f"len(map_mask.shape)): {len(map_mask.shape)}"
        sem_map = sem_map[None, ...]
        sem_map_prev = sem_map_prev[None, ...]
        map_mask = map_mask[None, ...]

    sem_map_prev = sem_map_prev * map_mask
    
    # update agent_position_map
    sem_map_prev[:, 0:1] = sem_map[:, 0:1]
    
    # update voxel_data based on current voxel_observability
    sem_map_prev[:, 1:-2] = (
        sem_map_prev[:, 1:-2] * ~sem_map[:, -1:]
        + sem_map[:, 1:-2] * sem_map[:, -1:]
    )
    
    # update voxel_occupancy based on current voxel_observability
    sem_map_prev[:, -2:-1] = (
        sem_map_prev[:, -2:-1] * ~sem_map[:, -1:]
        + sem_map[:, -2:-1] * sem_map[:, -1:]
    )

    # update volxe_observability via max pooling
    sem_map_prev[:, -1:] = torch.max(
        sem_map[:, -1:], sem_map_prev[:, -1:]
    )

    return sem_map_prev
