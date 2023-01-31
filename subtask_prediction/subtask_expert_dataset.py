import time
from typing import (
    Any,
    Optional,
    Sequence,
    List,
    Tuple,
    Union,
    Dict,
)
import os
import random
import numpy as np
import torch
from itertools import chain
from torch.utils.data.dataset import IterableDataset


class SubtaskExpertIterableDataset(IterableDataset):

    DATANAMES = [
        'unshuffle_rgb', 'walkthrough_rgb',
        'unshuffle_depth', 'walkthrough_depth',
        'unshuffle_semseg', 'walkthrough_semseg',
        'unshuffle_pos_rot_horizon', 'walkthrough_pos_rot_horizon',
        'unshuffle_pos_unity', 'walkthrough_pos_unity',
        'unshuffle_Tu2w', 'walkthrough_Tu2w',
        'unshuffle_Tw2c', 'walkthrough_Tw2c',
        'expert_subtask_action',
        'inventory',        
    ]

    def __init__(
        self,
        episode_paths: List[str],
        shuffle: bool = False,
        mean_rgb: Optional[List[float]] = None,
        std_rgb: Optional[List[float]] = None,
        mean_depth: Optional[List[float]] = None,
        std_depth: Optional[List[float]] = None,
    ) -> None:

        if shuffle:
            episode_paths = random.sample(
                episode_paths,
                len(episode_paths)
            )
        self.episode_paths = episode_paths

        self.should_normalize_rgb = (
            (mean_rgb is not None) and (std_rgb is not None)
        )
        self.mean_rgb, self.std_rgb = mean_rgb, std_rgb

        self.should_normalize_depth = (
            (mean_depth is not None) and (std_depth is not None)
        )
        self.mean_depth, self.std_depth = mean_depth, std_depth
        self.start_index = 0
        
        # a = time.time()
        # self.len_rollouts = [
        #     np.load(os.path.join(episode_path, 'inventory.npy')).shape[0]
        #     for episode_path in self.episode_paths
        # ]
        # self.total_len_rollouts = sum(self.len_rollouts)
        # print(f'{time.time() - a} sec consumed for loading... {self.total_len_rollouts}')

    @property
    def num_episodes(self):
        return len(self.episode_paths)

    # def process_rollout(
    #     self,
    #     index: int
    # ):
    #     obs = {
    #         obs_key: np.load(
    #             os.path.join(
    #                 self.episode_paths[index],
    #                 f"{obs_key}.npy"
    #             )
    #         )
    #         for obs_key in self.DATANAMES
    #     }
    #     print(f'index: {index}, shape: {obs["inventory"].shape[0]}')
    #     for step in range(obs['inventory'].shape[0]):
    #         worker = torch.utils.data.get_worker_info()
    #         worker_id = worker.id if worker is not None else -1
    #         out = {}
    #         for k, v in obs.items():
    #             if k.endswith('rgb'):
    #                 if np.issubdtype(v[step].dtype, np.uint8):
    #                     out[k] = v[step].astype(np.float32) / 255.0
    #                 if self.should_normalize_rgb:
    #                     out[k] -= self.mean_rgb
    #                     out[k] /= self.std_rgb
    #             elif k.endswith('depth'):
    #                 out[k] = v[step]
    #                 if self.should_normalize_depth:
    #                     out[k] -= self.mean_depth
    #                     out[k] /= self.std_depth
    #             else:
    #                 out[k] = v[step]
    #         print(f'step: {step}, index: {index}')
    #         yield {
    #             **out,
    #             'episode_id': index,
    #             'masks': 0.0 if step == 0 else 1.0,
    #         }, worker_id

    def __iter__(self):

        # return chain.from_iterable(map(self.process_rollout, range(self.num_episodes)))
        for index in range(self.num_episodes):
            obs = {
                obs_key: np.load(
                    os.path.join(
                        self.episode_paths[index],
                        f"{obs_key}.npy"
                    )
                )
                for obs_key in self.DATANAMES
            }
            # print(f'index: {index}, shape: {obs["inventory"].shape[0]}')
            for step in range(obs['inventory'].shape[0]):
                worker = torch.utils.data.get_worker_info()
                worker_id = worker.id if worker is not None else -1
                out = {}
                for k, v in obs.items():
                    if k.endswith('rgb'):
                        if np.issubdtype(v[step].dtype, np.uint8):
                            out[k] = v[step].astype(np.float32) / 255.0
                        if self.should_normalize_rgb:
                            out[k] -= self.mean_rgb
                            out[k] /= self.std_rgb
                    elif k.endswith('depth'):
                        out[k] = v[step]
                        if self.should_normalize_depth:
                            out[k] -= self.mean_depth
                            out[k] /= self.std_depth
                    else:
                        out[k] = v[step]
                # print(f'step: {step}, index: {index}')
                yield {
                    **out,
                    'episode_id': index + self.start_index,
                    'masks': 0.0 if step == 0 else 1.0,
                }, worker_id