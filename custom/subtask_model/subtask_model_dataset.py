from typing import Union, List, Dict
import os
import random
import json
from PIL import Image
import torch
import numpy as np
from itertools import cycle, islice, chain
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms, utils


class SubtaskModelDataset(Dataset):
    """
    Dataset for train subtask model.
    Dataset consists of ...
    root_dir
        |-- mode("train")
                |-- scene("FloorPlanXXX__train__YY")
                |     |-- rgb
                |          |-- 000.ttt
                |          |-- 001.ttt
                |          |-- ...
                |     |-- unshuffled_rgb
                |     |-- action_history
                |     |-- expert_action_history
                |     |-- expert_subtask_history
                |     |-- npz_data 

    """

    def __init__(self, root_dir: str, mode: str, shuffle: bool = False) -> None:
        
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.mode = mode
        self.mode_dir = os.path.abspath(os.path.join(self.root_dir, self.mode))
        self.shuffle = shuffle
        if self.shuffle:
            self.episode_list = random.sample(os.listdir(self.mode_dir), len(os.listdir(self.mode_dir)))
        else:
            self.episode_list = os.listdir(self.mode_dir)
        self.episode_dirs = [
            os.path.join(self.mode_dir, episode)
            for episode in self.episode_list
        ]
        self.num_episodes = len(self.episode_list)
        self.num_rollouts = [
            len(os.listdir(os.path.join(episode_dir, "rgb")))
            for episode_dir in self.episode_dirs
        ]
        self.total_num_rollouts = sum(self.num_rollouts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _sum = 0
        epi_idx = -1
        for i in range(self.num_episodes):
            if idx >= _sum:
                data_num = idx - _sum
                _sum += self.num_rollouts[i]
                if idx < _sum:
                   epi_idx = i
                   break
        
        episode = self.episode_dirs[epi_idx]
        datanames = os.listdir(episode)
        return {
            **{
                k: v
                for dataname in datanames
                for k, v in self.parse_data(
                    episode_path=episode,
                    dataname=dataname,
                    datanum=data_num,
                ).items()
            },
            'episode_id': epi_idx,
            'masks': 0.0 if data_num == 0 else 1.0,
        }

    def __len__(self):
        return self.total_num_rollouts
        
    def parse_data(self, episode_path, dataname, datanum) -> Dict[str, np.ndarray]:
        if "history" in dataname:
            """
            json file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}')
            f = open(datapath)
            data_ = json.load(f)
            data_ = {
                dataname: np.array(data_[-1])
            }
            f.close()
        elif "rgb" in dataname:
            """
            png file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}.png')
            data_ = Image.open(datapath)
            data_ = {
                dataname: np.asarray(data_)
            }
        else:
            """
            npz file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}.npz')
            data_ = np.load(datapath)
            data_ = {
                key: data_[key]
                for key in data_.files
            }

        return data_


class SubtaskModelIterableDataset(IterableDataset):
    """
    Dataset for train subtask model.
    Dataset consists of ...
    root_dir
        |-- mode("train")
                |-- scene("FloorPlanXXX__train__YY")
                |     |-- rgb
                |          |-- 000.ttt
                |          |-- 001.ttt
                |          |-- ...
                |     |-- unshuffled_rgb
                |     |-- action_history
                |     |-- expert_action_history
                |     |-- expert_subtask_history
                |     |-- npz_data 

    """

    def __init__(self, root_dir: str, mode: str, shuffle: bool = False) -> None:
        
        self.root_dir = root_dir
        self.mode = mode
        self.mode_dir = os.path.join(root_dir, mode)
        self.shuffle = shuffle
        if self.shuffle:
            self.episode_list = random.sample(os.listdir(self.mode_dir), len(os.listdir(self.mode_dir)))
        else:
            self.episode_list = os.listdir(self.mode_dir)
        self.episode_dirs = [
            os.path.join(self.mode_dir, episode)
            for episode in self.episode_list
        ]
        self.num_episodes = len(self.episode_list)
        self.num_rollouts = [
            len(os.listdir(os.path.join(episode_dir, "rgb")))
            for episode_dir in self.episode_dirs
        ]
        self.total_num_rollouts = sum(self.num_rollouts)

    def process_rollout(self, episode: List):
        datanames = os.listdir(episode)
        episode_idx = self.episode_dirs.index(episode)
        for step in range(self.num_rollouts[episode_idx]):
            yield {
                **{
                    k: v
                    for dataname in datanames
                    for k, v in self.parse_data(
                        episode_path=episode,
                        dataname=dataname,
                        datanum=step,
                    ).items()
                },
                'episode_id': episode_idx,
                'masks': 0.0 if step == 0 else 1.0,
            }

    def get_stream(self, episode_list):
        return chain.from_iterable(map(self.process_rollout, episode_list))

    def __len__(self):
        return self.total_num_rollouts

    def __iter__(self):
        return self.get_stream(self.episode_dirs)

    # def parse_episode(self, episode_path):
    #     datanames = os.listdir(episode_path)
    #     return [
    #         {
    #             k: v
    #             for dataname in datanames
    #             for k, v in self.parse_data(
    #                 episode_path=episode_path,
    #                 dataname=dataname,
    #                 datanum=datanum
    #             ).items()
    #         }
    #         for datanum in range(self.num_rollouts[self.episode_dirs.index(episode_path)])
    #     ]
        
    def parse_data(self, episode_path, dataname, datanum) -> Dict[str, np.ndarray]:
        if "history" in dataname:
            """
            json file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}')
            f = open(datapath)
            data_ = json.load(f)
            data_ = {
                dataname: np.array(data_[-1])
            }
            f.close()
        elif "rgb" in dataname:
            """
            png file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}.png')
            data_ = Image.open(datapath)
            data_ = {
                dataname: np.asarray(data_)
            }
        else:
            """
            npz file
            """
            datapath = os.path.join(episode_path, dataname, f'{datanum:03d}.npz')
            data_ = np.load(datapath)
            data_ = {
                key: data_[key]
                for key in data_.files
            }

        return data_

