import os
from typing import List
import torch
import numpy as np
from itertools import cycle, islice, chain
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms, utils
import random


class TestIterableDataset(IterableDataset):

    def __init__(self, data_list, shuffle: bool = False):
        self.shuffle = shuffle
        if shuffle:
            self.data_list = random.sample(data_list, len(data_list))
        else:
            self.data_list = data_list

    def process_data(self, data: List):
        for x in data:
            yield x

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, data_list))

    def __iter__(self):
        return self.get_stream(self.data_list)

