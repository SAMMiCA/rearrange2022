from typing import List, Callable, Optional, Any, cast, Dict, Sequence

import os
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.preprocessors.resnet import ResNetEmbedder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils import spaces_utils as su
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from task_aware_rearrange.subtasks import NUM_SUBTASKS


class SubtaskActionExpertPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.input_uuids = input_uuids
        self.output_uuid = output_uuid
        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:

        return gym.spaces.Dict(
            [
                ("action_or_policy", gym.spaces.Discrete(len(RearrangeBaseExperimentConfig.actions()))),
                ("expert_success", gym.spaces.Discrete(2)),
            ]
        )

    def to(self, device: torch.device) -> "SubtaskActionExpertPreprocessor":
        self.device = device
        return self

    def flatten_output(self, unflattened):
        return su.flatten(
            self.observation_space,
            su.torch_point(self.observation_space, unflattened),
        )

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        expert_subtask_and_action = obs[self.input_uuids[0]].to(self.device)    # B x 4
        
        return expert_subtask_and_action[..., -2:]


class SubtaskExpertPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
    ):
        self.input_uuids = input_uuids
        self.output_uuid = output_uuid
        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Dict:

        return gym.spaces.Dict(
            [
                ("action_or_policy", gym.spaces.Discrete(NUM_SUBTASKS)),
                ("expert_success", gym.spaces.Discrete(2)),
            ]
        )

    def to(self, device: torch.device) -> "SubtaskExpertPreprocessor":
        self.device = device
        return self

    def flatten_output(self, unflattened):
        return su.flatten(
            self.observation_space,
            su.torch_point(self.observation_space, unflattened),
        )

    def process(
        self,
        obs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        expert_subtask_and_action = obs[self.input_uuids[0]].to(self.device)    # B x 4
        
        return expert_subtask_and_action[..., :2]
