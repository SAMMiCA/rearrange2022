from collections import OrderedDict
from typing import List, Callable, Optional, Any, Tuple, cast, Dict, Sequence

import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils import spaces_utils as su

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from task_aware_rearrange.constants import UNKNOWN_OBJECT_STR
from semseg.semseg_constants import (
    CLASS_TO_COLOR,
    ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE,
    ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS,
)


# Based on the MaSS Segmentation Sensor (Mask R-CNN)
class SemanticPreprocessor(Preprocessor):

    def __init__(
        self,
        input_uuids: List[str], 
        output_uuid: str, 
        ordered_object_types: Sequence[str],
        class_to_color: Dict[str, Tuple[int, ...]] = CLASS_TO_COLOR,
        class_mapping: List[int] = ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE,
        height: int = 224,
        width: int = 224,
        detection_threshold: float = 0.8,
        model_weight_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any
    ) -> None:
        self.input_uuids = input_uuids
        self.output_uuid = output_uuid

        self.ordered_object_types = ordered_object_types
        self.object_type_to_idx = {ot: i for i, ot in enumerate(self.ordered_object_types)}
        self.num_classes = len(self.ordered_object_types)
        # self.object_type_to_idx[UNKNOWN_OBJECT_STR] = len(self.ordered_object_types)
        # self.num_classes = len(self.ordered_object_types) + 1
        self.class_to_color = class_to_color
        assert len(self.class_to_color) == self.num_classes, (
            f"length of color map {len(self.class_to_color)} is not matched "
            f"to the number of object types {self.num_classes}"
        )
        self.class_mapping = class_mapping
        self.height = height
        self.width = width
        self.detection_threshold = detection_threshold
        self.model_weight_path = model_weight_path
        self.cfg = None
        
        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        
        self._model: Optional[nn.Module] = None
        observation_space = gym.spaces.MultiDiscrete(
            np.full(
                (
                    cast(int, self.height), 
                    cast(int, self.width), 
                    1
                ), 
                self.num_classes
            )
        )
        
        super().__init__(**prepare_locals_for_super(locals()))
        
    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model, self.cfg = self.load_maskrcnn_model(
                self.num_classes,
                self.model_weight_path,
            )
            self._model.to(self.device)
        return self._model
        
    @staticmethod
    def load_maskrcnn_model(num_classes: int, model_weight_path: Optional[str]) -> nn.Module:
        cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        cfg.MODEL.WEIGHTS = model_weight_path
        
        model = build_model(cfg)
        model.eval()
        
        if model_weight_path is not None:
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
        
        return model, cfg
    
    def to(self, device: torch.device) -> "SemanticPreprocessor":
        self._model = self.model.to(device)
        self.device = device
        return self
    
    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """
        Input observation is RAW RGB image divided by 255
        It requires RGB input format.
        """
        batch_size = obs[self.input_uuids[0]].shape[0]
        model_inputs = [
            {
                # converts RGB input to BGR input format
                "image": 255 * obs[self.input_uuids[0]][i][..., [2, 1, 0]].permute(2, 0, 1),
            }
            for i in range(batch_size)
        ]
        outputs = self.model(model_inputs)

        semantic_seg = torch.zeros(
            batch_size,
            self.height,
            self.width,
            # self.num_classes,
            len(self.class_mapping),
            device=self.device,
            dtype=torch.float32,
        )
        
        for i in range(batch_size):
            output = outputs[i]
            for j in range(len(output['instances'])):
                object_score = output['instances'].scores[j]

                if object_score < self.detection_threshold:
                    continue

                object_class = output['instances'].pred_classes[j]

                semantic_seg[i, :, :, self.class_mapping[object_class]] += (
                    output['instances'].pred_masks[j].to(torch.float32)
                )

        semantic_seg = semantic_seg.argmax(dim=-1, keepdim=True)
        
        return semantic_seg