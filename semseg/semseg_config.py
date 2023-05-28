from typing import Tuple, Sequence, Optional, Dict, Any, Type, List
import os

from allenact.base_abstractions.preprocessor import Preprocessor

from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.base_abstractions.sensor import Sensor, SensorSuite

from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, ClosestUnshuffledRGBRearrangeSensor
from experiments.two_phase.two_phase_ta_base import TwoPhaseTaskAwareRearrangeExperimentConfig
from task_aware_rearrange.sensors import PoseSensor
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from semseg.semseg_sensors import SemanticRearrangeSensor
from semseg.semseg_constants import (
    CLASS_TO_COLOR_ORIGINAL,
    ORDERED_CLASS_TO_COLOR,
    ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE
)
from semseg.semseg_preprocessors import SemanticPreprocessor


class SemSegConfig(TwoPhaseTaskAwareRearrangeExperimentConfig):
    
    REQUIRE_SEMANTIC_SEGMENTATION = True
    REFERENCE_SEGMENTATION = True
    SEMANTIC_SEGMENTATION_UUID = "semantic"
    SEMANTIC_SEGMENTATION_BASE = "mask-rcnn"
    SEMANTIC_SEGMENTATION_USE_MASS = False
    SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
        "semseg",
        "checkpoints",
        "model_final.pth"
    )
    
    RGB_NORMALIZATION = False

    if SEMANTIC_SEGMENTATION_USE_MASS:
        CLASS_TO_COLOR = CLASS_TO_COLOR_ORIGINAL
        ORDERED_OBJECT_TYPES = list(CLASS_TO_COLOR.keys())
        CLASS_MAPPING = ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE
    else:
        CLASS_TO_COLOR = ORDERED_CLASS_TO_COLOR
        CLASS_MAPPING = list(range(len(CLASS_TO_COLOR)))
    DETECTION_THRESHOLD = 0.8

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        mean, stdev = None, None
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
            if pretraining_type.strip().lower() == "clip":
                from allenact_plugins.clip_plugin.clip_preprocessors import (
                    ClipResNetPreprocessor,
                )

                mean = ClipResNetPreprocessor.CLIP_RGB_MEANS
                stdev = ClipResNetPreprocessor.CLIP_RGB_STDS
            else:
                mean = IMAGENET_RGB_MEANS
                stdev = IMAGENET_RGB_STDS
                
        sensors = [
            RGBRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=cls.RGB_NORMALIZATION,
                uuid=cls.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            DepthRearrangeSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                uuid=cls.DEPTH_UUID,
                use_normalization=cls.DEPTH_NORMALIZATION,
            ),
            PoseSensor(
                reference_pose=cls.REFERENCE_POSE,
                uuid=cls.POSE_UUID,
            ),
            InWalkthroughPhaseSensor(),
        ]
        if cls.REFERENCE_SEGMENTATION:
            sensors.append(
                SemanticRearrangeSensor(
                    ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                    class_to_color=cls.CLASS_TO_COLOR,
                    height=cls.SCREEN_SIZE,
                    width=cls.SCREEN_SIZE,
                    uuid=cls.SEMANTIC_SEGMENTATION_UUID,
                )
            )
        if not cls.REFERENCE_SEGMENTATION:
            if cls.RGB_NORMALIZATION:
                sensors.append(
                    RGBRearrangeSensor(
                        height=cls.SCREEN_SIZE,
                        width=cls.SCREEN_SIZE,
                        use_resnet_normalization=False,
                        uuid=cls.EGOCENTRIC_RAW_RGB_UUID,
                        mean=mean,
                        stdev=stdev,
                    )
                )
        
        return sensors
    
    
    @classmethod
    def tag(cls) -> str:
        return "semseg"
    
    @classmethod
    def create_maskrcnn_semseg_preprocessor_builder(
        cls,
        in_uuids: Sequence[str],
        out_uuid: str,
    ):
        return SemanticPreprocessor(
            input_uuids=in_uuids,
            output_uuid=out_uuid,
            ordered_object_types=cls.ORDERED_OBJECT_TYPES,
            class_to_color=cls.CLASS_TO_COLOR,
            class_mapping=cls.CLASS_MAPPING,
            detection_threshold=cls.DETECTION_THRESHOLD,
            model_weight_path=cls.SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH,
            device=cls.DEVICE,
        )
    
    @classmethod
    def preprocessors(cls) -> Sequence[Preprocessor]:
        preprocessors = super().preprocessors()
        # preprocessors = []
        if not cls.REFERENCE_SEGMENTATION:
            if cls.RGB_NORMALIZATION:
                in_uuids = [cls.EGOCENTRIC_RAW_RGB_UUID]
            else:
                in_uuids = [cls.EGOCENTRIC_RGB_UUID]
            out_uuid = cls.SEMANTIC_SEGMENTATION_UUID

            if cls.SEMANTIC_SEGMENTATION_BASE == "mask-rcnn":
                preprocessors.append(
                    cls.create_maskrcnn_semseg_preprocessor_builder(
                        in_uuids=in_uuids,
                        out_uuid=out_uuid,
                    )
                )
            elif cls.SEMANTIC_SEGMENTATION_BASE == "solq":
                pass
        
        return preprocessors