from typing import Tuple, Sequence, Optional, Dict, Any, Type, List

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from rearrange.sensors import DepthRearrangeSensor, RGBRearrangeSensor, InWalkthroughPhaseSensor, ClosestUnshuffledRGBRearrangeSensor
from rearrange.tasks import RearrangeTaskSampler
from experiments.ta_base import TaskAwareBaseExperimentConfig
from task_aware_rearrange.sensors import PoseSensor, SemanticSegmentationSensor


class TwoPhaseTaskAwareRearrangeExperimentConfig(TaskAwareBaseExperimentConfig):
    
    TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH: int = 1
    IS_WALKTHROUGH_PHASE_EMBEDING_DIM: int = 32
    
    RGB_NORMALIZATION = True
    
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
        
        if cls.REQUIRE_SEMANTIC_SEGMENTATION:
            if cls.REFERENCE_SEGMENTATION:
                sensors.append(
                    SemanticSegmentationSensor(
                        ordered_object_types=cls.ORDERED_OBJECT_TYPES,
                        height=cls.SCREEN_SIZE,
                        width=cls.SCREEN_SIZE,
                        uuid=cls.SEMANTIC_SEGMENTATION_UUID,
                    )
                )
            elif cls.RGB_NORMALIZATION:
                # add raw rgb sensors to infer semantic segmentation masks
                sensors.append(
                    RGBRearrangeSensor(
                        height=cls.SCREEN_SIZE,
                        width=cls.SCREEN_SIZE,
                        use_resnet_normalization=False,
                        uuid=cls.EGOCENTRIC_RAW_RGB_UUID,
                    )
                )
                
        return sensors
    
    @classmethod
    def preprocessors(cls) -> Sequence[Preprocessor]:
        preprocessors = []
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            preprocessors.append(
                cls.create_resnet_bulder(
                    in_uuid=cls.EGOCENTRIC_RGB_UUID,
                    out_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                )
            )
        
        return preprocessors
    
    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        only_one_unshuffle_per_walkthrough: bool = False,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=True,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    )
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            unshuffle_runs_per_walkthrough=cls.TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH 
            if (not only_one_unshuffle_per_walkthrough) and stage == "train"
            else None,
            epochs=epochs,
            **kwargs,
        )