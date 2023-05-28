import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import cv2
import numpy as np
import torch
from allenact.utils.tensor_utils import batch_observations
# from semseg.semseg_config import SemSegConfig
from experiments.one_phase.ablation_001 import OnePhaseAblation001ExerimentConfig as Config
from semseg.semseg_sensors import SemanticRearrangeSensor
from semseg.semseg_preprocessors import SemanticPreprocessor
from rearrange.tasks import UnshuffleTask, WalkthroughTask
from task_aware_rearrange.constants import ORDERED_OBJECT_TYPES
from semseg.semseg_constants import (
    ORDERED_CLASS_TO_COLOR,
    CLASS_TO_COLOR_ORIGINAL,
    SEMSEG_DATA_DIR,
    ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE,
)
from detectron2.data import Metadata
import detectron2.utils.visualizer as visualizer


def test_sensor_preprocessor(stage: str = "train", verbose: bool = False):
    # Test for sensors & preprocessors
    task_sampler_params = Config.stagewise_task_sampler_args(
        stage=stage, process_ind=0, total_processes=1, devices=[0],
    )
    task_sampler_params["thor_controller_kwargs"].update(
        dict(
            renderSemanticSegmentation=True,
            renderInstanceSegmentation=True,
        )
    )
    task_sampler_params["sensors"].append(
        SemanticRearrangeSensor(
            ordered_object_types=Config.ORDERED_OBJECT_TYPES,
            class_to_color=Config.CLASS_TO_COLOR,
            height=Config.SCREEN_SIZE,
            width=Config.SCREEN_SIZE,
            uuid="gt_semantic",
        )
    )
    task_sampler = Config.make_sampler_fn(
        **task_sampler_params,
        force_cache_reset=True,
        epochs=1,
        # only_one_unshuffle_per_walkthrough=True,
    )

    sensor_preprocessor_graph = Config.create_preprocessor_graph(
        mode=stage, additional_output_uuids="gt_semantic"
    ).to(torch.device(0))
    print(sensor_preprocessor_graph.compute_order)

    task = task_sampler.next_task()
    if Config.__name__.startswith("TwoPhase"):
        assert isinstance(task, WalkthroughTask)
    else:
        assert isinstance(task, UnshuffleTask)

    obs = task.get_observations()

    if verbose:
        print('[ observations ]')
        for k, v in obs.items():
            if not isinstance(v, dict):
                print(
                    f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
                    + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                )
            else:
                print(f'KEY [{k}] | TYPE [{type(v)}]')
                for k1, v1 in v.items():
                    print(
                        f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
                        + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                    )
    import pdb; pdb.set_trace()
    batch = batch_observations([obs], device=torch.device(0))
    if verbose:
        print('[ batch ]')
        for k, v in batch.items():
            if not isinstance(v, dict):
                print(
                    f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
                    + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                )
            else:
                print(f'KEY [{k}] | TYPE [{type(v)}]')
                for k1, v1 in v.items():
                    print(
                        f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
                        + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                    )
    import pdb; pdb.set_trace()
    preprocessed_obs = sensor_preprocessor_graph.get_observations(batch)
    if verbose:
        print("[ preprocessed obs ]")
        for k, v in preprocessed_obs.items():
            if not isinstance(v, dict):
                print(
                    f'KEY [{k}] | TYPE [{type(v)}] | SHAPE [{v.shape if hasattr(v, "shape") else None}]'
                    + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                )
            else:
                print(f'KEY [{k}] | TYPE [{type(v)}]')
                for k1, v1 in v.items():
                    print(
                        f'    KEY [{k1}] | TYPE [{type(v1)}] | SHAPE [{v1.shape if hasattr(v1, "shape") else None}]'
                        + (f' | DEVICE [{v.device}]' if hasattr(v, "device") else '')
                    )
    
    import pdb; pdb.set_trace()


def test_visualization(
    file_path: str,
    stage: str = "train",
    base: str = "mass",
    verbose: str = True,
):
    if base == "mass":
        obj_order = list(CLASS_TO_COLOR_ORIGINAL.keys())
        class2color = CLASS_TO_COLOR_ORIGINAL
    elif base == "tidee":
        pass
    else:
        obj_order = ORDERED_OBJECT_TYPES
        class2color = ORDERED_CLASS_TO_COLOR

    metadata = Metadata()

    metadata.thing_classes = list(ORDERED_CLASS_TO_COLOR.keys())
    metadata.stuff_classes = list(ORDERED_CLASS_TO_COLOR.keys())

    metadata.thing_colors = list(ORDERED_CLASS_TO_COLOR.values())
    metadata.stuff_colors = list(ORDERED_CLASS_TO_COLOR.values())

    with open(file_path, "r") as f:
        anno_data = json.load(f)

    anno_data["file_name"] = os.path.join(
        SEMSEG_DATA_DIR, stage, anno_data["file_name"]
    )
    anno_data["sem_seg_file_name"] = os.path.join(
        SEMSEG_DATA_DIR, stage, anno_data["sem_seg_file_name"]
    )
    anno_data["pan_seg_file_name"] = os.path.join(
        SEMSEG_DATA_DIR, stage, anno_data["pan_seg_file_name"]
    )

    sem_seg_file_path = anno_data.pop("sem_seg_file_name")
    anno_data.pop("pan_seg_file_name")

    img_id = anno_data["image_id"]

    if os.path.exists(anno_data["file_name"]):
        image = cv2.imread(anno_data["file_name"])
        semantic_image = cv2.imread(sem_seg_file_path)[..., 0:1]

        anno_image = visualizer.Visualizer(
            image[..., ::-1], metadata=metadata
        ).draw_dataset_dict(anno_data).get_image()[..., ::-1]

        cv2.imwrite(f"{img_id:07d}-vis.png", anno_image)

    if base == "mass":
        semantic_preprocessor = SemanticPreprocessor(
            input_uuids=["rgb_raw"],
            output_uuid="semantic",
            ordered_object_types=obj_order,
            class_to_color=class2color,
            class_mapping=ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE,
            detection_threshold=0.8,
            model_weight_path=Config.SEMANTIC_SEGMENTATION_MODEL_WEIGHT_PATH,
            device=torch.device(0)
        )

    obs = {
        "rgb_raw": (
            torch.tensor(image[..., ::-1].copy()).type(torch.float32) / 255
        ).view(1, *image.shape)
    }
    processed_semantic_image = semantic_preprocessor.process(obs).squeeze(0)
    
    gt_unique_ids = np.unique(semantic_image).tolist()
    pred_unique_ids = torch.unique(processed_semantic_image).cpu().tolist()
    if verbose:
        print(f"From GT Semantic Segmentation")
        print(
            f"\tUnique class ids in the current image: "
            f"{gt_unique_ids}"
        )
        print(
            f"\tUnique class names in the current image: "
            f"{[ORDERED_OBJECT_TYPES[id] for id in gt_unique_ids]}\n"
        )

        print(f"From Predicted Semantic Segmentation")
        print(
            f"\tUnique PRED class ids in the current image: "
            f"{pred_unique_ids}"
        )
        print(
            f"\tUnique PRED class names in the current image: "
            f"{[ORDERED_OBJECT_TYPES[id] for id in pred_unique_ids]}"
        )

    pred_image = visualizer.Visualizer(
        image[..., ::-1], metadata=metadata
    ).draw_sem_seg(
        sem_seg=processed_semantic_image.squeeze(-1).cpu()
    ).get_image()[..., ::-1]

    cv2.imwrite(f"{img_id:07d}-pred-vis.png", pred_image)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    stage = "train"

    # test_visualization(
    #     file_path=os.path.join(SEMSEG_DATA_DIR, stage, "annotations", "0000000.json"),
    #     stage=stage
    # )
    test_sensor_preprocessor(verbose=True)