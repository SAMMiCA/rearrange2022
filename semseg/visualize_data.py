import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import cv2
import glob
import argparse

from detectron2.data import Metadata
import detectron2.utils.visualizer as visualizer
from semseg.semseg_constants import (
    CLASS_TO_COLOR,
    ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE,
    ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS,
    ORDERED_CLASS_TO_COLOR,
    SEMSEG_DATA_DIR,
    NUM_OBJECT_TYPES
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualization Tool")
    parser.add_argument(
        "--logdir",
        type=str,
        default=SEMSEG_DATA_DIR
    )
    args = parser.parse_args()

    os.makedirs(
        os.path.join(
            args.logdir, "visualizations"
        ),
        exist_ok=True,
    )

    metadata = Metadata()

    # metadata.thing_classes = [
    #     list(CLASS_TO_COLOR.keys())[
    #         ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS[i]
    #     ]
    #     for i in range(NUM_OBJECT_TYPES)
    # ]
    # metadata.stuff_classes = [
    #     list(CLASS_TO_COLOR.keys())[
    #         ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS[i]
    #     ]
    #     for i in range(NUM_OBJECT_TYPES)
    # ]

    # metadata.thing_colors = [
    #     list(CLASS_TO_COLOR.values())[
    #         ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS[i]
    #     ]
    #     for i in range(NUM_OBJECT_TYPES)
    # ]
    # metadata.stuff_colors = [
    #     list(CLASS_TO_COLOR.values())[
    #         ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS[i]
    #     ]
    #     for i in range(NUM_OBJECT_TYPES)
    # ]
    metadata.thing_classes = list(ORDERED_CLASS_TO_COLOR.keys())
    metadata.stuff_classes = list(ORDERED_CLASS_TO_COLOR.keys())

    metadata.thing_colors = list(ORDERED_CLASS_TO_COLOR.values())
    metadata.stuff_colors = list(ORDERED_CLASS_TO_COLOR.values())

    # metadata.thing_classes = list(CLASS_TO_COLOR.keys())
    # metadata.stuff_classes = list(CLASS_TO_COLOR.keys())

    # metadata.thing_colors = list(CLASS_TO_COLOR.values())
    # metadata.stuff_colors = list(CLASS_TO_COLOR.values())

    for annotation in glob.glob(
        os.path.join(
            args.logdir, "annotations/*.json"
        )
    ):
        with open(annotation, "r") as f:
            annotation_data = json.load(f)
        
        annotation_data["file_name"] = os.path.join(
            args.logdir, annotation_data["file_name"]
        )

        annotation_data["sem_seg_file_name"] = os.path.join(
            args.logdir, annotation_data["sem_seg_file_name"]
        )

        annotation_data["pan_seg_file_name"] = os.path.join(
            args.logdir, annotation_data["pan_seg_file_name"]
        )

        annotation_data.pop("sem_seg_file_name")
        annotation_data.pop("pan_seg_file_name")

        image_id = annotation_data["image_id"]

        if os.path.exists(annotation_data["file_name"]):
            image = cv2.imread(annotation_data["file_name"])

            annotated_image = visualizer.Visualizer(
                image[..., ::-1], metadata=metadata
            ).draw_dataset_dict(annotation_data).get_image()[..., ::-1]

            cv2.imwrite(
                os.path.join(
                    args.logdir, "visualizations", f"{image_id:07d}-vis.png"
                ),
                annotated_image
            )