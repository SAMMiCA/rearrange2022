import json
from typing import Any, Dict, List
from tqdm import tqdm
import os
import numpy as np
import argparse
from pycocotools.coco import COCO
from custom.constants import ORDERED_OBJECT_TYPES

from data_collection.coco_utils import binary_mask_to_polygon


def get_args():
    parser = argparse.ArgumentParser(
        description="make_coco_annotations", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base_dir",
        type=str,
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1e4,
    )

    args = parser.parse_args()

    return args

def convert_dirname(dirname: str):
    return "__".join(
        [
            f"{dirname.split('__')[0][:9]}{int(dirname.split('__')[0][9:]):03d}",
            dirname.split('__')[1],
            f"{int(dirname.split('__')[-1]):02d}"
        ]
    )

def revert_dirname(c_dirname: str):
    return "__".join(
        [
            f"{c_dirname.split('__')[0][:9]}{int(c_dirname.split('__')[0][9:])}",
            c_dirname.split('__')[1],
            f"{int(c_dirname.split('__')[-1])}"
        ]
    )

def sort_listdir(listdir: List[str], args):
    listdir = [dir for dir in listdir if os.path.isdir(os.path.join(args.base_dir, dir))]
    converted_listdir = [
        convert_dirname(dirname)
        for dirname in listdir
    ]
    sorted_c_listdir = sorted(converted_listdir)
    return [
        revert_dirname(c_dirname)
        for c_dirname in sorted_c_listdir
    ]

def save_coco(images: List[Dict[str, Any]], annotations: List[Dict[str, Any]], args):
    data = dict(
        images=images,
        annotations=annotations,
        categories=[
            {
                'id': it + 1,
                'name': name,
            }
            for it, name in enumerate(ORDERED_OBJECT_TYPES)
        ],
    )
    with open(os.path.join(args.base_dir, 'annotations.json'), 'w') as f:
        json.dump(data, f, indent=4)

def main():
    args = get_args()

    if not os.path.exists(args.base_dir):
        raise ValueError(f"Wrong directory is assigned.")

    image_id = 0
    coco_id = 0
    images = []
    annotations = []

    listdir = sort_listdir(os.listdir(args.base_dir), args)
    for dir in tqdm(listdir, desc="make COCO annotations"):
        for fname in sorted(os.listdir(os.path.join(args.base_dir, dir, 'npz_data'))):
            fpath = os.path.join(args.base_dir, dir, 'npz_data', fname)
            data = np.load(fpath)
            try:
                data = {
                    key: data[key]
                    for key in data.files
                }
            except:
                import pdb; pdb.set_trace()
                continue

            inst_detected = data["instseg_inst_detected"]
            add_image = False

            for nonzero_obj_id in inst_detected.nonzero()[0]:
                object_id = nonzero_obj_id.item()
                for i in range(inst_detected[object_id]):
                    mask = data["instseg_inst_masks"][object_id] & (2 ** i)
                    pos = np.where(mask)
                    xmin = np.min(pos[1]).item()
                    xmax = np.max(pos[1]).item()
                    ymin = np.min(pos[0]).item()
                    ymax = np.max(pos[0]).item()
                    width = xmax - xmin
                    height = ymax - ymin
                    if width < 15 and height < 15:
                        continue

                    poly = binary_mask_to_polygon(mask)
                    bbox = [xmin, ymin, width, height]
                    area = width * height

                    data_anno = dict(
                        image_id=image_id,
                        id=coco_id,
                        category_id=object_id+1,
                        bbox=bbox,
                        area=area,
                        segmentation=poly,
                        iscrowd=0,
                    )
                    annotations.append(data_anno)
                    coco_id += 1
                    add_image = True

                    if (coco_id + 1) % args.save_interval == 0:
                        save_coco(images, annotations, args)
            
            if add_image:
                images.append(
                    dict(
                        id=image_id,
                        file_name=os.path.relpath(
                            os.path.join(args.base_dir, dir, 'rgb', f'{fname.split(".")[0]}.png'),
                            args.base_dir
                        ),
                        height=224,
                        width=224,
                    )
                )
            image_id += 1

    save_coco(images, annotations, args)


if __name__ == "__main__":
    main()