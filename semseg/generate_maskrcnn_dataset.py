# https://github.com/brandontrabucco/mass/blob/main/generate_maskrcnn_dataset.py
import numpy as np
import torch
import cv2

import argparse
import os
import sys
import json
import shutil

import detectron2.structures as structures


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.experiment_utils import (
    TimeoutDueToUnityCrash,
    NumpyJSONEncoder,
    handle_read_only,
    run_experiment_with_restart,
)
from rearrange.tasks import RearrangeTaskSampler
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from semseg.semseg_config import SemSegConfig as Config


def create_dataset(args):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """
    
    padding = 5
    dilate_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * padding + 1, 2 * padding + 1),
        (padding, padding)
    )
    
    padding = 3
    erode_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * padding + 1, 2 * padding + 1),
        (padding, padding)
    )
    
    # Create arguments for the trainig and testing tasks
    task_sampler_params = Config.stagewise_task_sampler_args(
        stage=args.stage,
        process_ind=0,
        total_processes=1,
        devices=[0]
    )
        
    task_sampler_params['thor_controller_kwargs'].update(
        dict(
            renderSemanticSegmentation=True,
            renderInstanceSegmentation=True,
        )
    )
    # Generate a task sampler for training and testing evaluation
    with TimeoutDueToUnityCrash(300):
        task_sampler: RearrangeTaskSampler = Config.make_sampler_fn(
            **task_sampler_params,
            force_cache_reset=True,
            epochs=1,
            only_one_unshuffle_per_walkthrough=True,
        )
    
    # Skip the tasks until it meets the start_task
    for task_id in range(args.start_task):
        with TimeoutDueToUnityCrash():
            next(task_sampler.task_spec_iterator)
            
    # Perform the evaluation using every task in the task sampler
    for task_id in range(
        args.start_task,
        args.start_task + args.total_tasks * args.every_tasks,
        args.every_tasks
    ):
        def callback(obs: dict, image_index):
            """Callback function executed at every step during an episode
            that processes the current observation and serializes data
            for training a mask rcnn model at a later stage.

            Arguments:

            obs: dict
                the current observation from the environment, which contains
                ground truth semantic segmentation at the 'semantic' key.

            """
            image_id = args.start_task * args.images_per_task + image_index
            
            # Construct the disk path to the rgb and semantic images
            fname = f"images/{image_id:07d}-rgb.png"
            semseg_fname = f"images/{image_id:07d}-sem.png"
            panseg_fname = f"images/{image_id:07d}-pan.png"
            
            # Store semantic segmentation annotations in a dictionary format
            training_example = dict(
                image_id=image_id,
                file_name=fname,
                sem_seg_file_name=semseg_fname,
                pan_seg_file_name=panseg_fname,
                height=obs["semantic"].shape[0],
                width=obs["semantic"].shape[1],
                segments_info=[],
                annotations=[],
            )
            
            # Create a buffer to store instance ids for every object
            # differs from the ground truth instances
            total_instances = 0     
            instances = np.zeros(obs["semantic"].shape[:2], dtype=np.int32)
            training_example["segments_info"].append(
                dict(
                    category_id=0,
                    id=0,
                    isthing=False,
                )
            )
            
            # Generate a set of instance ids from the environment
            instance_segmentation = (
                task.env.last_event.instance_segmentation_frame
            )
            instance_segmentation = (
                instance_segmentation[..., 0:1] + 
                instance_segmentation[..., 1:2] * 256 + 
                instance_segmentation[..., 2:3] * 256 * 256
            )
            
            # Iterate through every detected class except the background
            for category_id in (
                set(np.unique(obs["semantic"]).tolist()) - {0} # 0 for Unknown
            ):
                # Generate a mask that shows where category_id is present
                semantic_mask = (category_id == obs["semantic"]).astype(np.uint8)
                
                # Iterate through each object instance in the frame
                for instance_id in (
                    np.unique(instance_segmentation[np.nonzero(semantic_mask)]).tolist()
                ):
                    # Generate a mask that shows where instance_id is present
                    instance_mask = (instance_id == instance_segmentation).astype(np.uint8)
                    instance_mask = cv2.erode(
                        cv2.dilate(
                            instance_mask[..., 0], 
                            dilate_element,
                        ),
                        erode_element,
                    )
                    
                    # For this object instance, generate a set of polygons
                    pts = cv2.findContours(
                        instance_mask,
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE
                    )[-2]
                    
                    pts = [x for x in pts if x.shape[0] > 2]
                    if len(pts) == 0:   # Skip the instance if it is too small
                        continue
                    
                    # Record the current object as a detected instance for panoptic segmentation
                    total_instances += 1
                    cv2.fillPoly(instances, pts=pts, color=total_instances)
                    training_example["segments_info"].append(
                        dict(
                            category_id=category_id,
                            id=total_instances,
                            isthing=True,
                        )
                    )
                    
                    # Record the location of the detected object instance
                    # in the frame with a bounding box, mask, and category id
                    training_example["annotations"].append(
                        dict(
                            category_id=category_id,
                            bbox=cv2.boundingRect(np.concatenate(pts, axis=0)),
                            bbox_mode=structures.BoxMode.XYWH_ABS,
                            segmentation=[
                                x.flatten().tolist() for x in pts
                            ],
                        )
                    )
            
            # Construct a RGB image for panoptic segmentation
            instance_rgb = np.zeros(
                [instances.shape[0], instances.shape[1], 3],
                dtype=np.uint8
            )
            for idx in range(3):
                instance_rgb[..., idx] =  (instances // (256 ** idx)) % 256
                # instance_rgb[..., idx] =  (
                #     instance_segmentation.reshape(*instances.shape) * instances // (256 ** idx)
                # ) % 256
                
            if len(training_example["annotations"]) == 0:
                return 0    # images without objects are not used
            
            # Write the observed image and its generated semantic segmentation labels
            # to the disk for training mask r-cnn with detectron
            
            cv2.imwrite(os.path.join(args.logdir, fname), 255 * obs["rgb"][..., ::-1])
            cv2.imwrite(os.path.join(args.logdir, semseg_fname), obs["semantic"][..., 0])
            cv2.imwrite(os.path.join(args.logdir, panseg_fname), instance_rgb[..., ::-1])
            
            # Open the target file for this image and write
            # the annotations to a json file
            with open(
                os.path.join(args.logdir, f"annotations/{image_id:07d}.json"), "w"
            ) as f:
                json.dump(training_example, f, indent=4, cls=NumpyJSONEncoder)
                
            return 1
        
        with TimeoutDueToUnityCrash():
            task = task_sampler.next_task()

        valid_positions = task.env.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        
        valid_positions = [
            dict(
                position=position,
                rotation=dict(x=0, y=rotation, z=0),
                horizon=horizon,
                standing=standing,
            )
            for position in valid_positions
            for rotation in (0, 90, 180, 270)
            for horizon in (-30, 0, 30, 60)
            for standing in (True, False)
        ]
        
        valid_positions = [
            valid_positions[idx] 
            for idx in np.random.permutation(len(valid_positions))
        ]
        
        num_images = 0
        
        while num_images < args.images_per_task // 2:
            task.env.controller.step(
                action="TeleportFull",
                **valid_positions.pop(0)
            )
            num_images += callback(task.get_observations(), num_images)
            
        with TimeoutDueToUnityCrash():
            task = task_sampler.next_task()

        valid_positions = task.env.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        
        valid_positions = [
            dict(
                position=position,
                rotation=dict(x=0, y=rotation, z=0),
                horizon=horizon,
                standing=standing,
            )
            for position in valid_positions
            for rotation in (0, 90, 180, 270)
            for horizon in (-30, 0, 30, 60)
            for standing in (True, False)
        ]
        
        valid_positions = [
            valid_positions[idx] 
            for idx in np.random.permutation(len(valid_positions))
        ]
        
        while num_images < args.images_per_task:
            task.env.controller.step(
                action="TeleportFull",
                **valid_positions.pop(0)
            )
            num_images += callback(task.get_observations(), num_images)
        
        for _ in range(args.every_tasks - 1):
            with TimeoutDueToUnityCrash():
                next(task_sampler.task_spec_iterator)
            
        args.total_tasks -= 1
        args.start_task += args.every_tasks


def run_experiments(args):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """
    
    name = (
        f"{args.start_task}-"
        f"{args.start_task + args.total_tasks * args.every_tasks}"
    )
    
    os.makedirs(os.path.join(args.logdir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "images"), exist_ok=True)
    
    os.makedirs(os.path.join(args.logdir, f"tmp-{name}"), exist_ok=True)
    os.environ["HOME"] = os.path.join(args.logdir, f"tmp-{name}")
    
    run_experiment_with_restart(create_dataset, args)
    
    shutil.rmtree(os.environ["HOME"], onerror=handle_read_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Semantic Segmentation Data Collection Agent")
    
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join(
            ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
            "semseg_data"
        )
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--start-task",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--every-tasks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--images-per-task",
        type=int,
        default=500,
    )
    
    run_experiments(parser.parse_args())