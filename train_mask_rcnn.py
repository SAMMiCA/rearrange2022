import datetime
import math
import os
import argparse
import sys
import time

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from custom.constants import NUM_OBJECT_TYPES, ORDERED_OBJECT_TYPES
from custom.segmentation_model.models import get_model_instance_segmentation
from custom.segmentation_model import utils, coco_utils
from custom.segmentation_model.engine import train_one_epoch, evaluate
from custom.segmentation_model.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups


# https://github.com/pytorch/vision/blob/369317f45354248582884f6e2f500b7cebea2236/references/detection/train.py#L24
def get_args():
    parser = argparse.ArgumentParser(
        description="Distributed Training Script for SubtaskPredictionModel", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="~/dataset/rearrange2022/segmentation",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=26,
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./custom/segmentation_model/models"
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="env://",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--aspect-ratio-group-factor",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--rpn-score-thresh",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        action="store_true",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    return args


def main(args):
    
    torch.cuda.manual_seed(args.seed)
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading...
    print("Loading data...")

    train_dataset = coco_utils.get_dataset(args.root_dir, "train", coco_utils.get_transform(True))
    valid_dataset = coco_utils.get_dataset(args.root_dir, "valid", coco_utils.get_transform(False))
    print("Data loading DONE.")

    print("Creating data loaders...")
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, sampler=valid_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    print("Creating data loaders DONE.")

    print("Creating model...")
    model = get_model_instance_segmentation(
        num_classes=NUM_OBJECT_TYPES,
        hidden_size=args.hidden_size,
    ).to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    optimizer = optim.Adam(
        params=parameters,
        lr=args.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )
    print("Creating model DONE.")

    print("Start Training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"checkpoint.pth"))

        # Evaluate after every epoch
        evaluate(model, valid_dataloader, device=device)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args()
    main(args)