import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from custom.constants import MAP_TYPES, NUM_OBJECT_TYPES
from custom.subtask_model.subtask_model_dataset import SubtaskModelDataset
from custom.models import SubtaskPredictionModel


def get_args():
    parser = argparse.ArgumentParser(
        description="Distributed Training Script for SubtaskPredictionModel", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        required=False,
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        required=False,
    )
    parser.set_defaults(shuffle=False)

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="~/dataset/rearrange2022/subtask_model",
        required=False,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        required=False,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        metavar='N',
        help='Local process rank.'
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    args.is_master = args.local_rank == 0

    args.device = torch.cuda.device(args.local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    torch.cuda.manual_seed_all(args.seed)

    model = SubtaskPredictionModel(hidden_size=args.hidden_size)
    model = model.to(args.device)

    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    dataset = SubtaskModelDataset(
        root_dir=args.root_dir,
        mode="train",
        shuffle=args.shuffle,
    )
    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
    )

    loss = nn.NLLLoss()
    optimizer = optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )

    for epoch in range(args.num_epochs):
        model.train()
        dist.barrier()

        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            for batch in tepoch:
                batch = {
                    k: v.to(args.device)
                    for k, v in batch.items()
                }

                updated_semantic_map_unshuffle = batch["updated_semmap_unshuffle"]
                updated_semantic_map_unshuffle[:, 0] = batch["semmap"][:, 0]
                updated_semantic_map_walkthrough = batch["updated_semmap_walkthrough"]
                updated_semantic_map_walkthrough[:, 0] = batch["unshuffled_semmap"][:, 0]

                updated_semantic_maps = torch.stack(
                    (updated_semantic_map_unshuffle, updated_semantic_map_walkthrough), dim=1
                )

                (
                    subtask_type_logprobs,
                    subtask_arg_logprobs,
                    subtask_target_map_type_logprobs,
                ) = model(
                    semantic_maps=updated_semantic_maps.float(),
                    inventory_vectors=batch["inventory"],
                    subtask_index_history=batch["expert_subtask_history"],
                    seq_masks=batch["masks"],
                    nsteps=args.batch_size,
                    nsamplers=1,
                )

                subtask_gt = batch["expert_subtask_history"]
                subtask_type_gt = (subtask_gt / (NUM_OBJECT_TYPES * len(MAP_TYPES))).long().view(-1)
                subtask_arg_gt = (subtask_gt % (NUM_OBJECT_TYPES * len(MAP_TYPES)) / len(MAP_TYPES)).long().view(-1)
                subtask_target_map_type_gt = (subtask_gt % len(MAP_TYPES)).long().view(-1)

                type_loss = loss(input=subtask_type_logprobs, target=subtask_type_gt)
                arg_loss = loss(input=subtask_arg_logprobs, target=subtask_arg_gt)
                target_map_type_loss = loss(input=subtask_target_map_type_logprobs, target=subtask_target_map_type_gt)

                total_loss = type_loss + arg_loss + target_map_type_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=total_loss.item())
        
        scheduler.step()

        if args.local_rank == 0:
            save_dict = {
                "model_state_dict": model.module.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "seed": args.seed,
            }
            torch.save(save_dict, f'model_{epoch}.pt')