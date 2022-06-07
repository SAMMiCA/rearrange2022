from multiprocessing import current_process
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
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

    args = parser.parse_args()

    return args

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")

def cleanup():
    dist.destroy_process_group()

def prepare_dataloader(dataset: Dataset, rank: int, world_size: int, batch_size: int, pin_memory: bool = False, num_workers: int = 0):
    sampler = DistributedSampler(
        dataset=dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False, 
        drop_last=False
    )
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        pin_memory=pin_memory, 
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader

def train(rank, world_size):
    args = get_args()

    setup(rank, world_size)
    torch.cuda.manual_seed_all(args.seed)

    dataset = SubtaskModelDataset(
        root_dir=args.root_dir,
        mode="train",
        shuffle=True,
    )
    dataloader = prepare_dataloader(
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
    )

    model = SubtaskPredictionModel(
        hidden_size=args.hidden_size,
    ).to(rank)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )

    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        # with tqdm(dataloader, desc=f"[Rank {rank}] Epoch {epoch + 1} - ", unit="batch", position=rank) as tepoch:
        for batch_id, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            batch = {
                k: v.to(rank)
                for k, v in batch.items()
            }
            updated_semantic_map_unshuffle = batch["updated_semmap_unshuffle"]
            updated_semantic_map_unshuffle[:, 0] = batch["semmap"][:, 0]
            updated_semantic_map_walkthrough = batch["updated_semmap_walkthrough"]
            updated_semantic_map_walkthrough[:, 0] = batch["unshuffled_semmap"][:, 0]
            updated_semantic_maps = torch.stack(
                (updated_semantic_map_unshuffle, updated_semantic_map_walkthrough), dim=1
            )
            bs = updated_semantic_maps.shape[0]

            (
                subtask_type_logprobs,
                subtask_arg_logprobs,
                subtask_target_map_type_logprobs,
            ) = model.forward(
                semantic_maps=updated_semantic_maps.float(),
                inventory_vectors=batch["inventory"],
                subtask_index_history=batch["expert_subtask_history"],
                seq_masks=batch["masks"],
                # nsteps=args.batch_size,
                nsteps=bs,
                nsamplers=1,
            )

            subtask_gt = batch["expert_subtask_history"]
            subtask_type_gt = (subtask_gt / (NUM_OBJECT_TYPES * len(MAP_TYPES))).long().view(-1)
            subtask_arg_gt = (subtask_gt % (NUM_OBJECT_TYPES * len(MAP_TYPES)) / len(MAP_TYPES)).long().view(-1)
            subtask_target_map_type_gt = (subtask_gt % len(MAP_TYPES)).long().view(-1)

            type_loss = loss_fn(input=subtask_type_logprobs, target=subtask_type_gt)
            arg_loss = loss_fn(input=subtask_arg_logprobs, target=subtask_arg_gt)
            target_map_type_loss = loss_fn(input=subtask_target_map_type_logprobs, target=subtask_target_map_type_gt)

            total_loss = type_loss + arg_loss + target_map_type_loss
            
            total_loss.backward()
            optimizer.step()
            # tepoch.set_postfix(loss=total_loss.item())
            # tepoch.update(rank)
            if rank == 0:
                if (batch_id + 1) % args.log_interval == 0:
                    print(f'Train Epoch: {epoch + 1}, [{batch_id + 1}/{len(dataloader)} ({(batch_id+1) / len(dataloader) * 100:.2f}%)] \t Loss: {total_loss.item():.6f}')

        if rank == 0:
            print(f'Train Epoch: {epoch + 1}, [{batch_id + 1}/{len(dataloader)} ({(batch_id+1) / len(dataloader) * 100:.2f}%)] \t Loss: {total_loss.item():.6f}')

        scheduler.step()
        if rank == 0:
            save_dict = {
                "model_state_dict": model.module.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "seed": args.seed,
            }
            torch.save(save_dict, f'model_{epoch}.pt')

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()