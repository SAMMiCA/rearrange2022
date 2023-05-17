import os
import argparse
from tqdm import tqdm
import time
import clip
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torch.utils.tensorboard import SummaryWriter

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph, Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor, ResNetEmbedder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor, ClipResNetEmbedder

from task_aware_rearrange.constants import NUM_OBJECT_TYPES
from task_aware_rearrange.preprocessors import SubtaskActionExpertPreprocessor, SubtaskExpertPreprocessor, Semantic3DMapPreprocessor
from task_aware_rearrange.voxel_utils import GridParameters, image_to_semantic_maps
from task_aware_rearrange.mapping_utils import update_semantic_map
from task_aware_rearrange.layers import (
    EgocentricViewEncoderPooled,
    Semantic2DMapWithInventoryEncoderPooled,
    SemanticMap2DEncoderPooled,
    SubtaskHistoryEncoder,
    SubtaskPredictor,
)
from task_aware_rearrange.subtasks import NUM_SUBTASKS
from subtask_prediction.subtask_expert_dataset import SubtaskExpertIterableDataset
from subtask_prediction.models import SubtaskPredictionModel
from experiments.test_exp import ExpertTestExpConfig


def get_args():
    parser = argparse.ArgumentParser(
        description="train_subtask_predictor", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--exp_name",
        type=str,
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
        "-r",
        "--resume",
        dest="resume",
        action="store_true",
        required=False,
    )
    parser.set_defaults(shuffle=False)

    parser.add_argument(
        "--data_dir",
        type=str,
        default="expert_data",
        required=False,
    )

    # Data Preprocessing
    parser.add_argument(
        "--resnet",
        type=lambda s: s.lower() in ['true', '1'],
        default="true",
        required=False
    )
    parser.add_argument(
        "--cnn_type",
        type=str,
        default="RN50",
        required=False,
    )
    parser.add_argument(
        "--cnn_pretrain_type",
        type=str,
        default="clip",
        required=False,
    )
    parser.add_argument(
        "--prev_action",
        type=lambda s: s.lower() in ['true', '1'],
        default="true",
        required=False
    )
    parser.add_argument(
        "--semantic_map_with_inventory",
        type=lambda s: s.lower() in ['true', '1'],
        default="false",
        required=False
    )
    parser.add_argument(
        "--semantic_map",
        type=lambda s: s.lower() in ['true', '1'],
        default="false",
        required=False
    )
    parser.add_argument(
        "--inventory",
        type=lambda s: s.lower() in ['true', '1'],
        default="false",
        required=False
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


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = dataset.num_episodes // worker_info.num_workers

    dataset.episode_paths = dataset.episode_paths[
        worker_id * split_size: (worker_id + 1) * split_size
    ]


def process_resnet(resnet, rgb: torch.Tensor, device: torch.device):
    return resnet(
        rgb.to(device).permute(0, 3, 1, 2)
    ).float()


def process_semantic_map(
    semseg: torch.Tensor,
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    pos: torch.Tensor,
    num_classes: int = 73,
    hfov: int = 90,
    grid_params: GridParameters = GridParameters(),
    device: torch.device = torch.device('cuda'),
):
    semseg = F.one_hot(semseg.long(), num_classes).to(device).permute(0, 3, 1, 2)       # bxcxhxw
    depth = depth.to(device).permute(0, 3, 1, 2)                                        # bx1xhxw
    extrinsics = extrinsics.to(device)                                                  # bx4x4
    pos = pos.to(device)                                                                # bx3

    apos_in_maps = torch.zeros_like(pos, dtype=torch.int32, device=device)
    for i in range(3):
        apos_in_maps[:, i:i+1] = (
            (pos[:, i:i+1] - grid_params.GRID_ORIGIN[i]) / grid_params.GRID_RES
        ).int()

    apos_maps = []
    map_shape = (
        int(grid_params.GRID_SIZE_X / grid_params.GRID_RES),
        int(grid_params.GRID_SIZE_Y / grid_params.GRID_RES),
        int(grid_params.GRID_SIZE_Z / grid_params.GRID_RES),
    )
    for apos in apos_in_maps:
        apos_map = torch.zeros((1, *map_shape), device=device)
        apos_map[0, apos.long()] = 1.0
        apos_maps.append(apos_map)

    apos_maps = torch.stack(apos_maps, dim=0)

    semmaps = image_to_semantic_maps(
        scene=semseg,
        depth=depth,
        extrinsics4f=extrinsics,
        hfov_deg=hfov,
        grid_params=grid_params,
    )

    return torch.cat(
        (
            apos_maps,
            semmaps,
        ),
        dim=1
    ).type(torch.bool)


# def train_one_epoch(model, loss_fn, optimizer, scheduler, dataloader, device, writer, epoch, accum_batch, accum_iter, args):
#     train_loss = 0.0
#     train_acc = 0.0
#     avg_loss = 0.0
#     epoch_batches = 0
#     epoch_num_data = 0
    
#     model.train()

#     with tqdm(dataloader, unit=" batch") as tepoch:
#         prev_actions = torch.zeros((args.batch_size + 1)).long()
#         for batch, _ in tepoch:
#             epoch_batches += 1
#             accum_batch += 1
#             """
#             batch: Dict[str, torch.Tensor]
#                 keys: 
#                     '(unshuffled/walkthrough)_rgb' => Preprocessed Egocentric RGB Image. (bsize, 224, 224, 3)
#                     '(unshuffled/walkthrough)_depth' => Preprocessed Egocentric Depth Image. (bsize, 224, 224, 1)
#                     '(unshuffled/walkthrough)_semseg' => Egocentric Semantic Segmentation Label Image. (bsize, 224, 224)
#                     '(unshuffled/walkthrough)_pos_rot_horizon' => Agent Position/Rotation/Horizon Angle (in meters, degrees) (bsize, 5)
#                     '(unshuffled/walkthrough)_pos_unity' => Agent position in "Unity" coordinates system. (in meters) (bsize, 3)
#                     '(unshuffled/walkthrough)_Tu2w' => Transformation Matrix from unity to world coordinates. (bsize, 4, 4)
#                     '(unshuffled/walkthrough)_Tw2c' => Transformation Matrix from world to camera. (bsize, 4, 4)
#                     'inventory' => Class-wise onehot vector for holding object (bsize, num_objs+1)
#                     'expert_subtask_action' => expert subtask & action obtained from expert subtask & action sensor. (bsize, 4)
#                     'episode_id' => id for distinguish episodes along rollouts. (bsize)
#                     'masks' => indicator for start of episode. (0: start of new episode / 1: during episode) (bsize)
#             """
#             model_inputs = {}

#             # Previous action is same with the batch["expert_subtask_action"][1:, -2].
#             bsize = batch["masks"].shape[0]
#             epoch_num_data += bsize
#             accum_iter += bsize
#             prev_actions[0] = prev_actions[-1]
#             prev_actions[1:bsize+1] = batch["expert_subtask_action"][:, -2]
#             model_inputs["prev_actions"] = prev_actions[:bsize].to(device)
#             model_inputs["masks"] = batch["masks"].bool().to(device)

#             # Preprocessing for rgb inputs
#             unshuffle_rgb_resnet = None
#             walkthrough_rgb_resnet = None
#             if args.resnet:
#                 if resnet is not None:
#                     unshuffle_rgb_resnet = process_resnet(resnet, batch["unshuffle_rgb"], device)
#                     walkthrough_rgb_resnet = process_resnet(resnet, batch["walkthrough_rgb"], device)

#             model_inputs["unshuffle_rgb_resnet"] = unshuffle_rgb_resnet
#             model_inputs["walkthrough_rgb_resnet"] = walkthrough_rgb_resnet

#             # Preprocessing for Semantic 3D Mapping
#             unshuffle_semmap = None
#             walkthrough_semmap = None
#             if args.semantic_map:
#                 unshuffle_semmap = []
#                 walkthrough_semmap = []

#                 cur_unshuffle_semmap = process_semantic_map(
#                     batch["unshuffle_semseg"],
#                     batch["unshuffle_depth"],
#                     batch["unshuffle_Tw2c"],
#                     batch["unshuffle_pos_rot_horizon"][..., :3],
#                     num_classes=NUM_OBJECT_TYPES,
#                     grid_params=grid_params,
#                     device=device,
#                 )
#                 cur_walkthrough_semmap = process_semantic_map(
#                     batch["walkthrough_semseg"],
#                     batch["walkthrough_depth"],
#                     batch["walkthrough_Tw2c"],
#                     batch["walkthrough_pos_rot_horizon"][..., :3],
#                     num_classes=NUM_OBJECT_TYPES,
#                     grid_params=grid_params,
#                     device=device,
#                 )

#                 # Update Semantic 3D Map Step-wise
#                 map_masks = batch["masks"].to(device)[:, None, None, None, None].bool()
#                 for i in range(cur_unshuffle_semmap.shape[0]):
#                     acc_unshuffle_semmap = update_semantic_map(
#                         sem_map=cur_unshuffle_semmap[i],
#                         sem_map_prev=acc_unshuffle_semmap,
#                         map_mask=map_masks[i],
#                     ).squeeze(0)
#                     acc_walkthrough_semmap = update_semantic_map(
#                         sem_map=cur_walkthrough_semmap[i],
#                         sem_map_prev=acc_walkthrough_semmap,
#                         map_mask=map_masks[i],
#                     ).squeeze(0)
#                     unshuffle_semmap.append(acc_unshuffle_semmap)
#                     walkthrough_semmap.append(acc_walkthrough_semmap)

#                 unshuffle_semmap = torch.stack(unshuffle_semmap)
#                 walkthrough_semmap = torch.stack(walkthrough_semmap)
                
#             model_inputs["unshuffle_semmap"] = unshuffle_semmap
#             model_inputs["walkthrough_semmap"] = walkthrough_semmap

#             # Inventory Vector
#             if args.inventory:
#                 model_inputs["inventory"] = batch["inventory"].to(device)
            
#             # Data for Subtask/Action Histories
#             model_inputs["subtask_history"] = batch["expert_subtask_action"][:, 0].to(device)
#             model_inputs["episode_id"] = batch["episode_id"].to(device)

#             output = model(model_inputs)
#             labels = batch["expert_subtask_action"][:, 0].to(device)

#             # Loss calculation
#             loss = loss_fn(
#                 input=output,
#                 target=labels,
#             )

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             avg_loss += loss.item()
#             pred_labels = torch.argmax(output, axis=1)
#             train_acc += (pred_labels == labels).sum().item()

#             tepoch.set_description_str(f'Epoch {epoch + 1}] Loss: {train_loss.item():.04f}')

#             if accum_batch % args.log_interval == 0:
#                 writer.add_scalar("Loss/train", train_loss / args.log_interval, accum_batch)
#                 train_loss = 0.0

#                 writer.add_scalar("Accuracy/train", train_acc / args.log_interval, accum_batch)
#                 train_acc = 0.0

#         scheduler.step()

#         writer.add_scalar("Training Loss", avg_loss / epoch_batches, epoch)
#         save_dict = {
#             "model_state_dict": model.state_dict(),
#             "total_steps": accum_iter,
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict(),
#             "args": args.__dict__,
#             "epochs": epoch,
#             "num_batches": epoch_batches,
#             "num_data": epoch_num_data,
#         }
#         torch.save(save_dict, f"subtask_out/checkpoints/{args.exp_name}/{args.exp_name}_epoch_{epoch}.pt")

#     return avg_loss, accum_batch, accum_iter


if __name__ == "__main__":

    args = get_args()
    device = torch.device('cuda')

    if not os.path.exists(f'subtask_out/tb/{args.exp_name}'):
        os.makedirs(f'subtask_out/tb/{args.exp_name}')
    if not os.path.exists(f'subtask_out/checkpoints/{args.exp_name}'):
        os.makedirs(f'subtask_out/checkpoints/{args.exp_name}')

    writer = SummaryWriter(f'subtask_out/tb/{args.exp_name}')

    if args.resnet:
        if args.cnn_type == "RN50":
            mean, std = None, None
            resnet = None
            if args.cnn_pretrain_type == "clip":
                # import clip
                # clip.load(args.cnn_type, "cpu")
                mean, std = ClipResNetPreprocessor.CLIP_RGB_MEANS, ClipResNetPreprocessor.CLIP_RGB_STDS
                resnet = ClipResNetEmbedder(
                    clip.load(args.cnn_type, device=device)[0], pool=False
                ).to(device)
                for module in resnet.modules():
                    if "BatchNorm" in type(module).__name__:
                        module.momentum = 0.0
                resnet.eval()
            elif args.cnn_pretrain_type == "imagenet":
                mean, std = IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
                resnet = ResNetEmbedder(
                    torchvision.models.resnet50(pretrained=True).to(device), pool=False
                ).to(device)
        
    train_data = SubtaskExpertIterableDataset(
        episode_paths=[
            os.path.join(
                args.data_dir, "train", episode
            )
            for episode in os.listdir(
                os.path.join(
                    args.data_dir, "train"
                )
            )
        ],
        shuffle=True,
    )
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)

    val_data = SubtaskExpertIterableDataset(
        episode_paths=[
            os.path.join(
                args.data_dir, "val", episode
            )
            for episode in os.listdir(
                os.path.join(
                    args.data_dir, "val"
                )
            )
        ],
        shuffle=True,
    )
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size)


    acc_unshuffle_semmap = None
    acc_walkthrough_semmap = None
    if args.semantic_map or args.semantic_map_with_inventory:
        grid_params = GridParameters()
        map_shape = (
            int(grid_params.GRID_SIZE_X / grid_params.GRID_RES),
            int(grid_params.GRID_SIZE_Y / grid_params.GRID_RES),
            int(grid_params.GRID_SIZE_Z / grid_params.GRID_RES),
        )
        acc_unshuffle_semmap = torch.zeros((NUM_OBJECT_TYPES + 3, *map_shape), device=device, dtype=torch.bool)
        acc_walkthrough_semmap = torch.zeros((NUM_OBJECT_TYPES + 3, *map_shape), device=device, dtype=torch.bool)

    # Training Models
    hidden_size = 512
    prev_action_emb_dim = 32
    egoview_emb_dim = 2048 if (args.resnet and args.cnn_type == "RN50") else None   # Should be edited
    model = SubtaskPredictionModel(
        hidden_size=hidden_size,
        prev_action_embedding_dim=prev_action_emb_dim,
        egoview_embedding_dim=egoview_emb_dim,
        resnet_embed=args.resnet,
        prev_action_embd=args.prev_action,
        semantic_map_with_inventory=args.semantic_map_with_inventory,
        semantic_map_embed=args.semantic_map,
        inventory=args.inventory,
    ).to(device)
    model.train()

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch,
    )

    start_time = time.time()
    acc_iter = 0
    acc_batch = 0
    acc_val_iter = 0
    acc_val_batch = 0

    for epoch in range(args.num_epochs):
        epoch_iter = 0
        epoch_batch = 0
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        running_iter = 0
        with tqdm(train_data_loader, unit=" batch") as tepoch:
            tepoch.set_description_str(f'Epoch {epoch + 1}]')
            prev_actions = torch.zeros((args.batch_size + 1)).long()
            for batch, worker_id in tepoch:
                acc_batch += 1
                epoch_batch += 1

                """
                batch: Dict[str, torch.Tensor]
                    keys: 
                        '(unshuffled/walkthrough)_rgb' => Preprocessed Egocentric RGB Image. (bsize, 224, 224, 3)
                        '(unshuffled/walkthrough)_depth' => Preprocessed Egocentric Depth Image. (bsize, 224, 224, 1)
                        '(unshuffled/walkthrough)_semseg' => Egocentric Semantic Segmentation Label Image. (bsize, 224, 224)
                        '(unshuffled/walkthrough)_pos_rot_horizon' => Agent Position/Rotation/Horizon Angle (in meters, degrees) (bsize, 5)
                        '(unshuffled/walkthrough)_pos_unity' => Agent position in "Unity" coordinates system. (in meters) (bsize, 3)
                        '(unshuffled/walkthrough)_Tu2w' => Transformation Matrix from unity to world coordinates. (bsize, 4, 4)
                        '(unshuffled/walkthrough)_Tw2c' => Transformation Matrix from world to camera. (bsize, 4, 4)
                        'inventory' => Class-wise onehot vector for holding object (bsize, num_objs+1)
                        'expert_subtask_action' => expert subtask & action obtained from expert subtask & action sensor. (bsize, 4)
                        'episode_id' => id for distinguish episodes along rollouts. (bsize)
                        'masks' => indicator for start of episode. (0: start of new episode / 1: during episode) (bsize)
                """
                model_inputs = {}
                # Previous action is same with the batch["expert_subtask_action"][1:, -2].
                bsize = batch["masks"].shape[0]
                acc_iter += bsize
                epoch_iter += bsize
                running_iter += bsize
                prev_actions[0] = prev_actions[-1]
                prev_actions[1:bsize+1] = batch["expert_subtask_action"][:, -2]
                model_inputs["prev_actions"] = prev_actions[:bsize].to(device)
                model_inputs["masks"] = batch["masks"].bool().to(device)

                # Preprocessing for rgb inputs
                unshuffle_rgb_resnet = None
                walkthrough_rgb_resnet = None
                if args.resnet:
                    if resnet is not None:
                        unshuffle_rgb_resnet = process_resnet(resnet, batch["unshuffle_rgb"], device)
                        walkthrough_rgb_resnet = process_resnet(resnet, batch["walkthrough_rgb"], device)
                    # import pdb; pdb.set_trace()
                model_inputs["unshuffle_rgb_resnet"] = unshuffle_rgb_resnet
                model_inputs["walkthrough_rgb_resnet"] = walkthrough_rgb_resnet

                # Preprocessing for Semantic 3D Mapping
                unshuffle_semmap = None
                walkthrough_semmap = None
                if args.semantic_map or args.semantic_map_with_inventory:
                    unshuffle_semmap = []
                    walkthrough_semmap = []

                    cur_unshuffle_semmap = process_semantic_map(
                        batch["unshuffle_semseg"],
                        batch["unshuffle_depth"],
                        batch["unshuffle_Tw2c"],
                        batch["unshuffle_pos_rot_horizon"][..., :3],
                        num_classes=NUM_OBJECT_TYPES,
                        grid_params=grid_params,
                        device=device,
                    )
                    cur_walkthrough_semmap = process_semantic_map(
                        batch["walkthrough_semseg"],
                        batch["walkthrough_depth"],
                        batch["walkthrough_Tw2c"],
                        batch["walkthrough_pos_rot_horizon"][..., :3],
                        num_classes=NUM_OBJECT_TYPES,
                        grid_params=grid_params,
                        device=device,
                    )

                    # Update Semantic 3D Map Step-wise
                    map_masks = batch["masks"].to(device)[:, None, None, None, None].bool()
                    for i in range(cur_unshuffle_semmap.shape[0]):
                        acc_unshuffle_semmap = update_semantic_map(
                            sem_map=cur_unshuffle_semmap[i],
                            sem_map_prev=acc_unshuffle_semmap,
                            map_mask=map_masks[i],
                        ).squeeze(0)
                        acc_walkthrough_semmap = update_semantic_map(
                            sem_map=cur_walkthrough_semmap[i],
                            sem_map_prev=acc_walkthrough_semmap,
                            map_mask=map_masks[i],
                        ).squeeze(0)
                        unshuffle_semmap.append(acc_unshuffle_semmap)
                        walkthrough_semmap.append(acc_walkthrough_semmap)

                    unshuffle_semmap = torch.stack(unshuffle_semmap)
                    walkthrough_semmap = torch.stack(walkthrough_semmap)
                    # import pdb; pdb.set_trace()
                model_inputs["unshuffle_semmap"] = unshuffle_semmap
                model_inputs["walkthrough_semmap"] = walkthrough_semmap

                # Inventory Vector
                if args.inventory or args.semantic_map_with_inventory:
                    model_inputs["inventory"] = batch["inventory"].to(device)
                
                # Data for Subtask/Action Histories
                model_inputs["subtask_history"] = batch["expert_subtask_action"][:, 0].to(device)
                model_inputs["episode_id"] = batch["episode_id"].to(device)
                
                output = model(model_inputs)
                labels = batch["expert_subtask_action"][:, 0].to(device)

                # Loss calculation
                loss = loss_fn(
                    input=output,
                    target=labels,
                )

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_train_loss += loss.item()
                pred_labels = torch.argmax(output, axis=1)
                acc = (pred_labels == labels).sum()
                epoch_train_acc += acc.item()
                running_acc += acc.item()

                tepoch.set_description_str(f'Epoch {epoch + 1}] Training Loss: {loss.item():.04f} | Training Accuracy {(acc.item() / bsize):.04f}')

                # Log every (args.log_interval) minibatches
                if epoch_batch % int(args.log_interval / (args.batch_size / 64)) == 0:
                    writer.add_scalar("training_loss_iter", running_loss / args.log_interval, acc_iter)
                    writer.add_scalar("training_accuracy_iter", (running_acc / running_iter) / args.log_interval, acc_iter)
                    running_loss = 0.0
                    running_acc = 0.0
                    running_iter = 0
                    
        lr_scheduler.step()

        avg_loss = epoch_train_loss / epoch_batch
        avg_acc = epoch_train_acc / epoch_iter
        print(f"Epoch {epoch + 1} Training Ended.")
        print(f"Epoch train loss: {avg_loss}, train accuracy: {avg_acc}, eBatch: {epoch_batch}, current data #: {epoch_iter}")
        writer.add_scalar("training_loss_epoch", avg_loss, epoch)
        writer.add_scalar("training_accuracy_epoch", avg_acc, epoch)
        print(f"Total data #: {acc_iter}, # iBatch: {acc_batch}")
        # print(f"Last training loss: {running_loss / (args.log_interval if iter % args.log_interval == 0 else (iter % args.log_interval))}")
        save_dict = {
            "model_state_dict": model.state_dict(),
            "total_steps": acc_iter,
            "total_batches": acc_batch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "args": args.__dict__,
            "epochs": epoch,
            "num_iterations": epoch_iter,
            "num_batches": epoch_batch,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_acc,
        }
        torch.save(save_dict, f"subtask_out/checkpoints/{args.exp_name}/{args.exp_name}_epoch_{epoch}.pt")
    
        # Validation
        model.eval()
        running_vloss = 0.0
        running_vacc = 0.0
        running_viter = 0
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        vepoch_iter = 0
        vepoch_batch = 0
        with tqdm(val_data_loader, unit=" batch") as vepoch:
            vepoch.set_description_str(f'Epoch {epoch + 1}]')
            prev_actions = torch.zeros((args.batch_size + 1)).long()
            if args.semantic_map or args.semantic_map_with_inventory:
                acc_unshuffle_semmap.zero_()
                acc_walkthrough_semmap.zero_()
            with torch.no_grad():
                for vbatch, worker_id in vepoch:
                    vepoch_batch += 1
                    acc_val_batch += 1

                    vmodel_inputs = {}
                    # Previous action is same with the batch["expert_subtask_action"][1:, -2].
                    bsize = vbatch["masks"].shape[0]
                    vepoch_iter += bsize
                    acc_val_iter += bsize
                    running_viter += bsize
                    prev_actions[0] = prev_actions[-1]
                    prev_actions[1:bsize+1] = vbatch["expert_subtask_action"][:, -2]
                    vmodel_inputs["prev_actions"] = prev_actions[:bsize].to(device)
                    vmodel_inputs["masks"] = vbatch["masks"].bool().to(device)

                    # Preprocessing for rgb inputs
                    unshuffle_rgb_resnet = None
                    walkthrough_rgb_resnet = None
                    if args.resnet:
                        if resnet is not None:
                            unshuffle_rgb_resnet = process_resnet(resnet, vbatch["unshuffle_rgb"], device)
                            walkthrough_rgb_resnet = process_resnet(resnet, vbatch["walkthrough_rgb"], device)
                        # import pdb; pdb.set_trace()
                    vmodel_inputs["unshuffle_rgb_resnet"] = unshuffle_rgb_resnet
                    vmodel_inputs["walkthrough_rgb_resnet"] = walkthrough_rgb_resnet

                    # Preprocessing for Semantic 3D Mapping
                    unshuffle_semmap = None
                    walkthrough_semmap = None
                    if args.semantic_map or args.semantic_map_with_inventory:
                        unshuffle_semmap = []
                        walkthrough_semmap = []

                        cur_unshuffle_semmap = process_semantic_map(
                            vbatch["unshuffle_semseg"],
                            vbatch["unshuffle_depth"],
                            vbatch["unshuffle_Tw2c"],
                            vbatch["unshuffle_pos_rot_horizon"][..., :3],
                            num_classes=NUM_OBJECT_TYPES,
                            grid_params=grid_params,
                            device=device,
                        )
                        cur_walkthrough_semmap = process_semantic_map(
                            vbatch["walkthrough_semseg"],
                            vbatch["walkthrough_depth"],
                            vbatch["walkthrough_Tw2c"],
                            vbatch["walkthrough_pos_rot_horizon"][..., :3],
                            num_classes=NUM_OBJECT_TYPES,
                            grid_params=grid_params,
                            device=device,
                        )

                        # Update Semantic 3D Map Step-wise
                        map_masks = vbatch["masks"].to(device)[:, None, None, None, None].bool()
                        for i in range(cur_unshuffle_semmap.shape[0]):
                            acc_unshuffle_semmap = update_semantic_map(
                                sem_map=cur_unshuffle_semmap[i],
                                sem_map_prev=acc_unshuffle_semmap,
                                map_mask=map_masks[i],
                            ).squeeze(0)
                            acc_walkthrough_semmap = update_semantic_map(
                                sem_map=cur_walkthrough_semmap[i],
                                sem_map_prev=acc_walkthrough_semmap,
                                map_mask=map_masks[i],
                            ).squeeze(0)
                            unshuffle_semmap.append(acc_unshuffle_semmap)
                            walkthrough_semmap.append(acc_walkthrough_semmap)

                        unshuffle_semmap = torch.stack(unshuffle_semmap)
                        walkthrough_semmap = torch.stack(walkthrough_semmap)
                        # import pdb; pdb.set_trace()
                    vmodel_inputs["unshuffle_semmap"] = unshuffle_semmap
                    vmodel_inputs["walkthrough_semmap"] = walkthrough_semmap

                    # Inventory Vector
                    if args.inventory or args.semantic_map_with_inventory:
                        vmodel_inputs["inventory"] = vbatch["inventory"].to(device)
                    
                    # Data for Subtask/Action Histories
                    vmodel_inputs["subtask_history"] = vbatch["expert_subtask_action"][:, 0].to(device)
                    vmodel_inputs["episode_id"] = vbatch["episode_id"].to(device)

                    voutput = model(vmodel_inputs)
                    vlabels = vbatch["expert_subtask_action"][:, 0].to(device)
                    vloss = loss_fn(voutput, vlabels)

                    running_vloss += vloss.item()
                    epoch_val_loss += vloss.item()
                    vpred_labels = torch.argmax(voutput, axis=1)
                    val_acc = (vpred_labels == vlabels).sum()
                    epoch_val_acc += val_acc.item()
                    running_vacc += val_acc.item()

                    vepoch.set_description_str(f'Epoch {epoch + 1}] Validation Loss: {vloss.item():.04f} | Validation Accuracy: {(val_acc.item() / bsize):.04f}')
                    if vepoch_batch % int(args.log_interval / (args.batch_size / 64) / 4) == 0:
                        writer.add_scalar("validation_loss_iter", running_vloss / (args.log_interval / 4), acc_val_iter)
                        writer.add_scalar("validation_accuracy_iter", (running_vacc / running_viter) / (args.log_interval / 4), acc_val_iter)
                        running_vloss = 0.0
                        running_vacc = 0.0
                        running_viter = 0

            avg_vloss = epoch_val_loss / vepoch_batch
            avg_vacc = epoch_val_acc / vepoch_iter
            print(f"Epoch {epoch + 1} Validation Ended.")
            print(f"Epoch validation loss: {avg_vloss}, validation accuracy: {avg_vacc}, vbatch: {vepoch_batch}")
            writer.add_scalar("validation_loss_epoch", avg_vloss, epoch)
            writer.add_scalar("validation_accuracy_epoch", avg_vacc, epoch)
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch,
            )
            writer.add_scalars(
                "Training vs. Validation Accuracy",
                {"Training": avg_acc, "Validation": avg_vacc},
                epoch,
            )

            writer.flush()
            