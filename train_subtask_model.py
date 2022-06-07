import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from custom.constants import MAP_TYPES, NUM_OBJECT_TYPES
from custom.subtask_model.subtask_model_dataset import SubtaskModelDataset
from custom.models import SubtaskPredictionModel


def get_args():
    parser = argparse.ArgumentParser(
        description="train_subtask_model", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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


if __name__ == "__main__":

    args = get_args()
    # datapath = os.path.join(
    #     os.path.expanduser("~"), 
    #     "dataset/rearrange2022/subtask_model/test"
    # )
    # batch_size = 64

    train_data = SubtaskModelDataset(
        root_dir=args.root_dir,
        mode="train",
        shuffle=args.shuffle,
    )
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)

    device = "cuda"
    subtask_model = SubtaskPredictionModel(
        hidden_size=512,
    ).to(device)
    subtask_model.train()

    loss = nn.NLLLoss()
    optimizer = optim.Adam(
        params=[p for p in subtask_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )

    for epoch in range(args.num_epochs):
        with tqdm(train_data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                """
                batch: Dict[str, torch.Tensor]
                    keys: '(unshuffled_)rgb' => Egocentric RGB Image.
                        '(unshuffled_)depth' => Egocentric Depth Image.
                        '(unshuffled_)semseg' => Egocentric Semantic Segmentation Masks.
                        '(unshuffled_)instseg_inst_masks' => Egocentric Instance Segmentation Masks.
                        '(unshuffled_)instseg_inst_detected' => Class-wise number of detected instances.
                        '(unshuffled_)semmap => Semantic 3D Map based on "current" egocentric views.
                        '(unshuffled_)pose_cam_horizon_deg' => Camera Horizon Angle (in Degrees)
                        '(unshuffled_)pose_cam_pos_enu' => Camera Position in "Local" coordinates system.
                        '(unshuffled_)pose_rot_3d_enu_deg' => Agent rotation in "Local" coordinates system
                        '(unshuffled_)pose_agent_pos' => Agent position in "Local" coordinates system.
                        '(unshuffled_)pose_agent_pos_unity' => Agent position in "Unity" coordinates system.
                        '(unshuffled_)pose_T_unity_to_world' => Transformation Matrix from unity to world coordinates.
                        '(unshuffled_)pose_T_world_to_cam' => Transformation Matrix from world to camera.
                        'updated_semmap_(unshuffle/walkthrough)' => Semantic 3D Map that is updated through episode rollouts.
                        'inventory' => Class-wise onehot vector for holding object
                        'expert_subtask_history' => expert subtask obtained from expert subtask sensor.
                        'expert_action_history' => expert action obtained from expert action sensor (based on expert subtask.)
                        'action_history' => Actual action executed by agent.
                        'episode_id' => id for distinguish episodes along rollouts.
                        'masks' => indicator for start of episode. (0: start of new episode / 1: during episode)
                """

                tepoch.set_description(f"Epoch {epoch + 1}")
                # batch["updated_semmap_(unshuffle/walkthrough)"][:, 0] should be replaced with 
                # batch["(unshuffled_)semmap"][:, 0] due to the error in update_semantic_map() method
                updated_semantic_map_unshuffle = batch["updated_semmap_unshuffle"].to(device)
                updated_semantic_map_unshuffle[:, 0] = batch["semmap"][:, 0]
                updated_semantic_map_walkthrough = batch["updated_semmap_walkthrough"].to(device)
                updated_semantic_map_walkthrough[:, 0] = batch["unshuffled_semmap"][:, 0]

                updated_semantic_maps = torch.stack(
                    (updated_semantic_map_unshuffle, updated_semantic_map_walkthrough), dim=1
                )

                (
                    subtask_type_logprobs,
                    subtask_arg_logprobs,
                    subtask_target_map_type_logprobs,
                ) = subtask_model.forward(
                    semantic_maps=updated_semantic_maps.float(),
                    inventory_vectors=batch["inventory"].to(device),
                    subtask_index_history=batch["expert_subtask_history"].to(device),
                    seq_masks=batch["masks"].to(device),
                    nsteps=args.batch_size,
                    nsamplers=1,
                )

                subtask_gt = batch["expert_subtask_history"].to(device)
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
                
                # if (it + 1) % args.log_interval == 0:
                #     current_num_rollouts = (it + 1) * args.batch_size
                #     total_batch_length = int(train_data_loader.dataset.total_num_rollouts / args.batch_size) + 1
                #     print(f'Train Epoch: {epoch + 1}, [{current_num_rollouts}/{train_data_loader.dataset.total_num_rollouts} ({(it+1) / total_batch_length * 100:.2f}%)] \t Loss: {total_loss.item():.6f}')
                            
        scheduler.step()

