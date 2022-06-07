from typing import Sequence, Union, List
import torch
import stringcase
from custom.hlsm.ops.misc import index_to_onehot
from custom.hlsm.voxel_grid import VoxelGrid, GridParameters
from custom.constants import NUM_OBJECT_TYPES, IDX_TO_OBJECT_TYPE, OBJECT_TYPES_TO_IDX, ORDERED_OBJECT_TYPES, IDX_TO_SUBTASK_TYPE, SUBTASK_TYPES_TO_IDX, SUBTASK_TYPES, NAV_SUBTASK_TYPES, INTERACT_SUBTASK_TYPES, IDX_TO_MAP_TYPE, MAP_TYPES_TO_IDX, MAP_TYPES
from example_utils import ForkedPdb


class Subtask:
    def __init__(
        self,
        subtask_type: torch.tensor,
        argument_vector: torch.tensor,
        target_map_type: torch.tensor,
        argument_mask: Union[torch.tensor, None] = None,
        ordered_object_types: Sequence[str] = ORDERED_OBJECT_TYPES,
    ):
        """
        ordered_object_types: list of object types
        subtask_type: B-length vector of integers indicating subtask type
        target_map_vector: Bx2 matrix of one-hot target map vectors
        argument_vector: BxN matrix of one-hot argument vectors
        argument_mask: BxWxLxH tensor of masks indicating interaction positions
        """
        self.ordered_object_types = ordered_object_types
        self.subtask_type = subtask_type
        self.argument_vector = argument_vector
        self.target_map_type = target_map_type
        self.argument_mask = argument_mask
        self.device = self.subtask_type.device

    def to(self, device):
        self.subtask_type = self.subtask_type.to(device)
        self.argument_vector = self.argument_vector.to(device)
        self.target_map_type = self.target_map_type.to(device)
        self.argument_mask = self.argument_mask.to(device) if self.argument_mask else None
        self.device = device
        
        return self

    def type_id(self):
        assert len(self.subtask_type) == 1, "Only single subtask can have a single ID."
        return self.subtask_type[0].item()

    @classmethod
    def collate(cls, lst: List["Subtask"]) -> "Subtask":
        subtask_types = [l.subtask_type for l in lst]
        subtask_types = torch.cat(subtask_types, dim=0)
        arg_vectors = [l.argument_vector for l in lst]
        arg_vectors = torch.cat(arg_vectors, dim=0)
        target_map_types = [l.target_map_type for l in lst]
        target_map_types = torch.cat(target_map_types, dim=0)
        arg_masks = [l.argument_mask for l in lst]

        for i, arg_mask in enumerate(arg_masks):
            if arg_mask is None:
                arg_masks[i] = VoxelGrid.create_empty(params=GridParameters(), device=subtask_types.device)

        arg_masks = VoxelGrid.collate(arg_masks)
        return Subtask(subtask_types, arg_vectors, target_map_types, arg_masks)

    def disperse(self) -> List["Subtask"]:
        out = []
        for i in range(len(self.subtask_type)):
            out.append(
                Subtask(
                    self.subtask_type[i:i+1],
                    self.argument_vector[i:i+1],
                    self.target_map_type[i:i+1],
                    self.argument_mask[i:i+1] if self.argument_mask[i] else None
                )
            )
        
        return out

    @classmethod
    def subtask_type_str_to_intid(cls, subtask_type_str: str) -> int:
        return SUBTASK_TYPES_TO_IDX[subtask_type_str]

    @classmethod
    def subtask_type_intid_to_str(cls, subtask_type_intid: int) -> str:
        return IDX_TO_SUBTASK_TYPE[subtask_type_intid]

    @classmethod
    def target_map_type_str_to_intid(cls, target_map_type_str: str) -> int:
        return MAP_TYPES_TO_IDX[target_map_type_str]

    @classmethod
    def target_map_type_intid_to_str(cls, target_map_type_intid: int) -> str:
        return IDX_TO_MAP_TYPE[target_map_type_intid]
    
    @classmethod
    def from_type_str_arg_vector_and_target_map_str(cls, type_str, arg_vec, tmap_str):
        type_id = cls.subtask_type_str_to_intid(type_str)
        type_vec = torch.tensor([type_id], device=arg_vec.device)
        tmap_id = cls.target_map_type_str_to_intid(tmap_str)
        tmap_vec = torch.tensor([tmap_id], device=arg_vec.device)
        return cls(type_vec, arg_vec, tmap_vec)

    @classmethod
    def from_type_arg_target_map_id(cls, type_id, arg_id, tmap_id):
        type_vec = torch.tensor([type_id])
        arg_vec = torch.zeros([cls.get_subtask_arg_space_dim()])
        arg_vec[arg_id + 1] = 1 # arg_id(-1): none
        arg_vec = arg_vec[None, :]
        tmap_vec = torch.tensor([tmap_id])
        return cls(type_vec, arg_vec, tmap_vec)

    @classmethod
    def from_type_str_arg_target_map_id(cls, type_str, arg_id, tmap_id):
        type_id = cls.subtask_type_str_to_intid(type_str)
        return cls.from_type_arg_target_map_id(type_id, arg_id, tmap_id)

    @classmethod
    def from_type_str_arg_target_map_id_with_mask(cls, type_str, arg_id, tmap_id, mask):
        subtask = cls.from_type_str_arg_target_map_id(type_str, arg_id, tmap_id)
        subtask.argument_mask = mask
        return subtask

    @classmethod
    def from_type_target_map_str_and_arg_id(cls, type_str, arg_id, tmap_str):
        tmap_id = cls.target_map_type_str_to_intid(tmap_str)
        return cls.from_type_str_arg_target_map_id(type_str, arg_id, tmap_id)

    @classmethod
    def from_type_target_map_str_and_arg_id_with_mask(cls, type_str, arg_id, tmap_str, mask):
        subtask = cls.from_type_target_map_str_and_arg_id(type_str, arg_id, tmap_str)
        subtask.mask = mask
        return subtask

    @classmethod
    def extract_touch_argument(cls, action: str):
        if action.startswith('pickup'):
            arg_str = action.split('_')[1:]
        elif action.startswith('open_by_type'):
            arg_str = action.split('_')[3:]
        else:
            return -1

        arg_str = stringcase.pascalcase(arg_str)
        arg_id = OBJECT_TYPES_TO_IDX[arg_str]
        
        return arg_id

    @classmethod
    def get_subtask_type_space_dim(cls) -> int:
        return len(SUBTASK_TYPES_TO_IDX)

    @classmethod
    def get_subtask_arg_space_dim(cls) -> int:
        return NUM_OBJECT_TYPES

    @classmethod
    def get_target_map_type_space_dim(cls) -> int:
        return len(MAP_TYPES_TO_IDX)

    def type_intid(self):
        assert len(self.subtask_type) == 1
        return int(self.subtask_type.item())

    def arg_intid(self):
        return int(self.argument_vector.argmax(dim=1)) - 1

    def target_map_intid(self):
        assert len(self.target_map_type) == 1
        return int(self.target_map_type.item())

    def object_vector(self):
        return self.argument_vector[:, 1:]

    def to_tensor(self, device="cpu", dtype=torch.int64):
        if len(self.subtask_type) == 1:
            type_id = self.type_intid()
            arg_id = self.arg_intid()
            tmap_id = self.target_map_intid()
            return torch.tensor([[type_id, arg_id, tmap_id]], device=device, dtype=dtype)
        else:
            type_id = self.subtask_type[:, None]
            arg_id = self.argument_vector.argmax(dim=1, keepdim=True) - 1
            tmap_id = self.target_map_type[:, None]
            return torch.cat([type_id, arg_id, tmap_id], dim=1)

    def type_str(self):
        if len(self.subtask_type) == 1:
            return self.subtask_type_intid_to_str(self.subtask_type[0].item())
        else:
            return [self.subtask_type_intid_to_str(i.item()) for i in self.subtask_type]

    def target_map_type_str(self):
        if len(self.target_map_type) == 1:
            return self.target_map_type_intid_to_str(self.target_map_type[0].item())
        else:
            return [self.target_map_type_intid_to_str(i.item()) for i in self.target_map_type]
    
    def arg_str(self):
        arg_ids = self.argument_vector.argmax(dim=1, keepdim=True) - 1
        if len(self.arg_ids) == 1:
            arg_intid = self.arg_intid()
            if arg_intid == -1:
                return "NIL"
            elif arg_intid >= NUM_OBJECT_TYPES:
                return "OutOfBounds"
            else:
                return ORDERED_OBJECT_TYPES[arg_intid]
        else:
            out = []
            arg_intids = [i.item() for i in arg_ids]
            for i in arg_intids:
                if i == -1:
                    out.append("NIL")
                elif i >= NUM_OBJECT_TYPES:
                    out.append("OutOfBounds")
                else:
                    out.append(ORDERED_OBJECT_TYPES[i])
            return out
    
    def type_oh(self):
        types = self.subtask_type
        oh = index_to_onehot(types, self.get_subtask_type_space_dim())
        return oh

    def arg_oh(self):
        return self.argument_vector

    def target_map_type_oh(self):
        tmap_types = self.target_map_type
        oh = index_to_onehot(tmap_types, self.get_target_map_type_space_dim())
        return oh

    def build_spatial_arg_proposal(
        self, 
        unshuffle_comb_map: torch.Tensor,
        walkthrough_comb_map: torch.Tensor,
    ):
        """
        comb_map: [sampler, map_channels, width, length, height]
                  channels: num_object_class + 1 (others) + 1 (occupancy) + 1 (observability)
                  2 types of semantic maps: Unshuffle / Walkthrough
        """
        ns, nc, w, l, h = unshuffle_comb_map.shape
        comb_map_stack = torch.stack((unshuffle_comb_map, walkthrough_comb_map), dim=0)
        selected_comb_map = comb_map_stack[self.target_map_type.tolist(), list(range(ns))]
        spatial_argument = torch.einsum(
            "bc,bcwlh->bwlh", self.argument_vector[:, 1:], selected_comb_map[:, :-3].type(self.argument_vector.dtype)
        )
        spatial_argument = spatial_argument[:, None, :, :, :]
        return spatial_argument
