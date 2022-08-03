import enum
from typing import Dict, Union, Optional
import stringcase
from task_aware_rearrange.constants import IDX_TO_OBJECT_TYPE, OBJECT_TYPE_TO_IDX, PICKUPABLE_OBJECTS, OPENABLE_OBJECTS, UNKNOWN_OBJECT_STR

"""
Subtasks
Explore (1): To explore to find out the difference between the semantic maps.
Goto(Find) (2 x Pickupable + 1 x Openable): To move to the designated object type with given map masks.
PickupObject (1 x Pickupable): To pickup a given object in the "Unshuffle" environment.
PutObject (1 x Pickupable): To put down a holding object on the target place. 
OpenObject (1 x Openable): To re-open an object with given object type.
Stop (1): To stop an episode and call next task.
"""
IDX_TO_SUBTASK_TYPE = {
    0: "Explore",
    1: "Goto",
    2: "PickupObject",
    3: "PutObject",
    4: "OpenObject",
    5: "Stop",
}

SUBTASK_TYPE_TO_IDX = {v: k for k, v in IDX_TO_SUBTASK_TYPE.items()}
SUBTASK_TYPES = [IDX_TO_SUBTASK_TYPE[i] for i in range(len(IDX_TO_SUBTASK_TYPE))]
NUM_SUBTASK_TYPES = len(SUBTASK_TYPES)

NAV_SUBTASK_TYPES = [
    "Explore",
    "Goto"
]

INTERACT_SUBTASK_TYPES = [
    # "Goto",
    "PickupObject",
    "PutObject",
    "OpenObject",
    # "CloseObject"     # NOT required as open action close the object and re-open as it is unshuffled
]

IDX_TO_MAP_TYPE = {
    0: "Unshuffle",
    1: "Walkthrough"
}
MAP_TYPE_TO_IDX = {v: k for k, v in IDX_TO_MAP_TYPE.items()}
MAP_TYPES = [IDX_TO_MAP_TYPE[i] for i in range(len(IDX_TO_MAP_TYPE))]
NUM_MAP_TYPES = len(MAP_TYPES)

SUBTASKS_EXPLORE = (
    ("Explore", None, None),
)
SUBTASKS_GOTO = (
    tuple(
        ("Goto", pickupable, target_map)
        for target_map in MAP_TYPES
        for pickupable in sorted(PICKUPABLE_OBJECTS)
    ) + 
    tuple(
        ("Goto", openable, "Unshuffle")
        for openable in sorted(OPENABLE_OBJECTS)
    )
)
SUBTASKS_PICKUP = tuple(
    ("PickupObject", pickupable, None)
    for pickupable in sorted(PICKUPABLE_OBJECTS)
)
SUBTASKS_PUT = tuple(
    ("PutObject", pickupable, None)
    for pickupable in sorted(PICKUPABLE_OBJECTS)
)
SUBTASKS_OPEN = tuple(
    ("OpenObject", openable, None)
    for openable in sorted(OPENABLE_OBJECTS)
)
SUBTASKS_STOP = (
    ("Stop", None, None),
)
SUBTASKS = (
    SUBTASKS_EXPLORE
    + SUBTASKS_GOTO
    + SUBTASKS_PICKUP
    + SUBTASKS_PUT
    + SUBTASKS_OPEN
    + SUBTASKS_STOP
)
IDX_TO_SUBTASK = {i: subtask for i, subtask in enumerate(SUBTASKS)}
SUBTASK_TO_IDX = {v: k for k, v in IDX_TO_SUBTASK.items()}
SUBTASK_TYPE_TO_SUBTASKS = {
    "Explore": SUBTASKS_EXPLORE,
    "Goto": SUBTASKS_GOTO,
    "PickupObject": SUBTASKS_PICKUP,
    "PutObject": SUBTASKS_PUT,
    "OpenObject": SUBTASKS_OPEN,
    "Stop": SUBTASKS_STOP,
}
NUM_SUBTASKS = len(SUBTASKS)

SUBTASK_TARGET_OBJECTS = (
    (UNKNOWN_OBJECT_STR,)
    + tuple(
        f"{pickupable}_{target_map}"
        for target_map in MAP_TYPES
        for pickupable in PICKUPABLE_OBJECTS
    ) + tuple(
        f"{openable}_Unshuffle"
        for openable in OPENABLE_OBJECTS
    )
)
IDX_TO_SUBTASK_TARGET_OBJECT = {i: target_object for i, target_object in enumerate(SUBTASK_TARGET_OBJECTS)}
SUBTASK_TARGET_OBJECT_TO_IDX = {v: k for k, v in IDX_TO_SUBTASK_TARGET_OBJECT.items()}

SUBTASK_TARGET_OBJECT_TO_OBJECT_TYPE = {
    target_object: target_object.split('_')[0] 
    for target_object in SUBTASK_TARGET_OBJECTS
}
SUBTASK_TARGET_OBJECT_TO_OBJECT_TYPE_IDX = {
    target_object: OBJECT_TYPE_TO_IDX[target_object.split('_')[0]]
    for target_object in SUBTASK_TARGET_OBJECTS
}
NUM_SUBTASK_TARGET_OBJECTS = len(SUBTASK_TARGET_OBJECTS)


class Subtask:
    def __init__(
        self, 
        subtask_type: Optional[Union[str, int]] = None, 
        object_type: Optional[Union[str, int]] = None, 
        target_map: Optional[Union[str, int]] = None,
        max_steps_for_subtask: int = 50,
    ):
        self.subtask_type = self.set_subtask_type(subtask_type)
        self.object_type = self.set_object_type(object_type)
        self.target_map = self.set_target_map(target_map)

        self.subtask_count = 0
        self.max_steps = max_steps_for_subtask

    def subtask_type_str(self, subtask_type: Optional[Union[int, str]]):
        if subtask_type is not None:
            assert isinstance(subtask_type, (int, str)), f"Invalid variable type for type: {type(subtask_type)}, it should be 'int' or 'str'."
        if isinstance(subtask_type, int):
            subtask_type = self.type_idx_to_str(subtask_type)
        return subtask_type

    def object_type_str(self, obj_type: Optional[Union[int, str]]):
        if obj_type is not None:
            assert isinstance(obj_type, (int, str)), f"Invalid variable type for type: {type(obj_type)}, it should be 'int' or 'str'."
        if isinstance(obj_type, int):
            obj_type = self.object_idx_to_str(obj_type)
        return obj_type

    def target_map_str(self, target_map: Optional[Union[int, str]]):
        if target_map is not None:
            assert isinstance(target_map, (int, str)), f"Invalid variable type for type: {type(target_map)}, it should be 'int' or 'str'."
        if isinstance(target_map, int):
            target_map = self.target_map_idx_to_str(target_map)
        return target_map

    @staticmethod
    def type_str_to_idx(type_str: str):
        assert type_str is not None and isinstance(type_str, str)
        return SUBTASK_TYPE_TO_IDX[type_str]

    @staticmethod
    def type_idx_to_str(type_idx: int):
        assert type_idx is not None and isinstance(type_idx, int)
        return IDX_TO_SUBTASK_TYPE[type_idx]

    @staticmethod
    def object_str_to_idx(obj_str: str):
        assert obj_str is not None and isinstance(obj_str, str)
        return OBJECT_TYPE_TO_IDX[obj_str]

    @staticmethod
    def object_idx_to_str(obj_idx: int):
        assert obj_idx is not None and isinstance(obj_idx, int)
        return IDX_TO_OBJECT_TYPE[obj_idx]

    @staticmethod
    def target_map_str_to_idx(tmap_str: str):
        assert tmap_str is not None and isinstance(tmap_str, str)
        return MAP_TYPE_TO_IDX[tmap_str]

    @staticmethod
    def target_map_idx_to_str(tmap_idx: int):
        assert tmap_idx is not None and isinstance(tmap_idx, int)
        return IDX_TO_MAP_TYPE

    def get_subtask_idx(self):
        if self.subtask_type is None:
            return None
        subtask_tuple = (self.subtask_type, self.object_type, self.target_map)
        return SUBTASK_TO_IDX[subtask_tuple]

    def is_interact_subtask(self):
        if self.subtask_type in INTERACT_SUBTASK_TYPES:
            return True
        
        return False

    def reset(self):
        self.subtask_type = None
        self.object_type = None
        self.target_map = None
        self.subtask_count = 0

    def get_expert_action_str(self):
        assert self.is_interact_subtask()
        if self.subtask_type == "PickupObject":
            return f"pickup_{stringcase.snakecase(self.object_type)}"
        elif self.subtask_type == "OpenObject":
            return f"open_by_type_{stringcase.snakecase(self.object_type)}"
        else:
            return f"drop_held_object_with_snap"
        
    def set_subtask_type(self, subtask_type: Optional[Union[int, str]]):
        self.subtask_type = self.subtask_type_str(subtask_type)

    def set_object_type(self, obj_type: Optional[Union[int, str]]):
        self.object_type = self.object_type_str(obj_type)

    def set_target_map(self, target_map: Optional[Union[int, str]]):
        self.target_map = self.target_map_str(target_map)

    def set_subtask(
        self, 
        subtask_type: Union[int, str], 
        obj_type: Optional[Union[int, str]] = None, 
        target_map: Optional[Union[int, str]] = None
    ):
        if self.subtask_type_str(subtask_type) != self.subtask_type:
            self.subtask_count = 0
        self.set_subtask_type(subtask_type)
        self.set_object_type(obj_type)
        self.set_target_map(target_map)

    def next_subtask(
        self, 
        obj_type: Optional[Union[int, str]] = None, 
        target_map: Optional[Union[int, str]] = None, 
        held_object: Optional[Dict] = None,
    ):
        if self.subtask_type == "Explore":
            if self.subtask_count > self.max_steps:
                self.set_subtask("Stop", None, None)
            else:
                assert obj_type is not None and target_map is not None
                self.set_subtask("Goto", obj_type, target_map)

        elif self.subtask_type == "Goto":
            if self.object_type in PICKUPABLE_OBJECTS:
                if held_object:
                    self.set_subtask("PutObject", held_object["objectType"], None)
                else:
                    self.set_subtask("PickupObject", self.object_type, None)
            elif self.object_type in OPENABLE_OBJECTS:
                self.set_subtask("OpenObject", self.object_type, None)
        
        elif self.subtask_type == "PickupObject":
            self.set_subtask("Goto", self.object_type, "Walkthrough")

        elif self.subtask_type in ("PutObject", "OpenObject"):
            if obj_type is None and target_map is None:
                self.set_subtask("Explore", None, None)
            else:
                self.set_subtask("Goto", obj_type, target_map)
        
        else:
            raise NotImplementedError(f"Not Implement Subtask Type...")

    def __str__(self) -> str:
        return f"{(self.subtask_type, self.object_type, self.target_map)}"