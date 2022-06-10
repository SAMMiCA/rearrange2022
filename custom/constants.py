from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS

UNKNOWN_OBJECT_STR = "Unknown"
ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))
NUM_OBJECT_TYPES = len(ORDERED_OBJECT_TYPES) + 1    # including "Others"
ADDITIONAL_MAP_CHANNELS = 3 # 1 for agent_pos_map, 1 for occupancy map 1 for observability map

IDX_TO_OBJECT_TYPE = {i: ot for i, ot in enumerate(ORDERED_OBJECT_TYPES)}
IDX_TO_OBJECT_TYPE[len(ORDERED_OBJECT_TYPES)] = UNKNOWN_OBJECT_STR

OBJECT_TYPES_TO_IDX = {v: k for k, v in IDX_TO_OBJECT_TYPE.items()}

IDX_TO_SUBTASK_TYPE = {
    0: "Explore",
    # 1: "Goto",
    1: "PickupObject",
    2: "PutObject",
    3: "OpenObject",
    4: "Stop",
}

SUBTASK_TYPES_TO_IDX = {v: k for k, v in IDX_TO_SUBTASK_TYPE.items()}
SUBTASK_TYPES = [IDX_TO_SUBTASK_TYPE[i] for i in range(len(IDX_TO_SUBTASK_TYPE))]
NUM_SUBTASK_TYPES = len(SUBTASK_TYPES)

NAV_SUBTASK_TYPES = [
    "Explore"
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
MAP_TYPES_TO_IDX = {v: k for k, v in IDX_TO_MAP_TYPE.items()}
MAP_TYPES = [IDX_TO_MAP_TYPE[i] for i in range(len(IDX_TO_MAP_TYPE))]
NUM_MAP_TYPES = len(MAP_TYPES)

OBJECTS_TO_BE_FILTERED = [
    "AluminumFoil", "Apple", "Bread", "Cloth", "HandTowel", "KeyChain", "LaundryHamper", "Lettuce", "Pillow", "Potato", "TableTopDecor", "Tomato", "Towel", 
]