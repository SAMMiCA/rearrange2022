from rearrange.constants import OPENABLE_OBJECTS, PICKUPABLE_OBJECTS

UNKNOWN_OBJECT_STR = "Unknown"
ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))
ORDERED_PICKUPABLES = [obj for obj in ORDERED_OBJECT_TYPES if obj in PICKUPABLE_OBJECTS]
ORDERED_OPENABLES = [obj for obj in ORDERED_OBJECT_TYPES if obj in OPENABLE_OBJECTS]
NUM_OBJECT_TYPES = len(ORDERED_OBJECT_TYPES) + 1    # including "Others"
ADDITIONAL_MAP_CHANNELS = 3 # 1 for agent_pos_map, 1 for occupancy map 1 for observability map

IDX_TO_OBJECT_TYPE = {i: ot for i, ot in enumerate(ORDERED_OBJECT_TYPES)}
IDX_TO_OBJECT_TYPE[len(ORDERED_OBJECT_TYPES)] = UNKNOWN_OBJECT_STR

OBJECT_TYPE_TO_IDX = {v: k for k, v in IDX_TO_OBJECT_TYPE.items()}

OBJECTS_TO_BE_FILTERED = [
    "AluminumFoil", "Apple", "Bread", "Cloth", "HandTowel", "KeyChain", "LaundryHamper", "Lettuce", "Pillow", "Potato", "TableTopDecor", "Tomato", "Towel", 
]

NUM_PICKUPABLE_OBJECTS = len(PICKUPABLE_OBJECTS)
NUM_OPENABLE_OBJECTS = len(OPENABLE_OBJECTS)
