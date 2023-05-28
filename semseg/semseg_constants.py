from collections import OrderedDict
import numpy as np
import os
import json

from task_aware_rearrange.constants import UNKNOWN_OBJECT_STR, ORDERED_OBJECT_TYPES, NUM_OBJECT_TYPES
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR


SEMSEG_DATA_DIR = os.path.join(
    ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
    "semseg_data"
)

PICKUPABLE_TO_COLOR = OrderedDict([
    ('Candle', (233, 102, 178)),
    ('SoapBottle', (168, 222, 137)),
    ('ToiletPaper', (162, 204, 152)),
    ('SoapBar', (43, 97, 155)),
    ('SprayBottle', (89, 126, 121)),
    ('TissueBox', (98, 43, 249)),
    ('DishSponge', (166, 58, 136)),
    ('PaperTowelRoll', (144, 173, 28)),
    ('Book', (43, 31, 148)),
    ('CreditCard', (56, 235, 12)),
    ('Dumbbell', (45, 57, 144)),
    ('Pen', (239, 130, 152)),
    ('Pencil', (177, 226, 23)),
    ('CellPhone', (227, 98, 136)),
    ('Laptop', (20, 107, 222)),
    ('CD', (65, 112, 172)),
    ('AlarmClock', (184, 20, 170)),
    ('Statue', (243, 75, 41)),
    ('Mug', (8, 94, 186)),
    ('Bowl', (209, 182, 193)),
    ('TableTopDecor', (126, 204, 158)),
    ('Box', (60, 252, 230)),
    ('RemoteControl', (187, 19, 208)),
    ('Vase', (83, 152, 69)),
    ('Watch', (242, 6, 88)),
    ('Newspaper', (19, 196, 2)),
    ('Plate', (188, 154, 128)),
    ('WateringCan', (147, 67, 249)),
    ('Fork', (54, 200, 25)),
    ('PepperShaker', (5, 204, 214)),
    ('Spoon', (235, 57, 90)),
    ('ButterKnife', (135, 147, 55)),
    ('Pot', (132, 237, 87)),
    ('SaltShaker', (36, 222, 26)),
    ('Cup', (35, 71, 130)),
    ('Spatula', (30, 98, 242)),
    ('WineBottle', (53, 130, 252)),
    ('Knife', (211, 157, 122)),
    ('Pan', (246, 212, 161)),
    ('Ladle', (174, 98, 216)),
    ('Egg', (240, 75, 163)),
    ('Kettle', (7, 83, 48)),
    ('Bottle', (64, 80, 115))
])

OPENABLE_TO_COLOR = OrderedDict([
    ('Drawer', (155, 30, 210)),
    ('Toilet', (21, 27, 163)),
    ('ShowerCurtain', (60, 12, 39)),
    ('ShowerDoor', (36, 253, 61)),
    ('Cabinet', (210, 149, 89)),
    ('Blinds', (214, 223, 197)),
    ('LaundryHamper', (35, 109, 26)),
    ('Safe', (198, 238, 160)),
    ('Microwave', (54, 96, 202)),
    ('Fridge', (91, 156, 207))
])

CLASS_TO_COLOR_ORIGINAL = OrderedDict(
    [(UNKNOWN_OBJECT_STR, (243, 246, 208))]
    + list(PICKUPABLE_TO_COLOR.items())
    + list(OPENABLE_TO_COLOR.items()))

EXTRA_PICKUPABLE_TO_COLOR = OrderedDict([
    ('Plunger', (74, 209, 56)),
    ('ScrubBrush', (222, 148, 80)),
    ('BasketBall', (97, 58, 36)),
    ('Footstool', (74, 187, 51)),
    ('TeddyBear', (229, 73, 134)),
    ('TennisRacket', (138, 71, 107)),
    ('BaseballBat', (171, 20, 38)),
    ('Boots', (121, 126, 101)),
    ('AluminumFoil', (181, 163, 89))
])

# Assign color to remaining object type
REMAINING_PICKUPABLE_TO_COLOR = OrderedDict([
    ('Apple', (235, 54, 22)),
    ('Bread', (166, 58, 136)),
    ('Cloth', (235, 222, 6)),
    ('HandTowel', (11, 51, 121)),
    ('KeyChain', (49, 156, 213)),
    ('Lettuce', (231, 52, 109)),
    ('Pillow', (55, 33, 114)),
    ('Potato', (48, 227, 158)),
    ('Tomato', (55, 33, 114)),
    ('Towel', (243, 75, 41)),
])


# Scripts for generating color maps
# REMAINING_PICKUPABLE_TO_COLOR = OrderedDict()
# # Check existing color maps for misc objects
# with open(os.path.join(ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR, 'data_2022.json'), 'r') as f:
#     data = json.load(f)

# misc_object_maps = {
#     k: tuple(v)
#     for k, v in data['miscs']['merged'].items()
# }
# misc_object_maps.update(CLASS_TO_COLOR)
# for obj in ORDERED_OBJECT_TYPES:
#     if (obj in CLASS_TO_COLOR):
#         continue
#     color = tuple(np.random.randint(0, 255, 3).tolist())
#     while color not in misc_object_maps.values():
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#     REMAINING_PICKUPABLE_TO_COLOR[obj] = color

CLASS_TO_COLOR = OrderedDict(
    list(CLASS_TO_COLOR_ORIGINAL.items())
    + list(EXTRA_PICKUPABLE_TO_COLOR.items())
    + list(REMAINING_PICKUPABLE_TO_COLOR.items()))

ID_MAP_COLOR_CLASS_TO_ORDERED_OBJECT_TYPE = [
    ORDERED_OBJECT_TYPES.index(color_class)
    for color_class in CLASS_TO_COLOR.keys()
]

ID_MAP_ORDERED_OBJECT_TYPE_TO_COLOR_CLASS = [
    list(CLASS_TO_COLOR.keys()).index(obj)
    for obj in ORDERED_OBJECT_TYPES
]

ORDERED_CLASS_TO_COLOR = OrderedDict(
    [
        (obj, CLASS_TO_COLOR[obj])
        for obj in ORDERED_OBJECT_TYPES
    ]
)