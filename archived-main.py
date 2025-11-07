import random
import multiprocessing as mp
from pathlib import Path

import torch

from helper2 import (
    get_devices,
    load_coco_ids,
    load_pil_images,
    set_seed,
    YOLOModel,
    RTDETRModel,
)
from constants import BATCH_SIZE, MODEL_CFG, SUBSET_SIZE


VERBOSE = False


set_seed()
device_list = get_devices()

