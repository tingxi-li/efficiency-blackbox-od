import json

with open("constants.json") as f:
    constants = json.load(f)

RANDOM_SEED = constants["WORLD_CFG"]["RANDOM_SEED"]
NGPUS = constants["WORLD_CFG"]["NGPUS"]
COCO_ROOT = constants["WORLD_CFG"]["COCO_ROOT"]
DIR_TO_COCO_IMG = COCO_ROOT + constants["WORLD_CFG"]["DIR_TO_COCO_IMG"]
DIR_TO_COCO_ANN = COCO_ROOT + constants["WORLD_CFG"]["DIR_TO_COCO_ANN"]

BATCH_SIZE = constants["INFERENCE_CFG"]["BATCH_SIZE"]
SUBSET_SIZE = constants["INFERENCE_CFG"]["SUBSET_SIZE"]

MODEL_CFG = constants["MODEL_CFG"]