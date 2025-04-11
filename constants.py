# constants.py

#
# MobileNetV3 Constants
#
MODEL_NAME_MNV3 = "mobilenetv3"
EXPERIMENT_NAME_MNV3 = "mobilenetv3-experiment-1"
EPOCHS_MNV3 = 5
BATCH_SIZE_MNV3 = 256
IMAGE_SIZE_MNV3 = 224
NUM_WORKERS_MNV3 = 2
LR_MNV3 = 1e-4
PRETRAINED_MNV3 = True
FREEZE_MNV3 = True

#
# VGG16 Constants
#
MODEL_NAME_VGG16 = "vgg16"
EXPERIMENT_NAME_VGG16 = "vgg16-experiment-1"
EPOCHS_VGG16 = 100
BATCH_SIZE_VGG16 = 128
IMAGE_SIZE_VGG16 = 224
NUM_WORKERS_VGG16 = 2
LR_VGG16 = 1e-4
PRETRAINED_VGG16 = True
FREEZE_VGG16 = True

#
# EfficientNet-B0 Constants
#
MODEL_NAME_EFN_B0 = "efficientnet_b0"
EPOCHS_EFN_B0 = 12
BATCH_SIZE_EFN_B0 = 64
IMAGE_SIZE_EFN_B0 = 224
NUM_WORKERS_EFN_B0 = 4
LR_EFN_B0 = 5e-4
PRETRAINED_EFN_B0 = True
FREEZE_EFN_B0 = False

#
# ViT Constants
#
MODEL_NAME_VIT = "deit_base_patch16_224"
EPOCHS_VIT = 30
BATCH_SIZE_VIT = 128
IMAGE_SIZE_VIT = 224
NUM_WORKERS_VIT = 4
LR_VIT = 3e-5
PRETRAINED_VIT = True

#
# Shared Constants (Data Paths, etc.)
#
DATA_DIR = "data/"
SAVE_DIR = "results"
