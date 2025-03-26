# constants.py

#
# MobileNetV3 Constants
#
MODEL_NAME_MNV3 = "mobilenetv3"
EPOCHS_MNV3 = 10
BATCH_SIZE_MNV3 = 64
IMAGE_SIZE_MNV3 = 224
NUM_WORKERS_MNV3 = 2
LR_MNV3 = 1e-3
PRETRAINED_MNV3 = True
FREEZE_MNV3 = False

#
# VGG16 Constants
#
MODEL_NAME_VGG16 = "vgg16"
EPOCHS_VGG16 = 8
BATCH_SIZE_VGG16 = 32
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
# Shared Constants (Data Paths, etc.)
#
DATA_DIR = "data"
SAVE_DIR = "checkpoints"
