import os
import shutil
import torch
from torchvision import datasets, transforms
import kagglehub
from utils.custom_tranformation import *

def download_data(data_dir="data"):
    """
    Downloads the FER-2013 dataset from Kaggle and places its contents directly into data_dir.
    """
    # 0. Check if the dataset is already downloaded
    if os.path.exists(data_dir):
        print(f"Data already exists in '{data_dir}'. Skipping download.")
        return
    # 1. Download the dataset
    path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
    # For example, path might end up as "...some_temp_dir/1" after download/unzip.

    # 2. Ensure data_dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 3. Move everything inside path -> data_dir
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        
        if os.path.isdir(src):
            # Recursively copy directories
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            # Copy individual files
            shutil.copy2(src, dst)
    
    # 4. Remove the now-empty folder "1"
    # os.rmdir(path)

    print(f"Data downloaded and placed in '{data_dir}'")

def split_data(data_dir="data", val_split=0.2):
    """
    Splits the data in data_dir into train and val folders.
    """
    # 0. Check if the dataset is already split
    if os.path.exists(os.path.join(data_dir, "val")):
        print(f"Validation data already exists in '{data_dir}/val'. Skipping split.")
        return

    # 1. Ensure data_dir exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    # 2. Create val folder
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 3. Read the data from train folder and split
    for class_name in os.listdir(os.path.join(data_dir, "train")):
        class_dir = os.path.join(data_dir, "train", class_name)
        files = os.listdir(class_dir)
        split_idx = int(len(files) * val_split)
        val_files = files[:split_idx]

        # Move val files to val folder
        for file in val_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(val_dir, class_name, file)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            shutil.move(src, dst)

    print(f"Data split into train and val folders in '{data_dir}'")

def get_data_loaders(
    data_dir="data",
    batch_size=64,
    image_size=224,
    num_workers=os.cpu_count(),
):
    """
    Returns PyTorch DataLoaders for training, validation, and test sets.
    Expects a folder structure like:
      data/train/<class_name>/*.png
      data/val/<class_name>/*.png
      data/test/<class_name>/*.png
    """

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        CLAHEEqualization(clip_limit=4.0, tile_grid_size=(4, 4)),  # CLAHE,
        GammaCorrection(gamma=1.5),  # Gamma correction
        transforms.RandomHorizontalFlip(p=0.5),  # Flip face left-right (safe for emotion/ID)
        transforms.RandomRotation(degrees=10),   # Slight head tilt
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),              # Slight shift
            scale=(0.95, 1.05),                  # Minor scaling
            shear=5                              # Slight slant
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])       # Grayscale normalization
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("Data loaders created:")
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    print(f"   Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader, test_loader, len(train_dataset.classes)
