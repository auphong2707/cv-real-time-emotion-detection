import os
import torch
from torchvision import datasets, transforms
from utils.custom_tranformation import *
from huggingface_hub import HfApi
import zipfile

def download_data(data_dir="data"):
    """
    Downloads the FER-2013 dataset from Kaggle and places its contents directly into data_dir.
    """
    # 0. Check if the dataset is already downloaded
    if os.path.exists(data_dir):
        print(f"Data already exists in '{data_dir}'. Skipping download.")
        return
    
    # 1. Ensure data_dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 2. Download the dataset
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.hf_hub_download(
        repo_id="auphong2707/affect-net",
        filename="archive.zip",
        repo_type="dataset",
        revision="main",
        local_dir="./data",
    )

    # 3. Unzip the dataset
    archive_path = "./data/archive.zip"
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall("./data")

    print("Archive unzipped successfully.")

def get_data_loaders(
    data_dir="data/AffectNet",
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

        # Optional: Slight brightness/contrast/saturation changes
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # CLAHE and gamma correction (use your custom transforms here)
        CLAHEEqualization(clip_limit=4.0, tile_grid_size=(4, 4)),
        GammaCorrection(gamma=1.5),

        # Horizontal flip is safe and effective
        transforms.RandomHorizontalFlip(p=0.5),

        # Small rotation, shift, scaling, and shearing
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5
        ),

        # Slight blur to simulate different image sharpness
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

        # Random erasing simulates occlusion (like glasses, hands)
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.01, 0.05),   # smaller min and max scale
            ratio=(0.5, 2.0)      # keep it roughly square to rectangular
        )
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
