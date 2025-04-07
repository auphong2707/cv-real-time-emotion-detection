# scripts/train_vgg16.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import *
from utils.train_utils import *
from models.vgg16 import *  # <-- Make sure you have this file

import constants

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VGG16] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_VGG16  # Add this to your constants
    EPOCHS = constants.EPOCHS_VGG16
    BATCH_SIZE = constants.BATCH_SIZE_VGG16
    IMAGE_SIZE = constants.IMAGE_SIZE_VGG16
    NUM_WORKERS = constants.NUM_WORKERS_VGG16
    LR = constants.LR_VGG16
    PRETRAINED = constants.PRETRAINED_VGG16
    FREEZE = constants.FREEZE_VGG16

    print("Hyperparameters and Constants:")
    print(f"   MODEL_NAME: {MODEL_NAME}")
    print(f"   EPOCHS: {EPOCHS}")
    print(f"   BATCH_SIZE: {BATCH_SIZE}")
    print(f"   IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"   NUM_WORKERS: {NUM_WORKERS}")
    print(f"   LR: {LR}")
    print(f"   PRETRAINED: {PRETRAINED}")
    print(f"   FREEZE: {FREEZE}")
    print(f"   DATA_DIR: {constants.DATA_DIR}")
    print(f"   SAVE_DIR: {constants.SAVE_DIR}")

    # ---------------------------
    # 2. Data Loading
    # ---------------------------
    print("Downloading data...")
    download_data()

    print("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        data_dir=constants.DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )

    # ---------------------------
    # 3. Model Creation
    # ---------------------------
    print("Creating model...")
    model = get_vgg16(
        num_classes=num_classes,
        pretrained=PRETRAINED,
        freeze=FREEZE
    )
    model.to(device)

    # ---------------------------
    # 4. Define Loss & Optimizer
    # ---------------------------
    print("Defining loss and optimizer...")
    criterion = nn.CrossEntropyLoss()
    if FREEZE:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------------------------
    # 5. Training Loop
    # ---------------------------
    print("Starting training loop...")
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        train_loss, train_acc = train_one_epoch(
            model, 
            tqdm(train_loader, desc="Training"),
            criterion, 
            optimizer, 
            device
        )
        
        val_loss, val_acc = validate(
            model, 
            tqdm(val_loader, desc="Validating"),
            criterion, 
            device
        )

        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val   Loss: {val_loss:.4f},   Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(constants.SAVE_DIR):
                os.makedirs(constants.SAVE_DIR)
            model_path = os.path.join(constants.SAVE_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"   > Saved best model to {model_path}")

if __name__ == "__main__":
    main()
