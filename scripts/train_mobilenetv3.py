# scripts/train_mobilenetv3.py

import os
import sys
import time
import huggingface_hub
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import *
from utils.train_utils import *
from models.mobilenetv3 import *

# Import model-specific and shared constants
import constants
import sys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MobileNetV3] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_MNV3
    EPOCHS = constants.EPOCHS_MNV3
    BATCH_SIZE = constants.BATCH_SIZE_MNV3
    IMAGE_SIZE = constants.IMAGE_SIZE_MNV3
    NUM_WORKERS = constants.NUM_WORKERS_MNV3
    LR = constants.LR_MNV3
    PRETRAINED = constants.PRETRAINED_MNV3
    FREEZE = constants.FREEZE_MNV3
    EXPERIMENT_NAME_MNV3 = constants.EXPERIMENT_NAME_MNV3
    EXPERIMENT_SAVE_DIR = constants.SAVE_DIR + '/' + EXPERIMENT_NAME_MNV3 + '/'

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
    print(f"   SAVE_DIR: {EXPERIMENT_SAVE_DIR}")

    # ---------------------------
    # 2. Set up API
    # ---------------------------
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(
        project=os.environ['WANDB_PROJECT'],
        name=EXPERIMENT_NAME_MNV3,
        config={
            "model_name": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LR,
            "pretrained": PRETRAINED,
            "freeze": FREEZE
        }
    )
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # ---------------------------
    # 3. Data Loading
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
    # 4. Model Creation
    # ---------------------------
    print("Creating model...")
    model = get_mobilenetv3(
        num_classes=num_classes,
        pretrained=PRETRAINED,
        freeze=FREEZE
    )
    model.to(device)

    # ---------------------------
    # 5. Define Loss & Optimizer
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
    # 6. Training Loop
    # ---------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        EPOCHS=EPOCHS,
        MODEL_NAME=MODEL_NAME,
        SAVE_DIR=EXPERIMENT_SAVE_DIR,
    )

    # ---------------------------
    # 7. Evaluation
    # ---------------------------
    print("Evaluating model on test set...")

    # --- Load the best model ---
    best_model_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{MODEL_NAME}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    # --- Set model to evaluation mode ---    
    model.eval()

    # --- Ensure accurate timing on GPU ---
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # --- Run evaluation ---
    test_results = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    # --- Stop timing ---
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    # --- Compute FPS ---
    elapsed_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    fps = num_samples / elapsed_time

    # --- Log to WandB ---
    wandb.log({
        "test_loss": test_results['loss'],
        "test_accuracy": test_results['accuracy'],
        "test_precision": test_results['precision'],
        "test_recall": test_results['recall'],
        "test_f1_score": test_results['f1_score'],
        "test_fps": fps
    })

    # --- Print Results ---
    print("Test Results:")
    print(f"   Loss: {test_results['loss']:.4f}")
    print(f"   Accuracy: {test_results['accuracy']:.2f}%")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall: {test_results['recall']:.4f}")
    print(f"   F1 Score: {test_results['f1_score']:.4f}")
    print(f"   FPS (Frames/sec): {fps:.2f}")
    print("Training and evaluation completed.")
    
    # ---------------------------
    # 8. Upload models to Hugging Face
    # ---------------------------
    print("Uploading model to Hugging Face...")
    api = huggingface_hub.HfApi()
    api.upload_large_folder(
        folder_path=constants.SAVE_DIR,
        repo_id="auphong2707/cv-real-time-emotion-detection",
        repo_type="model",
        private=False
    )
    print("Model uploaded to Hugging Face.")


if __name__ == "__main__":
    main()
