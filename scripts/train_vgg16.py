# scripts/train_vgg16.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import set_seed
set_seed(42)

from utils.dataset import *
from utils.train_utils import *
from models.vgg16 import *

import os
import time
import huggingface_hub
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import constants

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VGG16] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_VGG16
    EPOCHS = constants.EPOCHS_VGG16
    BATCH_SIZE = constants.BATCH_SIZE_VGG16
    IMAGE_SIZE = constants.IMAGE_SIZE_VGG16
    NUM_WORKERS = constants.NUM_WORKERS_VGG16
    LR = constants.LR_VGG16
    PRETRAINED = constants.PRETRAINED_VGG16
    FREEZE = constants.FREEZE_VGG16
    EXPERIMENT_NAME = constants.EXPERIMENT_NAME_VGG16
    EXPERIMENT_SAVE_DIR = constants.SAVE_DIR + '/' + EXPERIMENT_NAME + '/'

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
        name=EXPERIMENT_NAME,
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
    model = get_vgg16(
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
    best_model_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{MODEL_NAME}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.eval()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    test_results = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        confusion_matrix_save_path=os.path.join(EXPERIMENT_SAVE_DIR, "confusion_matrix.png"),
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    fps = num_samples / elapsed_time

    wandb.log({
        "test_loss": test_results['loss'],
        "test_accuracy": test_results['accuracy'],
        "test_precision": test_results['precision'],
        "test_recall": test_results['recall'],
        "test_f1_score": test_results['f1_score'],
        "test_fps": fps
    })

    print("Test Results:")
    print(f"   Loss: {test_results['loss']:.4f}")
    print(f"   Accuracy: {test_results['accuracy']:.2f}%")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall: {test_results['recall']:.4f}")
    print(f"   F1 Score: {test_results['f1_score']:.4f}")
    print(f"   FPS (Frames/sec): {fps:.2f}")
    print("Training and evaluation completed.")

    results_file = os.path.join(EXPERIMENT_SAVE_DIR, "results.txt")
    with open(results_file, "w") as f:
        f.write("Test Results:\n")
        f.write(f"   Loss: {test_results['loss']:.4f}\n")
        f.write(f"   Accuracy: {test_results['accuracy']:.2f}%\n")
        f.write(f"   Precision: {test_results['precision']:.4f}\n")
        f.write(f"   Recall: {test_results['recall']:.4f}\n")
        f.write(f"   F1 Score: {test_results['f1_score']:.4f}\n")
        f.write(f"   FPS (Frames/sec): {fps:.2f}\n")

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
