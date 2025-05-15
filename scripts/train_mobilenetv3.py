import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import set_seed, measure_fps
set_seed(42)

from utils.dataset import *
from utils.train_utils import *
from models.mobilenetv3 import *

import huggingface_hub
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import shutil
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler

import constants
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train MobileNetV3 model for emotion detection.")
parser.add_argument("--training_time_limit", type=int, default=39600, help="Training time limit in seconds (default: 39600 seconds or 11 hours).")
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MobileNetV3] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_MNV3
    EXPERIMENT_NAME = constants.EXPERIMENT_NAME_MNV3
    EPOCHS = constants.EPOCHS_MNV3
    BATCH_SIZE = constants.BATCH_SIZE_MNV3
    IMAGE_SIZE = constants.IMAGE_SIZE_MNV3
    NUM_WORKERS = constants.NUM_WORKERS_MNV3
    LR = constants.LR_MNV3
    WEIGHT_DECAY = constants.WEIGHT_DECAY_MNV3
    PRETRAINED = constants.PRETRAINED_MNV3
    FREEZE = constants.FREEZE_MNV3
    EXPERIMENT_SAVE_DIR = constants.SAVE_DIR + '/' + EXPERIMENT_NAME + '/'

    print("Hyperparameters and Constants:")
    print(f"   MODEL_NAME: {MODEL_NAME}")
    print(f"   EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    print(f"   EPOCHS: {EPOCHS}")
    print(f"   BATCH_SIZE: {BATCH_SIZE}")
    print(f"   IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"   NUM_WORKERS: {NUM_WORKERS}")
    print(f"   LR: {LR}")
    print(f"   WEIGHT_DECAY: {WEIGHT_DECAY}")
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
            "weight_decay": WEIGHT_DECAY,
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
    # Use the original call to get_data_loaders without passing transforms
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        data_dir=constants.DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )

    # Weighted sampling for class imbalance
    class_counts = torch.bincount(torch.tensor(train_loader.dataset.labels))
    total = sum(class_counts)
    class_weights = total / (len(class_counts) * class_counts.float())
    sample_weights = [class_weights[label] for label in train_loader.dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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

    # Unfreeze later blocks since FREEZE=True
    if FREEZE:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last few blocks (adjust indices based on MobileNetV3 structure)
        for name, param in model.named_parameters():
            if "features.9" in name or "features.10" in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Add dropout to classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[0].in_features, num_classes)
    )
    model.to(device)

    # ---------------------------
    # 5. Define Loss, Optimizer & Scheduler
    # ---------------------------
    print("Defining loss and optimizer...")
    # Use class weights in loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    # Optimizer for frozen layers
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Use CosineAnnealingWarmRestarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ---------------------------
    # 6. Training with Early Stopping
    # ---------------------------
    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)
    latest_ckpt = os.path.join(EXPERIMENT_SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', 0.0)
    else:
        print("Training from scratch.")
        start_epoch = 0
        best_metric = 0.0

    best_f1 = best_metric
    patience = 10
    counter = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_f1_per_class = f1_score(val_labels, val_preds, average=None, labels=range(num_classes))
        wandb.log({
            "val_f1": val_f1,
            "val_f1_per_class": val_f1_per_class,
            "epoch": epoch
        })

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_SAVE_DIR, f"{MODEL_NAME}_best.pth"))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_f1
        }, latest_ckpt)

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

    fps_gpu = None
    test_results = None
    if torch.cuda.is_available():
        fps_gpu, test_results = measure_fps(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device='cuda',
            experiment_save_dir=EXPERIMENT_SAVE_DIR,
            save_confusion_matrix=True
        )
        print(f"GPU FPS: {fps_gpu:.2f}")

    fps_cpu, test_results_cpu = measure_fps(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device='cpu',
        experiment_save_dir=EXPERIMENT_SAVE_DIR,
        save_confusion_matrix=False
    )
    print(f"CPU FPS: {fps_cpu:.2f}")

    final_test_results = test_results if test_results is not None else test_results_cpu

    model.to(device)

    wandb.log({
        "test_loss": final_test_results['loss'],
        "test_accuracy": final_test_results['accuracy'],
        "test_precision": final_test_results['precision'],
        "test_recall": final_test_results['recall'],
        "test_f1_score": final_test_results['f1_score'],
        "test_fps_gpu": fps_gpu if fps_gpu is not None else 0.0,
        "test_fps_cpu": fps_cpu,
    })

    print("Test Results:")
    print(f"   Loss: {final_test_results['loss']:.4f}")
    print(f"   Accuracy: {final_test_results['accuracy']:.2f}%")
    print(f"   Precision: {final_test_results['precision']:.4f}")
    print(f"   Recall: {final_test_results['recall']:.4f}")
    print(f"   F1 Score: {final_test_results['f1_score']:.4f}")
    print(f"   FPS GPU (Frames/sec): {fps_gpu:.2f}" if fps_gpu is not None else "   FPS GPU: N/A")
    print(f"   FPS CPU (Frames/sec): {fps_cpu:.2f}")

    with open(os.path.join(EXPERIMENT_SAVE_DIR, "results.txt"), "w") as f:
        f.write("Test Results:\n")
        f.write(f"   Loss: {final_test_results['loss']:.4f}\n")
        f.write(f"   Accuracy: {final_test_results['accuracy']:.2f}%\n")
        f.write(f"   Precision: {final_test_results['precision']:.4f}\n")
        f.write(f"   Recall: {final_test_results['recall']:.4f}\n")
        f.write(f"   F1 Score: {final_test_results['f1_score']:.4f}\n")
        f.write(f"   FPS GPU (Frames/sec): {fps_gpu:.2f}\n" if fps_gpu is not None else "   FPS GPU: N/A\n")
        f.write(f"   FPS CPU (Frames/sec): {fps_cpu:.2f}\n")

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