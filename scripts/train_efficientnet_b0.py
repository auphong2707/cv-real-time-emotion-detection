import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import set_seed, measure_fps
set_seed(42)

from utils.dataset import *
from utils.train_utils import *
from models.efficientnet_b0 import *

import huggingface_hub
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import shutil

import constants
import argparse

parser = argparse.ArgumentParser(description="Train EfficientNet-B0 model for emotion detection.")
parser.add_argument("--training_time_limit", type=int, default=39600, help="Training time limit in seconds (default: 39600 seconds or 11 hours).")
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EfficientNet-B0] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_EFFICIENTNET_B0
    EPOCHS = constants.EPOCHS_EFFICIENTNET_B0
    BATCH_SIZE = constants.BATCH_SIZE_EFFICIENTNET_B0
    IMAGE_SIZE = constants.IMAGE_SIZE_EFFICIENTNET_B0
    NUM_WORKERS = constants.NUM_WORKERS_EFFICIENTNET_B0
    LR = constants.LR_EFFICIENTNET_B0
    WEIGHT_DECAY = constants.WEIGHT_DECAY_EFFICIENTNET_B0
    PRETRAINED = constants.PRETRAINED_EFFICIENTNET_B0
    FREEZE = constants.FREEZE_EFFICIENTNET_B0
    EXPERIMENT_NAME = constants.EXPERIMENT_NAME_EFFICIENTNET_B0
    EXPERIMENT_SAVE_DIR = constants.SAVE_DIR + '/' + EXPERIMENT_NAME + '/'

    print("Hyperparameters and Constants:")
    print(f"   MODEL_NAME: {MODEL_NAME}")
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
    model = get_efficientnet_b0(
        num_classes=num_classes,
        pretrained=PRETRAINED,
        freeze=FREEZE
    )
    model.to(device)

    # ---------------------------
    # 5. Loss, Optimizer, Scheduler
    # ---------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()) if FREEZE else model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    def get_linear_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps=0):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


    # ----------------------------
    # 6. Checkpoint Loading
    # ----------------------------
    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)
    latest_ckpt = os.path.join(EXPERIMENT_SAVE_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', 0.0)
    else:
        print("Train from beginning.")
        start_epoch = 0
        best_metric = 0.0

    # ---------------------------
    # 7. Training Loop
    # ---------------------------
    finished_training = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        EPOCHS=EPOCHS,
        MODEL_NAME=MODEL_NAME,
        SAVE_DIR=EXPERIMENT_SAVE_DIR,
        start_epoch=start_epoch,
        best_metric=best_metric,
        training_time_limit=args.training_time_limit,
        scheduler=scheduler,
    )
    if not finished_training:
        shutil.rmtree(constants.DATA_DIR, ignore_errors=True)
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")
        return

    # ---------------------------
    # 8. Evaluation
    # ---------------------------
    best_model_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{MODEL_NAME}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.eval()

    # Measure FPS on GPU (if available)
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

    # Measure FPS on CPU
    fps_cpu, test_results_cpu = measure_fps(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device='cpu',
        experiment_save_dir=EXPERIMENT_SAVE_DIR,
        save_confusion_matrix=False
    )
    print(f"CPU FPS: {fps_cpu:.2f}")

    # Use GPU test results if available, else CPU results
    final_test_results = test_results if test_results is not None else test_results_cpu

    # Move model back to original device
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
    print("Training and evaluation completed.")

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
    huggingface_hub.HfApi().upload_large_folder(
        folder_path=constants.SAVE_DIR,
        repo_id="auphong2707/cv-real-time-emotion-detection",
        repo_type="model",
        private=False
    )
    print("✅ Upload complete.")

if __name__ == "__main__":
    main()