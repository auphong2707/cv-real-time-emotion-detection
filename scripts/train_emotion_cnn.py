import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import set_seed, measure_fps
set_seed(42)

from utils.dataset import *
from utils.train_utils import *
from models.emotion_cnn import EmotionCNN
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import shutil
import huggingface_hub
import argparse
import constants

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train EmotionCNN model for emotion detection.")
parser.add_argument("--training_time_limit", type=int, default=39600, help="Training time limit in seconds (default: 39600 seconds or 11 hours).")
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EmotionCNN] Using device: {device}")

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    MODEL_NAME = constants.MODEL_NAME_EMOTIONCNN
    EPOCHS = constants.EPOCHS_EMOTIONCNN
    BATCH_SIZE = constants.BATCH_SIZE_EMOTIONCNN
    IMAGE_SIZE = constants.IMAGE_SIZE_EMOTIONCNN
    NUM_WORKERS = constants.NUM_WORKERS_EMOTIONCNN
    LR = constants.LR_EMOTIONCNN
    WEIGHT_DECAY = constants.WEIGHT_DECAY_EMOTIONCNN
    PRETRAINED = constants.PRETRAINED_EMOTIONCNN
    FREEZE = constants.FREEZE_EMOTIONCNN
    EXPERIMENT_NAME = constants.EXPERIMENT_NAME_EMOTIONCNN
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
    model = EmotionCNN(num_classes=num_classes, in_channels=3, image_size=IMAGE_SIZE)
    model.to(device)

    # ---------------------------
    # 5. Define Loss, Optimizer & Scheduler
    # ---------------------------
    print("Defining loss and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Define a Hugging Face-style linear scheduler with warmup
    def get_linear_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps=0):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Total number of training steps
    total_steps = EPOCHS * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_training_steps=total_steps
    )

    # ----------------------------
    # 6. Checkpoint Loading
    # ----------------------------
    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

    def find_latest_checkpoint(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        return None
    
    latest_ckpt = find_latest_checkpoint(EXPERIMENT_SAVE_DIR)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
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
        print("Deleting raw data to save space...")
        shutil.rmtree(constants.DATA_DIR)
        # Delete temporary directory
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")
        print("Training stopped before completion due to time limit. Exiting training...")
        return
    
    # ---------------------------
    # 8. Evaluation
    # ---------------------------
    print("Evaluating model on test set...")
    best_model_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{MODEL_NAME}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.eval()
    
    # Use GPU test results if available, else CPU results
    test_result = validate(model, test_loader, criterion, device, confusion_matrix_save_path=os.path.join(EXPERIMENT_SAVE_DIR, "confusion_matrix.png"))

    wandb.log({
        "test_loss": test_result['loss'],
        "test_accuracy": test_result['accuracy'],
        "test_precision": test_result['precision'],
        "test_recall": test_result['recall'],
        "test_f1_score": test_result['f1_score'],
    })

    metrics_per_class = {}
    for idx, label in ID2LABEL.items():
        wandb.log({
            f"test_precision_{label}": test_result['precision_per_class'][idx],
            f"test_recall_{label}":    test_result['recall_per_class'][idx],
            f"test_f1_score_{label}":  test_result['f1_score_per_class'][idx],
        })

        metrics_per_class[label] = {
            'precision': test_result['precision_per_class'][idx],
            'recall':    test_result['recall_per_class'][idx],
            'f1_score':  test_result['f1_score_per_class'][idx],
        }

    print("Test Results:")

    print("=== Overall Metrics ===")
    print(f"   Loss: {test_result['loss']:.4f}")
    print(f"   Accuracy: {test_result['accuracy']:.2f}%")
    print(f"   Precision: {test_result['precision']:.4f}")
    print(f"   Recall: {test_result['recall']:.4f}")
    print(f"   F1 Score: {test_result['f1_score']:.4f}")
    
    print("\n=== Per-Class Metrics ===")
    for label, metrics in metrics_per_class.items():
        print(f"   {label}:")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
        print(f"      F1 Score:  {metrics['f1_score']:.4f}")

    print("Training and evaluation completed.")

    with open(os.path.join(EXPERIMENT_SAVE_DIR, "results.txt"), "w") as f:
        f.write("Test Results:\n")

        f.write("=== Overall Metrics ===\n")
        f.write(f"   Loss: {test_result['loss']:.4f}\n")
        f.write(f"   Accuracy: {test_result['accuracy']:.2f}%\n")
        f.write(f"   Precision: {test_result['precision']:.4f}\n")
        f.write(f"   Recall: {test_result['recall']:.4f}\n")
        f.write(f"   F1 Score: {test_result['f1_score']:.4f}\n")

        f.write("\n=== Per-Class Metrics ===\n")
        for label, metrics in metrics_per_class.items():
            f.write(f"   {label}:\n")
            f.write(f"      Precision: {metrics['precision']:.4f}\n")
            f.write(f"      Recall:    {metrics['recall']:.4f}\n")
            f.write(f"      F1 Score:  {metrics['f1_score']:.4f}\n")

    print(f"Results saved to {os.path.join(EXPERIMENT_SAVE_DIR, 'results.txt')}")

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