import os
import torch
import time
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

ID2LABEL = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprise"
}

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    total = 0

    epoch_lr = 0.0
    epoch_grad_norm = 0.0

    for step, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # --- Compute gradient norm for this batch ---
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        epoch_grad_norm = total_norm ** 0.5  # from the last batch

        optimizer.step()
        scheduler.step()

        # --- Get learning rate for this batch ---
        epoch_lr = optimizer.param_groups[0]['lr']

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    return {
        'loss': epoch_loss,
        'learning_rate': epoch_lr,
        'gradient_norm': epoch_grad_norm
    }

def validate(model, dataloader, criterion, device, confusion_matrix_save_path=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    # Overall metrics (weighted average)
    precision_overall = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_overall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_overall = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    result = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision_overall,
        'recall': recall_overall,
        'f1_score': f1_overall,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_score_per_class': f1_per_class.tolist()
    }

    # Optionally save confusion matrix
    if confusion_matrix_save_path:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[ID2LABEL[i] for i in range(len(ID2LABEL))], yticklabels=[ID2LABEL[i] for i in range(len(ID2LABEL))])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(confusion_matrix_save_path)
        plt.close()

    return result

def save_checkpoint(state, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, filename)

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    EPOCHS,
    MODEL_NAME,
    SAVE_DIR,
    eval_metrics="f1_score",  # can be: "f1_score", "precision", "recall"
    start_epoch=0,
    best_metric=0.0,
    training_time_limit=41400,
    scheduler=None,
):
    print("Starting training loop...")
    # Track the start time of training
    start_time = time.time()

    best_metric = 0.0

    for epoch in range(start_epoch, EPOCHS):
        # Check if the training time limit has been reached
        if time.time() - start_time > training_time_limit:
            return False
        
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # --- Training ---
        train_result = train_one_epoch(
            model,
            tqdm(train_loader, desc="Training"),
            criterion,
            optimizer,
            scheduler,
            device,
        )

        # --- Validation ---
        eval_result = validate(
            model,
            tqdm(val_loader, desc="Validating"),
            criterion,
            device
        )

        # --- Step the scheduler if provided ---
        if scheduler is not None:
            scheduler.step()

        # --- Logging ---
        print(f"Epoch {epoch + 1} Summary:\n")
        print("Training Results:")
        print(f"{' - Loss:':<20} {train_result['loss']:.4f}")
        print(f"{' - Learning Rate:':<20} {train_result['learning_rate']:.6f}")
        print(f"{' - Gradient Norm:':<20} {train_result['gradient_norm']:.6f}")

        print("\nValidation Results (Overall):")
        print(f"{' - Loss:':<20} {eval_result['loss']:.4f}")
        print(f"{' - Accuracy:':<20} {eval_result['accuracy']:.2f}%")
        print(f"{' - Precision:':<20} {eval_result['precision']:.4f}")
        print(f"{' - Recall:':<20} {eval_result['recall']:.4f}")
        print(f"{' - F1 Score:':<20} {eval_result['f1_score']:.4f}")

        print("\nValidation Results (Per-Class):")
        for i, (p, r, f) in enumerate(zip(
            eval_result['precision_per_class'],
            eval_result['recall_per_class'],
            eval_result['f1_score_per_class']
        )):
            print(f"  Class {ID2LABEL[i]}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

        # --- WandB Logging ---
        log_data = {
            "epoch": epoch + 1,
            "train/loss": train_result['loss'],
            "train/learning_rate": train_result['learning_rate'],
            "train/gradient_norm": train_result['gradient_norm'],
            "val/loss": eval_result['loss'],
            "val/accuracy": eval_result['accuracy'],
            "val/precision": eval_result['precision'],
            "val/recall": eval_result['recall'],
            "val/f1_score": eval_result['f1_score'],
        }

        for i, (p, r, f) in enumerate(zip(
            eval_result['precision_per_class'],
            eval_result['recall_per_class'],
            eval_result['f1_score_per_class']
        )):
            log_data[f"val/precision_class_{ID2LABEL[i]}"] = p
            log_data[f"val/recall_class_{ID2LABEL[i]}"] = r
            log_data[f"val/f1_score_class_{ID2LABEL[i]}"] = f

        wandb.log(log_data)

        # --- Save best model based on overall metric ---
        if eval_result[eval_metrics] > best_metric:
            best_metric = eval_result[eval_metrics]

            print(f"\nNew best {eval_metrics}: {best_metric:.4f}, saving model...")

            os.makedirs(SAVE_DIR, exist_ok=True)
            model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), model_path)

            print(f"> Saved best model to {model_path}")

        # --- Save checkpoint every epoch ---
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
        }
        save_checkpoint(checkpoint, SAVE_DIR)

    print("Training complete.")

    return True