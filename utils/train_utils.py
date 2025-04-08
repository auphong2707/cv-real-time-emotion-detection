import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import wandb

def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

def validate(model, dataloader, criterion, device):
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

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    result = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


    return result

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
    eval_metrics="f1_score"
):
    print("Starting training loop...")
    best_metric = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # --- Training ---
        train_result = train_one_epoch(
            model,
            tqdm(train_loader, desc="Training"),
            criterion,
            optimizer,
            device
        )

        # --- Validation ---
        eval_result = validate(
            model,
            tqdm(val_loader, desc="Validating"),
            criterion,
            device
        )

        # --- Logging ---
        print(f"Epoch {epoch + 1} Summary:\n")
        print("Training Results:")
        print(f"{' - Loss:':<15} {train_result['loss']:.4f}")
        print(f"{' - Learning Rate:':<15} {train_result['learning_rate']:.6f}")
        print(f"{' - Gradient Norm:':<15} {train_result['gradient_norm']:.6f}")

        print("\nValidation Results:")
        print(f"{' - Loss:':<15} {eval_result['loss']:.4f}")
        print(f"{' - Accuracy:':<15} {eval_result['accuracy']:.2f}%")
        print(f"{' - Precision:':<15} {eval_result['precision']:.4f}")
        print(f"{' - Recall:':<15} {eval_result['recall']:.4f}")
        print(f"{' - F1 Score:':<15} {eval_result['f1_score']:.4f}")

        # --- WandB Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_result['loss'],
            "train/learning_rate": train_result['learning_rate'],
            "train/gradient_norm": train_result['gradient_norm'],
            "val/loss": eval_result['loss'],
            "val/accuracy": eval_result['accuracy'],
            "val/precision": eval_result['precision'],
            "val/recall": eval_result['recall'],
            "val/f1_score": eval_result['f1_score']
        })

        # --- Save best model ---
        if eval_result[eval_metrics] > best_metric:
            best_metric = eval_result[eval_metrics]

            print(f"New best {eval_metrics}: {best_metric:.4f}, saving model...")

            os.makedirs(SAVE_DIR, exist_ok=True)
            model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), model_path)

            print(f"> Saved best model to {model_path}")

    print("Training complete.")
