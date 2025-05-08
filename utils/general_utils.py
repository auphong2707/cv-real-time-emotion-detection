import os
import random
import time
import numpy as np
import torch
from train_utils import validate

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def measure_fps(model, test_loader, criterion, device, experiment_save_dir, save_confusion_matrix=True):
    # Ensure device is a torch.device object
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    # Warm-up run to avoid initialization overhead
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            model(inputs)
            break  # One batch is enough for warm-up

    # Measure FPS
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()  # Use perf_counter for higher precision
    test_results = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        confusion_matrix_save_path=os.path.join(experiment_save_dir, "confusion_matrix.png") if save_confusion_matrix else None,
    )
    
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    fps = num_samples / elapsed_time
    
    return fps, test_results
