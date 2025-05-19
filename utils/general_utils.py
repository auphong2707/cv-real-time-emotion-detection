import os
import random
import time
import numpy as np
import torch
from utils.train_utils import validate
from models.efficientnet_b0 import get_efficientnet_b0
from models.mobilenetv2 import get_mobilenetv2
from models.vgg16 import get_vgg16
from models.emotion_cnn import EmotionCNN

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

def load_model(model_type, model_path, device):
    """
    Load a model from a state_dict checkpoint based on model_type.
    """
    try:
        if model_type == "efficientnet_b0":
            model = get_efficientnet_b0()  
        elif model_type == "mobilenetv2":
            model = get_mobilenetv2()
        elif model_type == "vgg16":
            model = get_vgg16()
        elif model_type == "custom":
            model = EmotionCNN(8)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        model.eval()
        model.to(device)

        if model_path is None:
            return model
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)