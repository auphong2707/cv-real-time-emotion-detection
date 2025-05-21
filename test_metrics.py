import torch
import argparse
import torch.nn as nn

from utils.general_utils import load_model
from utils.dataset import get_data_loaders
from utils.train_utils import validate

from pprint import pprint
import os
import json

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

def main():
    parser = argparse.ArgumentParser(description="Test FPS of a PyTorch CV model.")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the model .pth/.pt file")
    parser.add_argument("--model_type", type=str, choices=["efficientnet_b0", "mobilenetv3", "vgg16", "custom"], required=True, help="Model architecture type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 224, 224],
                        help="Input shape as batch_size channels height width")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model_type} model from: {args.model_path}")
    model = load_model(args.model_type, args.model_path, device)

    _, _, test_loader, _  = get_data_loaders()

    result = validate(model, test_loader, nn.CrossEntropyLoss(), device)

    overall = {
        'loss': result['loss'],
        'accuracy': result['accuracy'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1_score': result['f1_score'],
    }

    metrics_per_class = {}
    for idx, label in ID2LABEL.items():
        metrics_per_class[label] = {
            'precision': result['precision_per_class'][idx],
            'recall':    result['recall_per_class'][idx],
            'f1_score':  result['f1_score_per_class'][idx],
        }

    print("=== Overall Metrics ===")
    pprint(overall)

    print("\n=== Per-Class Metrics ===")
    pprint(metrics_per_class)

    # Export results to file in results folder, include model name in file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else args.model_type
    results_path = os.path.join(results_dir, f"test_metrics_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump({
            "overall": overall,
            "per_class": metrics_per_class
        }, f, indent=4)
    print(f"\nResults exported to {results_path}")


if __name__ == "__main__":
    main()
