import argparse
import time
import torch
from utils.general_utils import load_model
import os
import json

def measure_fps(model, input_shape, device, iterations=100):
    dummy_input = torch.randn(*input_shape).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    for _ in range(iterations):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - start_time
    return iterations / total_time

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

    print(f"Measuring FPS with input shape: {args.input_shape}")
    fps = measure_fps(model, args.input_shape, device, args.iterations)

    print(f"[RESULT] {args.model_type} FPS: {fps:.2f} frames/second")

    # Export result to file with model name
    os.makedirs("results", exist_ok=True)
    model_base = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else args.model_type
    result_filename = f"results/test_fps_{model_base}.json"
    with open(result_filename, "w") as f:
        json.dump({"model": args.model_type, "model_file": model_base, "fps": round(fps, 2)}, f, indent=2)

if __name__ == "__main__":
    main()


# EfficientNet-B0
## python test_fps.py --model_type efficientnet_b0 --model_path deploy_models/efficientnet_b0_best.pth

# MobileNetV3
## python test_fps.py --model_type mobilenetv3 --model_path deploy_models/mobilenetv3_best.pth

# VGG16
## python test_fps.py --model_type vgg16 --model_path deploy_models/vgg16_best.pth

# EmotionCNN
## python test_fps.py --model_type custom --model_path deploy_models/emotioncnn_best.pth
