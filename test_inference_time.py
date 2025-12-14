import torch
import time
import numpy as np
from utils import factory
import argparse
import json
import os

def test_inference_time(args, model_path, input_shape=(3, 224, 224), num_runs=100):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # Initialize model using factory
    # model = factory.get_model(args["model_name"], args)._network
    
    # Load saved weights
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location = 'cpu')
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model.to(device)
    model.eval()
    
    # Create random test data
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # Ensure GPU computations are completed
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    
    print(f"Device: {device}")
    print(f"Model: {args['model_name']}")
    print(f"Backbone: {args['convnet_type']}")
    print(f"Input shape: {input_shape}")
    print(f"Average inference time: {mean_time:.2f} ms")
    print(f"Standard deviation: {std_time:.2f} ms")
    print(f"FPS: {1000/mean_time:.2f}")
    
    return mean_time, std_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"exps/POD_foster.json")
    parser.add_argument("--model_path", type=str, default=r"saved_pth\hrrp9\POD_foster\time_2025_04_09_17_32_28_cil_1993_M=500.pth")
    parser.add_argument("--input_shape", type=str, default="512")
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Run inference time test
    test_inference_time(
        args=config,
        model_path=args.model_path,
        input_shape=input_shape,
        num_runs=args.num_runs
    )