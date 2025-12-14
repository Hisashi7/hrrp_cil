import torch
import time
import numpy as np
import pickle
import os
import argparse
import json
from sklearn.svm import SVC

def test_svm_inference_time(model_path, input_shape=(3, 224, 224), num_runs=100):
    """Test inference time for SVM model
    Args:
        model_path: Path to saved SVM model
        input_shape: Input shape of test data (channels, height, width)
        num_runs: Number of inference runs
    """
    # Load saved model
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create random test data
    total_dims = np.prod(input_shape)
    dummy_input = np.random.randn(1, total_dims)
    
    # Warmup runs
    for _ in range(10):
        _ = model.predict(dummy_input)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    
    print(f"Input shape: {input_shape}")
    print(f"Average inference time: {mean_time:.2f} ms")
    print(f"Standard deviation: {std_time:.2f} ms")
    print(f"FPS: {1000/mean_time:.2f}")
    
    return mean_time, std_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exps/svm.json")
    parser.add_argument("--model_path", type=str, default="saved_pth/svm/model_2.pkl")
    parser.add_argument("--input_shape", type=str, default="512")
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Run inference time test
    test_svm_inference_time(
        model_path=args.model_path,
        input_shape=input_shape, 
        num_runs=args.num_runs
    )