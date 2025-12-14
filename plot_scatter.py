import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from utils import factory
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
import logging

def plot_features(args):
    # Set random seed for reproducibility 
    torch.manual_seed(1)
    np.random.seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # # Initialize model using factory like in trainer.py
    # model = factory.get_model(args["model_name"], args)

    # # Initialize the model similar to incremental_train
    # model._network.update_fc(args["init_cls"])
    
    # # Load the initial model state
    # if not args["init_train"]:
    #     model._network.convnets[0] = torch.load(args["convnet_path"])
    #     model._network.fc = torch.load(args["fc_path"])
    # model._network.to(device)
    # model._network.eval()

    model = torch.load(r"saved_pth/hrrp9/POD_foster/time_2025_03_30_15_37_43_cil_1993_M=500.pth")
    model.to(device)
    
    # Data manager setup matching trainer.py
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"], 
        args["seed"],
        args["init_cls"],
        args["increment"]
    )

    # Create train loader
    train_dataset = data_manager.get_dataset(
        # np.arange(args["init_cls"]),
        np.arange(9),
        source="train",
        mode="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True
    )
    
    # Extract features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        for _, inputs, targets in train_loader:
            inputs = inputs.to(device)
            
            # Get feature vectors using model's outputs
            outputs = model._network(inputs)
            feature = outputs["features"]  # Extract features from model output dict
            feature = feature.cpu().numpy()
            
            features.append(feature)
            labels.append(targets.numpy())
    
    # Merge features and labels
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20')
    # plt.colorbar(scatter)
    
    # plt.title(f'Feature Space Visualization ({args["dataset"]}, Task 0)')
    # plt.xlabel('t-SNE dimension 1')
    # plt.ylabel('t-SNE dimension 2')
    
    # Save visualization
    save_dir = f'visualization/{args["dataset"]}/{args["model_name"]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'task0_feature_scatter.png'))
    plt.close()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

if __name__ == '__main__':
    # Import json at the top of the file
    import json
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Plot feature space visualization.')
    parser.add_argument('--config', type=str, default='./exps/POD_foster.json',
                       help='Json file of settings.')
    args = parser.parse_args()
    
    # Load parameters from json
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to dict
    args.update(param)  # Add parameters from json
    
    plot_features(args)