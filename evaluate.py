import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml
import logging
import os
import json
from typing import Dict, Tuple

import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from tqdm import tqdm
import torchvision.models as models



Dataset = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
ModelOutput = Tuple[nn.Module, optim.Optimizer, nn.CrossEntropyLoss]

from train import create_model
from load_date import load_and_split_data
from load_date import load_config

config = load_config("params.yaml")
data_dir = config["data"]["local_dir"]

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute and save other metrics as needed
    metrics = {"accuracy": accuracy / 100.0}
    return metrics

def main():
    config = load_config("params.yaml")
    data_dir = config["data"]["local_dir"]
    # Load the dataset
    _, _, test_dataset = load_and_split_data(data_dir)

    # Create the data loader
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

    # Load the trained model
    model, _, _ = create_model()
    model.load_state_dict(torch.load(config["artifacts"]["output_dir"] + "/best_model.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Evaluate the model
    metrics = evaluate(model, test_loader, device)

    output_dir = config["artifacts"]["output_dir"]
    metrics_file = os.path.join(output_dir, "metrics.json")

    # Check if the file exists
    if os.path.isfile(metrics_file):
        # File exists, load existing metrics and append new ones
        with open(metrics_file, "r") as f:
            existing_metrics = json.load(f)
        existing_metrics.update(metrics)
        metrics_to_save = existing_metrics
    else:
        # File doesn't exist, create a new one
        metrics_to_save = metrics

    # Save the metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
