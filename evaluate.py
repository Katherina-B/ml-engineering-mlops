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

from train import split_data, create_model

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_data(data_dir: str) -> Dataset:
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["input_shape"][0], config["dataset"]["input_shape"][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Add or remove any other necessary transforms
    ])

    data_dir = os.path.join(data_dir, "jpg")

    # Check if data is already downloaded
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please download the dataset.")

    # Check if dataset is already loaded
   
    test_dataset = torchvision.datasets.Flowers102(root=data_dir,    transform=transform, download=True)
    return test_dataset

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
    # Load the dataset
    test_dataset = load_data(config["data"]["local_dir"])

    # Create the data loader
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])

    # Load the trained model
    model, _, _ = create_model()
    model.load_state_dict(torch.load(config["artifacts"]["output_dir"] + "/model.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Evaluate the model
    metrics = evaluate(model, test_loader, device)

    # Save the metrics
    output_dir = config["artifacts"]["output_dir"]
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
