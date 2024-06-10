import matplotlib
matplotlib.use('agg')
import os
import torch
from captum.attr import GradientShap
import matplotlib.pyplot as plt
import wandb
import logging
import json
from typing import Dict, Tuple
import random
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from load_date import load_and_split_data
from train import create_model
import numpy as np

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

ind = config["training"]["optimizer"]["lr"]
wandb.init(
    project="lb5_intr",
    name=f"run_{ind}",
    config={
        "learning_rate": config["training"]["optimizer"],
        "dataset": "Flower-102",
        "epochs": config["training"]["epochs"],
    })

def interpret_model(config):
    logger = logging.getLogger(__name__)
    train_dataset, val_dataset, test_dataset = load_and_split_data(config["data"]["local_dir"])
    test_loader = DataLoader(test_dataset, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_fn = create_model()
    model = model.to(device)

    output_dir = "interpretation_results"
    os.makedirs(output_dir, exist_ok=True)
    incorrect_count = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_()
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        if preds != labels:
            incorrect_count += 1
            if incorrect_count > 100:
                break
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            attributions = images.grad  # Use Gradient * Input for attribution

            for i in range(len(images)):
                attr_img = attributions[i].cpu().detach().numpy().transpose(1, 2, 0)
                attr_img = np.clip(attr_img, 0, 1)  # Clip attribution values to [0, 1] range

                # Save original input image and attribution map side by side
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(images[i].cpu().detach().permute(1, 2, 0))  # Detach the tensor before displaying
                ax[0].axis('off')
                ax[0].set_title('Original Image')
                ax[1].imshow(attr_img)
                ax[1].axis('off')
                ax[1].set_title('Attribution Map')
                img_path = os.path.join(output_dir, f"combined_{incorrect_count}_{i}.png")
                plt.savefig(img_path)
                plt.close()
                wandb.log({"combined_image": [wandb.Image(img_path, caption=f"Label: {labels[i].item()}")]})
                os.remove(img_path)

if __name__ == "__main__":
    interpret_model(config)
    wandb.finish()
