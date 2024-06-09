import os
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import wandb
import logging

import json
from typing import Dict, Tuple
import random
import time

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt


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

wandb.init(
    # Set the project where this run will be logged
    project="lb5_intr",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"run_{ind}",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": config["training"]["optimizer"],
    "dataset": "Flower-102",
    "epochs": config["training"]["epochs"],
    })

def interpret_model(config):
  logger = setup_logger()
  _, _, test_dataset = load_and_split_data(config["data"]["local_dir"])
  test_loader = DataLoader(test_dataset, batch_size=1)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, optimizer, loss_fn = create_model()
  ig = IntegratedGradients(model)
  output_dir = "interpretation_results"
  os.makedirs(output_dir, exist_ok=True)
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    attributions, delta = ig.attribute(images, target=labels, return_convergence_delta=True)
    for i in range(len(images)):
      attr_img = attributions[i].cpu().detach().numpy().transpose(1, 2, 0)
      plt.imshow(attr_img)
      plt.axis('off')
      img_path = os.path.join(output_dir, f"attr_{i}.png")
      plt.savefig(img_path)
      plt.close()  
      wandb.log({"interpretation": [wandb.Image(img_path, caption=f"Label: {labels[i].item()}")]})
      os.remove(img_path)  
            
if __name__ == "__main__":
  interpret_model(config)
  
