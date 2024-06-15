import matplotlib
matplotlib.use('agg')
import os
import torch
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import Occlusion
import matplotlib.pyplot as plt
import wandb
import logging
import numpy as np
import yaml
from load_date import load_and_split_data
from train import create_model
from torch.utils.data import DataLoader

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from lime import lime_image
import os
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import wandb
from model import ResNet50

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
    original_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor()
    ])
    data_dir = config['data']['local_dir']
    data_dir = os.path.join(data_dir, "jpg")
    
    logger = logging.getLogger(__name__)
    test_dataset = datasets.Flowers102(root=data_dir, split="test", transform=original_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_fn = create_model()
    model = model.to(device)

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
    wandb.finish()
