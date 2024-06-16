import matplotlib
matplotlib.use('agg')
import os
import torch
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop, IntegratedGradients
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
        transforms.Resize((224, 224)),  # Resize the input images to (224, 224)
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

    grad_cam = LayerGradCam(model, model.layer4[-1])
    guided_backprop = GuidedBackprop(model)
    integrated_gradients = IntegratedGradients(model)
    occlusion = Occlusion(model)

    output_dir = "interpretation_results"
    os.makedirs(output_dir, exist_ok=True)
    incorrect_count = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        if preds != labels:
            incorrect_count += 1
            if incorrect_count > 1024:
                break

            # Grad-CAM Attribution Computation
            grad_cam_attr = grad_cam.attribute(images, target=labels)
            grad_cam_attr = grad_cam_attr.squeeze(0)  # Reduce dimensionality to 3D

            # Create a 3D tensor for Grad-CAM attribution interpolation
            input_tensor = torch.ones(1, device=device)
            input_tensor_3d = input_tensor.expand(1, images.shape[-2] // 28, images.shape[-1] // 28)
            output_tensor_3d = input_tensor_3d.clone().detach()

            input_tensor_3d = input_tensor_3d.detach().clone()
            output_tensor_3d = output_tensor_3d.detach().clone()

            # Upsample the Grad-CAM attribution to match the input image size
            upsampled_grad_cam_attr = LayerAttribution.interpolate(grad_cam_attr, input_tensor_3d, output_tensor_3d)
            attr_img_grad_cam = upsampled_grad_cam_attr[0].cpu().detach().numpy()

            # Guided Backpropagation Attribution Computation
            guided_backprop_attr = guided_backprop.attribute(images, target=labels)
            print(f"Guided Backpropagation attribution tensor shape: {guided_backprop_attr.shape}")  # Print shape for debugging

            # Integrated Gradients Attribution Computation
            integrated_gradients_attr = integrated_gradients.attribute(images, target=labels, n_steps=50)
            print(f"Integrated Gradients attribution tensor shape: {integrated_gradients_attr.shape}")  # Print shape for debugging

            # Specify the sliding_window_shapes for Occlusion
            sliding_window_shapes = (3, images.shape[-2] // 28, images.shape[-1] // 28)  # Set sliding window shapes to (3, 8, 8)
            occlusion_attr = occlusion.attribute(images, target=labels, sliding_window_shapes=sliding_window_shapes)
            print(f"Occlusion attribution tensor shape: {occlusion_attr.shape}")  # Print shape for debugging

            # Check if the attribution tensors contain valid values
            if torch.isnan(guided_backprop_attr).any() or torch.isinf(guided_backprop_attr).any():
                print("Warning: Guided Backpropagation attribution tensor contains NaN or infinity values.")
            if torch.isnan(integrated_gradients_attr).any() or torch.isinf(integrated_gradients_attr).any():
                print("Warning: Integrated Gradients attribution tensor contains NaN or infinity values.")
            if torch.isnan(occlusion_attr).any() or torch.isinf(occlusion_attr).any():
                print("Warning: Occlusion attribution tensor contains NaN or infinity values.")

            # Check if the sliding window shapes are compatible with the input image dimensions
            image_height, image_width = images.shape[-2], images.shape[-1]
            window_height, window_width = sliding_window_shapes[1], sliding_window_shapes[2]

            if image_height % window_height != 0 or image_width % window_width != 0:
                print("Warning: Sliding window shapes are not compatible with the input image dimensions.")

            for i in range(len(images)):
                attr_img_guided_backprop = guided_backprop_attr[i].cpu().detach().numpy().transpose(1, 2, 0)
                attr_img_integrated_gradients = integrated_gradients_attr[i].cpu().detach().numpy().transpose(1, 2, 0)
                attr_img_occlusion = occlusion_attr[i].cpu().detach().numpy().transpose(1, 2, 0)

                # Print attribution tensor shapes for comparison
                print(f"Grad-CAM attribution tensor shape: {attr_img_grad_cam.shape}")
                print(f"Guided Backpropagation attribution tensor shape: {attr_img_guided_backprop.shape}")
                print(f"Integrated Gradients attribution tensor shape: {attr_img_integrated_gradients.shape}")
                print(f"Occlusion attribution tensor shape: {attr_img_occlusion.shape}")

                fig, ax = plt.subplots(1, 5, figsize=(30, 6))
                ax[0].imshow(images[i].cpu().detach().permute(1, 2, 0))
                ax[0].axis('off')
                ax[0].set_title('Original Image')
                ax[1].imshow(attr_img_grad_cam, cmap='viridis')
                ax[1].axis('off')
                ax[1].set_title('Grad-CAM Attribution')
                ax[2].imshow(attr_img_guided_backprop, cmap='viridis')
                ax[2].axis('off')
                ax[2].set_title('Guided Backpropagation Attribution')
                ax[3].imshow(attr_img_integrated_gradients, cmap='viridis')
                ax[3].axis('off')
                ax[3].set_title('Integrated Gradients Attribution')
                ax[4].imshow(attr_img_occlusion, cmap='viridis')
                ax[4].axis('off')
                ax[4].set_title('Occlusion Attribution')
                img_path = os.path.join(output_dir, f"combined_{incorrect_count}_{i}.png")

if __name__ == "__main__":
    interpret_model(config)
