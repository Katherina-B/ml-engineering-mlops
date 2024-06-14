import matplotlib
matplotlib.use('agg')
import os
import torch
from captum.attr import LayerGradCam, LayerAttribution
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



import matplotlib
matplotlib.use('agg')
import os
import torch
from captum.attr import LayerGradCam, Occlusion
import matplotlib.pyplot as plt
import wandb
import logging
import numpy as np
from train import create_model
from skimage.segmentation import slic
from lime import lime_image

def interpret_model(config):
    original_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor()
    ])
    data_dir=config['data']['local_dir']
    data_dir = os.path.join(data_dir, "jpg")
    
    logger = logging.getLogger(__name__)
    test_dataset = datasets.Flowers102(root=data_dir, split="test", transform=original_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_fn = create_model()
    model = model.to(device)

    grad_cam = LayerGradCam(model, model.layer4[-1])
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

            grad_cam_attr = grad_cam.attribute(images, target=labels)

            # Specify the sliding_window_shapes for Occlusion
            sliding_window_shapes = (3, images.shape[-2] // 8, images.shape[-1] // 8)
            occlusion_attr = occlusion.attribute(images, target=labels, sliding_window_shapes=sliding_window_shapes)

            for i in range(len(images)):
                attr_img_grad_cam = grad_cam_attr[i].cpu().detach().numpy().transpose(1, 2, 0)
                attr_img_occlusion = occlusion_attr[i].cpu().detach().numpy().transpose(1, 2, 0)

                # Use LIME for interpretation
                lime_explainer = lime_image.LimeImageExplainer()
                explanation = lime_explainer.explain_instance(images[i].cpu().detach().permute(1, 2, 0).numpy(), model.predict, top_labels=5, hide_color=0, num_samples=1000)

                fig, ax = plt.subplots(1, 4, figsize=(24, 6))
                ax[0].imshow(images[i].cpu().detach().permute(1, 2, 0))
                ax[0].axis('off')
                ax[0].set_title('Original Image')
                ax[1].imshow(attr_img_grad_cam, cmap='viridis')
                ax[1].axis('off')
                ax[1].set_title('Grad-CAM Attribution')
                ax[2].imshow(attr_img_occlusion, cmap='viridis')
                ax[2].axis('off')
                ax[2].set_title('Occlusion Attribution')
                lime_image = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
                ax[3].imshow(lime_image)
                ax[3].axis('off')
                ax[3].set_title('LIME Explanation')
                img_path = os.path.join(output_dir, f"combined_{incorrect_count}_{i}.png")
                plt.savefig(img_path)
                plt.close()
                wandb.log({"combined_image": [wandb.Image(img_path, caption=f"Label: {labels[i].item()}")]})
                os.remove(img_path)


if __name__ == "__main__":
    interpret_model(config)
    wandb.finish()
