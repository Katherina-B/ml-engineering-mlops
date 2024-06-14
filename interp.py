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

    grad_cam = LayerGradCam(model, model.layer4[-1])
    saliency = LayerAttribution(model, model.layer4[-1])

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
            saliency_attr = saliency.attribute(images, target=labels)

            for i in range(len(images)):
                attr_img_grad_cam = grad_cam_attr[i].cpu().detach().numpy().transpose(1, 2, 0)
                attr_img_saliency = saliency_attr[i].cpu().detach().numpy().transpose(1, 2, 0)

                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                ax[0].imshow(images[i].cpu().detach().permute(1, 2, 0))
                ax[0].axis('off')
                ax[0].set_title('Original Image')
                ax[1].imshow(attr_img_grad_cam, cmap='viridis')
                ax[1].axis('off')
                ax[1].set_title('Grad-CAM Attribution')
                ax[2].imshow(attr_img_saliency, cmap='viridis')
                ax[2].axis('off')
                ax[2].set_title('Saliency Attribution')
                img_path = os.path.join(output_dir, f"combined_{incorrect_count}_{i}.png")
                plt.savefig(img_path)
                plt.close()
                wandb.log({"combined_image": [wandb.Image(img_path, caption=f"Label: {labels[i].item()}")]})
                os.remove(img_path)

if __name__ == "__main__":
    interpret_model(config)
    wandb.finish()
