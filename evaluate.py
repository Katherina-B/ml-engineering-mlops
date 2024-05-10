import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train import split_data, create_model

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

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
    _, _, test_dataset = load_data(config["data"]["local_dir"])

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
    with open(config["artifacts"]["output_dir"] + "/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
