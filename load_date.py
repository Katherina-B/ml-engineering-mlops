import yaml
import os
import requests
import tarfile

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
 
def download_and_extract_archive(url, destination_folder, config=None):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
 
    # Extract the file name from the URL to use as the local filename
    filename = os.path.join(destination_folder, url.split("/")[-1])
 
    # Download the archive
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
 
    # Extract the contents of the archive
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(destination_folder)
 
    # Remove the downloaded archive file
    os.remove(filename)
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_split_data(data_dir):
    config = load_config("params.yaml")

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["input_shape"][0], config["dataset"]["input_shape"][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join(data_dir, "jpg")

    # Load and split the dataset
    train_dataset = datasets.Flowers102(root=data_dir, split="train", transform=transform, download=True)
    val_dataset = datasets.Flowers102(root=data_dir, split="val", transform=transform, download=True)
    test_dataset = datasets.Flowers102(root=data_dir, split="test", transform=transform, download=True)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    config_file = "params.yaml"
    config = load_config(config_file)
    archive_url = config['data']['dataset_url']
    destination_folder = config['data']['local_dir']

    download_and_extract_archive(archive_url, destination_folder, config)
    print(f"Archive extracted to {destination_folder}")
    train_dataset, val_dataset, test_dataset = load_and_split_data(destination_folder)

    # Save the split datasets to pickle files
    import pickle
    with open('data/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('data/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open('data/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

 
