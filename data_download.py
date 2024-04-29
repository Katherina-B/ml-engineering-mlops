import os
import tarfile
import urllib.request

from config import config

def download_data():
    # Create the data directory if it doesn't exist
    data_dir = config["data"]["local_dir"]
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset
    dataset_url = config["data"]["dataset_url"]
    label_url = config["data"]["label_url"]

    # Download and extract the dataset
    urllib.request.urlretrieve(dataset_url, os.path.join(data_dir, "dataset.tgz"))
    with tarfile.open(os.path.join(data_dir, "dataset.tgz"), "r:gz") as tar:
        tar.extractall(path=data_dir)

    # Download the labels (if applicable)
    urllib.request.urlretrieve(label_url, os.path.join(data_dir, "labels.mat"))

if __name__ == "__main__":
    download_data()
