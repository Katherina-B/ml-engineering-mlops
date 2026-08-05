## ML Engineering Pipeline — Flower Image Classification with MLOps

A complete ML engineering project built across 5 labs as part of the **ML Engineering course** at NTU KhPI (2024).

End-to-end image classification pipeline using **ResNet50** on the Oxford Flowers 102 dataset, with full MLOps tooling integrated at each stage.

---

## Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **96.96%** (Epoch 19) |
| Best Validation Loss | 0.1296 |
| Test Set Size | 1,020 images |
| Training Set Size | 6,149 images |
| Classes | 102 flower categories |

---

## Model

- **Architecture:** ResNet50 (pretrained on ImageNet)
- **Dataset:** [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Optimizer:** AdamW (lr=0.0001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 20
- **Framework:** PyTorch

---

## Pipeline Stages

| Lab | Focus | Key Tools |
|-----|-------|-----------|
| 1 | Basic training pipeline | Python, logging, type hints, ruff, black |
| 2 | Automated dataset extension | Batch splitting, data registry |
| 3 | Data version control | DVC, params.yaml |
| 4 | Experiment tracking | MLflow |
| 5 | Advanced tracking + interpretability | Weights & Biases, Captum (GradCAM) |

---

## Tech Stack

- **Model:** ResNet50 (PyTorch / torchvision)
- **Dataset:** Oxford Flowers 102 (102 categories)
- **Experiment tracking:** Weights & Biases, MLflow
- **Interpretability:** Captum (GradCAM + Saliency Maps)
- **Config management:** YAML-based (`params.yaml`)
- **Dependency management:** Poetry (`pyproject.toml`)
- **Logging:** Python logging module
- **Code quality:** ruff, black, mypy, isort

---

## Interpretability

Lab 5 includes model interpretability using **Captum**:
- **Grad-CAM** — highlights regions most influential 
  for the model's prediction
- **Saliency Maps** — pixel-level attribution visualisation
- Results logged as images to **Weights & Biases**
- Correct and incorrect predictions saved separately   for error analysis

---
## Project Structure

    ├── train.py              # Training loop + ResNet50 model definition
    ├── load_date.py          # Data loading, splitting, augmentation
    ├── interp.py             # GradCAM + Saliency interpretability
    ├── check_cuda.py         # GPU availability check
    ├── params.yaml           # Centralised configuration
    ├── pyproject.toml        # Poetry dependency management
    ├── run_pipeline.ipynb    # Colab notebook to run full pipeline
    └── result.txt            # Training logs and metrics
---

## How to Run

```bash
# Install dependencies
poetry install

# Check GPU
python check_cuda.py

# Train model
python train.py

# Run interpretability analysis
python interp.py
```

> **Note:** Developed and trained in Google Colab with GPU. 
> Update `data.local_dir` and `artifacts.output_dir` 
> in `params.yaml` for local execution.

---

## Configuration

All hyperparameters and paths are controlled via `params.yaml`:

```yaml
model:
  name: resnet50
  pretrained: True
  num_classes: 102

training:
  epochs: 20
  batch_size: 32
  optimizer:
    name: AdamW
    lr: 0.0001
```

---

*Part of the ML Engineering course — NTU KhPI, 2024*
*Instructor: Maksym Tatariants, PhD (ML Engineer @ Toshiba)*

---

## How to Run

```bash
# Install dependencies
poetry install

# Check GPU
python check_cuda.py

# Train model
python train.py

# Run interpretability analysis
python interp.py
```

> **Note:** Developed and trained in Google Colab with GPU. 
> Update `data.local_dir` and `artifacts.output_dir` 
> in `params.yaml` for local execution.

---

## Configuration

All hyperparameters and paths are controlled via `params.yaml`:

```yaml
model:
  name: resnet50
  pretrained: True
  num_classes: 102

training:
  epochs: 20
  batch_size: 32
  optimizer:
    name: AdamW
    lr: 0.0001
```

---

*Part of the ML Engineering course — NTU KhPI, 2024*
*Instructor: Maksym Tatariants, PhD (ML Engineer @ Toshiba)*
## Project Structure
## Project Structure
