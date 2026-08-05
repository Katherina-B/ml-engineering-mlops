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
