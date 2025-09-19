# RankMe: A Comprehensive PyTorch Metrics Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <img src="docs/RANKME.svg" alt="RankMe logo" width="440" />
</p>

<p align="center">
    <em>A comprehensive PyTorch-based metrics library for machine learning tasks, featuring the RankMe score for feature learning evaluation alongside standard classification and regression metrics.</em>
</p>


## Features

### 🧠 Feature Learning Metrics
- **RankMe**: Measures the effective rank of learned representations using spectral entropy
- Supports batched operations and gradient computation
- Configurable centering, logarithm base, and numerical stability

### 📊 Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- Intersection over Union (IoU)
- Support for binary, multiclass, and multilabel scenarios
- Micro, macro, and weighted averaging

### 📈 Regression Metrics
- Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)
- Support for multioutput regression

## Installation

### Using PDM (Recommended)
```bash
git clone https://github.com/sirbastiano/rankme.git
cd rankme
make install-dev  # or just 'make' for full dev setup
```

### Using pip
```bash
pip install rankme
```

## Quick Start

### RankMe Score for Feature Learning
```python
import torch
from rankme import RankMe

# Single batch of embeddings
Z = torch.randn(1024, 256)  # N=1024 samples, D=256 features
rankme = RankMe(center=False)
score = rankme(Z)  # Returns value in [0, 1]
print(f"RankMe score: {score:.3f}")

# Batched embeddings
Z_batch = torch.randn(8, 512, 128)  # B=8 batches
scores = rankme(Z_batch)  # Returns shape (8,)
```

### Classification Metrics
```python
import torch
from rankme import Accuracy, F1Score, IoU

y_true = torch.tensor([0, 1, 2, 1, 0])
y_pred = torch.tensor([0, 1, 2, 2, 0])

# Accuracy
acc = Accuracy(task='multiclass', num_classes=3)
accuracy = acc(y_pred, y_true)

# F1 Score
f1 = F1Score(task='multiclass', num_classes=3, average='macro')
f1_score = f1(y_pred, y_true)

# IoU for semantic segmentation
iou = IoU(task='multiclass', num_classes=3)
iou_score = iou(y_pred, y_true)
```

### Regression Metrics
```python
import torch
from rankme import MSE, MAE, R2Score

y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])

# Mean Squared Error
mse = MSE()
mse_value = mse(y_pred, y_true)

# R² Score
r2 = R2Score()
r2_value = r2(y_pred, y_true)
```

## Advanced Usage

### RankMe with Custom Configuration
```python
from rankme import RankMe

# Custom configuration
rankme = RankMe(
    log_base=2.0,      # Use log base 2
    center=True,       # Mean-center features
    eps=1e-10,         # Custom numerical stability
    detach=True        # Compute without gradients
)

# Works with any embedding dimension
embeddings = model.encode(data)  # shape: (batch_size, embed_dim)
diversity_score = rankme(embeddings)
```

### Integration with Training Loops
```python
import torch
import torch.nn as nn
from rankme import RankMe, Accuracy

model = nn.Sequential(...)
rankme = RankMe(center=True)
accuracy = Accuracy(task='multiclass', num_classes=10)

for batch in dataloader:
    x, y = batch
    
    # Forward pass
    logits = model(x)
    embeddings = model[:-1](x)  # Get embeddings before final layer
    
    # Compute metrics
    acc = accuracy(logits.argmax(dim=1), y)
    rank_score = rankme(embeddings)
    
    print(f"Accuracy: {acc:.3f}, RankMe: {rank_score:.3f}")
```

## Development

### Setup Development Environment
```bash
make dev-setup  # Install deps + pre-commit hooks
```

### Running Tests
```bash
make test           # Run all tests
make test-cov       # Run tests with coverage
make quick-test     # Run quick test subset
```

### Code Quality
```bash
make format         # Format code with black & isort
make lint           # Run flake8 & mypy
make check          # Run all checks (format, lint, test)
```

### Building Documentation
```bash
make docs           # Build documentation
make docs-serve     # Serve docs locally on :8000
```

## API Reference

### Base Classes
All metrics inherit from `BaseMetric`, providing consistent interface:
- `forward()`: Compute metric value
- `reset()`: Reset internal state for stateful metrics
- `update()`: Update internal state (for accumulated metrics)

### Task-Specific Modules
- `rankme.feature_learning`: RankMe and related metrics
- `rankme.classification`: Classification metrics
- `rankme.regression`: Regression metrics

## Mathematical Background

### RankMe Score
The RankMe score quantifies the effective dimensionality of learned representations:

1. Compute SVD of embedding matrix Z ∈ ℝ^(N×D)
2. Convert singular values to probabilities: p_i = s_i / Σⱼ s_j  
3. Return normalized Shannon entropy: -Σᵢ p_i log(p_i) / log(D)

This yields a score in [0,1] where:
- 0: All information concentrated in one dimension
- 1: Information uniformly distributed across all dimensions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use RankMe in your research, please cite:

```bibtex
@article{rankme2023,
  title={RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank},
  author={Garrido, Quentin and Najman, Laurent and LeCun, Yann},
  journal={arXiv preprint arXiv:2210.02885},
  year={2023}
}
```

## Acknowledgments

- RankMe implementation based on the original paper by Garrido et al.
- Inspired by torchmetrics for API design patterns
- Built with PyTorch for seamless integration with deep learning workflows