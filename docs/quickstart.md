# Quick Start

## Basic Usage

```python
import torch
from rankme import RankMe

# Create some embeddings
embeddings = torch.randn(1000, 256)

# Calculate RankMe score
rankme = RankMe()
score = rankme(embeddings)
print(f"RankMe score: {score:.3f}")
```

## Classification Metrics

```python
from rankme import Accuracy, F1Score

# Binary classification
y_true = torch.tensor([0, 1, 1, 0, 1])
y_pred = torch.tensor([0, 1, 0, 0, 1])

acc = Accuracy(task='binary')
accuracy = acc(y_pred, y_true)
print(f"Accuracy: {accuracy:.3f}")
```

## Regression Metrics

```python
from rankme import MSE, R2Score

# Regression data
y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])

mse = MSE()
r2 = R2Score()

print(f"MSE: {mse(y_pred, y_true):.3f}")
print(f"R²: {r2(y_pred, y_true):.3f}")
```