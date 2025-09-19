"""Example usage of the RankMe metrics library."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rankme import RankMe, Accuracy, F1Score, MSE, R2Score


def create_sample_model(input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )


def example_feature_learning_evaluation():
    """Example: Evaluating representation quality with RankMe."""
    print("=== Feature Learning Evaluation with RankMe ===")
    
    # Create sample data and model
    batch_size, input_dim, hidden_dim = 128, 784, 256
    model = create_sample_model(input_dim, hidden_dim, 10)
    
    # Create sample input data
    x = torch.randn(batch_size, input_dim)
    
    # Extract embeddings from hidden layer
    with torch.no_grad():
        # Forward pass to get hidden representations
        hidden = model[:3](x)  # Up to second ReLU
        
    # Initialize RankMe metric
    rankme = RankMe(center=True)  # Mean-center features
    
    # Compute RankMe score
    rankme_score = rankme(hidden)
    
    print(f"Hidden layer embeddings shape: {hidden.shape}")
    print(f"RankMe score: {rankme_score:.4f}")
    print(f"Score interpretation: {rankme_score:.4f} (0=low diversity, 1=high diversity)")
    
    # Compare with different network initializations
    print("\nComparing different initialization schemes:")
    
    for init_name, init_fn in [
        ("Xavier Normal", nn.init.xavier_normal_),
        ("Kaiming Normal", nn.init.kaiming_normal_),
        ("Normal (std=0.01)", lambda w: nn.init.normal_(w, std=0.01)),
    ]:
        model_init = create_sample_model(input_dim, hidden_dim, 10)
        
        # Apply initialization
        for layer in model_init:
            if isinstance(layer, nn.Linear):
                init_fn(layer.weight)
                
        with torch.no_grad():
            hidden_init = model_init[:3](x)
            
        rankme_init = RankMe(center=True)
        score_init = rankme_init(hidden_init)
        
        print(f"  {init_name}: {score_init:.4f}")


def example_classification_evaluation():
    """Example: Evaluating classification performance."""
    print("\n=== Classification Evaluation ===")
    
    # Generate sample classification data
    num_samples, num_classes = 1000, 5
    
    # Simulate model predictions and ground truth
    torch.manual_seed(42)
    logits = torch.randn(num_samples, num_classes)
    y_true = torch.randint(0, num_classes, (num_samples,))
    y_pred = logits.argmax(dim=1)
    
    # Initialize classification metrics
    accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    f1_macro = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    f1_micro = F1Score(task='multiclass', num_classes=num_classes, average='micro')
    
    # Compute metrics
    acc_score = accuracy(y_pred, y_true)
    f1_macro_score = f1_macro(y_pred, y_true)
    f1_micro_score = f1_micro(y_pred, y_true)
    
    print(f"Classification Results:")
    print(f"  Accuracy: {acc_score:.4f}")
    print(f"  F1-Score (Macro): {f1_macro_score:.4f}")
    print(f"  F1-Score (Micro): {f1_micro_score:.4f}")
    
    # Example with top-k accuracy
    top_k_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
    top_k_score = top_k_acc(logits, y_true)  # Use logits for top-k
    print(f"  Top-3 Accuracy: {top_k_score:.4f}")


def example_regression_evaluation():
    """Example: Evaluating regression performance."""
    print("\n=== Regression Evaluation ===")
    
    # Generate sample regression data
    num_samples = 1000
    
    torch.manual_seed(42)
    y_true = torch.randn(num_samples)
    # Add some noise to create predictions
    y_pred = y_true + 0.2 * torch.randn(num_samples)
    
    # Initialize regression metrics
    mse = MSE()
    r2 = R2Score()
    
    # Compute metrics
    mse_score = mse(y_pred, y_true)
    r2_score = r2(y_pred, y_true)
    
    print(f"Regression Results:")
    print(f"  MSE: {mse_score:.4f}")
    print(f"  R² Score: {r2_score:.4f}")
    
    # Example with multioutput regression
    y_true_multi = torch.randn(num_samples, 3)
    y_pred_multi = y_true_multi + 0.1 * torch.randn(num_samples, 3)
    
    r2_multi = R2Score(multioutput='raw_values')
    r2_multi_scores = r2_multi(y_pred_multi, y_true_multi)
    
    print(f"Multioutput Regression:")
    print(f"  R² Scores per output: {r2_multi_scores}")
    print(f"  Average R² Score: {r2_multi_scores.mean():.4f}")


def example_training_loop_integration():
    """Example: Integrating metrics into a training loop."""
    print("\n=== Training Loop Integration ===")
    
    # Create model and data
    model = create_sample_model(784, 128, 10)
    batch_size = 64
    
    # Initialize metrics
    rankme = RankMe(center=True)
    accuracy = Accuracy(task='multiclass', num_classes=10)
    
    # Simulate training loop
    print("Simulating training epochs...")
    
    for epoch in range(3):
        # Simulate a training batch
        x = torch.randn(batch_size, 784)
        y_true = torch.randint(0, 10, (batch_size,))
        
        with torch.no_grad():
            # Forward pass
            logits = model(x)
            
            # Extract hidden representations
            hidden = model[:3](x)  # Get hidden layer output
            
            # Compute metrics
            acc = accuracy(logits.argmax(dim=1), y_true)
            diversity = rankme(hidden)
            
        print(f"Epoch {epoch + 1}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Representation Diversity (RankMe): {diversity:.4f}")


def example_batched_evaluation():
    """Example: Batched evaluation of metrics."""
    print("\n=== Batched Evaluation ===")
    
    # Create batched data
    batch_size, seq_len, embed_dim = 8, 512, 128
    
    # Simulate batched embeddings (e.g., from transformer layers)
    embeddings_batch = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize RankMe for batched evaluation
    rankme_batch = RankMe(center=True)
    
    # Compute RankMe scores for each batch item
    batch_scores = rankme_batch(embeddings_batch)
    
    print(f"Batched Embeddings Shape: {embeddings_batch.shape}")
    print(f"RankMe Scores per Batch Item:")
    for i, score in enumerate(batch_scores):
        print(f"  Batch {i}: {score:.4f}")
    print(f"Average RankMe Score: {batch_scores.mean():.4f}")
    print(f"Score Standard Deviation: {batch_scores.std():.4f}")


if __name__ == "__main__":
    print("RankMe Metrics Library - Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_feature_learning_evaluation()
    example_classification_evaluation()
    example_regression_evaluation()
    example_training_loop_integration()
    example_batched_evaluation()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output above for detailed results.")
    print("For more information, see the documentation and test files.")