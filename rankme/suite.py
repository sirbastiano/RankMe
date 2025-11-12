import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import time

# Import all available metrics
from rankme import RankMe, Accuracy, F1Score, IoU, Precision, Recall, MSE, MAE, R2Score, RMSE
from rankme.network import ParamCount, ModelSizeMB, Flops, PrecisionBits, InferenceTime


class ModelBenchmark:
    """Comprehensive benchmark suite for PyTorch models.
    
    This class provides a unified interface to compute all available metrics
    for a given model including network metrics, feature learning metrics,
    and task-specific performance metrics.
    """
    
    def __init__(self):
        """Initialize the benchmark suite with all available metrics."""
        # Network metrics (stateless)
        self.network_metrics = {
            'param_count': ParamCount(),
            'model_size_mb': ModelSizeMB(), 
            'flops': Flops(),
            'precision_bits': PrecisionBits(),
            'inference_time': InferenceTime(),
        }
        
        # Feature learning metrics
        self.feature_metrics = {
            'rankme': RankMe(center=False),
            'rankme_centered': RankMe(center=True),
        }
        
        # Classification metrics
        self.classification_metrics = {
            'accuracy': None,  # Will be initialized with num_classes
            'f1_score': None,
            'iou': None, 
            'precision': None,
            'recall': None,
        }
        
        # Regression metrics
        self.regression_metrics = {
            'mse': MSE(),
            'mae': MAE(),
            'rmse': RMSE(),
            'r2_score': R2Score(),
        }
    
    def benchmark_network(
        self, 
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Benchmark network-level metrics.
        
        Args:
            model: PyTorch model to benchmark
            input_size: Input size for FLOPs and inference time estimation
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict containing all network metrics
        """
        results = {}
        
        print("🔍 Computing Network Metrics...")
        
        # Parameter count
        total_params = self.network_metrics['param_count'](model, include_non_trainable=True)
        trainable_params = self.network_metrics['param_count'](model, include_non_trainable=False)
        results['total_parameters'] = total_params
        results['trainable_parameters'] = trainable_params
        
        # Model size
        size_params = self.network_metrics['model_size_mb'](model, include_buffers=False)
        size_buffers = self.network_metrics['model_size_mb'](model, include_buffers=True)
        results['size_params_mb'] = size_params
        results['size_total_mb'] = size_buffers
        
        # FLOPs estimation
        try:
            flops = self.network_metrics['flops'](model, input_size=input_size)
            results['flops'] = flops
        except Exception as e:
            results['flops'] = f"Error: {str(e)}"
        
        # Precision bits
        avg_bits = self.network_metrics['precision_bits'](model)
        dtype_summary = self.network_metrics['precision_bits'].get_dtype_summary(model)
        results['avg_precision_bits'] = avg_bits
        results['dtype_summary'] = dtype_summary
        
        # Inference time
        try:
            cpu_time, gpu_time = self.network_metrics['inference_time'](
                model, input_size=input_size, **kwargs
            )
            results['cpu_inference_time_ms'] = cpu_time * 1000
            results['gpu_inference_time_ms'] = gpu_time * 1000 if not torch.isnan(torch.tensor(gpu_time)) else None
        except Exception as e:
            results['cpu_inference_time_ms'] = f"Error: {str(e)}"
            results['gpu_inference_time_ms'] = None
        
        return results
    
    def benchmark_features(
        self,
        embeddings: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Benchmark feature learning metrics.
        
        Args:
            embeddings: Feature embeddings tensor of shape (N, D) or (B, N, D)
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict containing feature learning metrics
        """
        results = {}
        
        print("📊 Computing Feature Learning Metrics...")
        
        # RankMe scores
        try:
            rankme_score = self.feature_metrics['rankme'](embeddings)
            rankme_centered_score = self.feature_metrics['rankme_centered'](embeddings)
            
            if embeddings.dim() == 3:  # Batched
                results['rankme_scores'] = rankme_score.tolist()
                results['rankme_centered_scores'] = rankme_centered_score.tolist()
                results['avg_rankme'] = rankme_score.mean().item()
                results['avg_rankme_centered'] = rankme_centered_score.mean().item()
            else:  # Single batch
                results['rankme_score'] = rankme_score.item()
                results['rankme_centered_score'] = rankme_centered_score.item()
                
        except Exception as e:
            results['feature_metrics_error'] = str(e)
        
        return results
    
    def benchmark_classification(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        num_classes: int,
        task: str = 'multiclass',
        average: str = 'macro',
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Benchmark classification metrics.
        
        Args:
            y_pred: Predicted labels or logits
            y_true: True labels
            num_classes: Number of classes
            task: Task type ('binary', 'multiclass', 'multilabel')
            average: Averaging method for multi-class metrics
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict containing classification metrics
        """
        results = {}
        
        print("🎯 Computing Classification Metrics...")
        
        # Initialize metrics with proper parameters
        metrics = {
            'accuracy': Accuracy(task=task, num_classes=num_classes),
            'f1_score': F1Score(task=task, num_classes=num_classes, average=average),
            'iou': IoU(task=task, num_classes=num_classes),
            'precision': Precision(task=task, num_classes=num_classes, average=average),
            'recall': Recall(task=task, num_classes=num_classes, average=average),
        }
        
        for metric_name, metric in metrics.items():
            try:
                score = metric(y_pred, y_true)
                results[metric_name] = score.item()
            except Exception as e:
                results[f'{metric_name}_error'] = str(e)
        
        return results
    
    def benchmark_regression(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Benchmark regression metrics.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict containing regression metrics
        """
        results = {}
        
        print("📈 Computing Regression Metrics...")
        
        for metric_name, metric in self.regression_metrics.items():
            try:
                score = metric(y_pred, y_true)
                results[metric_name] = score.item()
            except Exception as e:
                results[f'{metric_name}_error'] = str(e)
        
        return results
    
    def full_benchmark(
        self,
        model: nn.Module,
        embeddings: Optional[torch.Tensor] = None,
        y_pred_class: Optional[torch.Tensor] = None,
        y_true_class: Optional[torch.Tensor] = None,
        y_pred_reg: Optional[torch.Tensor] = None,
        y_true_reg: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        classification_task: str = 'multiclass',
        classification_average: str = 'macro',
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run complete benchmark suite on a model.
        
        Args:
            model: PyTorch model to benchmark
            embeddings: Feature embeddings for feature learning metrics
            y_pred_class: Classification predictions
            y_true_class: Classification ground truth
            y_pred_reg: Regression predictions  
            y_true_reg: Regression ground truth
            num_classes: Number of classes for classification
            input_size: Input size for network metrics
            classification_task: Classification task type
            classification_average: Averaging method for classification
            **kwargs: Additional arguments
            
        Returns:
            Dict containing all computed metrics
        """
        print("🚀 Starting Full Model Benchmark Suite")
        print("=" * 50)
        
        results = {
            'model_info': {
                'model_class': model.__class__.__name__,
                'input_size': input_size,
            }
        }
        
        # Network metrics (always computed)
        start_time = time.time()
        results['network'] = self.benchmark_network(model, input_size, **kwargs)
        results['network']['computation_time_s'] = time.time() - start_time
        
        # Feature learning metrics (if embeddings provided)
        if embeddings is not None:
            start_time = time.time()
            results['feature_learning'] = self.benchmark_features(embeddings, **kwargs)
            results['feature_learning']['computation_time_s'] = time.time() - start_time
        
        # Classification metrics (if classification data provided)
        if y_pred_class is not None and y_true_class is not None and num_classes is not None:
            start_time = time.time()
            results['classification'] = self.benchmark_classification(
                y_pred_class, y_true_class, num_classes, 
                classification_task, classification_average, **kwargs
            )
            results['classification']['computation_time_s'] = time.time() - start_time
        
        # Regression metrics (if regression data provided)
        if y_pred_reg is not None and y_true_reg is not None:
            start_time = time.time()
            results['regression'] = self.benchmark_regression(y_pred_reg, y_true_reg, **kwargs)
            results['regression']['computation_time_s'] = time.time() - start_time
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print benchmark results.
        
        Args:
            results: Results dictionary from benchmark methods
        """
        print("\n" + "=" * 60)
        print("📊 MODEL BENCHMARK RESULTS")
        print("=" * 60)
        
        # Model info
        if 'model_info' in results:
            info = results['model_info']
            print(f"\n🏗️  Model: {info['model_class']}")
            print(f"📐 Input Size: {info['input_size']}")
        
        # Network metrics
        if 'network' in results:
            print(f"\n🔍 NETWORK METRICS")
            print("-" * 30)
            net = results['network']
            if 'total_parameters' in net:
                print(f"Total Parameters: {net['total_parameters']:,}")
            if 'trainable_parameters' in net:
                print(f"Trainable Parameters: {net['trainable_parameters']:,}")
            if 'size_total_mb' in net:
                print(f"Model Size: {net['size_total_mb']:.2f} MB")
            if 'flops' in net:
                if isinstance(net['flops'], (int, float)):
                    print(f"FLOPs: {net['flops']:,.0f}")
                else:
                    print(f"FLOPs: {net['flops']}")
            if 'avg_precision_bits' in net:
                print(f"Avg Precision: {net['avg_precision_bits']:.1f} bits")
            if 'cpu_inference_time_ms' in net:
                if isinstance(net['cpu_inference_time_ms'], (int, float)):
                    print(f"CPU Inference: {net['cpu_inference_time_ms']:.2f} ms")
                else:
                    print(f"CPU Inference: {net['cpu_inference_time_ms']}")
            if 'gpu_inference_time_ms' in net and net['gpu_inference_time_ms'] is not None:
                print(f"GPU Inference: {net['gpu_inference_time_ms']:.2f} ms")
        
        # Feature learning metrics
        if 'feature_learning' in results:
            print(f"\n📊 FEATURE LEARNING METRICS")
            print("-" * 30)
            feat = results['feature_learning']
            if 'rankme_score' in feat:
                print(f"RankMe Score: {feat['rankme_score']:.4f}")
            if 'rankme_centered_score' in feat:
                print(f"RankMe (Centered): {feat['rankme_centered_score']:.4f}")
            if 'avg_rankme' in feat:
                print(f"Avg RankMe: {feat['avg_rankme']:.4f}")
            if 'avg_rankme_centered' in feat:
                print(f"Avg RankMe (Centered): {feat['avg_rankme_centered']:.4f}")
        
        # Classification metrics
        if 'classification' in results:
            print(f"\n🎯 CLASSIFICATION METRICS")
            print("-" * 30)
            clf = results['classification']
            for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'iou']:
                if metric in clf:
                    print(f"{metric.replace('_', ' ').title()}: {clf[metric]:.4f}")
        
        # Regression metrics
        if 'regression' in results:
            print(f"\n📈 REGRESSION METRICS")
            print("-" * 30)
            reg = results['regression']
            for metric in ['mse', 'mae', 'rmse', 'r2_score']:
                if metric in reg:
                    print(f"{metric.upper()}: {reg[metric]:.4f}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    # Example usage of the ModelBenchmark suite
    import torchvision
    # Load a sample model (ResNet18)
    model = torchvision.models.resnet18(weights=None)
    
    # Example usage with the ResNet18 model
    benchmark = ModelBenchmark()

    # Generate some sample data for demonstration
    batch_size, num_classes = 32, 10

    # Sample embeddings (e.g., from model's feature extractor)
    sample_embeddings = torch.randn(batch_size, 512)  # 512-dim features

    # Sample classification data
    y_true_class = torch.randint(0, num_classes, (batch_size,))
    y_pred_class = torch.randint(0, num_classes, (batch_size,))

    # Sample regression data  
    y_true_reg = torch.randn(batch_size)
    y_pred_reg = torch.randn(batch_size)

    # Run comprehensive benchmark
    print("Running comprehensive benchmark on ResNet18...")
    results = benchmark.full_benchmark(
        model=model,
        embeddings=sample_embeddings,
        y_pred_class=y_pred_class,
        y_true_class=y_true_class, 
        y_pred_reg=y_pred_reg,
        y_true_reg=y_true_reg,
        num_classes=num_classes,
        input_size=(1, 3, 224, 224),
        runs=20,  # For inference timing
        warmup=5
    )

    # Pretty print all results
    benchmark.print_results(results)

    # You can also access individual results programmatically
    print(f"\nQuick Summary:")
    print(f"Parameters: {results['network']['total_parameters']:,}")
    print(f"Model Size: {results['network']['size_total_mb']:.2f} MB") 
    if 'feature_learning' in results:
        print(f"RankMe Score: {results['feature_learning']['rankme_score']:.4f}")
    if 'classification' in results:
        print(f"Accuracy: {results['classification']['accuracy']:.4f}")
    if 'regression' in results:
        print(f"R² Score: {results['regression']['r2_score']:.4f}")