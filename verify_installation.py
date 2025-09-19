#!/usr/bin/env python3
"""Simple script to verify RankMe installation and basic functionality."""

import sys

def main():
    """Run basic verification tests."""
    print("RankMe Installation Verification")
    print("=" * 40)
    
    # Test imports
    print("1. Testing imports...")
    try:
        import rankme
        print("   ✓ rankme package imported successfully")
        
        from rankme import RankMe, Accuracy, MSE
        print("   ✓ Core metrics imported successfully")
        
        import torch
        print("   ✓ PyTorch imported successfully")
        
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        sys.exit(1)
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    try:
        # Test RankMe
        torch.manual_seed(42)
        Z = torch.randn(100, 32)
        rankme = RankMe()
        score = rankme(Z)
        assert 0 <= score <= 1, f"RankMe score {score} not in [0,1]"
        print(f"   ✓ RankMe score: {score:.4f}")
        
        # Test classification metric
        y_true = torch.randint(0, 3, (50,))
        y_pred = torch.randint(0, 3, (50,))
        acc = Accuracy(task='multiclass', num_classes=3)
        accuracy = acc(y_pred, y_true)
        assert 0 <= accuracy <= 1, f"Accuracy {accuracy} not in [0,1]"
        print(f"   ✓ Accuracy: {accuracy:.4f}")
        
        # Test regression metric
        y_true_reg = torch.randn(50)
        y_pred_reg = y_true_reg + 0.1 * torch.randn(50)
        mse = MSE()
        mse_val = mse(y_pred_reg, y_true_reg)
        assert mse_val >= 0, f"MSE {mse_val} is negative"
        print(f"   ✓ MSE: {mse_val:.4f}")
        
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        sys.exit(1)
    
    # Test package info
    print("\n3. Package information:")
    print(f"   Version: {rankme.__version__}")
    print(f"   Author: {rankme.__author__}")
    print(f"   Available metrics: {len(rankme.__all__)} total")
    
    print("\n✓ All verification tests passed!")
    print("RankMe is ready to use.")


if __name__ == "__main__":
    main()