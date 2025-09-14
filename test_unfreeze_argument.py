#!/usr/bin/env python3
"""
Test the --unfreeze-dino argument functionality
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add the current directory to path to import the training script
sys.path.insert(0, '.')
from train_yolov12_dino import parse_arguments, modify_yaml_config_for_custom_dino

def test_unfreeze_dino_argument():
    """Test that --unfreeze-dino argument works correctly."""
    
    print("🧪 Testing --unfreeze-dino Argument")
    print("="*50)
    
    # Test 1: Default behavior (weights should be frozen)
    print("\n1️⃣ Testing DEFAULT behavior (frozen weights)...")
    sys.argv = ['train_yolov12_dino.py', '--data', 'test.yaml', '--yolo-size', 's']
    args = parse_arguments()
    print(f"   Default args.unfreeze_dino: {args.unfreeze_dino}")
    assert args.unfreeze_dino == False, "Default should be False (frozen)"
    print("   ✅ Default: DINO weights FROZEN")
    
    # Test 2: With --unfreeze-dino flag (weights should be trainable)  
    print("\n2️⃣ Testing --unfreeze-dino flag (trainable weights)...")
    sys.argv = ['train_yolov12_dino.py', '--data', 'test.yaml', '--yolo-size', 's', '--unfreeze-dino']
    args = parse_arguments()
    print(f"   With --unfreeze-dino args.unfreeze_dino: {args.unfreeze_dino}")
    assert args.unfreeze_dino == True, "--unfreeze-dino should be True (trainable)"
    print("   ✅ With --unfreeze-dino: DINO weights TRAINABLE")
    
    # Test 3: Test YAML modification logic
    print("\n3️⃣ Testing YAML configuration modification...")
    
    # Create a test YAML config
    test_config = {
        'backbone': [
            [-1, 1, 'Conv', [32, 3, 2]],
            [0, 1, 'DINO3Preprocessor', ['DINO_MODEL_NAME', True, 3]],  # [model, freeze_backbone, output_channels]
            [-1, 1, 'Conv', [64, 3, 2]]
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='-preprocess.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        # Test with unfreeze_dino=False (should freeze weights)
        print("   Testing unfreeze_dino=False (freeze weights)...")
        modified_path = modify_yaml_config_for_custom_dino(
            temp_config_path, 'dinov3_vitb16', 's', unfreeze_dino=False
        )
        
        with open(modified_path, 'r') as f:
            modified_config = yaml.safe_load(f)
        
        freeze_backbone = modified_config['backbone'][1][3][1]
        print(f"   freeze_backbone parameter: {freeze_backbone}")
        assert freeze_backbone == True, "unfreeze_dino=False should set freeze_backbone=True"
        print("   ✅ unfreeze_dino=False → freeze_backbone=True (FROZEN)")
        
        # Test with unfreeze_dino=True (should unfreeze weights)
        print("   Testing unfreeze_dino=True (unfreeze weights)...")
        modified_path = modify_yaml_config_for_custom_dino(
            temp_config_path, 'dinov3_vitb16', 's', unfreeze_dino=True
        )
        
        with open(modified_path, 'r') as f:
            modified_config = yaml.safe_load(f)
        
        freeze_backbone = modified_config['backbone'][1][3][1]
        print(f"   freeze_backbone parameter: {freeze_backbone}")
        assert freeze_backbone == False, "unfreeze_dino=True should set freeze_backbone=False"
        print("   ✅ unfreeze_dino=True → freeze_backbone=False (TRAINABLE)")
        
    finally:
        # Clean up
        Path(temp_config_path).unlink()
        if modified_path != temp_config_path:
            Path(modified_path).unlink()
    
    print("\n🎉 All tests passed!")
    print("\n📚 Usage Examples:")
    print("   # Default (frozen weights):")
    print("   python train_yolov12_dino.py --data data.yaml --yolo-size s --dino-variant vitb16")
    print("   ")
    print("   # Trainable weights:")  
    print("   python train_yolov12_dino.py --data data.yaml --yolo-size s --dino-variant vitb16 --unfreeze-dino")

if __name__ == "__main__":
    test_unfreeze_dino_argument()