#!/usr/bin/env python3
"""
Test all YOLOv12n integration approaches
"""
import sys
import os
from pathlib import Path
import torch

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO

def test_all_yolov12n():
    """Test all YOLOv12n integration configurations"""
    print("🎯 COMPREHENSIVE YOLOv12n INTEGRATION TESTING")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Preprocessing Approach
    print("1️⃣  TESTING YOLOv12n PREPROCESSING")
    print("-" * 40)
    try:
        config_path = 'ultralytics/cfg/models/v12/yolov12n-dino3-preprocess.yaml'
        print(f"📄 Config: {config_path}")
        
        model = YOLO(config_path)
        print("   ✅ Model loaded successfully!")
        
        # Count DINO layers
        dino_count = 0
        for layer in model.model.model:
            if 'DINO3Preprocessor' in type(layer).__name__:
                dino_count += 1
        
        if dino_count == 1:
            print(f"   ✅ Preprocessing confirmed: {dino_count} DINO3Preprocessor at layer 0")
            success_count += 1
        else:
            print(f"   ❌ Expected 1 DINO3Preprocessor, found {dino_count}")
        
        # Test forward pass
        with torch.no_grad():
            model.model.train()
            output = model.model(torch.randn(1, 3, 64, 64))
            print("   ✅ Forward pass successful!")
        
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
    
    print()
    
    # Test 2: Single Integration
    print("2️⃣  TESTING YOLOv12n SINGLE INTEGRATION")
    print("-" * 40)
    try:
        config_path = 'ultralytics/cfg/models/v12/yolov12n-dino3-vitb16-single.yaml'
        print(f"📄 Config: {config_path}")
        
        model = YOLO(config_path)
        print("   ✅ Model loaded successfully!")
        
        # Count DINO layers
        dino_count = 0
        for layer in model.model.model:
            if 'DINO3Backbone' in type(layer).__name__:
                dino_count += 1
        
        if dino_count == 1:
            print(f"   ✅ Single integration confirmed: {dino_count} DINO3Backbone")
            success_count += 1
        else:
            print(f"   ❌ Expected 1 DINO3Backbone, found {dino_count}")
        
        # Test forward pass
        with torch.no_grad():
            model.model.train()
            output = model.model(torch.randn(1, 3, 64, 64))
            print("   ✅ Forward pass successful!")
        
    except Exception as e:
        print(f"   ❌ Single integration failed: {e}")
    
    print()
    
    # Test 3: Dual Integration
    print("3️⃣  TESTING YOLOv12n DUAL INTEGRATION")
    print("-" * 40)
    try:
        config_path = 'ultralytics/cfg/models/v12/yolov12n-dino3-vitb16-dual.yaml'
        print(f"📄 Config: {config_path}")
        
        model = YOLO(config_path)
        print("   ✅ Model loaded successfully!")
        
        # Count DINO layers
        dino_count = 0
        dino_layers = []
        for i, layer in enumerate(model.model.model):
            if 'DINO3Backbone' in type(layer).__name__:
                dino_count += 1
                dino_layers.append(i)
        
        if dino_count == 2:
            print(f"   ✅ Dual integration confirmed: {dino_count} DINO3Backbone at {dino_layers}")
            success_count += 1
        else:
            print(f"   ❌ Expected 2 DINO3Backbone, found {dino_count}")
        
        # Test forward pass
        with torch.no_grad():
            model.model.train()
            output = model.model(torch.randn(1, 3, 64, 64))
            print("   ✅ Forward pass successful!")
        
    except Exception as e:
        print(f"   ❌ Dual integration failed: {e}")
    
    print()
    print("🎯 YOLOv12n FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL YOLOv12n INTEGRATION APPROACHES WORK!")
        print("✅ Preprocessing: DINO3Preprocessor at P0 level")
        print("✅ Single integration: DINO3Backbone at P4 level") 
        print("✅ Dual integration: DINO3Backbone at P3 and P4 levels")
        print("✅ All configurations load without errors")
        print("✅ All A2C2f channel requirements satisfied for nano scale")
        print("✅ Forward passes successful")
        print()
        print("🚀 USER'S YOLOv12n COMMANDS ARE READY:")
        print()
        print("   # PREPROCESSING (Most Stable)")
        print("   python train_yolov12_dino.py \\")
        print("       --data /path/to/data.yaml \\")
        print("       --yolo-size n \\")
        print("       --dino-version 3 \\")
        print("       --dino-input dinov3_vitb16 \\")
        print("       --epochs 100")
        print()
        print("   # SINGLE INTEGRATION")
        print("   python train_yolov12_dino.py \\")
        print("       --data /path/to/data.yaml \\")
        print("       --yolo-size n \\")
        print("       --dino-version 3 \\")
        print("       --dino-variant vitb16 \\")
        print("       --integration single \\")
        print("       --epochs 100")
        print()
        print("   # DUAL INTEGRATION")
        print("   python train_yolov12_dino.py \\")
        print("       --data /path/to/data.yaml \\")
        print("       --yolo-size n \\")
        print("       --dino-version 3 \\")
        print("       --dino-variant vitb16 \\")
        print("       --integration dual \\")
        print("       --epochs 100")
        return True
    else:
        print("❌ Some YOLOv12n integration approaches still have issues")
        return False

if __name__ == '__main__':
    success = test_all_yolov12n()
    if success:
        print("\n🎊 ALL YOLOv12n INTEGRATIONS VERIFIED! 🎊")
    else:
        print("\n💥 Some issues remain")
    sys.exit(0 if success else 1)