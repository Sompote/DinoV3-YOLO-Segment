#!/usr/bin/env python3
"""
Debug script to understand YOLOv12s channel dimensions
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
import yaml

def debug_base_yolo():
    """Debug base YOLOv12s to understand actual channel dimensions"""
    try:
        # Load base YOLOv12s
        print("🔍 Loading base YOLOv12s to understand channel flow...")
        model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')
        
        # Force scale to 's'
        model.model.args = {'scale': 's'}
        
        print("✅ Base YOLOv12s loaded!")
        
        # Create hook to capture layer outputs
        layer_outputs = {}
        def create_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'shape'):
                    layer_outputs[name] = output.shape
                elif isinstance(output, (list, tuple)):
                    layer_outputs[name] = [x.shape if hasattr(x, 'shape') else str(x) for x in output]
                else:
                    layer_outputs[name] = str(output)
            return hook
        
        # Register hooks for backbone layers
        for i, layer in enumerate(model.model.model[:9]):  # First 9 layers (backbone)
            layer.register_forward_hook(create_hook(f'layer_{i}'))
        
        # Test forward pass
        print("🔍 Running forward pass to capture layer outputs...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            model.model.eval()
            _ = model.model(dummy_input)
        
        print("\n📊 Layer Output Shapes:")
        for name, shape in layer_outputs.items():
            print(f"   {name}: {shape}")
            
        # Print model structure  
        print("\n🏗️ Model Structure:")
        for i, layer in enumerate(model.model.model):
            print(f"   {i}: {layer}")
            if i >= 8:  # Only show first 9 layers
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    debug_base_yolo()