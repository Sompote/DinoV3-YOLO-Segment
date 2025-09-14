#!/usr/bin/env python3
"""
Example usage of YOLOv12 + DINOv3 integration

This script demonstrates how to use the DINOv3-enhanced YOLOv12 models.
"""

import torch
from ultralytics import YOLO
from pathlib import Path

def example_model_loading():
    """Example of loading different DINOv3 variants."""
    print("🚀 Loading YOLOv12 + DINOv3 Models")
    print("=" * 40)
    
    # Available configurations
    configs = {
        'Small (Fast)': 'ultralytics/cfg/models/v12/yolov12-dino3-small.yaml',
        'Base (Balanced)': 'ultralytics/cfg/models/v12/yolov12-dino3.yaml', 
        'Large (Accurate)': 'ultralytics/cfg/models/v12/yolov12-dino3-large.yaml',
        'ConvNeXt (Hybrid)': 'ultralytics/cfg/models/v12/yolov12-dino3-convnext.yaml'
    }
    
    for name, config_path in configs.items():
        try:
            print(f"\n📦 Loading {name} variant...")
            model = YOLO(config_path)
            print(f"   ✅ {name} loaded successfully!")
            
            # Print model info
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"   📊 Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"   ❌ Error loading {name}: {e}")

def example_training():
    """Example training script."""
    print("\n🏋️  Training Example")
    print("=" * 40)
    
    # Load model
    model = YOLO('ultralytics/cfg/models/v12/yolov12-dino3.yaml')
    
    print("📝 To train the model, use:")
    print("""
# Basic training
model.train(
    data='coco.yaml',          # Dataset configuration
    epochs=100,                # Number of epochs
    batch=16,                  # Batch size
    imgsz=640,                 # Image size
    device=0,                  # GPU device
    freeze_backbone=True       # Keep DINOv3 frozen (recommended)
)

# Advanced training with fine-tuning
model.train(
    data='coco.yaml',
    epochs=50,
    batch=8,                   # Smaller batch for fine-tuning
    lr=1e-5,                   # Lower learning rate
    freeze_backbone=False      # Allow DINOv3 fine-tuning
)
    """)

def example_inference():
    """Example inference usage."""
    print("\n🔍 Inference Example")
    print("=" * 40)
    
    # Load pretrained model (would be available after training)
    print("📝 To run inference:")
    print("""
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/trained/yolov12-dino3.pt')

# Single image inference
results = model('path/to/image.jpg')

# Batch inference
results = model(['image1.jpg', 'image2.jpg'])

# Video inference
results = model('path/to/video.mp4')

# Real-time inference
results = model(source=0)  # Webcam
    """)

def example_model_analysis():
    """Analyze model architecture."""
    print("\n🔬 Model Analysis Example")
    print("=" * 40)
    
    try:
        model = YOLO('ultralytics/cfg/models/v12/yolov12-dino3.yaml')
        
        # Model summary
        print("📊 Model Summary:")
        model.info(detailed=False)
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model.model(dummy_input)
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shapes: {[o.shape for o in output]}")
            
    except Exception as e:
        print(f"❌ Error in model analysis: {e}")

def example_custom_training_script():
    """Complete training script example."""
    training_script = '''
#!/usr/bin/env python3
"""
Custom training script for YOLOv12 + DINOv3
"""

from ultralytics import YOLO
import torch

def train_yolov12_dino3():
    """Train YOLOv12 with DINOv3 backbone."""
    
    # Choose your configuration
    configs = {
        'development': 'ultralytics/cfg/models/v12/yolov12-dino3-small.yaml',
        'production': 'ultralytics/cfg/models/v12/yolov12-dino3.yaml',
        'research': 'ultralytics/cfg/models/v12/yolov12-dino3-large.yaml'
    }
    
    # Load model
    model = YOLO(configs['production'])
    
    # Phase 1: Train with frozen DINOv3
    print("Phase 1: Training with frozen DINOv3 backbone...")
    model.train(
        data='coco.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        project='yolov12_dino3',
        name='phase1_frozen',
        freeze_backbone=True,
        amp=True,  # Mixed precision
        save_period=10
    )
    
    # Phase 2: Fine-tune with unfrozen DINOv3 (optional)
    print("Phase 2: Fine-tuning with unfrozen DINOv3...")
    
    # Load best model from phase 1
    model = YOLO('yolov12_dino3/phase1_frozen/weights/best.pt')
    
    # Unfreeze DINOv3 backbone
    for module in model.model.modules():
        if hasattr(module, 'unfreeze_backbone'):
            module.unfreeze_backbone()
    
    model.train(
        data='coco.yaml',
        epochs=25,
        batch=8,   # Smaller batch for fine-tuning
        lr=1e-5,   # Lower learning rate
        device=0,
        project='yolov12_dino3',
        name='phase2_finetuned',
        amp=True
    )
    
    print("Training completed!")

if __name__ == "__main__":
    train_yolov12_dino3()
'''
    
    print("\n📜 Complete Training Script:")
    print("=" * 40)
    print("Save this as 'train_yolov12_dino3.py':")
    print(training_script)

def main():
    """Run all examples."""
    print("🎯 YOLOv12 + DINOv3 Integration Examples")
    print("=" * 50)
    
    examples = [
        example_model_loading,
        example_training,
        example_inference,
        example_model_analysis,
        example_custom_training_script
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
        print()

if __name__ == "__main__":
    main()