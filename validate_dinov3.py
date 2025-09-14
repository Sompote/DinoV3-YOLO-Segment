#!/usr/bin/env python3
"""
DINOv3 Validation Script

This script validates that we are actually loading DINOv3 models from the official 
Facebook Research repository, not DINOv2 or other fallback models.

Usage:
    python validate_dinov3.py --model dinov3_vitb16
    python validate_dinov3.py --test-all
"""

import argparse
import sys
import torch
from pathlib import Path

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.nn.modules.block import DINO3Backbone

def validate_model_source(model_name):
    """Validate that we're loading actual DINOv3 models"""
    print(f"\n🔍 Validating DINOv3 model: {model_name}")
    print("=" * 50)
    
    try:
        # Create DINO3Backbone with the model
        backbone = DINO3Backbone(model_name, freeze_backbone=True, output_channels=512)
        
        # Check model attributes to verify it's DINOv3
        model = backbone.dino_model
        
        validation_results = {
            'model_loaded': True,
            'is_dinov3': False,
            'source': 'unknown',
            'embed_dim': backbone.embed_dim,
            'model_type': backbone.model_type,
            'attributes': []
        }
        
        # Check for DINOv3 specific attributes
        model_str = str(type(model))
        print(f"📋 Model type: {model_str}")
        
        # Check if it has DINOv3 characteristics
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__
            print(f"📋 Class name: {class_name}")
            validation_results['attributes'].append(f"Class: {class_name}")
            
            # DINOv3 models typically have specific class names
            if 'dinov3' in class_name.lower() or 'DinoV3' in class_name:
                validation_results['is_dinov3'] = True
                validation_results['source'] = 'official_dinov3'
            elif 'dinov2' in class_name.lower() or 'DinoV2' in class_name:
                validation_results['source'] = 'dinov2_fallback'
            
        # Check model configuration
        if hasattr(model, 'config'):
            config = model.config
            print(f"📋 Config: {type(config)}")
            if hasattr(config, 'model_type'):
                model_type_config = config.model_type
                print(f"📋 Config model_type: {model_type_config}")
                validation_results['attributes'].append(f"Config type: {model_type_config}")
                
                if 'dinov3' in str(model_type_config).lower():
                    validation_results['is_dinov3'] = True
                    validation_results['source'] = 'official_dinov3'
        
        # Check embedding dimension consistency
        expected_dims = {
            'dinov3_vits16': 384,
            'dinov3_vitb16': 768,
            'dinov3_vitl16': 1024,
            'dinov3_vith16plus': 1280,
            'dinov3_vit7b16': 4096,
            'dinov3_convnext_base': 1024,
            'vitb16': 768,  # alias
            'convnext_base': 1024  # alias
        }
        
        expected_dim = expected_dims.get(model_name, 768)
        if backbone.embed_dim == expected_dim:
            print(f"✅ Embedding dimension correct: {backbone.embed_dim}")
            validation_results['dim_correct'] = True
        else:
            print(f"⚠️  Embedding dimension mismatch: got {backbone.embed_dim}, expected {expected_dim}")
            validation_results['dim_correct'] = False
        
        # Test forward pass
        print(f"🧪 Testing forward pass...")
        test_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = backbone(test_input)
        
        expected_shape = (1, 512, 40, 40)  # P4 level
        if output.shape == expected_shape:
            print(f"✅ Forward pass successful: {output.shape}")
            validation_results['forward_pass'] = True
        else:
            print(f"❌ Forward pass shape mismatch: got {output.shape}, expected {expected_shape}")
            validation_results['forward_pass'] = False
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {
            'model_loaded': False,
            'error': str(e),
            'is_dinov3': False,
            'source': 'failed'
        }

def print_validation_summary(results):
    """Print validation summary"""
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"\n🔍 {model_name}:")
        
        if result['model_loaded']:
            # Model source verification
            if result['source'] == 'official_dinov3':
                print(f"   ✅ Source: Official DINOv3")
            elif result['source'] == 'dinov2_fallback':
                print(f"   ⚠️  Source: DINOv2 Fallback")
            else:
                print(f"   ❓ Source: {result['source']}")
            
            # Embedding dimension
            dim_status = "✅" if result.get('dim_correct', False) else "❌"
            print(f"   {dim_status} Embed dim: {result['embed_dim']}")
            
            # Forward pass
            forward_status = "✅" if result.get('forward_pass', False) else "❌"
            print(f"   {forward_status} Forward pass")
            
            # Model type
            print(f"   📋 Type: {result['model_type']}")
            
            # Additional attributes
            for attr in result.get('attributes', []):
                print(f"   📋 {attr}")
                
        else:
            print(f"   ❌ Failed to load: {result.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description='Validate DINOv3 model loading')
    parser.add_argument('--model', type=str, default='dinov3_vitb16',
                       help='Model to validate')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all official DINOv3 models')
    
    args = parser.parse_args()
    
    print("🔬 DINOv3 Model Validation")
    print("Verifying we're loading official DINOv3 models from:")
    print("📍 https://github.com/facebookresearch/dinov3")
    
    if args.test_all:
        # Test all official DINOv3 models
        models_to_test = [
            'dinov3_vits16',
            'dinov3_vitb16', 
            'dinov3_vitl16',
            'dinov3_convnext_base',
            'vitb16',  # alias
            'convnext_base'  # alias
        ]
    else:
        models_to_test = [args.model]
    
    results = {}
    
    for model_name in models_to_test:
        results[model_name] = validate_model_source(model_name)
    
    # Print summary
    print_validation_summary(results)
    
    # Final verdict
    official_dinov3_count = sum(1 for r in results.values() 
                               if r.get('source') == 'official_dinov3')
    total_models = len([r for r in results.values() if r.get('model_loaded')])
    
    print(f"\n🎯 FINAL VERDICT:")
    print(f"   Official DINOv3 models: {official_dinov3_count}/{total_models}")
    
    if official_dinov3_count == total_models and total_models > 0:
        print(f"   ✅ SUCCESS: All models are using official DINOv3!")
        sys.exit(0)
    elif official_dinov3_count > 0:
        print(f"   ⚠️  PARTIAL: Some models using fallback methods")
        sys.exit(1)
    else:
        print(f"   ❌ FAILURE: No official DINOv3 models loaded")
        sys.exit(2)

if __name__ == '__main__':
    main()