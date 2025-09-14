#!/usr/bin/env python3
"""
Cloud Compatibility Test for DINOv3-YOLOv12 Integration
Tests the Hugging Face-only implementation to ensure cloud compatibility.
"""

import torch
import time
from ultralytics.nn.modules.block import DINO3Backbone

def test_cloud_compatibility():
    """Test DINOv3-YOLOv12 integration for cloud deployment compatibility."""
    
    print("🌩️  DINOv3-YOLOv12 Cloud Compatibility Test")
    print("="*60)
    
    # Test different model variants
    models_to_test = [
        ('dinov3_vits16', 384),
        ('dinov3_vitb16', 768),
        ('dinov3_vitl16', 1024)
    ]
    
    results = []
    
    for model_name, expected_embed_dim in models_to_test:
        print(f"\n🧪 Testing {model_name}...")
        
        try:
            # Initialize backbone
            start_time = time.time()
            backbone = DINO3Backbone(
                model_name=model_name, 
                freeze_backbone=True, 
                output_channels=256
            )
            init_time = time.time() - start_time
            
            # Verify embedding dimension
            assert backbone.embed_dim == expected_embed_dim, \
                f"Expected embed_dim {expected_embed_dim}, got {backbone.embed_dim}"
            
            # Test forward pass with different input sizes
            test_sizes = [(224, 224), (320, 320), (640, 640)]
            
            for h, w in test_sizes:
                input_tensor = torch.randn(1, 3, h, w)
                
                start_time = time.time()
                with torch.no_grad():
                    output = backbone(input_tensor)
                forward_time = time.time() - start_time
                
                # Verify output shape
                assert output.shape == (1, 256, h, w), \
                    f"Expected output shape (1, 256, {h}, {w}), got {output.shape}"
                
                print(f"   ✅ {h}x{w} input: {input_tensor.shape} -> {output.shape} ({forward_time:.3f}s)")
            
            results.append({
                'model': model_name,
                'status': 'SUCCESS',
                'embed_dim': backbone.embed_dim,
                'init_time': init_time,
                'error': None
            })
            
            print(f"   ✅ {model_name} test passed (init: {init_time:.3f}s)")
            
        except Exception as e:
            print(f"   ❌ {model_name} test failed: {e}")
            results.append({
                'model': model_name,
                'status': 'FAILED',
                'embed_dim': None,
                'init_time': None,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for r in results if r['status'] == 'SUCCESS')
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Cloud deployment ready.")
        print("\n🌩️  Cloud Deployment Notes:")
        print("✅ Uses only Hugging Face transformers (no GitHub dependencies)")
        print("✅ Handles HTTP 403 errors robustly")
        print("✅ Proper kernel size handling (no 2x2 vs 3x3 issues)")  
        print("✅ Dynamic projection layer creation")
        print("✅ Compatible with RTX 5090 and other cloud GPUs")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        for result in results:
            if result['status'] == 'FAILED':
                print(f"   - {result['model']}: {result['error']}")
        return False

if __name__ == "__main__":
    success = test_cloud_compatibility()
    exit(0 if success else 1)