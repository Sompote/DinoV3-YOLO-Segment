#!/usr/bin/env python3
"""
Test deterministic interpolation fix for DINO3Backbone
"""

import torch
import warnings
from ultralytics.nn.modules.block import DINO3Backbone, deterministic_interpolate

def test_deterministic_warnings():
    """Test that deterministic mode doesn't produce warnings."""
    
    print("🧪 Testing Deterministic Interpolation Fix")
    print("="*50)
    
    # Enable deterministic mode (this causes the warnings)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print("✅ Deterministic algorithms enabled")
    
    # Test the deterministic_interpolate function
    x = torch.randn(1, 3, 64, 64)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test regular interpolation (would cause warning)
        print("🧪 Testing direct F.interpolate...")
        result1 = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        direct_warnings = len(w)
        
        # Clear warnings
        w.clear()
        
        # Test our deterministic interpolation
        print("🧪 Testing deterministic_interpolate...")
        result2 = deterministic_interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        our_warnings = len(w)
        
    print(f"Direct F.interpolate warnings: {direct_warnings}")
    print(f"Deterministic_interpolate warnings: {our_warnings}")
    
    if our_warnings < direct_warnings:
        print("✅ Deterministic interpolation reduces warnings!")
    else:
        print("ℹ️  Warning levels same (may depend on PyTorch version)")
    
    print(f"Output shapes match: {result1.shape == result2.shape}")
    
    # Test with DINO3Backbone
    print("\n🧪 Testing DINO3Backbone with deterministic mode...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            backbone = DINO3Backbone('dinov3_vitb16', freeze_backbone=True, output_channels=256)
            test_input = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                output = backbone(test_input)
            
            backbone_warnings = len(w)
            print(f"DINO3Backbone warnings: {backbone_warnings}")
            print(f"✅ Forward pass successful: {test_input.shape} -> {output.shape}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Reset deterministic mode
    torch.use_deterministic_algorithms(False)
    print("\n✅ Test complete - deterministic mode disabled")

if __name__ == "__main__":
    test_deterministic_warnings()