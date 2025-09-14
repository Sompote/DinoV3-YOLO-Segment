#!/usr/bin/env python3
"""
Custom DINO Configuration Creator

This script creates a temporary YAML configuration file with custom DINO input
for use with --dino-input parameter in training and testing scripts.

Usage:
    python create_custom_dino_config.py --dino-input facebook/dinov2-base --yolo-size s --output temp_config.yaml
"""

import argparse
import yaml
import tempfile
from pathlib import Path

YOLO_BASE_CONFIG = {
    'nc': 80,
    'scales': {
        'n': [0.50, 0.25, 1024],
        's': [0.50, 0.50, 1024], 
        'm': [0.67, 0.75, 768],
        'l': [1.00, 1.00, 512],
        'x': [1.00, 1.25, 512]
    }
}

def create_custom_dino_config(dino_input, yolo_size='s', integration='single', output_channels=1024):
    """Create custom DINO configuration"""
    
    config = YOLO_BASE_CONFIG.copy()
    
    # Backbone configuration
    backbone = [
        [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]], # 1-P2/4
        [-1, 3, 'C3k2', [256, False, 0.25]],
        [-1, 1, 'Conv', [256, 3, 2]], # 3-P3/8
        [-1, 6, 'C3k2', [512, False, 0.25]],
    ]
    
    if integration == 'dual':
        # Add P3 DINOv3 enhancement for dual integration
        backbone.extend([
            [-1, 1, 'DINO3Backbone', [dino_input, True, 512]],  # 5: DINOv3 P3
            [-1, 1, 'Conv', [512, 3, 2]], # 6-P4/16
            [-1, 6, 'C3k2', [1024, True]],
            [-1, 1, 'DINO3Backbone', [dino_input, True, 1024]],  # 8: DINOv3 P4
            [-1, 1, 'Conv', [1024, 3, 2]], # 9-P5/32
            [-1, 3, 'C3k2', [1024, True]]
        ])
        
        # Dual-scale head
        head = [
            [-1, 1, 'nn.Upsample', [None, 2, "nearest"]], # 11
            [[-1, 8], 1, 'Concat', [1]], # cat P4 (DINOv3-enhanced)
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 13
            [-1, 1, 'nn.Upsample', [None, 2, "nearest"]], # 14
            [[-1, 5], 1, 'Concat', [1]], # cat P3 (DINOv3-enhanced)
            [-1, 3, 'C2fPSA', [512, 0.5]], # 16 (P3/8-small)
            [-1, 1, 'Conv', [512, 3, 2]], # 17
            [[-1, 13], 1, 'Concat', [1]], # cat head P4
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 19 (P4/16-medium)
            [-1, 1, 'Conv', [1024, 3, 2]], # 20
            [[-1, 10], 1, 'Concat', [1]], # cat head P5
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 22 (P5/32-large)
            [[16, 19, 22], 1, 'Detect', ['nc']] # Detect(P3, P4, P5)
        ]
    else:
        # Single-scale integration (P4 only)
        backbone.extend([
            [-1, 1, 'Conv', [512, 3, 2]], # 5-P4/16
            [-1, 6, 'C3k2', [1024, True]],
            [-1, 1, 'DINO3Backbone', [dino_input, True, output_channels]], # 7: DINOv3 P4
            [-1, 1, 'Conv', [1024, 3, 2]], # 8-P5/32
            [-1, 3, 'C3k2', [1024, True]]
        ])
        
        # Single-scale head
        head = [
            [-1, 1, 'nn.Upsample', [None, 2, "nearest"]], # 10
            [[-1, 7], 1, 'Concat', [1]], # cat P4 (DINOv3-enhanced)
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 12
            [-1, 1, 'nn.Upsample', [None, 2, "nearest"]], # 13
            [[-1, 4], 1, 'Concat', [1]], # cat backbone P3
            [-1, 3, 'C2fPSA', [512, 0.5]], # 15 (P3/8-small)
            [-1, 1, 'Conv', [512, 3, 2]], # 16
            [[-1, 12], 1, 'Concat', [1]], # cat head P4
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 18 (P4/16-medium)
            [-1, 1, 'Conv', [1024, 3, 2]], # 19
            [[-1, 9], 1, 'Concat', [1]], # cat head P5
            [-1, 3, 'C2fPSA', [1024, 0.5]], # 21 (P5/32-large)
            [[15, 18, 21], 1, 'Detect', ['nc']] # Detect(P3, P4, P5)
        ]
    
    config['backbone'] = backbone
    config['head'] = head
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Create custom DINO configuration')
    parser.add_argument('--dino-input', type=str, required=True,
                       help='Custom DINO model input/path')
    parser.add_argument('--yolo-size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--integration', type=str, default='single', choices=['single', 'dual'],
                       help='Integration type')
    parser.add_argument('--output', type=str, default=None,
                       help='Output config file path')
    parser.add_argument('--output-channels', type=int, default=1024,
                       help='Output channels for DINO backbone')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_custom_dino_config(
        args.dino_input, args.yolo_size, args.integration, args.output_channels
    )
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Create temporary file
        temp_dir = Path('ultralytics/cfg/models/v12')
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f'yolov12{args.yolo_size}-dino-custom-temp.yaml'
    
    # Save configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Custom DINO configuration created: {output_path}")
    print(f"   DINO input: {args.dino_input}")
    print(f"   YOLO size: {args.yolo_size}")
    print(f"   Integration: {args.integration}")
    print(f"   Output channels: {args.output_channels}")
    
    return str(output_path)

if __name__ == '__main__':
    main()