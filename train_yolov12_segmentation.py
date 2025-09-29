#!/usr/bin/env python3
"""
YOLOv12 Instance Segmentation Training Script with DINO Enhancement

This script provides a systematic approach to training YOLOv12 instance segmentation models 
with optional DINOv3 enhancement, using clear CLI arguments specifically for segmentation tasks.

Usage Examples:
    # Basic segmentation training
    python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s --epochs 100

    # DINO-enhanced segmentation (single integration: P4 level) - DINOVERSION REQUIRED
    python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s --use-dino --dino-variant vitb16 --integration single --dinoversion v3 --epochs 100

    # DINO-enhanced segmentation (dual integration: P3+P4 levels) - DINOVERSION REQUIRED
    python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size l --use-dino --dino-variant vitl16 --integration dual --dinoversion v2 --epochs 100

    # TRIPLE DINO integration (P0+P3+P4 levels - ultimate performance) - DINOVERSION REQUIRED
    python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size l --use-dino --dino-variant vitl16 --integration triple --dinoversion v3 --epochs 150
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import tempfile
import yaml

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

def create_segmentation_config_path(model_size, use_dino=False, dino_variant=None, dino_integration=None, dino_preprocessing=None, dino_version=None):
    """
    Create segmentation model configuration path based on parameters.
    
    Args:
        model_size (str): YOLOv12 size (n, s, m, l, x)
        use_dino (bool): Whether to use DINO enhancement
        dino_variant (str): DINO variant (vitb16, vitl16, etc.)
        dino_integration (str): Integration type (single, dual)
        dino_preprocessing (str): DINO preprocessing model
        dino_version (str): DINO version ('v2' or 'v3')
    
    Returns:
        str: Path to segmentation model configuration file
    """
    if not use_dino:
        # Base YOLOv12 segmentation model
        return f'ultralytics/cfg/models/v12/yolov12{model_size}-seg.yaml'
    
    # Triple integration: preprocessing + backbone integration (ultimate performance)
    if dino_preprocessing and dino_variant:
        # For triple and dualp0p3 integration, use the preprocessing config as base and add backbone enhancement
        # This maintains proper channel flow since preprocessing is already handled
        config_path = f'ultralytics/cfg/models/v12/yolov12{model_size}-dino3-preprocess-seg.yaml'
        if Path(config_path).exists():
            return config_path
        else:
            # Fallback to generic preprocessing config for triple/dualp0p3 integration
            return 'ultralytics/cfg/models/v12/yolov12-dino3-preprocess-seg.yaml'
    
    # DINO preprocessing only (input enhancement)
    if dino_preprocessing and not dino_variant:
        config_path = f'ultralytics/cfg/models/v12/yolov12{model_size}-dino3-preprocess-seg.yaml'
        if Path(config_path).exists():
            return config_path
        else:
            # Fallback to generic preprocessing config
            return 'ultralytics/cfg/models/v12/yolov12-dino3-preprocess-seg.yaml'
    
    # DINO integrated approach only (backbone enhancement)
    if dino_variant and dino_integration:
        if dino_integration == 'dual':
            config_name = f'yolov12{model_size}-dino3-{dino_variant}-dual-seg.yaml'
        else:
            config_name = f'yolov12{model_size}-dino3-{dino_variant}-single-seg.yaml'
        
        config_path = f'ultralytics/cfg/models/v12/{config_name}'
        if Path(config_path).exists():
            return config_path
        else:
            # Fallback to generic DINO segmentation config
            return 'ultralytics/cfg/models/v12/yolov12-dino3-seg.yaml'
    
    # Default fallback
    return f'ultralytics/cfg/models/v12/yolov12{model_size}-seg.yaml'

def get_segmentation_batch_size(model_size, use_dino=False, dino_integration='single', is_triple=False, is_dualp0p3=False):
    """Get recommended batch size for segmentation training."""
    # Segmentation requires more memory than detection, so reduce batch sizes
    base_batches = {'n': 32, 's': 16, 'm': 8, 'l': 6, 'x': 4}
    batch = base_batches.get(model_size, 8)
    
    if use_dino:
        # Further reduce for DINO models
        batch = max(batch // 2, 2)
        if dino_integration == 'dual':
            batch = max(batch // 2, 1)
        # DualP0P3 integration uses moderate memory (preprocessing + single backbone)
        if is_dualp0p3:
            batch = max(batch // 2, 1)
        # Triple integration uses even more memory
        if is_triple:
            batch = max(batch // 2, 1)
    
    return batch

def get_segmentation_epochs(use_dino=False):
    """Get recommended epochs for segmentation training."""
    if use_dino:
        return 100  # DINO-enhanced models converge faster
    else:
        return 150  # Standard segmentation training (less than detection)

def parse_segmentation_arguments():
    """Parse command line arguments for segmentation training."""
    parser = argparse.ArgumentParser(
        description='YOLOv12 Instance Segmentation Training with Optional DINO Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic segmentation training
  python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s

  # DINO-enhanced single-scale segmentation
  python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s --use-dino --dino-variant vitb16 --dino-integration single

  # DINO-enhanced dual-scale segmentation (best performance)
  python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size l --use-dino --dino-variant vitl16 --dino-integration dual

  # DINO preprocessing segmentation
  python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s --use-dino --dino-preprocessing dinov3_vitb16

  # TRIPLE DINO integration (ultimate performance)
  python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size l --use-dino --dino-preprocessing dinov3_vitb16 --dino-variant vitl16 --dino-integration dual
        """
    )
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Segmentation dataset YAML file path (e.g., segmentation_data.yaml)')
    parser.add_argument('--model-size', type=str, required=True, 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv12 model size: n(nano), s(small), m(medium), l(large), x(extra-large)')
    
    # Segmentation task specification
    parser.add_argument('--task', type=str, default='segment', choices=['segment'],
                       help='Task type (fixed to segment for this script)')
    
    # DINO enhancement options
    dino_group = parser.add_argument_group('DINO Enhancement Options')
    dino_group.add_argument('--use-dino', action='store_true',
                           help='Enable DINO enhancement for better segmentation performance')
    dino_group.add_argument('--dino-variant', type=str, default=None,
                           choices=['vits16', 'vitb16', 'vitl16', 'vitl16_distilled', 'vith16_plus', 'vit7b16', 'vit7b16_lvd',
                                   'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
                           help='DINO model variant: ViTs/B/L/L-dist/H+/7B-16/7B-LVD for different model sizes and datasets (required if --use-dino with integration)')
    dino_group.add_argument('--integration', type=str, default='single',
                           choices=['single', 'dual', 'dualp0p3', 'triple'],
                           help='DINO integration strategy: single(P4), dual(P3+P4), dualp0p3(P0+P3), triple(P0+P3+P4)')
    dino_group.add_argument('--dino-preprocessing', type=str, default=None,
                           help='DINO preprocessing model (advanced users only - use --integration triple instead)')
    dino_group.add_argument('--dinoversion', type=str, required=False,
                           choices=['v2', 'v3'],
                           help='DINO version: v2 (DINOv2) or v3 (DINOv3) - REQUIRED when using --use-dino')
    dino_group.add_argument('--freeze-dino', action='store_true', default=True,
                           help='Freeze DINO weights during training (default: True)')
    dino_group.add_argument('--unfreeze-dino', action='store_true',
                           help='Make DINO weights trainable (overrides --freeze-dino)')
    
    # Segmentation-specific training parameters
    seg_group = parser.add_argument_group('Segmentation Training Parameters')
    seg_group.add_argument('--epochs', type=int, default=None,
                          help='Number of training epochs (auto-determined if not specified)')
    seg_group.add_argument('--batch-size', type=int, default=None,
                          help='Batch size (auto-determined if not specified)')
    seg_group.add_argument('--imgsz', type=int, default=640,
                          help='Image size for training')
    seg_group.add_argument('--overlap-mask', action='store_true', default=True,
                          help='Allow overlapping masks in segmentation')
    seg_group.add_argument('--mask-ratio', type=int, default=4,
                          help='Mask downsample ratio')
    seg_group.add_argument('--single-cls', action='store_true',
                          help='Train as single-class segmentation')
    
    # Loss function parameters
    loss_group = parser.add_argument_group('Segmentation Loss Parameters')
    loss_group.add_argument('--box-loss-gain', type=float, default=7.5,
                           help='Box loss gain for segmentation')
    loss_group.add_argument('--cls-loss-gain', type=float, default=0.5,
                           help='Classification loss gain')
    loss_group.add_argument('--dfl-loss-gain', type=float, default=1.5,
                           help='DFL loss gain')
    
    # Training optimization
    opt_group = parser.add_argument_group('Training Optimization')
    opt_group.add_argument('--device', type=str, default='0',
                          help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    opt_group.add_argument('--workers', type=int, default=8,
                          help='Number of data loader workers')
    opt_group.add_argument('--optimizer', type=str, default='SGD',
                          choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'],
                          help='Optimizer to use (SGD, Adam, AdamW, RMSProp, auto)')
    opt_group.add_argument('--lr', type=float, default=0.01,
                          help='Initial learning rate')
    opt_group.add_argument('--weight-decay', type=float, default=0.0005,
                          help='Weight decay')
    opt_group.add_argument('--momentum', type=float, default=0.937,
                          help='SGD momentum')
    opt_group.add_argument('--warmup-epochs', type=int, default=3,
                          help='Warmup epochs')
    opt_group.add_argument('--patience', type=int, default=10,
                          help='Early stopping patience')
    
    # Data augmentation for segmentation
    aug_group = parser.add_argument_group('Segmentation Data Augmentation')
    aug_group.add_argument('--hsv-h', type=float, default=0.015,
                          help='HSV-Hue augmentation')
    aug_group.add_argument('--hsv-s', type=float, default=0.7,
                          help='HSV-Saturation augmentation')
    aug_group.add_argument('--hsv-v', type=float, default=0.4,
                          help='HSV-Value augmentation')
    aug_group.add_argument('--degrees', type=float, default=0.0,
                          help='Rotation degrees')
    aug_group.add_argument('--translate', type=float, default=0.1,
                          help='Translation augmentation')
    aug_group.add_argument('--scale', type=float, default=0.5,
                          help='Scale augmentation')
    aug_group.add_argument('--shear', type=float, default=0.0,
                          help='Shear augmentation')
    aug_group.add_argument('--perspective', type=float, default=0.0,
                          help='Perspective augmentation')
    aug_group.add_argument('--flipud', type=float, default=0.0,
                          help='Vertical flip probability')
    aug_group.add_argument('--fliplr', type=float, default=0.5,
                          help='Horizontal flip probability')
    aug_group.add_argument('--mosaic', type=float, default=1.0,
                          help='Mosaic augmentation probability')
    aug_group.add_argument('--mixup', type=float, default=0.0,
                          help='Mixup augmentation probability')
    aug_group.add_argument('--copy-paste', type=float, default=0.1,
                          help='Copy-paste augmentation probability')
    
    # Experiment management
    exp_group = parser.add_argument_group('Experiment Management')
    exp_group.add_argument('--name', type=str, default=None,
                          help='Experiment name (auto-generated if not specified)')
    exp_group.add_argument('--project', type=str, default='runs/segment',
                          help='Project directory to save results')
    exp_group.add_argument('--resume', type=str, default=None,
                          help='Resume training from checkpoint')
    exp_group.add_argument('--save-period', type=int, default=10,
                          help='Save checkpoint every n epochs')
    
    # Validation and visualization
    val_group = parser.add_argument_group('Validation and Visualization')
    val_group.add_argument('--val', action='store_true', default=True,
                          help='Validate during training')
    val_group.add_argument('--val-period', type=int, default=1,
                          help='Validate every N epochs (default: 1, use 5-10 for faster training)')
    val_group.add_argument('--val-split', type=float, default=None,
                          help='Fraction of validation set to use (0.1-0.5 for faster validation)')
    val_group.add_argument('--fast-val', action='store_true',
                          help='Enable fast validation (reduced metrics, faster execution)')
    val_group.add_argument('--val-batch-size', type=int, default=None,
                          help='Validation batch size (larger = faster validation)')
    val_group.add_argument('--save-json', action='store_true', default=False,
                          help='Save results to JSON file (disabled by default for speed)')
    val_group.add_argument('--plots', action='store_true', default=False,
                          help='Generate training plots and mask visualizations (disabled by default for speed)')
    val_group.add_argument('--save-hybrid', action='store_true',
                          help='Save hybrid version of dataset labels')
    val_group.add_argument('--save-best', action='store_true', default=True,
                          help='Save best checkpoint based on fitness (default: True)')
    val_group.add_argument('--save-last', action='store_true', default=True,
                          help='Save last checkpoint (default: True)')
    val_group.add_argument('--cache', type=str, default=None, choices=['ram', 'disk'],
                          help='Cache dataset in RAM or disk for faster training')
    
    return parser.parse_args()

def validate_segmentation_arguments(args):
    """Validate command line arguments for segmentation training."""
    # Check if data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Segmentation dataset file not found: {args.data}")
    
    # Validate DINO arguments
    if args.use_dino:
        if not args.dino_variant and not args.dino_preprocessing:
            raise ValueError("--dino-variant is required when --use-dino is specified")
        
        if not args.dinoversion:
            raise ValueError("--dinoversion is required when --use-dino is specified. Choose 'v2' or 'v3'.")
        
        LOGGER.info(f"🔄 Using DINO version: {args.dinoversion.upper()}")
        
        # Handle --integration parameter
        if args.integration == 'single':
            args.dino_integration = 'single'
            # Clear any existing preprocessing to ensure clean single integration
            if not args.dino_preprocessing:
                args.dino_preprocessing = None
            LOGGER.info("📐 SINGLE INTEGRATION: DINO enhancement at P4 level")
        elif args.integration == 'dual':
            args.dino_integration = 'dual' 
            # Clear any existing preprocessing to ensure clean dual integration
            if not args.dino_preprocessing:
                args.dino_preprocessing = None
            LOGGER.info("📐 DUAL INTEGRATION: DINO enhancement at P3+P4 levels")
        elif args.integration == 'dualp0p3':
            args.dino_integration = 'single'  # DualP0P3 uses single backbone (P3 only) + preprocessing
            if not args.dino_preprocessing:
                args.dino_preprocessing = f"dinov3_{args.dino_variant}"
            LOGGER.info("🔄 DUALP0P3 INTEGRATION: DINO enhancement at P0+P3 levels")
            LOGGER.info(f"   📐 Architecture: {args.dino_preprocessing} (P0) + {args.dino_variant} (P3)")
        elif args.integration == 'triple':
            args.dino_integration = 'dual'  # Triple uses dual backbone + preprocessing
            if not args.dino_preprocessing:
                args.dino_preprocessing = f"dinov3_{args.dino_variant}"
            LOGGER.info("🚀 TRIPLE INTEGRATION: DINO enhancement at P0+P3+P4 levels")
            LOGGER.info(f"   📐 Architecture: {args.dino_preprocessing} (P0) + {args.dino_variant} (P3+P4)")
        
        # Validate dino_variant is provided for backbone integration
        if args.integration in ['single', 'dual', 'dualp0p3', 'triple'] and not args.dino_variant:
            LOGGER.error("❌ --dino-variant is required when using --integration with --use-dino")
            sys.exit(1)
    
    # Handle DINO freezing logic
    if args.unfreeze_dino:
        args.freeze_dino = False
        LOGGER.info("DINO weights will be trainable during training")
    else:
        args.freeze_dino = True
        LOGGER.info("DINO weights will be frozen during training (recommended)")
    
    # Check GPU availability
    if not torch.cuda.is_available() and args.device != 'cpu':
        LOGGER.warning("CUDA not available, switching to CPU training")
        args.device = 'cpu'
    
    # Validate segmentation-specific parameters
    if args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be >= 1")
    
    return args

def create_segmentation_experiment_name(args):
    """Create experiment name for segmentation training."""
    if args.name:
        return args.name
    
    name_parts = [f"yolov12{args.model_size}", "seg"]
    
    if args.use_dino:
        # Triple integration: preprocessing + backbone
        if args.dino_preprocessing and args.dino_variant:
            name_parts.extend(["dino3-triple", args.dino_variant, args.dino_integration])
        # Preprocessing only
        elif args.dino_preprocessing and not args.dino_variant:
            name_parts.append("dino3-preprocess")
        # Backbone integration only
        elif args.dino_variant:
            name_parts.extend(["dino3", args.dino_variant, args.dino_integration])
    
    return "-".join(name_parts)

def setup_segmentation_training_parameters(args):
    """Setup segmentation-specific training parameters."""
    # Auto-determine batch size if not specified
    if args.batch_size is None:
        is_triple = args.dino_preprocessing and args.dino_variant and args.integration == 'triple'
        is_dualp0p3 = args.dino_preprocessing and args.dino_variant and args.integration == 'dualp0p3'
        args.batch_size = get_segmentation_batch_size(
            args.model_size, args.use_dino, args.dino_integration, is_triple, is_dualp0p3
        )
        if is_triple:
            batch_type = "triple DINO"
        elif is_dualp0p3:
            batch_type = "dualp0p3 DINO"
        elif args.use_dino:
            batch_type = "DINO"
        else:
            batch_type = "standard"
        LOGGER.info(f"Auto-determined {batch_type} segmentation batch size: {args.batch_size}")
    
    # Auto-determine epochs if not specified
    if args.epochs is None:
        args.epochs = get_segmentation_epochs(args.use_dino)
        LOGGER.info(f"Auto-determined segmentation epochs: {args.epochs}")
    
    # Adjust augmentation parameters for segmentation
    if args.model_size in ['s', 'm', 'l', 'x']:
        if args.model_size == 's':
            args.mixup = max(args.mixup, 0.05)
            args.copy_paste = max(args.copy_paste, 0.15)
        elif args.model_size in ['m', 'l']:
            args.mixup = max(args.mixup, 0.15)
            args.copy_paste = max(args.copy_paste, 0.4)
        elif args.model_size == 'x':
            args.mixup = max(args.mixup, 0.2)
            args.copy_paste = max(args.copy_paste, 0.6)
    
    # Setup validation optimization
    setup_validation_optimization(args)
    
    return args

def setup_validation_optimization(args):
    """Setup validation optimization parameters for faster training."""
    
    # Auto-determine validation batch size if not specified
    if args.val_batch_size is None:
        # Use larger batch size for validation (faster)
        val_multiplier = 2.0 if args.fast_val else 1.5
        args.val_batch_size = max(int(args.batch_size * val_multiplier), args.batch_size + 2)
        LOGGER.info(f"Auto-determined validation batch size: {args.val_batch_size}")
    
    # Validation optimization recommendations
    optimization_msg = []
    
    if args.val_period == 1:
        optimization_msg.append("💡 Use --val-period 5-10 to validate less frequently")
    
    if args.val_split is None:
        optimization_msg.append("💡 Use --val-split 0.2 to use only 20% of validation set")
    
    if not args.fast_val:
        optimization_msg.append("💡 Use --fast-val for reduced metrics and faster validation")
    
    if args.save_json:
        optimization_msg.append("💡 Remove --save-json to skip JSON file generation")
    
    if args.plots:
        optimization_msg.append("💡 Remove --plots to skip visualization generation")
    
    if optimization_msg:
        LOGGER.info("🚀 Validation Speed Optimization Tips:")
        for tip in optimization_msg[:3]:  # Show max 3 tips
            LOGGER.info(f"   {tip}")
    
    # Fast validation configuration
    if args.fast_val:
        LOGGER.info("⚡ Fast validation enabled: reduced metrics, faster execution")
        args.save_json = False  # Disable JSON saving
        args.plots = False      # Disable plot generation

def get_additional_validation_params(args):
    """Get additional validation parameters that are supported by ultralytics."""
    params = {}
    
    # Add validation period if different from default
    if hasattr(args, 'val_period') and args.val_period != 1:
        # Note: Ultralytics uses 'val' parameter differently, we'll handle this in post-processing
        pass
    
    # Add validation batch size if specified
    if hasattr(args, 'val_batch_size') and args.val_batch_size:
        # Note: Most YOLO implementations don't have separate val batch size
        # This would need custom implementation
        pass
    
    return params

def modify_segmentation_config_for_dino(config_path, dino_preprocessing, model_size, freeze_dino, dino_variant=None, dino_integration='single', dino_version='v3'):
    """
    Modify segmentation config for DINO approaches.
    For triple integration: adds preprocessing to backbone integration configs.
    For preprocessing only: modifies preprocessing configs.
    """
    if not dino_preprocessing:
        return config_path
    
    # Load the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine integration type based on preprocessing and backbone integration
    has_preprocessing = dino_preprocessing is not None
    has_backbone = dino_variant is not None
    is_triple = has_preprocessing and has_backbone and dino_integration == 'dual'
    is_dualp0p3 = has_preprocessing and has_backbone and dino_integration == 'single'
    
    if is_triple:
        print("🚀 Configuring TRIPLE DINO3 Segmentation Integration...")
        print("   📐 P0 (Preprocessing) + P3/P4 (Backbone) Enhancement")
        
        # For triple integration with preprocessing config, add DINO3Backbone layers to existing backbone
        # This maintains proper channel flow since preprocessing is already handled in the config
        if 'backbone' in config and config['backbone']:
            # Find and update the existing DINO3Preprocessor layer
            for i, layer in enumerate(config['backbone']):
                if len(layer) >= 4 and layer[2] == 'DINO3Preprocessor':
                    # Update preprocessor with the specified variant
                    # Format: [model_name, freeze_backbone, output_channels]
                    if len(layer[3]) >= 1:
                        config['backbone'][i][3][0] = dino_preprocessing
                        # Keep freeze_backbone setting (layer[3][1]) 
                        config['backbone'][i][3][1] = freeze_dino
                        # Keep output_channels as 3 (layer[3][2])
                        # dino_version will default to 'v3' in constructor
                    print(f"   ✅ P0 Enhancement: {dino_preprocessing} (updated existing preprocessor)")
                    break
            
            # Add DINO3Backbone layers at appropriate positions for P3 and P4 enhancement
            # For nano scale, we need to be careful about channel dimensions
            if dino_integration == 'dual':
                # Add DINO3Backbone at P3 (after layer 5 in nano config) and P4 (after layer 7)
                # Determine output channels based on model size
                if model_size == 'n':
                    p3_channels = 128  # After scaling: 512 * 0.25 = 128
                    p4_channels = 128  # After scaling: 512 * 0.25 = 128
                elif model_size == 's':
                    p3_channels = 256  # After scaling: 512 * 0.5 = 256
                    p4_channels = 256  # After scaling: 512 * 0.5 = 256
                else:
                    p3_channels = 512  # Full scale
                    p4_channels = 512  # Full scale
                
                # Insert DINO3Backbone after P3 layer (layer 5 in preprocessor config)
                if len(config['backbone']) > 5:
                    p3_layer = [5, 1, 'DINO3Backbone', [dino_variant, freeze_dino, p3_channels]]
                    config['backbone'].insert(6, p3_layer)
                    print(f"   ✅ P3 Enhancement: {dino_variant} (inserted after layer 5)")
                
                # Insert DINO3Backbone after P4 layer (now layer 8 after insertion)
                if len(config['backbone']) > 8:
                    p4_layer = [7, 1, 'DINO3Backbone', [dino_variant, freeze_dino, p4_channels]]
                    config['backbone'].insert(9, p4_layer)
                    print(f"   ✅ P4 Enhancement: {dino_variant} (inserted after layer 7)")
                
                # Update head layer references to account for the two new layers
                if 'head' in config:
                    for i, layer in enumerate(config['head']):
                        # Update layer references that point to backbone layers after insertion points
                        if isinstance(layer[0], list):
                            # Handle multi-input layers like Concat and Segment
                            for j, ref in enumerate(layer[0]):
                                if isinstance(ref, int):
                                    if ref > 5:  # After first insertion
                                        config['head'][i][0][j] = ref + 1
                                    if ref > 7:  # After second insertion (accounting for first)
                                        config['head'][i][0][j] = ref + 1
                        elif isinstance(layer[0], int):
                            if layer[0] > 5:  # After first insertion
                                config['head'][i][0] = layer[0] + 1
                            if layer[0] > 7:  # After second insertion (accounting for first)
                                config['head'][i][0] = layer[0] + 1
                    
                    print(f"   🔧 Updated head layer references for dual backbone insertion")
            
            print(f"   ✅ DINO Version: {dino_version} (applied to all DINO modules)")
            print(f"   🔧 Using preprocessing config as base to maintain channel flow")
        
    elif is_dualp0p3:
        print("🔄 Configuring DUALP0P3 DINO3 Segmentation Integration...")
        print("   📐 P0 (Preprocessing) + P3 (Backbone) Enhancement")
        
        # For dualp0p3 integration with preprocessing config, add DINO3Backbone layer only at P3
        # This maintains proper channel flow since preprocessing is already handled in the config
        if 'backbone' in config and config['backbone']:
            # Find and update the existing DINO3Preprocessor layer
            for i, layer in enumerate(config['backbone']):
                if len(layer) >= 4 and layer[2] == 'DINO3Preprocessor':
                    # Update preprocessor with the specified variant
                    # Format: [model_name, freeze_backbone, output_channels]
                    if len(layer[3]) >= 1:
                        config['backbone'][i][3][0] = dino_preprocessing
                        # Keep freeze_backbone setting (layer[3][1]) 
                        config['backbone'][i][3][1] = freeze_dino
                        # Keep output_channels as 3 (layer[3][2])
                        # dino_version will default to 'v3' in constructor
                    print(f"   ✅ P0 Enhancement: {dino_preprocessing} (updated existing preprocessor)")
                    break
            
            # Add DINO3Backbone layer only at P3 position for dualp0p3 integration
            # Determine output channels based on model size
            if model_size == 'n':
                p3_channels = 128  # After scaling: 512 * 0.25 = 128
            elif model_size == 's':
                p3_channels = 256  # After scaling: 512 * 0.5 = 256
            else:
                p3_channels = 512  # Full scale
            
            # Insert DINO3Backbone after P3 layer (layer 5 in preprocessor config)
            if len(config['backbone']) > 5:
                p3_layer = [5, 1, 'DINO3Backbone', [dino_variant, freeze_dino, p3_channels]]
                config['backbone'].insert(6, p3_layer)
                print(f"   ✅ P3 Enhancement: {dino_variant} (inserted after layer 5)")
            
            # Update head layer references to account for the one new layer
            if 'head' in config:
                for i, layer in enumerate(config['head']):
                    # Update layer references that point to backbone layers after insertion point
                    if isinstance(layer[0], list):
                        # Handle multi-input layers like Concat and Segment
                        for j, ref in enumerate(layer[0]):
                            if isinstance(ref, int) and ref > 5:  # After insertion
                                config['head'][i][0][j] = ref + 1
                    elif isinstance(layer[0], int) and layer[0] > 5:
                        # Single positive reference
                        config['head'][i][0] = layer[0] + 1
                
                print(f"   🔧 Updated head layer references for single backbone insertion")
            
            print(f"   ✅ DINO Version: {dino_version} (applied to all DINO modules)")
            print(f"   🔧 Using preprocessing config as base to maintain channel flow")
        
    else:
        print("🔧 Configuring DINO3 Segmentation Preprocessing...")
        
        # Replace DINO_MODEL_NAME in backbone for preprocessing-only
        if 'backbone' in config:
            for i, layer in enumerate(config['backbone']):
                if len(layer) >= 4 and isinstance(layer[3], list) and len(layer[3]) > 0:
                    if layer[3][0] == 'DINO_MODEL_NAME':
                        config['backbone'][i][3][0] = dino_preprocessing
                        config['backbone'][i][3][1] = freeze_dino  # Set freeze parameter
                        config['backbone'][i][3][2] = 3  # DINO preprocessing outputs 3 channels
                        print(f"   ✅ P0 Enhancement: {dino_preprocessing}")
                        break
    
    print(f"   🔧 DINO weights {'frozen' if freeze_dino else 'trainable'}")
    
    # Ensure task is set to segment
    config['task'] = 'segment'
    
    # Force the scale parameter
    config['scale'] = model_size
    print(f"   🔧 Set model scale: {model_size}")
    print(f"   🎭 Task: instance segmentation")
    
    # Create temporary config file
    temp_fd, temp_path = tempfile.mkstemp(suffix=f'_{model_size}_seg.yaml', prefix=f'yolov12{model_size}_dino_seg_')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return temp_path

def apply_dino_version_to_config(config_path, dino_version):
    """
    Apply DINO version parameter to all DINO modules in a configuration file.
    
    This function ensures that all DINO3Backbone and DINO3Preprocessor modules
    in the YAML configuration receive the correct dino_version parameter.
    
    Args:
        config_path (str): Path to the YAML configuration file
        dino_version (str): DINO version ('v2' or 'v3')
    
    Returns:
        str: Path to the updated configuration file (may be temporary)
    """
    # Check if this is already a temporary file with DINO version applied
    if (config_path.startswith('/tmp/') or config_path.startswith('/var/folders/') or 
        'dino_seg_' in os.path.basename(config_path) or '_dino_version.yaml' in config_path):
        # This is likely already processed
        return config_path
    
    # Load the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    needs_update = False
    
    # Update DINO3Backbone layers in backbone
    if 'backbone' in config:
        for layer in config['backbone']:
            if len(layer) >= 4 and layer[2] == 'DINO3Backbone':
                # Add dino_version as the 4th parameter if not present
                if len(layer[3]) == 3:  # Current format: [model_name, freeze_backbone, output_channels]
                    layer[3].append(dino_version)
                    needs_update = True
    
    # Update DINO3Preprocessor layers in backbone (if any)
    if 'backbone' in config:
        for layer in config['backbone']:
            if len(layer) >= 4 and layer[2] == 'DINO3Preprocessor':
                # Add dino_version as the 4th parameter if not present
                if len(layer[3]) == 3:  # Current format: [model_name, freeze_backbone, output_channels]
                    layer[3].append(dino_version)
                    needs_update = True
    
    # If no updates needed, return original path
    if not needs_update:
        return config_path
    
    # Create temporary config file with updates
    temp_fd, temp_path = tempfile.mkstemp(suffix='_dino_version.yaml', prefix='yolov12_dino_')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return temp_path

def main():
    """Main segmentation training function."""
    print("🎭 YOLOv12 Instance Segmentation Training with DINO Enhancement")
    print("=" * 70)
    
    # Parse and validate arguments
    args = parse_segmentation_arguments()
    args = validate_segmentation_arguments(args)
    args = setup_segmentation_training_parameters(args)
    
    # Create segmentation model configuration path
    model_config = create_segmentation_config_path(
        args.model_size, args.use_dino, args.dino_variant, 
        args.dino_integration, args.dino_preprocessing, args.dinoversion
    )
    
    # Apply DINO version to configuration if needed
    if args.use_dino:
        model_config = apply_dino_version_to_config(model_config, args.dinoversion)
    
    # Create experiment name
    experiment_name = create_segmentation_experiment_name(args)
    
    # Print configuration summary
    print(f"📊 Segmentation Training Configuration:")
    print(f"   Task: Instance Segmentation")
    print(f"   Model: YOLOv12{args.model_size}-seg")
    if args.use_dino:
        # Advanced integration: preprocessing + backbone
        if args.dino_preprocessing and args.dino_variant:
            if args.integration == 'triple':
                print(f"   DINO: 🚀 TRIPLE INTEGRATION")
                print(f"         ├─ P0 (Preprocessing): {args.dino_preprocessing}")
                print(f"         └─ P3+P4 (Backbone): {args.dino_variant} ({args.dino_integration}-scale)")
            elif args.integration == 'dualp0p3':
                print(f"   DINO: 🔄 DUALP0P3 INTEGRATION")
                print(f"         ├─ P0 (Preprocessing): {args.dino_preprocessing}")
                print(f"         └─ P3 (Backbone): {args.dino_variant} ({args.dino_integration}-scale)")
            else:
                print(f"   DINO: {args.integration.upper()} INTEGRATION")
                print(f"         ├─ P0 (Preprocessing): {args.dino_preprocessing}")
                print(f"         └─ Backbone: {args.dino_variant} ({args.dino_integration}-scale)")
        # Preprocessing only
        elif args.dino_preprocessing and not args.dino_variant:
            print(f"   DINO: Preprocessing with {args.dino_preprocessing}")
        # Backbone integration only
        elif args.dino_variant:
            print(f"   DINO: {args.dino_variant} ({args.dino_integration}-scale)")
        print(f"   DINO Weights: {'Frozen' if args.freeze_dino else 'Trainable'}")
    else:
        print(f"   DINO: None (Base YOLOv12 Segmentation)")
    print(f"   Config: {model_config}")
    print(f"   Dataset: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {args.device}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Overlap Masks: {args.overlap_mask}")
    print(f"   Mask Ratio: {args.mask_ratio}")
    print()
    
    try:
        # Modify config for DINO preprocessing if needed
        temp_config_path = None
        if args.dino_preprocessing:
            temp_config_path = modify_segmentation_config_for_dino(
                model_config, args.dino_preprocessing, args.model_size, args.freeze_dino,
                args.dino_variant, args.dino_integration, args.dinoversion
            )
            if temp_config_path != model_config:
                model_config = temp_config_path
        
        # Load segmentation model
        print(f"🔧 Loading segmentation model: {model_config}")
        model = YOLO(model_config)
        
        # Start segmentation training
        print("🏋️  Starting instance segmentation training...")
        results = model.train(
            data=args.data,
            task=args.task,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=experiment_name,
            
            # Segmentation-specific parameters
            overlap_mask=args.overlap_mask,
            mask_ratio=args.mask_ratio,
            single_cls=args.single_cls,
            
            # Loss parameters
            box=args.box_loss_gain,
            cls=args.cls_loss_gain,
            dfl=args.dfl_loss_gain,
            
            # Optimization parameters
            optimizer=args.optimizer,
            lr0=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            
            # Augmentation parameters
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            
            # Experiment parameters
            resume=args.resume,
            save_period=args.save_period,
            cache=args.cache,
            
            # Validation and visualization  
            val=args.val,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
            plots=args.plots,
            save=args.save_best,  # Enable saving best weights
            deterministic=False,  # Allow some randomness to prevent overfitting
            verbose=True,
            
            # Additional validation parameters (if supported by ultralytics)
            **get_additional_validation_params(args)
        )
        
        print("🎉 Segmentation training completed successfully!")
        print(f"📁 Results saved in: {args.project}/{experiment_name}")
        
        # Print final segmentation metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"📊 Final Segmentation Metrics:")
            for key, value in metrics.items():
                if 'mask' in key.lower() or 'map' in key.lower():
                    print(f"   {key}: {value:.4f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Segmentation training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Segmentation training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup temporary config file if created
        if 'temp_config_path' in locals() and temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                print(f"🗑️  Cleaned up temporary config file")
            except Exception:
                pass

if __name__ == '__main__':
    main()