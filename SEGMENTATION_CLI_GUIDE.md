# YOLOv12 Instance Segmentation CLI Guide 🎭

This guide provides comprehensive documentation for the `train_yolov12_segmentation.py` script, which offers clear CLI arguments specifically for instance segmentation training without confusion with object detection.

## 🚀 Quick Start Examples

### Basic Segmentation Training
```bash
# Train a small YOLOv12 segmentation model
python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s

# Train with specific batch size and epochs
python train_yolov12_segmentation.py --data segmentation_data.yaml --model-size s --batch-size 16 --epochs 100
```

### DINO-Enhanced Segmentation
```bash
# Single-scale DINO enhancement (recommended for balanced performance)
python train_yolov12_segmentation.py \
    --data segmentation_data.yaml \
    --model-size s \
    --use-dino \
    --dino-variant vitb16 \
    --dino-integration single

# Dual-scale DINO enhancement (best performance for complex scenes)
python train_yolov12_segmentation.py \
    --data segmentation_data.yaml \
    --model-size l \
    --use-dino \
    --dino-variant vitl16 \
    --dino-integration dual

# DINO preprocessing approach (most stable)
python train_yolov12_segmentation.py \
    --data segmentation_data.yaml \
    --model-size s \
    --use-dino \
    --dino-preprocessing dinov3_vitb16

# TRIPLE DINO integration (ultimate performance - P0+P3+P4)
python train_yolov12_segmentation.py \
    --data segmentation_data.yaml \
    --model-size l \
    --use-dino \
    --dino-preprocessing dinov3_vitb16 \
    --dino-variant vitl16 \
    --dino-integration dual
```

## 📋 Required Arguments

| Argument | Description | Choices | Example |
|----------|-------------|---------|---------|
| `--data` | Segmentation dataset YAML file | Any valid path | `segmentation_data.yaml` |
| `--model-size` | YOLOv12 model size | `n`, `s`, `m`, `l`, `x` | `s` (small) |

## 🔧 DINO Enhancement Options

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--use-dino` | Enable DINO enhancement | `False` | Required for any DINO features |
| `--dino-variant` | DINO model variant | `None` | Required with `--use-dino` |
| `--dino-integration` | Integration type | `single` | `single` or `dual` |
| `--dino-preprocessing` | DINO preprocessing model | `None` | Alternative to `--dino-variant` |
| `--freeze-dino` | Freeze DINO weights | `True` | Recommended for efficiency |
| `--unfreeze-dino` | Make DINO trainable | `False` | For advanced fine-tuning |

### Available DINO Variants
- **Vision Transformers**: `vits16`, `vitb16`, `vitl16`, `vith16_plus`, `vit7b16`
- **ConvNeXt Hybrid**: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

## 🎭 Segmentation-Specific Parameters

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--overlap-mask` | Allow overlapping masks | `True` | Boolean |
| `--mask-ratio` | Mask downsample ratio | `4` | ≥ 1 |
| `--single-cls` | Single-class segmentation | `False` | Boolean |
| `--box-loss-gain` | Box loss gain | `7.5` | > 0 |
| `--cls-loss-gain` | Classification loss gain | `0.5` | > 0 |
| `--dfl-loss-gain` | DFL loss gain | `1.5` | > 0 |

## ⚡ Training Optimization

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--epochs` | Training epochs | Auto | Auto-determined based on model |
| `--batch-size` | Batch size | Auto | Auto-determined based on GPU memory |
| `--imgsz` | Image size | `640` | 320, 640, 1280, etc. |
| `--device` | CUDA device | `0` | `0`, `0,1,2,3`, or `cpu` |
| `--workers` | Data loader workers | `8` | Based on CPU cores |
| `--lr` | Learning rate | `0.01` | 0.001 - 0.1 |
| `--weight-decay` | Weight decay | `0.0005` | L2 regularization |
| `--momentum` | SGD momentum | `0.937` | 0.8 - 0.99 |
| `--warmup-epochs` | Warmup epochs | `3` | 1 - 10 |
| `--patience` | Early stopping patience | `10` | Epochs to wait |

## 🎨 Data Augmentation for Segmentation

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--hsv-h` | HSV-Hue augmentation | `0.015` | 0.0 - 1.0 |
| `--hsv-s` | HSV-Saturation augmentation | `0.7` | 0.0 - 1.0 |
| `--hsv-v` | HSV-Value augmentation | `0.4` | 0.0 - 1.0 |
| `--degrees` | Rotation degrees | `0.0` | 0.0 - 45.0 |
| `--translate` | Translation fraction | `0.1` | 0.0 - 0.9 |
| `--scale` | Scale augmentation | `0.5` | 0.0 - 0.9 |
| `--shear` | Shear degrees | `0.0` | 0.0 - 10.0 |
| `--perspective` | Perspective transform | `0.0` | 0.0 - 0.001 |
| `--flipud` | Vertical flip probability | `0.0` | 0.0 - 1.0 |
| `--fliplr` | Horizontal flip probability | `0.5` | 0.0 - 1.0 |
| `--mosaic` | Mosaic probability | `1.0` | 0.0 - 1.0 |
| `--mixup` | Mixup probability | `0.0` | 0.0 - 1.0 |
| `--copy-paste` | Copy-paste probability | `0.1` | 0.0 - 1.0 |

## 📁 Experiment Management

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--name` | Experiment name | Auto-generated | Custom name for this run |
| `--project` | Project directory | `runs/segment` | Where to save results |
| `--resume` | Resume from checkpoint | `None` | Path to .pt file |
| `--save-period` | Save checkpoint frequency | `10` | Every N epochs |

## 🔍 Validation and Visualization

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--val` | Validate during training | `True` | Recommended |
| `--save-json` | Save results to JSON | `True` | For analysis |
| `--plots` | Generate training plots | `True` | Mask visualizations |
| `--save-hybrid` | Save hybrid labels | `False` | Advanced feature |
| `--cache` | Cache dataset | `None` | `ram` or `disk` |

## 💡 Best Practices and Recommendations

### Model Size Selection
- **Nano (`n`)**: Embedded systems, real-time applications
- **Small (`s`)**: General purpose, balanced performance
- **Medium (`m`)**: Higher accuracy requirements
- **Large (`l`)**: Research, high-end applications
- **Extra-Large (`x`)**: Maximum accuracy, research

### DINO Enhancement Strategy
```bash
# For most users (balanced performance)
--use-dino --dino-variant vitb16 --dino-integration single

# For maximum accuracy (complex scenes)
--use-dino --dino-variant vitl16 --dino-integration dual

# For stable training (preprocessing approach)
--use-dino --dino-preprocessing dinov3_vitb16

# For ultimate performance (triple integration - P0+P3+P4)
--use-dino --dino-preprocessing dinov3_vitb16 --dino-variant vitl16 --dino-integration dual
```

### 🚀 Triple DINO Integration (P0+P3+P4)

The **ultimate segmentation performance** can be achieved by combining DINO preprocessing with backbone integration:

```bash
# Triple DINO integration - maximum enhancement
python train_yolov12_segmentation.py \
    --data segmentation_data.yaml \
    --model-size l \
    --use-dino \
    --dino-preprocessing dinov3_vitb16 \
    --dino-variant vitl16 \
    --dino-integration dual \
    --epochs 150 \
    --batch-size 4
```

**Triple Integration Components**:
- **P0 (Preprocessing)**: `--dino-preprocessing dinov3_vitb16` enhances input images
- **P3+P4 (Backbone)**: `--dino-variant vitl16 --dino-integration dual` enhances feature extraction
- **Result**: Maximum possible DINO enhancement across all processing levels

**Performance Expectations**:
- **mAP Improvement**: +15-25% over baseline segmentation
- **Memory Usage**: ~16GB VRAM (reduce batch size accordingly)
- **Training Time**: 3-4x longer than baseline
- **Best For**: Research, maximum accuracy requirements, complex segmentation tasks

### Memory and Batch Size Guidelines
- **Segmentation uses more memory** than detection due to mask processing
- **Auto batch sizing** is recommended (omit `--batch-size`)
- **Manual batch sizes** for different GPUs:
  - **4GB VRAM**: Standard `--batch-size 4-8`, DINO `--batch-size 2-4`, Triple `--batch-size 1-2`
  - **8GB VRAM**: Standard `--batch-size 8-16`, DINO `--batch-size 4-8`, Triple `--batch-size 2-4`
  - **16GB+ VRAM**: Standard `--batch-size 16-32`, DINO `--batch-size 8-16`, Triple `--batch-size 4-8`
  - **32GB+ VRAM**: Standard `--batch-size 32+`, DINO `--batch-size 16+`, Triple `--batch-size 8+`

### Training Duration
- **Without DINO**: 150 epochs (auto-determined)
- **With DINO**: 100 epochs (auto-determined)
- **Triple DINO**: 150 epochs (recommended for convergence)
- **Custom**: Use `--epochs` for specific requirements

## 🚀 Complete Training Examples

### Production-Ready Segmentation
```bash
python train_yolov12_segmentation.py \
    --data my_segmentation_data.yaml \
    --model-size s \
    --epochs 100 \
    --device 0 \
    --name production-seg \
    --val \
    --plots \
    --save-json
```

### Research-Grade with DINO
```bash
python train_yolov12_segmentation.py \
    --data research_data.yaml \
    --model-size l \
    --use-dino \
    --dino-variant vitl16 \
    --dino-integration dual \
    --epochs 150 \
    --batch-size 8 \
    --lr 0.005 \
    --patience 15 \
    --name research-dino-dual \
    --cache ram
```

### Ultimate Performance with Triple DINO
```bash
python train_yolov12_segmentation.py \
    --data research_data.yaml \
    --model-size l \
    --use-dino \
    --dino-preprocessing dinov3_vitb16 \
    --dino-variant vitl16 \
    --dino-integration dual \
    --epochs 200 \
    --batch-size 4 \
    --lr 0.005 \
    --patience 20 \
    --name ultimate-triple-dino \
    --cache ram
```

### Multi-GPU Training
```bash
python train_yolov12_segmentation.py \
    --data large_dataset.yaml \
    --model-size x \
    --use-dino \
    --dino-variant vitb16 \
    --dino-integration single \
    --device 0,1,2,3 \
    --batch-size 32 \
    --workers 16 \
    --name multi-gpu-seg
```

### Single-Class Segmentation
```bash
python train_yolov12_segmentation.py \
    --data crack_segmentation.yaml \
    --model-size s \
    --single-cls \
    --overlap-mask \
    --mask-ratio 4 \
    --epochs 200 \
    --name crack-detection
```

## 🔧 Advanced Configuration

### Custom Augmentation for Medical Images
```bash
python train_yolov12_segmentation.py \
    --data medical_data.yaml \
    --model-size m \
    --degrees 0.0 \
    --perspective 0.0 \
    --mixup 0.0 \
    --copy-paste 0.0 \
    --hsv-h 0.005 \
    --hsv-s 0.3 \
    --hsv-v 0.2 \
    --name medical-seg
```

### High-Resolution Segmentation
```bash
python train_yolov12_segmentation.py \
    --data high_res_data.yaml \
    --model-size l \
    --imgsz 1280 \
    --batch-size 4 \
    --mask-ratio 2 \
    --name high-res-seg
```

## 📊 Output and Results

Training results are saved in the specified project directory:
```
runs/segment/
└── experiment-name/
    ├── weights/
    │   ├── best.pt          # Best checkpoint
    │   └── last.pt          # Last checkpoint
    ├── results.png          # Training curves
    ├── confusion_matrix.png # Confusion matrix
    ├── val_batch0_labels.jpg # Ground truth masks
    ├── val_batch0_pred.jpg   # Predicted masks
    └── args.yaml            # Training arguments
```

## 🎯 Key Differences from Detection Training

1. **Task-Specific**: Fixed to segmentation task
2. **Mask Parameters**: `--overlap-mask`, `--mask-ratio`
3. **Loss Functions**: Segmentation-specific loss gains
4. **Memory Usage**: Higher memory requirements for mask processing
5. **Batch Sizes**: Automatically reduced for segmentation workload
6. **Validation**: Includes mask mAP metrics
7. **Visualization**: Generates mask overlays and contours

This CLI interface ensures clear, unambiguous segmentation training without confusion with object detection parameters.