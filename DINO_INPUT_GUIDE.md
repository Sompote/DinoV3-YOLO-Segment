# Custom DINO Input Support (`--dino-input`)

This guide explains how to use the `--dino-input` parameter to load custom DINO models before P0 (the first processing layer) in YOLOv12 + DINOv3 integration.

## 🎯 What is `--dino-input`?

The `--dino-input` parameter allows you to specify custom DINO model sources instead of using the predefined DINOv3 variants. This enables:

- **Using any Hugging Face DINO/DINOv2 model**
- **Loading local model files** (`.pth`, `.pt`, `.ckpt`)
- **Specifying PyTorch Hub models** 
- **Testing experimental DINO variants**
- **Fine-tuned or custom-trained DINO models**

## 🚀 Supported Input Types

### 1. Official DINOv3 Models (Recommended)
```bash
# ViT variants from official Facebook Research repository
--dino-input dinov3_vits16       # 384-dim, efficient
--dino-input dinov3_vitb16       # 768-dim, standard choice  
--dino-input dinov3_vitl16       # 1024-dim, high performance
--dino-input dinov3_vith16plus   # 1280-dim, very high performance
--dino-input dinov3_vit7b16      # 4096-dim, maximum performance

# ConvNeXt variants (hybrid CNN-ViT)
--dino-input dinov3_convnext_tiny   # 768-dim, efficient hybrid
--dino-input dinov3_convnext_small  # 768-dim, balanced hybrid
--dino-input dinov3_convnext_base   # 1024-dim, strong hybrid
--dino-input dinov3_convnext_large  # 1536-dim, maximum hybrid

# Simplified aliases (automatically converted to full names)
--dino-input vitb16               # → dinov3_vitb16
--dino-input convnext_base        # → dinov3_convnext_base
```

### 2. Hugging Face Models (Fallback)
```bash
# DINOv3 Hugging Face models
--dino-input facebook/dinov3-vitb16-pretrain-lvd1689m
--dino-input facebook/dinov3-convnext-base-pretrain-lvd1689m

# DINOv2 fallback variants 
--dino-input facebook/dinov2-base        # 768-dim fallback
--dino-input facebook/dinov2-large       # 1024-dim fallback

# Custom Hugging Face models
--dino-input your-username/custom-dino-model
--dino-input organization/fine-tuned-dinov3
```

### 2. Local Model Files
```bash
# PyTorch model files
--dino-input /path/to/your/model.pth
--dino-input ./models/custom_dino.pt
--dino-input ../checkpoints/fine_tuned_dino.ckpt

# Models with config
--dino-input /path/to/model_with_config.pth  # Automatically detects embed_dim
```

### 3. PyTorch Hub Models
```bash
# Format: repository/model_name
--dino-input facebookresearch/dinov2:dinov2_vitb14
--dino-input custom-repo/experimental-dino
```

### 4. Experimental/Research Models
```bash
# Any DINO-compatible model identifier
--dino-input research-org/new-dino-variant
--dino-input experimental/dino-v3-beta
```

## 💻 Usage Examples

### Training with Custom DINO Input

#### Basic Usage
```bash
# Train with official DINOv3 ViT-B/16 (recommended)
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size s \
    --dino-version 3 \
    --dino-input dinov3_vitb16 \
    --epochs 100
```

#### Advanced Configuration
```bash
# High-performance setup with official DINOv3 ViT-L/16
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size l \
    --dino-version 3 \
    --dino-input dinov3_vitl16 \
    --integration dual \
    --epochs 200 \
    --batch-size 8
```

#### Local Model File
```bash
# Use local fine-tuned model
python train_yolov12_dino.py \
    --data custom_dataset.yaml \
    --yolo-size m \
    --dino-version 3 \
    --dino-input ./models/fine_tuned_dino.pth \
    --integration single \
    --freeze-dino \
    --epochs 50
```

### Testing Custom DINO Inputs

#### Quick Test
```bash
# Test custom input compatibility
python test_custom_dino_input.py
```

#### Comprehensive Variant Testing
```bash
# Test specific custom input with all integration types
python test_dino3_variants.py \
    --variant vitb16 \
    --dino-input facebook/dinov2-base \
    --yolo-size s \
    --integration single
```

#### Batch Testing with Custom Input
```bash
# Test multiple configurations with custom DINO
python run_dino_tests.py \
    --variant vitb16 \
    --dino-input facebook/dinov2-large \
    --test-both \
    --save-results custom_dino_results.json
```

## 🔧 Technical Implementation

### Automatic Model Detection
The system automatically detects and handles different input types:

1. **Hugging Face Detection**: Models with `/` and no file extension
2. **Local File Detection**: Paths that exist on filesystem
3. **PyTorch Hub Detection**: Repository/model format
4. **Fallback Handling**: Graceful fallback to DINOv2 base if loading fails

### Dynamic Embedding Dimension
```python
# The system automatically detects embedding dimensions:
facebook/dinov2-small  → 384-dim
facebook/dinov2-base   → 768-dim  
facebook/dinov2-large  → 1024-dim
facebook/dinov2-giant  → 1536-dim

# Custom models: inferred from model.config.hidden_size or state_dict
```

### Integration Before P0
The custom DINO model is loaded and integrated at the P4 level (or P3+P4 for dual integration) before normal YOLOv12 processing:

```
Input Image (3×640×640)
    ↓
YOLOv12 Backbone Layers (P1, P2, P3)
    ↓
Custom DINO Model Integration (P4) ← --dino-input loads here
    ↓  
YOLOv12 Head (Detection)
```

## 📋 Configuration Files

### Automatic Config Selection
When using `--dino-input`, the system automatically uses the custom configuration:
```yaml
# ultralytics/cfg/models/v12/yolov12-dino3-custom.yaml
backbone:
  # ... standard YOLOv12 layers ...
  - [-1, 1, DINO3Backbone, ['CUSTOM_DINO_INPUT', True, 1024]]
  # ... remaining layers ...
```

### Manual Config Creation
```bash
# Create custom config for specific requirements
python create_custom_dino_config.py \
    --dino-input facebook/dinov2-large \
    --yolo-size l \
    --integration dual \
    --output custom_config.yaml
```

## ⚡ Performance Considerations

### Memory Usage by Model Size
```bash
# Memory-efficient options
--dino-input facebook/dinov2-small    # ~2GB VRAM
--dino-input facebook/dinov2-base     # ~4GB VRAM

# High-performance options  
--dino-input facebook/dinov2-large    # ~8GB VRAM
--dino-input facebook/dinov2-giant    # ~16GB VRAM
```

### Batch Size Recommendations
```bash
# Nano/Small YOLO + Small DINO
--batch-size 32

# Medium YOLO + Base DINO
--batch-size 16  

# Large YOLO + Large DINO
--batch-size 8

# Extra Large YOLO + Giant DINO
--batch-size 4
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Failures
```bash
# Issue: Custom model fails to load
❌ Failed to load custom DINO model 'custom/model': ...

# Solutions:
✅ Check model path/identifier exists
✅ Verify internet connection for Hugging Face models  
✅ Ensure local files are accessible
✅ Try fallback models (facebook/dinov2-base)
```

#### 2. Embedding Dimension Mismatch
```bash
# Issue: Shape mismatch during integration
❌ Shape mismatch: got (2, 768, 40, 40), expected (2, 1024, 40, 40)

# Solutions:
✅ Check model's actual embedding dimension
✅ Verify output_channels parameter matches
✅ Use automatic dimension detection
```

#### 3. Memory Issues
```bash
# Issue: CUDA out of memory
❌ CUDA out of memory. Tried to allocate 2.00 GiB

# Solutions:  
✅ Reduce batch size: --batch-size 4
✅ Use smaller DINO model: facebook/dinov2-small
✅ Use CPU training: --device cpu
✅ Enable gradient checkpointing
```

## 📊 Testing and Validation

### Test Custom Input Support
```bash
# Basic functionality test
python test_custom_dino_input.py

# Comprehensive testing
python test_dino3_variants.py \
    --dino-input facebook/dinov2-base \
    --test-all \
    --save-results validation_results.json
```

### Validate Integration
```bash
# Test training pipeline
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size s \
    --dino-version 3 \
    --dino-input facebook/dinov2-base \
    --epochs 1 \
    --name validation_test
```

## 🎯 Best Practices

### 1. Model Selection
- **General use**: `facebook/dinov2-base` (balanced performance/memory)
- **Resource-limited**: `facebook/dinov2-small` (efficient) 
- **High-performance**: `facebook/dinov2-large` (best accuracy)
- **Research/experimental**: Custom fine-tuned models

### 2. Integration Strategy  
- **Single integration**: P4 only, memory-efficient
- **Dual integration**: P3+P4, maximum performance
- **Custom channels**: Match YOLO size requirements

### 3. Training Configuration
```bash
# Production-ready training
python train_yolov12_dino.py \
    --data your_dataset.yaml \
    --yolo-size m \
    --dino-version 3 \
    --dino-input facebook/dinov2-base \
    --integration single \
    --epochs 100 \
    --batch-size 16 \
    --freeze-dino \
    --name production_model
```

## 🔗 Related Files

- `ultralytics/nn/modules/block.py` - DINO3Backbone implementation with custom input support
- `train_yolov12_dino.py` - Training script with --dino-input parameter
- `test_dino3_variants.py` - Testing script for custom inputs
- `test_custom_dino_input.py` - Specialized custom input tests
- `ultralytics/cfg/models/v12/yolov12-dino3-custom.yaml` - Custom input configuration
- `create_custom_dino_config.py` - Manual config creation tool

The `--dino-input` parameter provides flexible support for custom DINO models while maintaining full compatibility with the existing YOLOv12 + DINOv3 integration architecture.