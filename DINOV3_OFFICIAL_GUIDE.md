# Official DINOv3 Integration with YOLOv12

This guide explains how to use **official DINOv3 models** from Facebook Research's repository with YOLOv12 integration.

## 🎯 Official DINOv3 Repository
We use models directly from: **https://github.com/facebookresearch/dinov3**

## 🚀 Available Official DINOv3 Models

### Vision Transformer (ViT) Models
```bash
# Official model names from facebookresearch/dinov3
dinov3_vits16       # ViT-S/16: 21M params, 384-dim embeddings
dinov3_vits16plus   # ViT-S+/16: 29M params, 384-dim embeddings  
dinov3_vitb16       # ViT-B/16: 86M params, 768-dim embeddings (RECOMMENDED)
dinov3_vitl16       # ViT-L/16: 300M params, 1024-dim embeddings
dinov3_vitl16plus   # ViT-L+/16: 300M params, 1024-dim embeddings
dinov3_vith16plus   # ViT-H+/16: 840M params, 1280-dim embeddings
dinov3_vit7b16      # ViT-7B/16: 6716M params, 4096-dim embeddings
```

### ConvNeXt Hybrid Models  
```bash
# Official hybrid CNN-ViT models
dinov3_convnext_tiny   # 29M params, 768-dim embeddings
dinov3_convnext_small  # 50M params, 768-dim embeddings
dinov3_convnext_base   # 89M params, 1024-dim embeddings
dinov3_convnext_large  # 198M params, 1536-dim embeddings
```

## 💻 Training Examples

### Basic Training with Official DINOv3
```bash
# Recommended: DINOv3 ViT-B/16 with YOLOv12-Small
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size s \
    --dino-version 3 \
    --dino-input dinov3_vitb16 \
    --integration single \
    --epochs 100

# High performance: DINOv3 ViT-L/16 with YOLOv12-Large (dual-scale)  
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size l \
    --dino-version 3 \
    --dino-input dinov3_vitl16 \
    --integration dual \
    --epochs 200 \
    --batch-size 8

# Hybrid CNN-ViT: DINOv3 ConvNeXt-Base with YOLOv12-Medium
python train_yolov12_dino.py \
    --data coco.yaml \
    --yolo-size m \
    --dino-version 3 \
    --dino-input dinov3_convnext_base \
    --integration single \
    --epochs 150
```

### Using Simplified Aliases
```bash
# These are automatically converted to official names
--dino-input vitb16         # → dinov3_vitb16
--dino-input vitl16         # → dinov3_vitl16
--dino-input convnext_base  # → dinov3_convnext_base
--dino-input vith16_plus    # → dinov3_vith16plus
```

## 🧪 Testing Official DINOv3 Models

### Quick Test
```bash
# Test official DINOv3 ViT-B/16
python test_custom_dino_input.py

# Test specific official model
python test_dino3_variants.py \
    --dino-input dinov3_vitl16 \
    --variant vitl16 \
    --integration dual
```

### Comprehensive Testing
```bash
# Test all official DINOv3 models
python test_dino3_variants.py --test-all

# Test with custom official DINOv3 input
python run_dino_tests.py \
    --variant vitb16 \
    --dino-input dinov3_convnext_base \
    --test-both
```

## 🔧 Technical Implementation

### Model Loading Priority
1. **Official DINOv3** (from facebookresearch/dinov3) - **HIGHEST PRIORITY**
2. Hugging Face DINOv3 models (if available)
3. Hugging Face DINOv2 models (fallback)
4. Local model files
5. Other PyTorch Hub models

### Loading Process
```python
# The system automatically:
1. Detects DINOv3 model identifiers
2. Loads from official repository: torch.hub.load('facebookresearch/dinov3', model_name)
3. Extracts embedding dimensions automatically  
4. Integrates at P4 level (or P3+P4 for dual integration)
5. Falls back gracefully if official models aren't available
```

### Embedding Dimensions (Auto-detected)
```python
# Automatically detected from official models:
dinov3_vits16*     → 384 dimensions
dinov3_vitb16      → 768 dimensions  
dinov3_vitl16*     → 1024 dimensions
dinov3_vith16plus  → 1280 dimensions
dinov3_vit7b16     → 4096 dimensions
dinov3_convnext_*  → varies (768-1536 dimensions)
```

## 📊 Model Performance Comparison

| Model | Parameters | Embed Dim | Memory | Speed | Accuracy |
|-------|-----------|-----------|--------|-------|----------|
| `dinov3_vits16` | 21M | 384 | Low | Fast | Good |
| `dinov3_vitb16` | 86M | 768 | Medium | Medium | **Best Balance** |
| `dinov3_vitl16` | 300M | 1024 | High | Slow | Excellent |
| `dinov3_vith16plus` | 840M | 1280 | Very High | Very Slow | Maximum |
| `dinov3_convnext_base` | 89M | 1024 | Medium | Medium | Excellent (Hybrid) |

## 🎯 Recommended Configurations

### For Production Use
```bash
# Balanced performance-efficiency
python train_yolov12_dino.py \
    --data your_data.yaml \
    --yolo-size s \
    --dino-input dinov3_vitb16 \
    --integration single \
    --epochs 100 \
    --batch-size 16
```

### For Research/Maximum Performance  
```bash
# Maximum accuracy setup
python train_yolov12_dino.py \
    --data research_data.yaml \
    --yolo-size l \
    --dino-input dinov3_vitl16 \
    --integration dual \
    --epochs 200 \
    --batch-size 4 \
    --freeze-dino
```

### For Resource-Constrained Environments
```bash
# Efficient setup
python train_yolov12_dino.py \
    --data mobile_data.yaml \
    --yolo-size n \
    --dino-input dinov3_vits16 \
    --integration single \
    --epochs 50 \
    --batch-size 32
```

## 🛠️ Troubleshooting Official DINOv3

### Common Issues

#### 1. Repository Access
```bash
# Issue: Cannot access facebookresearch/dinov3
❌ Failed to load official DINOv3: HTTP Error 404

# Solutions:
✅ Check internet connection
✅ Ensure PyTorch Hub cache is accessible  
✅ Try with trust_repo=True parameter
✅ Use fallback methods (Hugging Face DINOv2)
```

#### 2. Model Loading
```bash  
# Issue: Model architecture mismatch
❌ RuntimeError: Error(s) in loading state_dict

# Solutions:
✅ Verify model name is correct (check hubconf.py)
✅ Try loading without pretrained weights first
✅ Use --dino-input facebook/dinov2-base as fallback
✅ Check GPU memory availability
```

#### 3. Memory Issues
```bash
# Issue: CUDA out of memory with large models
❌ CUDA out of memory with dinov3_vit7b16

# Solutions:
✅ Use smaller models: dinov3_vitb16 or dinov3_vits16
✅ Reduce batch size: --batch-size 2
✅ Use CPU training: --device cpu
✅ Enable mixed precision training
```

## 📝 Integration with Existing Code

### Update Existing Configurations
If you have existing configurations, simply update the `--dino-input`:

```bash
# Old (using predefined variants)
python train_yolov12_dino.py --dino-variant vitb16

# New (using official DINOv3)  
python train_yolov12_dino.py --dino-input dinov3_vitb16
```

### YAML Configuration
The system automatically uses the correct configuration:
```yaml
# ultralytics/cfg/models/v12/yolov12-dino3-custom.yaml
backbone:
  # ... standard layers ...
  - [-1, 1, DINO3Backbone, ['CUSTOM_DINO_INPUT', True, 1024]]
  # CUSTOM_DINO_INPUT is replaced with your --dino-input value
```

## 🔗 Related Resources

- **Official DINOv3 Repository**: https://github.com/facebookresearch/dinov3
- **DINOv3 Paper**: "DINOv2: Learning Robust Visual Features without Supervision"  
- **Model Architectures**: Vision Transformer (ViT) and ConvNeXt variants
- **Pretraining Dataset**: LVD-1689M (Large-scale Vision Dataset)

## ✅ Verification

To verify you're using official DINOv3:
```bash
# Look for this message in logs:
✅ Successfully loaded official DINOv3: dinov3_vitb16

# Not this (indicates fallback):
⚠️ Using DINOv2 as compatible fallback for DINOv3 specs
```

The integration ensures you're using the **most advanced and official DINOv3 models** directly from Facebook Research, providing the best possible performance for your YOLOv12 object detection tasks.