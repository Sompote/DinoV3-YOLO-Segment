# 🚀 RTX 5090 Compatibility Guide

## ⚠️ **CUDA Compatibility Issue**

If you encounter this error with **RTX 5090**:
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

## ✅ **Solutions**

### **Option 1: Update to PyTorch Nightly (Recommended)**

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install PyTorch nightly with CUDA 12.1+ support for RTX 5090
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No CUDA\"}')"
```

### **Option 2: Use PyTorch 2.5+ with CUDA 12.4**

```bash
# Install latest PyTorch with extended CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### **Option 3: Force CUDA Architecture (Temporary)**

```bash
# Set environment variable to allow older kernels
export TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0 12.0"

# Then run training
python train_yolov12_dino.py --data data.yaml --yolo-size s --dino-input dinov3_vitb16 --epochs 200
```

## 🎯 **RTX 5090 Optimized Training**

With **32GB VRAM**, you can use much larger batch sizes:

```bash
# Recommended settings for RTX 5090
python train_yolov12_dino.py \
    --data /workspace/crack/data.yaml \
    --yolo-size l \
    --dino-version 3 \
    --dino-input dinov3_vitb16 \
    --batch-size 64 \
    --epochs 200
```

## 🔧 **Complete Fix Process**

```bash
# 1. Update PyTorch for RTX 5090 support
cd /workspace/DINOV3-YOLOV12
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

# 2. Verify RTX 5090 compatibility
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 3. Run optimized training
python train_yolov12_dino.py \
    --data /workspace/crack/data.yaml \
    --yolo-size s \
    --dino-version 3 \
    --dino-input dinov3_vitb16 \
    --batch-size 64 \
    --epochs 200
```

## 📝 **Why This Happens**

- **RTX 5090** uses newer **Ada Lovelace architecture** with CUDA compute capability `sm_120`
- **Standard PyTorch** builds only include kernels for older architectures up to `sm_90`
- **PyTorch nightly** builds include support for the latest GPUs
- This is a common issue with new GPU releases

## 🚀 **Performance Benefits**

After fixing CUDA compatibility, RTX 5090 provides:
- **3-5x faster training** compared to RTX 3090
- **32GB VRAM** allows much larger batch sizes (64+ vs 16-32)
- **Better memory bandwidth** for Vision Transformer operations
- **Optimal for DINOv3-YOLOv12** large model training

## 💡 **Pro Tips**

1. **Use PyTorch nightly** for best RTX 5090 performance
2. **Increase batch sizes** to 64+ to utilize full memory
3. **Consider larger models** (YOLOv12l/x) with available VRAM
4. **Monitor GPU utilization** with `nvidia-smi` during training