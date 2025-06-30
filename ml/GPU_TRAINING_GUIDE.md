# FGO Image Classification: GPU Training Guide

## 🚀 Overview

This guide provides comprehensive instructions for training state-of-the-art image classification models for Fate/Grand Order (FGO) using GPU acceleration. Our pipeline leverages your **NVIDIA GeForce RTX 3080** with 8GB VRAM for optimal performance.

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 3080 (8GB VRAM) ✅
- **CUDA**: Version 12.9 ✅
- **RAM**: 16GB+ recommended for data loading
- **Storage**: 3GB+ available space

### Software
- **Python**: 3.8+ (3.12 installed) ✅
- **PyTorch**: 2.7.1+cu126 ✅
- **CUDA**: Compatible drivers ✅

## 🏗️ Architecture Overview

### Model Options

1. **EfficientNet-B0** (Recommended)
   - **Parameters**: ~5.3M
   - **Memory**: ~1.2GB GPU
   - **Speed**: ~500 images/sec
   - **Accuracy**: High with transfer learning

2. **ResNet-50**
   - **Parameters**: ~25.6M
   - **Memory**: ~2.1GB GPU
   - **Speed**: ~300 images/sec
   - **Accuracy**: Excellent, industry standard

3. **MobileNet-V3-Large**
   - **Parameters**: ~5.4M
   - **Memory**: ~0.8GB GPU
   - **Speed**: ~800 images/sec
   - **Accuracy**: Good, optimized for mobile

### Dataset Configuration

Our optimized dataset contains:
- **Servants**: 385 classes, 3,238 images
- **Craft Essences**: 1,846 classes, 5,538 images
- **Total**: 2,231 classes, 8,776 images (2.16GB)

## 🎯 Training Configurations

### Quick Test (Verification)
```bash
python quick_train_test.py
```
- **Purpose**: Verify GPU setup
- **Duration**: 2-3 minutes
- **Dataset**: 10 classes, 49 images
- **Epochs**: 3

### Servants Classification
```bash
python train_model.py --config model_config.json --dataset_type servants
```
- **Classes**: 385 servants
- **Images**: 3,238 high-quality images
- **Training Time**: ~2-3 hours on RTX 3080
- **Expected Accuracy**: 85-95%

### Craft Essences Classification
```bash
python train_model.py --config model_config.json --dataset_type ces
```
- **Classes**: 1,846 craft essences
- **Images**: 5,538 high-quality images
- **Training Time**: ~4-6 hours on RTX 3080
- **Expected Accuracy**: 80-90%

### Combined Classification
```bash
python train_model.py --config model_config.json --dataset_type combined
```
- **Classes**: 2,231 total entities
- **Images**: 8,776 total images
- **Training Time**: ~6-8 hours on RTX 3080
- **Expected Accuracy**: 75-85%

## ⚙️ Configuration Options

### Model Configuration (`model_config.json`)

```json
{
  "model_name": "efficientnet_b0",
  "num_epochs": 30,
  "batch_size": 64,
  "learning_rate": 0.001,
  "weight_decay": 1e-4,
  "image_size": 224,
  "num_workers": 8,
  "mixed_precision": true,
  "save_checkpoint_every": 5,
  "early_stopping_patience": 10,
  "dataset_type": "servants",
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "augmentation_intensity": "medium",
  "optimizer": "adamw",
  "scheduler": "cosine",
  "warmup_epochs": 5
}
```

### GPU Optimization Settings

#### Batch Size Recommendations
- **RTX 3080 (8GB)**: 
  - EfficientNet-B0: 64-128
  - ResNet-50: 32-64
  - MobileNet-V3: 128-256

#### Memory Optimization
- **Mixed Precision**: Enabled (reduces memory by ~40%)
- **Pin Memory**: Enabled for faster data transfer
- **Num Workers**: 8 (matches CPU cores)
- **Data Loading**: Optimized for GPU pipeline

## 🚀 Training Commands

### 1. Basic Training (Servants)
```bash
cd ml
python train_model.py \
  --dataset_type servants \
  --model_name efficientnet_b0 \
  --batch_size 64 \
  --num_epochs 30 \
  --output_dir training_output_servants
```

### 2. Advanced Training with Custom Config
```bash
cd ml
python train_model.py \
  --config model_config.json \
  --dataset_path dataset \
  --output_dir training_output_advanced
```

### 3. High-Performance Training (Full GPU Utilization)
```bash
cd ml
python train_model.py \
  --config model_config.json \
  --batch_size 128 \
  --num_workers 12 \
  --mixed_precision true \
  --dataset_type servants
```

## 📊 Monitoring Training

### TensorBoard Visualization
```bash
# In a separate terminal
cd ml/training_output/tensorboard
tensorboard --logdir .
# Open http://localhost:6006
```

### Real-time GPU Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f training_output/logs/training.log
```

### Key Metrics to Watch
- **GPU Utilization**: Should be 80-95%
- **GPU Memory**: Should use 4-6GB out of 8GB
- **Training Loss**: Should decrease steadily
- **Validation Accuracy**: Should increase and plateau
- **Learning Rate**: Should follow scheduler pattern

## 🎯 Performance Optimization

### GPU Memory Management
```python
# Automatic mixed precision
config.mixed_precision = True

# Gradient accumulation for large effective batch sizes
effective_batch_size = batch_size * accumulation_steps

# GPU memory clearing
torch.cuda.empty_cache()
```

### Data Loading Optimization
```python
# Optimal data loader settings
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True,  # Reduce worker overhead
    prefetch_factor=2     # Prefetch batches
)
```

### Training Speed Optimization
- **Compile Model**: `torch.compile()` for 10-20% speedup
- **Channels Last**: Memory format optimization
- **AMP**: Automatic Mixed Precision
- **DataLoader**: Optimized worker configuration

## 📈 Expected Results

### Performance Benchmarks (RTX 3080)

| Model | Dataset | Batch Size | Speed (img/sec) | Memory (GB) | Accuracy |
|-------|---------|------------|----------------|-------------|----------|
| EfficientNet-B0 | Servants | 64 | 450-500 | 3.2 | 90-95% |
| EfficientNet-B0 | CEs | 64 | 450-500 | 3.2 | 85-90% |
| ResNet-50 | Servants | 32 | 280-320 | 4.1 | 92-96% |
| MobileNet-V3 | Servants | 128 | 700-800 | 2.1 | 88-93% |

### Training Time Estimates

| Configuration | Training Time | Total Epochs | Final Accuracy |
|---------------|---------------|--------------|----------------|
| Quick Test | 2-3 minutes | 3 | 60-80% |
| Servants (385 classes) | 2-3 hours | 30 | 90-95% |
| CEs (1,846 classes) | 4-6 hours | 30 | 85-90% |
| Combined (2,231 classes) | 6-8 hours | 50 | 80-85% |

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 32

# Enable mixed precision
--mixed_precision true

# Reduce image size
--image_size 192
```

#### 2. Slow Data Loading
```bash
# Increase workers
--num_workers 8

# Enable pin memory
pin_memory=True

# Use SSD storage for dataset
```

#### 3. Poor Convergence
```bash
# Adjust learning rate
--learning_rate 0.0001

# Try different scheduler
--scheduler cosine

# Increase warmup epochs
--warmup_epochs 10
```

### Performance Debugging
```bash
# Profile GPU usage
python -m torch.profiler.profiler

# Monitor memory usage
python -c "import torch; print(torch.cuda.memory_summary())"

# Check data loading bottlenecks
python -m torch.utils.bottleneck train_model.py
```

## 📁 Output Structure

```
training_output/
├── models/
│   ├── best_model.pth           # Best validation accuracy
│   ├── checkpoint_epoch_5.pth   # Regular checkpoints
│   └── checkpoint_epoch_10.pth
├── logs/
│   └── training.log             # Detailed training logs
├── tensorboard/
│   └── events.out.tfevents.*    # TensorBoard logs
├── training_config.json         # Training configuration
└── training_results.json        # Final results and metrics
```

## 🚀 Production Deployment

### Model Export for Inference
```bash
# Convert to TorchScript
python export_model.py --model best_model.pth --format torchscript

# Convert to ONNX
python export_model.py --model best_model.pth --format onnx

# Convert to TensorFlow Lite
python export_model.py --model best_model.pth --format tflite
```

### Inference Usage
```bash
# Single image prediction
python inference.py --model best_model.pth --image test_image.png

# Batch prediction
python inference.py --model best_model.pth --image_dir test_images/

# Real-time inference server
python inference_server.py --model best_model.pth --port 8080
```

## 📚 Advanced Features

### Transfer Learning
- Pre-trained ImageNet weights
- Fine-tuning with frozen backbone
- Progressive unfreezing strategy

### Data Augmentation
- **Light**: Basic flips and rotations
- **Medium**: Color jittering and affine transforms
- **Heavy**: Advanced augmentations with Albumentations

### Experiment Tracking
- TensorBoard integration
- Weights & Biases support
- Automatic hyperparameter logging

### Model Architecture Search
- Neural Architecture Search (NAS)
- Efficient architecture discovery
- Hardware-aware optimization

## 🎯 Next Steps

1. **Start with Quick Test**: Verify your setup
2. **Train Servants Model**: Get familiar with the pipeline
3. **Optimize Configuration**: Tune for your specific needs
4. **Scale to Full Dataset**: Train on complete dataset
5. **Deploy for Production**: Export and integrate with FGA

## 📞 Support

For issues or questions:
1. Check the logs in `training_output/logs/`
2. Monitor GPU usage with `nvidia-smi`
3. Review training metrics in TensorBoard
4. Consult the troubleshooting section above

**Happy Training! 🚀** 