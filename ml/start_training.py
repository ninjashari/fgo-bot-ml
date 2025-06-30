#!/usr/bin/env python3
"""
FGO GPU Training Launcher

This script provides an easy interface to start GPU-accelerated training
for FGO image classification with optimized settings for RTX 3080.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import time


def check_gpu():
    """Check GPU availability and specs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("❌ No GPU detected")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def create_optimized_config(dataset_type: str, model_name: str) -> dict:
    """Create optimized training configuration."""
    
    # Base configuration optimized for RTX 3080
    base_config = {
        "num_epochs": 30,
        "image_size": 224,
        "num_workers": 8,
        "mixed_precision": True,
        "save_checkpoint_every": 5,
        "early_stopping_patience": 10,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "augmentation_intensity": "medium",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "weight_decay": 1e-4
    }
    
    # Model-specific settings
    if model_name == "efficientnet_b0":
        base_config.update({
            "batch_size": 64,
            "learning_rate": 0.001,
        })
    elif model_name == "resnet50":
        base_config.update({
            "batch_size": 32,
            "learning_rate": 0.0008,
        })
    elif model_name == "mobilenet_v3_large":
        base_config.update({
            "batch_size": 128,
            "learning_rate": 0.001,
        })
    
    # Dataset-specific adjustments
    if dataset_type == "servants":
        # 385 classes - moderate complexity
        base_config.update({
            "num_epochs": 25,
            "early_stopping_patience": 8,
        })
    elif dataset_type == "ces":
        # 1,846 classes - high complexity
        base_config.update({
            "num_epochs": 40,
            "early_stopping_patience": 12,
            "learning_rate": base_config["learning_rate"] * 0.8,  # Slightly lower LR
        })
    elif dataset_type == "combined":
        # 2,231 classes - very high complexity
        base_config.update({
            "num_epochs": 50,
            "early_stopping_patience": 15,
            "learning_rate": base_config["learning_rate"] * 0.7,  # Lower LR
            "augmentation_intensity": "heavy",  # More augmentation
        })
    
    base_config.update({
        "model_name": model_name,
        "dataset_type": dataset_type
    })
    
    return base_config


def estimate_training_time(config: dict) -> str:
    """Estimate training time based on configuration."""
    
    # Base estimates for RTX 3080
    base_times = {
        "servants": 2.5,    # hours
        "ces": 5.0,         # hours
        "combined": 7.0,    # hours
    }
    
    dataset_type = config["dataset_type"]
    num_epochs = config["num_epochs"]
    
    # Base time for dataset type
    base_time = base_times.get(dataset_type, 4.0)
    
    # Adjust for epoch count (base estimates are for ~30 epochs)
    time_multiplier = num_epochs / 30
    estimated_hours = base_time * time_multiplier
    
    # Model-specific adjustments
    if config["model_name"] == "resnet50":
        estimated_hours *= 1.2  # ResNet is slower
    elif config["model_name"] == "mobilenet_v3_large":
        estimated_hours *= 0.7  # MobileNet is faster
    
    if estimated_hours < 1:
        return f"{estimated_hours * 60:.0f} minutes"
    else:
        hours = int(estimated_hours)
        minutes = int((estimated_hours - hours) * 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{hours}h"


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='FGO GPU Training Launcher')
    parser.add_argument('--dataset_type', type=str, default='servants',
                       choices=['servants', 'ces', 'combined'],
                       help='Dataset type to train on')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3_large'],
                       help='Model architecture')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name for outputs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without starting training')
    parser.add_argument('--monitor', action='store_true',
                       help='Open monitoring tools alongside training')
    
    args = parser.parse_args()
    
    print("🚀 FGO GPU Training Launcher")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        print("❌ GPU not available. Please check your setup.")
        return 1
    
    # Create optimized configuration
    config = create_optimized_config(args.dataset_type, args.model_name)
    
    # Estimate training time
    estimated_time = estimate_training_time(config)
    
    # Create run name
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model_name}_{args.dataset_type}_{timestamp}"
    
    output_dir = f"training_output_{run_name}"
    
    print(f"📊 Training Configuration:")
    print(f"   Model: {config['model_name']}")
    print(f"   Dataset: {config['dataset_type']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Mixed Precision: {config['mixed_precision']}")
    print(f"   Estimated Time: {estimated_time}")
    print(f"   Output Directory: {output_dir}")
    
    # Dataset info
    dataset_info = {
        "servants": "385 classes, 3,238 images",
        "ces": "1,846 classes, 5,538 images", 
        "combined": "2,231 classes, 8,776 images"
    }
    print(f"   Dataset Info: {dataset_info[args.dataset_type]}")
    
    if args.dry_run:
        print("\n📄 Configuration Preview:")
        print(json.dumps(config, indent=2))
        print("\n✅ Dry run completed. Use --monitor flag to start training.")
        return 0
    
    # Save configuration
    config_path = f"config_{run_name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to: {config_path}")
    
    # Prepare training command
    training_cmd = [
        "python", "train_model.py",
        "--config", config_path,
        "--dataset_path", "dataset",
        "--output_dir", output_dir
    ]
    
    print(f"\n🏃 Starting training...")
    print(f"Command: {' '.join(training_cmd)}")
    
    # Start monitoring if requested
    if args.monitor:
        print("\n📊 Starting monitoring tools...")
        
        # Start nvidia-smi monitoring in background
        nvidia_cmd = ["watch", "-n", "2", "nvidia-smi"]
        print("Starting GPU monitor: watch -n 2 nvidia-smi")
        
        # Note about TensorBoard
        tensorboard_dir = f"{output_dir}/tensorboard"
        print(f"📈 TensorBoard will be available at:")
        print(f"   Run: tensorboard --logdir {tensorboard_dir}")
        print(f"   Then open: http://localhost:6006")
    
    print(f"\n⏰ Estimated completion time: {estimated_time}")
    print(f"🎯 Expected accuracy: 85-95% for {args.dataset_type}")
    print("\n" + "=" * 50)
    
    # Confirm before starting
    if not args.dry_run:
        response = input("🚀 Start training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Training cancelled.")
            return 0
    
    # Start training
    try:
        print("\n🏃 Training started!")
        print("💡 Tip: Use Ctrl+C to stop training early")
        print("📊 Monitor progress in the output directory")
        
        # Execute training
        result = subprocess.run(training_cmd, check=True)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"🏆 Check training_results.json for final metrics")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 130


if __name__ == "__main__":
    exit(main()) 