#!/usr/bin/env python3
"""
Craft Essences Training Monitor

This script monitors the ongoing Craft Essences training progress
and provides regular updates on accuracy, loss, and estimated completion time.
"""

import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

def parse_log_file(log_path):
    """Parse the training log file to extract current progress."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_epoch = 0
    total_epochs = 20
    best_val_acc = 0.0
    current_train_acc = 0.0
    current_val_acc = 0.0
    train_loss = 0.0
    val_loss = 0.0
    learning_rate = 0.0
    epoch_time = 0.0
    
    # Parse the log to find latest metrics
    for line in lines:
        # Extract current epoch
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
        
        # Extract training metrics
        train_match = re.search(r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)%', line)
        if train_match:
            train_loss = float(train_match.group(1))
            current_train_acc = float(train_match.group(2))
        
        # Extract validation metrics
        val_match = re.search(r'Val Loss: ([\d.]+), Val Acc: ([\d.]+)%', line)
        if val_match:
            val_loss = float(val_match.group(1))
            current_val_acc = float(val_match.group(2))
        
        # Extract learning rate
        lr_match = re.search(r'Learning Rate: ([\d.]+)', line)
        if lr_match:
            learning_rate = float(lr_match.group(1))
        
        # Extract epoch time
        time_match = re.search(r'Epoch Time: ([\d.]+)s', line)
        if time_match:
            epoch_time = float(time_match.group(1))
        
        # Extract best validation accuracy
        best_match = re.search(r'New best model saved with validation accuracy: ([\d.]+)%', line)
        if best_match:
            best_val_acc = float(best_match.group(1))
    
    return {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'current_train_acc': current_train_acc,
        'current_val_acc': current_val_acc,
        'best_val_acc': best_val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
        'epoch_time': epoch_time
    }

def estimate_completion_time(current_epoch, total_epochs, avg_epoch_time):
    """Estimate remaining training time."""
    remaining_epochs = total_epochs - current_epoch
    remaining_seconds = remaining_epochs * avg_epoch_time
    
    if remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f} minutes"
    else:
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def main():
    """Monitor the training progress."""
    log_path = "ces_training_output.log"
    
    print("🎴 Craft Essences Training Monitor")
    print("=" * 50)
    print(f"📊 Dataset: 1,846 Craft Essence classes, 5,538 images")
    print(f"🔧 Model: EfficientNet-B0 with Transfer Learning")
    print(f"⚡ Hardware: RTX 3080 Laptop GPU (8GB)")
    print("=" * 50)
    
    while True:
        stats = parse_log_file(log_path)
        
        if stats is None:
            print("❌ Training log not found. Waiting...")
            time.sleep(10)
            continue
        
        if stats['current_epoch'] == 0:
            print("⏳ Training initializing...")
            time.sleep(10)
            continue
        
        # Calculate progress
        progress = (stats['current_epoch'] / stats['total_epochs']) * 100
        
        # Estimate completion time
        if stats['epoch_time'] > 0:
            eta = estimate_completion_time(stats['current_epoch'], stats['total_epochs'], stats['epoch_time'])
        else:
            eta = "Calculating..."
        
        # Display current status
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 📈 Training Progress Update")
        print(f"┌─ Epoch: {stats['current_epoch']}/{stats['total_epochs']} ({progress:.1f}% complete)")
        print(f"├─ Train Accuracy: {stats['current_train_acc']:.2f}%")
        print(f"├─ Validation Accuracy: {stats['current_val_acc']:.2f}%")
        print(f"├─ Best Val Accuracy: {stats['best_val_acc']:.2f}%")
        print(f"├─ Train Loss: {stats['train_loss']:.4f}")
        print(f"├─ Val Loss: {stats['val_loss']:.4f}")
        print(f"├─ Learning Rate: {stats['learning_rate']:.6f}")
        print(f"├─ Epoch Time: {stats['epoch_time']:.1f}s")
        print(f"└─ ETA: {eta}")
        
        # Check if training is complete
        if stats['current_epoch'] >= stats['total_epochs']:
            print("\n🎉 Training Complete!")
            print(f"🏆 Final Best Validation Accuracy: {stats['best_val_acc']:.2f}%")
            break
        
        # Wait before next update
        time.sleep(30)

if __name__ == "__main__":
    main() 