#!/usr/bin/env python3
"""
Quick Training Test for FGO Image Classification

This script performs a quick training test to verify GPU acceleration
and demonstrate the training pipeline functionality with a small subset
of the dataset. Useful for testing before full training runs.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import from our training script
sys.path.append(str(Path(__file__).parent))
from train_model import FGODataset, load_dataset, TrainingConfig


def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_quick_model(num_classes: int) -> nn.Module:
    """
    Create a lightweight model for quick testing.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        PyTorch model ready for training
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace the classifier with a simpler one for quick testing
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model


def quick_train_epoch(model: nn.Module, dataloader: DataLoader, 
                     criterion: nn.Module, optimizer: optim.Optimizer, 
                     device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch with progress tracking.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Training device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
        
        # Update progress bar
        accuracy = 100 * correct_predictions / total_samples
        progress_bar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy


def quick_validate(model: nn.Module, dataloader: DataLoader, 
                  criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Training device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy


def main():
    """Main function for quick training test."""
    setup_logging()
    
    # Configuration for quick test
    config = TrainingConfig(
        model_name="efficientnet_b0",
        num_epochs=3,  # Just a few epochs for testing
        batch_size=16,  # Smaller batch size for quick test
        learning_rate=0.001,
        image_size=224,
        num_workers=4,
        dataset_type="servants",  # Start with servants
    )
    
    logging.info("🚀 Starting Quick Training Test for FGO Image Classification")
    logging.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Show initial GPU memory
        logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    # Load dataset
    dataset_path = "dataset"
    logging.info(f"Loading dataset from: {dataset_path}")
    
    try:
        image_paths, labels, label_names = load_dataset(dataset_path, config)
        logging.info(f"Dataset loaded: {len(image_paths)} images, {len(label_names)} classes")
        
        # Show some example classes
        logging.info(f"Sample classes: {label_names[:5]}...")
        
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # Create a smaller subset for quick testing (limit to first 10 classes and 5 images per class)
    quick_image_paths = []
    quick_labels = []
    quick_label_names = label_names[:10]  # First 10 classes
    
    for label_idx in range(min(10, len(label_names))):
        class_images = [path for path, label in zip(image_paths, labels) if label == label_idx]
        # Take up to 5 images per class
        selected_images = class_images[:5]
        quick_image_paths.extend(selected_images)
        quick_labels.extend([label_idx] * len(selected_images))
    
    logging.info(f"Quick test subset: {len(quick_image_paths)} images, {len(quick_label_names)} classes")
    
    # Split dataset (ensure we have enough samples per class)
    from collections import Counter
    label_counts = Counter(quick_labels)
    
    # Only use classes with at least 2 samples for splitting
    valid_classes = {label: count for label, count in label_counts.items() if count >= 2}
    
    if len(valid_classes) < len(set(quick_labels)):
        logging.info(f"Filtering to {len(valid_classes)} classes with >=2 samples")
        
        # Filter dataset
        filtered_paths = []
        filtered_labels = []
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes.keys())}
        
        for path, label in zip(quick_image_paths, quick_labels):
            if label in valid_classes:
                filtered_paths.append(path)
                filtered_labels.append(label_mapping[label])
        
        quick_image_paths, quick_labels = filtered_paths, filtered_labels
        quick_label_names = [quick_label_names[old_label] for old_label in valid_classes.keys()]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        quick_image_paths, quick_labels, test_size=0.3, random_state=42, stratify=quick_labels
    )
    
    logging.info(f"Split: {len(train_paths)} train, {len(val_paths)} validation")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FGODataset(train_paths, train_labels, quick_label_names, train_transform)
    val_dataset = FGODataset(val_paths, val_labels, quick_label_names, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    logging.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    num_classes = len(quick_label_names)
    model = create_quick_model(num_classes)
    model = model.to(device)
    
    logging.info(f"Model created with {num_classes} output classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    logging.info(f"Optimizer: AdamW, Learning rate: {config.learning_rate}")
    
    # Show GPU memory after model loading
    if torch.cuda.is_available():
        logging.info(f"GPU Memory after model loading: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    # Training loop
    logging.info("🏃 Starting training...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        logging.info("-" * 40)
        
        # Training phase
        train_loss, train_acc = quick_train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = quick_validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        
        # Show GPU memory usage
        if torch.cuda.is_available():
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated")
    
    total_time = time.time() - start_time
    
    logging.info("\n🎉 Quick Training Test Completed!")
    logging.info("=" * 60)
    logging.info(f"Total training time: {total_time:.2f}s")
    logging.info(f"Average time per epoch: {total_time/config.num_epochs:.2f}s")
    logging.info(f"Final validation accuracy: {val_acc:.2f}%")
    
    # GPU utilization summary
    if torch.cuda.is_available():
        logging.info(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
        logging.info(f"GPU Utilization: ✅ Successfully used GPU acceleration")
    
    # Test inference speed
    logging.info("\n🔍 Testing inference speed...")
    model.eval()
    
    # Create a batch for timing
    dummy_input = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Time inference
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_start = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = (time.time() - inference_start) / 10
    
    images_per_second = config.batch_size / inference_time
    logging.info(f"Inference speed: {images_per_second:.1f} images/second")
    logging.info(f"Inference time per batch: {inference_time*1000:.1f}ms")
    
    # Save a test model
    output_dir = Path("quick_test_output")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "quick_test_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'label_names': quick_label_names,
        'final_val_accuracy': val_acc,
        'training_time': total_time
    }, model_path)
    
    logging.info(f"Test model saved to: {model_path}")
    
    logging.info("\n✅ Quick training test completed successfully!")
    logging.info("🚀 Your GPU setup is ready for full-scale training!")


if __name__ == "__main__":
    main() 