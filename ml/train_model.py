#!/usr/bin/env python3
"""
FGO Image Classification Training Pipeline

This module implements a GPU-accelerated training pipeline for Fate/Grand Order
image classification using PyTorch. It supports both servant and craft essence
classification with transfer learning from pre-trained models.

Key Features:
- GPU acceleration with automatic device detection
- Transfer learning with EfficientNet and ResNet architectures
- Advanced data augmentation and preprocessing
- Mixed precision training for improved performance
- Comprehensive logging and model checkpointing
- TensorBoard integration for training visualization
- Automatic learning rate scheduling
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights

import timm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    model_name: str = "efficientnet_b0"  # efficientnet_b0, resnet50, mobilenet_v3_large
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 4
    mixed_precision: bool = True
    save_checkpoint_every: int = 5
    early_stopping_patience: int = 10
    dataset_type: str = "servants"  # servants, ces, combined
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    augmentation_intensity: str = "medium"  # light, medium, heavy
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5


class FGODataset(Dataset):
    """
    Custom dataset class for FGO images.
    
    Handles loading and preprocessing of servant and craft essence images
    with configurable data augmentation.
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 label_names: List[str], transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of integer labels corresponding to each image
            label_names: List of class names
            transform: Torchvision transforms to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.label_names = label_names
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, label)
        """
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logging.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            black_image = torch.zeros(3, 224, 224)
            return black_image, 0


def get_data_transforms(config: TrainingConfig) -> Dict[str, transforms.Compose]:
    """
    Create data transforms for training, validation, and testing.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with train, val, and test transforms
    """
    img_size = config.image_size
    
    # Base transforms for all phases
    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    # Training transforms with augmentation
    train_transforms = []
    
    if config.augmentation_intensity == "light":
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    elif config.augmentation_intensity == "medium":
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    elif config.augmentation_intensity == "heavy":
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ])
    
    train_transforms.extend(base_transform)
    
    return {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(base_transform),
        'test': transforms.Compose(base_transform)
    }


def load_dataset(dataset_path: str, config: TrainingConfig) -> Tuple[List[str], List[int], List[str]]:
    """
    Load dataset images and create labels.
    
    Args:
        dataset_path: Path to the dataset directory
        config: Training configuration
        
    Returns:
        Tuple of (image_paths, labels, label_names)
    """
    image_paths = []
    labels = []
    label_names = []
    
    if config.dataset_type == "servants":
        data_dir = Path(dataset_path) / "servants"
    elif config.dataset_type == "ces":
        data_dir = Path(dataset_path) / "ces"
    else:  # combined
        data_dir = Path(dataset_path)
        
    logging.info(f"Loading dataset from: {data_dir}")
    
    # Get all entity directories
    entity_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    entity_dirs.sort()  # Ensure consistent ordering
    
    for label_idx, entity_dir in enumerate(entity_dirs):
        entity_name = entity_dir.name
        label_names.append(entity_name)
        
        # Get all images in this entity directory
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(entity_dir.glob(ext)))
        
        for image_file in image_files:
            image_paths.append(str(image_file))
            labels.append(label_idx)
    
    logging.info(f"Loaded {len(image_paths)} images from {len(label_names)} classes")
    return image_paths, labels, label_names


def create_model(num_classes: int, model_name: str = "efficientnet_b0") -> nn.Module:
    """
    Create a pre-trained model for transfer learning.
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the model architecture
        
    Returns:
        PyTorch model ready for training
    """
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace the classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, 
                scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Training device (CPU/GPU)
        scaler: Mixed precision scaler
        
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
        
        if scaler is not None:
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
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


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Training device (CPU/GPU)
        
    Returns:
        Tuple of (average_loss, accuracy, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)


def setup_logging(log_dir: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='FGO Image Classification Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--dataset_path', type=str, default='dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='training_output',
                       help='Output directory for models and logs')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3_large'])
    parser.add_argument('--dataset_type', type=str, default='servants',
                       choices=['servants', 'ces', 'combined'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
        
        # Override with command line arguments
        if args.model_name:
            config.model_name = args.model_name
        if args.dataset_type:
            config.dataset_type = args.dataset_type
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.num_epochs:
            config.num_epochs = args.num_epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tensorboard_dir = output_dir / "tensorboard"
    
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(str(log_dir))
    
    # Save config
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    logging.info(f"Training configuration: {asdict(config)}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset
    image_paths, labels, label_names = load_dataset(args.dataset_path, config)
    
    # Filter out classes with too few samples for stratified splitting
    from collections import Counter
    label_counts = Counter(labels)
    
    # Find classes with at least 3 samples (minimum for train/val/test split)
    valid_classes = {label: count for label, count in label_counts.items() if count >= 3}
    
    logging.info(f"Classes with <3 samples: {len(label_counts) - len(valid_classes)}")
    logging.info(f"Using {len(valid_classes)} classes with sufficient samples")
    
    # Filter dataset to only include valid classes
    filtered_paths = []
    filtered_labels = []
    label_mapping = {}  # Map old labels to new consecutive labels
    new_label_names = []
    
    for old_label, count in valid_classes.items():
        new_label = len(label_mapping)
        label_mapping[old_label] = new_label
        new_label_names.append(label_names[old_label])
    
    for path, label in zip(image_paths, labels):
        if label in valid_classes:
            filtered_paths.append(path)
            filtered_labels.append(label_mapping[label])
    
    logging.info(f"Filtered dataset: {len(filtered_paths)} images, {len(new_label_names)} classes")
    
    # Update variables
    image_paths, labels, label_names = filtered_paths, filtered_labels, new_label_names
    
    # Split dataset - Handle high class count datasets
    from sklearn.model_selection import train_test_split
    
    # Calculate minimum samples needed for stratified split
    num_classes = len(set(labels))
    min_samples_needed = num_classes * 2  # At least 2 samples per class for stratified split
    
    # Adjust ratios for datasets with many classes
    if config.dataset_type == "ces" or num_classes > 1000:
        # For datasets with many classes, we need to avoid stratified split as it's not feasible
        # Use a simple 80/10/10 split for high-class datasets
        adjusted_train_ratio = 0.80
        adjusted_val_ratio = 0.10
        adjusted_test_ratio = 0.10
        
        logging.info(f"Adjusted ratios for high-class dataset: train={adjusted_train_ratio:.2f}, val={adjusted_val_ratio:.2f}, test={adjusted_test_ratio:.2f}")
    else:
        adjusted_test_ratio = config.test_ratio
        adjusted_val_ratio = config.val_ratio
        adjusted_train_ratio = config.train_ratio
    
    # Check if stratified split is possible
    test_size = int(len(image_paths) * adjusted_test_ratio)
    use_stratify = test_size >= num_classes
    
    if not use_stratify:
        logging.warning(f"Cannot use stratified split: {test_size} test samples < {num_classes} classes. Using random split.")
        stratify_param = None
    else:
        stratify_param = labels
    
    # First split: train + val, test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=adjusted_test_ratio, random_state=42, stratify=stratify_param
    )
    
    # Second split: train, val
    val_size = adjusted_val_ratio / (adjusted_train_ratio + adjusted_val_ratio)
    val_test_size = int(len(train_val_paths) * val_size)
    use_stratify_val = val_test_size >= len(set(train_val_labels))
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size, 
        random_state=42, 
        stratify=train_val_labels if use_stratify_val else None
    )
    
    logging.info(f"Dataset split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    # Create transforms
    transforms_dict = get_data_transforms(config)
    
    # Create datasets
    train_dataset = FGODataset(train_paths, train_labels, label_names, transforms_dict['train'])
    val_dataset = FGODataset(val_paths, val_labels, label_names, transforms_dict['val'])
    test_dataset = FGODataset(test_paths, test_labels, label_names, transforms_dict['test'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    # Create model
    num_classes = len(label_names)
    model = create_model(num_classes, config.model_name)
    model = model.to(device)
    
    logging.info(f"Model: {config.model_name}, Classes: {num_classes}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, 
                             momentum=0.9, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.num_epochs//3, gamma=0.1)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision and device.type == 'cuda' else None
    
    # TensorBoard writer
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training loop
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    
    logging.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        logging.info("-" * 50)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation phase
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if config.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logging.info(f"Learning Rate: {current_lr:.6f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.save_checkpoint_every == 0:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'config': asdict(config),
                'label_names': label_names
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping and best model saving
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = model_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'config': asdict(config),
                'label_names': label_names
            }, best_model_path)
            logging.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    total_time = time.time() - start_time
    logging.info(f"\nTraining completed in {total_time:.2f}s")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Final evaluation on test set
    logging.info("\nEvaluating on test set...")
    
    # Load best model
    best_model_path = model_dir / "best_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device)
    
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted')
    logging.info(f"Test Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save final results
    results = {
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'total_training_time': total_time,
        'config': asdict(config),
        'label_names': label_names
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    logging.info(f"Training results saved to: {results_path}")


if __name__ == "__main__":
    main() 