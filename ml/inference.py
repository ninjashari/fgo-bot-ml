#!/usr/bin/env python3
"""
FGO Image Classification Inference

This module provides inference capabilities for trained FGO image classification models.
It supports both single image and batch prediction with confidence scoring and 
visualization capabilities.

Key Features:
- GPU-accelerated inference
- Batch processing support
- Confidence scoring and top-k predictions
- Visualization of results
- Support for multiple model architectures
- Image preprocessing pipeline
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet50
import timm

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


class FGOInferenceDataset(Dataset):
    """Dataset class for inference on new images."""
    
    def __init__(self, image_paths: List[str], transform=None):
        """
        Initialize the inference dataset.
        
        Args:
            image_paths: List of paths to image files
            transform: Torchvision transforms to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, image_path)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, image_path
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            black_image = torch.zeros(3, 224, 224)
            return black_image, image_path


class FGOClassifier:
    """
    FGO Image Classifier for inference.
    
    This class handles loading trained models and performing inference
    on new images with confidence scoring and visualization.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use for inference ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_names = None
        self.config = None
        self.transform = None
        
        self.load_model(model_path)
        self.setup_transforms()
    
    def load_model(self, model_path: str) -> None:
        """
        Load the trained model from checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
        """
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model information
        self.config = checkpoint['config']
        self.label_names = checkpoint['label_names']
        num_classes = len(self.label_names)
        
        print(f"Model: {self.config['model_name']}, Classes: {num_classes}")
        
        # Create model architecture
        model_name = self.config['model_name']
        if model_name == "efficientnet_b0":
            self.model = efficientnet_b0(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == "resnet50":
            self.model = resnet50(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v3_large":
            self.model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def setup_transforms(self) -> None:
        """Set up image preprocessing transforms."""
        img_size = self.config.get('image_size', 224)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.label_names)))
                
                predictions = []
                for i in range(top_probs.size(1)):
                    class_idx = top_indices[0][i].item()
                    confidence = top_probs[0][i].item()
                    class_name = self.label_names[class_idx]
                    
                    predictions.append({
                        'class_name': class_name,
                        'class_index': class_idx,
                        'confidence': confidence
                    })
            
            return {
                'image_path': image_path,
                'original_size': original_size,
                'predictions': predictions,
                'top_prediction': predictions[0] if predictions else None
            }
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'predictions': [],
                'top_prediction': None
            }
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32, 
                     top_k: int = 5) -> List[Dict]:
        """
        Predict the classes of a batch of images.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction dictionaries
        """
        # Create dataset and dataloader
        dataset = FGOInferenceDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="Processing batches"):
                batch_images = batch_images.to(self.device)
                
                # Perform inference
                outputs = self.model(batch_images)
                probabilities = F.softmax(outputs, dim=1)
                
                # Process each image in the batch
                for i, image_path in enumerate(batch_paths):
                    # Get top-k predictions for this image
                    image_probs = probabilities[i]
                    top_probs, top_indices = torch.topk(image_probs, k=min(top_k, len(self.label_names)))
                    
                    predictions = []
                    for j in range(len(top_probs)):
                        class_idx = top_indices[j].item()
                        confidence = top_probs[j].item()
                        class_name = self.label_names[class_idx]
                        
                        predictions.append({
                            'class_name': class_name,
                            'class_index': class_idx,
                            'confidence': confidence
                        })
                    
                    all_predictions.append({
                        'image_path': image_path,
                        'predictions': predictions,
                        'top_prediction': predictions[0] if predictions else None
                    })
        
        return all_predictions
    
    def visualize_prediction(self, image_path: str, prediction: Dict, 
                           save_path: Optional[str] = None, 
                           show_top_k: int = 3) -> None:
        """
        Visualize prediction results on the image.
        
        Args:
            image_path: Path to the original image
            prediction: Prediction dictionary from predict_single
            save_path: Optional path to save the visualization
            show_top_k: Number of top predictions to show
        """
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Show predictions
        if 'predictions' in prediction and prediction['predictions']:
            predictions = prediction['predictions'][:show_top_k]
            
            # Create bar chart of confidences
            class_names = [pred['class_name'] for pred in predictions]
            confidences = [pred['confidence'] for pred in predictions]
            
            bars = ax2.barh(range(len(class_names)), confidences)
            ax2.set_yticks(range(len(class_names)))
            ax2.set_yticklabels(class_names)
            ax2.set_xlabel('Confidence')
            ax2.set_title(f'Top {len(predictions)} Predictions')
            ax2.set_xlim(0, 1)
            
            # Color bars based on confidence
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                if i == 0:  # Top prediction
                    bar.set_color('green')
                elif conf > 0.1:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
                
                # Add confidence text
                ax2.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.3f}', va='center')
        
        else:
            ax2.text(0.5, 0.5, 'No predictions available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Predictions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='FGO Image Classification Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, 
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to return')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image_dir")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize classifier
    print("Initializing FGO Classifier...")
    classifier = FGOClassifier(args.model, args.device)
    
    if args.image:
        # Single image prediction
        print(f"Predicting single image: {args.image}")
        prediction = classifier.predict_single(args.image, args.top_k)
        
        # Print results
        print(f"\nPrediction for {args.image}:")
        print("-" * 50)
        
        if prediction.get('error'):
            print(f"Error: {prediction['error']}")
        elif prediction['predictions']:
            for i, pred in enumerate(prediction['predictions']):
                print(f"{i+1}. {pred['class_name']}: {pred['confidence']:.4f}")
        
        # Save results
        results_path = output_dir / "single_image_prediction.json"
        with open(results_path, 'w') as f:
            json.dump(prediction, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Visualize if requested
        if args.visualize:
            viz_path = output_dir / "prediction_visualization.png"
            classifier.visualize_prediction(args.image, prediction, str(viz_path))
    
    elif args.image_dir:
        # Batch prediction
        image_dir = Path(args.image_dir)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f"*{ext}")))
            image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Processing {len(image_paths)} images from {image_dir}")
        
        # Batch prediction
        predictions = classifier.predict_batch([str(p) for p in image_paths], 
                                             args.batch_size, args.top_k)
        
        # Print summary
        print(f"\nBatch Prediction Summary:")
        print("-" * 50)
        
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred in predictions:
            if pred['top_prediction']:
                image_name = Path(pred['image_path']).name
                top_class = pred['top_prediction']['class_name']
                confidence = pred['top_prediction']['confidence']
                print(f"{image_name}: {top_class} ({confidence:.4f})")
        
        # Save batch results
        results_path = output_dir / "batch_predictions.json"
        with open(results_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nBatch results saved to: {results_path}")
        
        # Create summary report
        summary = {
            'total_images': total_predictions,
            'model_path': args.model,
            'predictions_summary': {}
        }
        
        # Count predictions by class
        for pred in predictions:
            if pred['top_prediction']:
                class_name = pred['top_prediction']['class_name']
                if class_name not in summary['predictions_summary']:
                    summary['predictions_summary'][class_name] = 0
                summary['predictions_summary'][class_name] += 1
        
        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main() 