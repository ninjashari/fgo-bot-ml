"""
FGO-Bot-ML: Dataset Analysis and Quality Assessment

This script analyzes the collected servant and craft essence datasets to:
1. Assess data quality and completeness
2. Generate distribution statistics  
3. Identify potential issues for ML training
4. Create data quality reports
5. Validate image formats and integrity

Author: FGO-Bot-ML Team
Date: 2025-06-30
"""
import os
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import hashlib

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImageInfo:
    """Container for image metadata and analysis results."""
    path: str
    size_bytes: int
    dimensions: Tuple[int, int]
    format: str
    mode: str
    is_valid: bool
    file_hash: str

@dataclass  
class DatasetStats:
    """Container for dataset statistics."""
    total_entities: int
    total_images: int
    total_size_mb: float
    avg_images_per_entity: float
    image_formats: Dict[str, int]
    image_dimensions: Dict[Tuple[int, int], int]
    corrupted_images: List[str]
    duplicate_hashes: Dict[str, List[str]]
    entities_without_images: List[str]
    entities_without_metadata: List[str]
    common_dimensions: List[Tuple[Tuple[int, int], int]]

class DatasetAnalyzer:
    """Comprehensive dataset analysis tool for FGO ML datasets."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.servants_dir = self.dataset_dir / "servants"
        self.ces_dir = self.dataset_dir / "ces"
        self.servants_stats = None
        self.ces_stats = None
        self.overall_stats = {}
        logger.info(f"Initialized DatasetAnalyzer for {self.dataset_dir}")
    
    def analyze_image(self, image_path: Path) -> ImageInfo:
        """Analyze a single image file for metadata and integrity."""
        try:
            size_bytes = image_path.stat().st_size
            
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    dimensions = img.size
                    format_type = img.format or "UNKNOWN"
                    mode = img.mode
                    is_valid = True
            else:
                dimensions = (0, 0)
                format_type = image_path.suffix.upper().replace('.', '')
                mode = "UNKNOWN"
                is_valid = True
                
            return ImageInfo(
                path=str(image_path),
                size_bytes=size_bytes,
                dimensions=dimensions,
                format=format_type,
                mode=mode,
                is_valid=is_valid,
                file_hash=file_hash
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze image {image_path}: {e}")
            return ImageInfo(
                path=str(image_path),
                size_bytes=0,
                dimensions=(0, 0),
                format="UNKNOWN",
                mode="UNKNOWN", 
                is_valid=False,
                file_hash=""
            )
    
    def analyze_entity_directory(self, entity_dir: Path) -> Tuple[List[ImageInfo], bool]:
        """Analyze a single entity directory (servant or CE)."""
        images = []
        has_metadata = False
        
        if not entity_dir.is_dir():
            return images, has_metadata
        
        # Check for metadata.json
        metadata_path = entity_dir / "metadata.json"
        has_metadata = metadata_path.exists()
        
        # Analyze all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        for pattern in image_extensions:
            for image_file in entity_dir.glob(pattern):
                images.append(self.analyze_image(image_file))
        
        return images, has_metadata
    
    def calculate_dataset_stats(self, dataset_dir: Path, dataset_name: str) -> DatasetStats:
        """Calculate comprehensive statistics for a dataset directory."""
        logger.info(f"Analyzing {dataset_name} dataset at {dataset_dir}")
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory does not exist: {dataset_dir}")
            return DatasetStats(
                total_entities=0, total_images=0, total_size_mb=0.0,
                avg_images_per_entity=0.0, image_formats={}, image_dimensions={},
                corrupted_images=[], duplicate_hashes={}, entities_without_images=[],
                entities_without_metadata=[], common_dimensions=[]
            )
        
        total_entities = 0
        total_images = 0
        total_size_bytes = 0
        image_formats = Counter()
        image_dimensions = Counter()
        corrupted_images = []
        file_hashes = defaultdict(list)
        entities_without_images = []
        entities_without_metadata = []
        
        # Analyze each entity directory
        for entity_dir in dataset_dir.iterdir():
            if not entity_dir.is_dir():
                continue
                
            total_entities += 1
            images, has_metadata = self.analyze_entity_directory(entity_dir)
            
            if not images:
                entities_without_images.append(entity_dir.name)
                
            if not has_metadata:
                entities_without_metadata.append(entity_dir.name)
            
            # Process image statistics
            for img_info in images:
                total_images += 1
                total_size_bytes += img_info.size_bytes
                
                if img_info.is_valid:
                    image_formats[img_info.format] += 1
                    if img_info.dimensions != (0, 0):
                        image_dimensions[img_info.dimensions] += 1
                    if img_info.file_hash:
                        file_hashes[img_info.file_hash].append(img_info.path)
                else:
                    corrupted_images.append(img_info.path)
        
        # Find duplicates
        duplicate_hashes = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
        
        # Calculate averages and common dimensions
        avg_images_per_entity = total_images / total_entities if total_entities > 0 else 0
        total_size_mb = total_size_bytes / (1024 * 1024)
        common_dimensions = image_dimensions.most_common(10)
        
        return DatasetStats(
            total_entities=total_entities,
            total_images=total_images,
            total_size_mb=total_size_mb,
            avg_images_per_entity=avg_images_per_entity,
            image_formats=dict(image_formats),
            image_dimensions=dict(image_dimensions),
            corrupted_images=corrupted_images,
            duplicate_hashes=duplicate_hashes,
            entities_without_images=entities_without_images,
            entities_without_metadata=entities_without_metadata,
            common_dimensions=common_dimensions
        )
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on both servant and CE datasets."""
        logger.info("Starting full dataset analysis...")
        
        # Analyze servants dataset
        if self.servants_dir.exists():
            self.servants_stats = self.calculate_dataset_stats(self.servants_dir, "Servants")
        else:
            logger.warning(f"Servants directory not found: {self.servants_dir}")
        
        # Analyze CEs dataset
        if self.ces_dir.exists():
            self.ces_stats = self.calculate_dataset_stats(self.ces_dir, "Craft Essences")
        else:
            logger.warning(f"CEs directory not found: {self.ces_dir}")
        
        # Calculate overall statistics
        self.calculate_overall_stats()
        
        logger.info("Dataset analysis completed!")
        return self.get_analysis_results()
    
    def calculate_overall_stats(self):
        """Calculate combined statistics across all datasets."""
        total_entities = 0
        total_images = 0
        total_size_mb = 0.0
        
        if self.servants_stats:
            total_entities += self.servants_stats.total_entities
            total_images += self.servants_stats.total_images
            total_size_mb += self.servants_stats.total_size_mb
            
        if self.ces_stats:
            total_entities += self.ces_stats.total_entities
            total_images += self.ces_stats.total_images
            total_size_mb += self.ces_stats.total_size_mb
        
        self.overall_stats = {
            'total_entities': total_entities,
            'total_images': total_images,
            'total_size_mb': total_size_mb,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all analysis results as a dictionary."""
        return {
            'overall': self.overall_stats,
            'servants': asdict(self.servants_stats) if self.servants_stats else None,
            'craft_essences': asdict(self.ces_stats) if self.ces_stats else None
        }
    
    def print_summary_report(self):
        """Print a human-readable summary of the analysis results."""
        print("\n" + "="*80)
        print("📊 FGO-Bot-ML Dataset Analysis Report")
        print("="*80)
        
        if self.overall_stats:
            print(f"📈 Overall Statistics:")
            print(f"   Total Entities: {self.overall_stats.get('total_entities', 0):,}")
            print(f"   Total Images: {self.overall_stats.get('total_images', 0):,}")
            print(f"   Total Size: {self.overall_stats.get('total_size_mb', 0):.1f} MB")
            print(f"   Analysis Time: {self.overall_stats.get('analysis_timestamp', 'Unknown')}")
            print()
        
        if self.servants_stats:
            print(f"⚔️  Servants Dataset:")
            self.print_dataset_summary(self.servants_stats, "   ")
            
        if self.ces_stats:
            print(f"🎴 Craft Essences Dataset:")
            self.print_dataset_summary(self.ces_stats, "   ")
        
        print("\n" + "="*80)
    
    def print_dataset_summary(self, stats: DatasetStats, indent: str):
        """Print summary for a specific dataset."""
        print(f"{indent}Entities: {stats.total_entities:,}")
        print(f"{indent}Images: {stats.total_images:,}")
        print(f"{indent}Size: {stats.total_size_mb:.1f} MB")
        print(f"{indent}Avg Images/Entity: {stats.avg_images_per_entity:.1f}")
        
        if stats.corrupted_images:
            print(f"{indent}⚠️  Corrupted Images: {len(stats.corrupted_images)}")
            
        if stats.duplicate_hashes:
            print(f"{indent}🔄 Duplicate Images: {len(stats.duplicate_hashes)} sets")
            
        if stats.entities_without_images:
            print(f"{indent}📁 Empty Entities: {len(stats.entities_without_images)}")
            
        if stats.entities_without_metadata:
            print(f"{indent}📝 Missing Metadata: {len(stats.entities_without_metadata)}")
        
        print(f"{indent}Image Formats: {dict(stats.image_formats)}")
        
        if stats.common_dimensions:
            print(f"{indent}Common Dimensions:")
            for (w, h), count in stats.common_dimensions[:5]:
                print(f"{indent}   {w}x{h}: {count} images")
        print()
    
    def export_detailed_report(self, output_path: str = "dataset_analysis_report.json"):
        """Export detailed analysis results to JSON file."""
        results = self.get_analysis_results()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Detailed analysis report exported to {output_path}")

def main():
    """Main execution function for the dataset analysis script."""
    parser = argparse.ArgumentParser(description="Analyze FGO-Bot-ML datasets")
    parser.add_argument("--detailed", action="store_true", 
                       help="Print detailed analysis information")
    parser.add_argument("--export-report", action="store_true",
                       help="Export detailed report to JSON file")
    parser.add_argument("--dataset-dir", default="dataset",
                       help="Path to dataset directory (default: dataset)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.dataset_dir)
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    # Print summary
    analyzer.print_summary_report()
    
    # Export detailed report if requested
    if args.export_report:
        analyzer.export_detailed_report()
    
    # Print detailed information if requested
    if args.detailed:
        print("\n📋 Detailed Analysis Results:")
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
