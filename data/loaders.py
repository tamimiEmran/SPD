"""
loaders module for V2X-Seq project.

This module provides high-level data loading functionality, including:
- Function to load datasets with different configurations
- Factory functions for creating train/val/test dataloaders
- Utility functions for determining appropriate dataset parameters
- Support for different fusion strategies and data splits
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader


# add M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\v2x_tracking\data
# to the sys.path
import sys
sys.path.append("M:/Documents/Mwasalat/dataset/Full Dataset (train & val)-20250313T155844Z/Full Dataset (train & val)/V2X-Seq-SPD/V2X-Seq-SPD/v2x_tracking/data")

from dataset import V2XDataset, V2XBaseDataset
from preprocessing.transform import transform_points_to_veh_coordinate
from preprocessing.augmentation import augment_point_cloud

logger = logging.getLogger(__name__)


def load_dataset_config(config_path: Union[str, Dict]) -> Dict:
    """
    Load dataset configuration from YAML file or dictionary.
    
    Args:
        config_path: Path to YAML config file or dictionary with config
        
    Returns:
        Dictionary with dataset configuration
    """
    if isinstance(config_path, dict):
        return config_path
    
    # Load from file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def create_dataloader(
    dataset_path: str,
    config: Union[str, Dict],
    split: str = 'train',
    distributed: bool = False,
    **kwargs
) -> Tuple[Dataset, DataLoader]:
    """
    Create dataset and dataloader based on configuration.
    
    Args:
        dataset_path: Path to dataset root directory
        config: Path to config file or config dictionary
        split: Data split ('train', 'val', or 'test')
        distributed: Whether to use distributed training
        **kwargs: Additional arguments to override config values
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    # Load configuration
    config_dict = load_dataset_config(config)
    
    # Update config with kwargs
    if 'dataset' in config_dict:
        dataset_config = config_dict['dataset'].copy()
    else:
        dataset_config = config_dict.copy()
        
    # Override with kwargs
    for k, v in kwargs.items():
        dataset_config[k] = v
    
    # Create dataset
    fusion_strategy = dataset_config.pop('fusion_strategy', 'late_fusion')
    simulate_latency = dataset_config.pop('simulate_latency', False)
    latency_ms = dataset_config.pop('latency_ms', 200)
    batch_size = dataset_config.pop('batch_size', 4)
    num_workers = dataset_config.pop('num_workers', 4)
    segment_length = dataset_config.pop('segment_length', 10)
    use_image = dataset_config.pop('use_image', False)
    shuffle = dataset_config.pop('shuffle', None)
    drop_last = dataset_config.pop('drop_last', None)
    augment = dataset_config.pop('augment', None)
    
    # Determine augmentation based on split if not specified
    if augment is None:
        augment = (split == 'train')
    
    # Create dataset
    dataset = V2XDataset(
        dataset_path=dataset_path,
        split=split,
        fusion_strategy=fusion_strategy,
        simulate_latency=simulate_latency,
        latency_ms=latency_ms,
        segment_length=segment_length,
        use_image=use_image,
        augment=augment,
        **dataset_config
    )
    
    # Create sampler for distributed training
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
        shuffle = False  # Shuffle is handled by sampler
    
    # Determine shuffle and drop_last if not specified
    if shuffle is None:
        shuffle = (split == 'train' and sampler is None)
    if drop_last is None:
        drop_last = (split == 'train')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=V2XBaseDataset.collate_fn,
        pin_memory=True,
        drop_last=drop_last,
        sampler=sampler
    )
    
    return dataset, dataloader


def create_train_val_dataloaders(
    dataset_path: str,
    config: Union[str, Dict],
    distributed: bool = False,
    val_batch_size: Optional[int] = None,
    **kwargs
) -> Dict[str, Tuple[Dataset, DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_path: Path to dataset root directory
        config: Path to config file or config dictionary
        distributed: Whether to use distributed training
        val_batch_size: Batch size for validation dataloader (defaults to train batch size)
        **kwargs: Additional arguments to override config values
        
    Returns:
        Dictionary with 'train' and 'val' keys containing (dataset, dataloader) tuples
    """
    # Load config
    config_dict = load_dataset_config(config)
    
    # Get batch sizes
    if 'dataset' in config_dict:
        train_batch_size = config_dict['dataset'].get('batch_size', 4)
    else:
        train_batch_size = config_dict.get('batch_size', 4)
    
    # Use provided val_batch_size or fallback to train_batch_size
    if val_batch_size is None:
        val_batch_size = train_batch_size
    
    # Create train dataloader
    train_dataset, train_dataloader = create_dataloader(
        dataset_path=dataset_path,
        config=config,
        split='train',
        distributed=distributed,
        **kwargs
    )
    
    # Create validation dataloader with potentially larger batch size
    val_kwargs = kwargs.copy()
    val_kwargs['batch_size'] = val_batch_size
    val_kwargs['augment'] = False  # No augmentation for validation
    
    val_dataset, val_dataloader = create_dataloader(
        dataset_path=dataset_path,
        config=config,
        split='val',
        distributed=distributed,
        **val_kwargs
    )
    
    return {
        'train': (train_dataset, train_dataloader),
        'val': (val_dataset, val_dataloader)
    }


def create_test_dataloader(
    dataset_path: str,
    config: Union[str, Dict],
    distributed: bool = False,
    **kwargs
) -> Tuple[Dataset, DataLoader]:
    """
    Create test dataloader.
    
    Args:
        dataset_path: Path to dataset root directory
        config: Path to config file or config dictionary
        distributed: Whether to use distributed training
        **kwargs: Additional arguments to override config values
        
    Returns:
        Tuple of (test_dataset, test_dataloader)
    """
    # Set test-specific defaults
    test_kwargs = kwargs.copy()
    test_kwargs['augment'] = False  # No augmentation for test
    
    # Create test dataloader
    test_dataset, test_dataloader = create_dataloader(
        dataset_path=dataset_path,
        config=config,
        split='test',
        distributed=distributed,
        **test_kwargs
    )
    
    return test_dataset, test_dataloader


class DatasetBuilder:
    """Builder class for creating datasets with different configurations."""
    
    def __init__(self, dataset_path: str, config: Union[str, Dict] = None):
        """
        Initialize dataset builder.
        
        Args:
            dataset_path: Path to dataset root directory
            config: Optional configuration file path or dictionary
        """
        self.dataset_path = dataset_path
        
        if config is not None:
            self.config = load_dataset_config(config)
        else:
            self.config = {}
        
        self.dataset_params = {
            'fusion_strategy': 'late_fusion',
            'simulate_latency': False,
            'latency_ms': 200,
            'segment_length': 10,
            'use_image': False,
            'augment': False
        }
        
        # Update with config values if present
        if 'dataset' in self.config:
            self.dataset_params.update(self.config['dataset'])
        else:
            self.dataset_params.update(self.config)
    
    def with_fusion(self, fusion_strategy: str) -> 'DatasetBuilder':
        """Set fusion strategy."""
        self.dataset_params['fusion_strategy'] = fusion_strategy
        return self
    
    def with_latency(self, simulate_latency: bool, latency_ms: int = 200) -> 'DatasetBuilder':
        """Set latency simulation parameters."""
        self.dataset_params['simulate_latency'] = simulate_latency
        self.dataset_params['latency_ms'] = latency_ms
        return self
    
    def with_segment_length(self, segment_length: int) -> 'DatasetBuilder':
        """Set segment length."""
        self.dataset_params['segment_length'] = segment_length
        return self
    
    def with_images(self, use_image: bool) -> 'DatasetBuilder':
        """Set whether to use images."""
        self.dataset_params['use_image'] = use_image
        return self
    
    def with_augmentation(self, augment: bool) -> 'DatasetBuilder':
        """Set whether to apply augmentation."""
        self.dataset_params['augment'] = augment
        return self
    
    def with_param(self, key: str, value: Any) -> 'DatasetBuilder':
        """Set arbitrary parameter."""
        self.dataset_params[key] = value
        return self
    
    def build(self, split: str) -> Dataset:
        """
        Build dataset with configured parameters.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            Configured dataset
        """
        # Make a copy to avoid modifying the original
        params = self.dataset_params.copy()
        
        # Create dataset
        dataset = V2XDataset(
            dataset_path=self.dataset_path,
            split=split,
            **params
        )
        
        return dataset
    
    def build_dataloader(
        self, 
        split: str, 
        batch_size: int = 4, 
        num_workers: int = 4,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
        distributed: bool = False
    ) -> Tuple[Dataset, DataLoader]:
        """
        Build dataset and dataloader with configured parameters.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            distributed: Whether to use distributed training
            
        Returns:
            Tuple of (dataset, dataloader)
        """
        # Build dataset
        dataset = self.build(split)
        
        # Set defaults based on split if not specified
        if shuffle is None:
            shuffle = (split == 'train')
        if drop_last is None:
            drop_last = (split == 'train')
        
        # Create sampler for distributed training
        sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Shuffle is handled by sampler
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=num_workers,
            collate_fn=V2XBaseDataset.collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            sampler=sampler
        )
        
        return dataset, dataloader


def get_class_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Calculate class distribution in dataset.
    
    Args:
        dataset: V2X-Seq dataset
        
    Returns:
        Dictionary mapping class names to counts
    """
    class_counts = {}
    
    # Sample a subset of data for efficiency
    sample_size = min(100, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size].tolist()
    
    for idx in indices:
        sample = dataset[idx]
        
        # Extract labels from cooperative labels if available, otherwise from vehicle labels
        labels = []
        if 'cooperative_labels' in sample and sample['cooperative_labels']:
            for frame_labels in sample['cooperative_labels']:
                for obj in frame_labels:
                    if 'type' in obj:
                        labels.append(obj['type'])
        elif 'vehicle_labels' in sample:
            for frame_labels in sample['vehicle_labels']:
                for obj in frame_labels:
                    if 'type' in obj:
                        labels.append(obj['type'])
        
        # Update counts
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
    
    # Scale counts to represent full dataset
    scale_factor = len(dataset) / sample_size
    for label in class_counts:
        class_counts[label] = int(class_counts[label] * scale_factor)
    
    return class_counts


def get_dataset_statistics(dataset_path: str, split: str = 'train') -> Dict:
    """
    Calculate statistics for the dataset.
    
    Args:
        dataset_path: Path to dataset root directory
        split: Data split to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    # Load minimal dataset to analyze
    dataset = V2XDataset(
        dataset_path=dataset_path,
        split=split,
        segment_length=1,  # Minimal segment length for efficiency
        use_infrastructure=True,
        use_image=False
    )
    
    # Calculate statistics
    stats = {
        'num_sequences': len(dataset.sequences),
        'num_frames': sum(len(dataset.sequence_frames[seq_id]) for seq_id in dataset.sequences),
        'class_distribution': get_class_distribution(dataset),
    }
    
    # Calculate point cloud statistics from a sample
    sample_size = min(10, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size].tolist()
    
    total_points = 0
    max_points = 0
    min_points = float('inf')
    
    for idx in indices:
        sample = dataset[idx]
        for points in sample['vehicle_points']:
            num_points = len(points)
            total_points += num_points
            max_points = max(max_points, num_points)
            min_points = min(min_points, num_points)
    
    if sample_size > 0:
        stats['point_cloud'] = {
            'avg_points_per_frame': total_points / (sample_size * len(sample['vehicle_points'])),
            'max_points': max_points,
            'min_points': min_points
        }
    
    return stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    dir_path = r"M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD"

    infra_path = os.path.join(dir_path, "infrastructure-side")
    vehicle_path = os.path.join(dir_path, "vehicle-side")
    cooperative_path = os.path.join(dir_path, "cooperative")

    



    parser = argparse.ArgumentParser(description="V2X-Seq data loader utilities")
    parser.add_argument("--dataset_path", type=str, default= "no", choices= ["infra", "vehicle", "coop", "no"] , help="Path to dataset")


    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], 
                       help="Dataset split")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    
    args = parser.parse_args()

    if args.dataset_path == "infra":
        args.dataset_path = infra_path
    elif args.dataset_path == "vehicle":
        args.dataset_path = vehicle_path
    elif args.dataset_path == "coop":
        args.dataset_path = cooperative_path

    elif args.dataset_path == "no":
        args.dataset_path = dir_path
    
    else:
        raise ValueError(f"Unsupported dataset path: {args.dataset_path}")


    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.stats:
        # Print dataset statistics
        stats = get_dataset_statistics(args.dataset_path, args.split)
        print(f"Dataset statistics for {args.split} split:")
        print(f"Number of sequences: {stats['num_sequences']}")
        print(f"Number of frames: {stats['num_frames']}")
        print("Class distribution:")
        for cls, count in sorted(stats['class_distribution'].items()):
            print(f"  {cls}: {count}")
        
        if 'point_cloud' in stats:
            print("Point cloud statistics:")
            print(f"  Average points per frame: {stats['point_cloud']['avg_points_per_frame']:.2f}")
            print(f"  Maximum points: {stats['point_cloud']['max_points']}")
            print(f"  Minimum points: {stats['point_cloud']['min_points']}")
    else:
        # Create a dataloader example
        if args.config:
            config = args.config
        else:
            # Default config
            config = {
                'fusion_strategy': 'late_fusion',
                'segment_length': 5,
                'batch_size': 2,
                'num_workers': 2
            }
        
        # Create dataloader
        dataset, dataloader = create_dataloader(
            dataset_path=args.dataset_path,
            config=config,
            split=args.split
        )
        
        print(f"Created dataloader for {args.split} split")
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Print a sample batch
        sample_batch = next(iter(dataloader))
        print("\nSample batch keys:")
        for key in sample_batch:
            if isinstance(sample_batch[key], list):
                print(f"  {key}: List of length {len(sample_batch[key])}")
            else:
                print(f"  {key}: {type(sample_batch[key])}")