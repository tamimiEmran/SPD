"""
base_model module for V2X-Seq project.

This module provides abstract base classes and common functionality for models
used in the Vehicle-Infrastructure Cooperative 3D Tracking (VIC3D) task.
It includes:
- BaseModel: Abstract base class for all models
- DetectionModel: Base class for 3D object detection models
- FusionModel: Base class for different fusion strategies
- LossCalculator: Utility for calculating and combining losses
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the V2X-Seq project.
    
    This class defines the common interface and functionality that all models
    should implement, including saving/loading checkpoints, computing loss,
    and forward propagation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize base model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config
        self.device = None
        
    @abstractmethod
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass for the model.
        
        Args:
            batch_dict: Dictionary containing batch data
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch_dict: Dict, outputs: Dict) -> Dict:
        """
        Compute loss for the model.
        
        Args:
            batch_dict: Dictionary containing batch data
            outputs: Dictionary containing model outputs
            
        Returns:
            Dictionary containing computed losses
        """
        pass
    
    def to_device(self, device: torch.device) -> 'BaseModel':
        """
        Move model to specified device.
        
        Args:
            device: Device to move model to
            
        Returns:
            Self with model moved to device
        """
        self.device = device
        return self.to(device)
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       epoch: int = 0, metadata: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint to
            optimizer: Optional optimizer to save state
            scheduler: Optional learning rate scheduler to save state
            epoch: Current epoch number
            metadata: Additional metadata to save with checkpoint
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'metadata': metadata or {}
        }
        
        # Add optimizer and scheduler states if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       strict: bool = True) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            optimizer: Optional optimizer to load state
            scheduler: Optional learning rate scheduler to load state
            strict: Whether to strictly enforce that the keys in state_dict
                   match the keys returned by this module's state_dict()
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        # Check if checkpoint exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Model checkpoint loaded from {path}")
        
        # Return metadata
        return {
            'epoch': checkpoint.get('epoch', 0),
            'config': checkpoint.get('config', {}),
            'metadata': checkpoint.get('metadata', {})
        }
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def train_mode(self) -> 'BaseModel':
        """
        Set model to training mode.
        
        Returns:
            Self with model in training mode
        """
        self.train()
        return self
    
    def eval_mode(self) -> 'BaseModel':
        """
        Set model to evaluation mode.
        
        Returns:
            Self with model in evaluation mode
        """
        self.eval()
        return self
    
    def summary(self) -> str:
        """
        Generate a summary of the model.
        
        Returns:
            String containing model summary
        """
        # Count number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Build summary string
        summary_str = (
            f"{self.__class__.__name__} Summary:\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
        )
        
        return summary_str


class DetectionModel(BaseModel):
    """
    Base class for 3D object detection models.
    
    This class extends BaseModel with functionality specific to 3D
    object detection, such as generating 3D bounding boxes and
    handling detection-specific losses.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize detection model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__(config)
        self.num_classes = config.get('num_classes', 4)  # Default to 4 classes
        self.detection_threshold = config.get('detection_threshold', 0.5)
        
        # Initialize loss functions
        self._init_loss_functions()
    
    def _init_loss_functions(self) -> None:
        """Initialize loss functions for detection."""
        # Classification loss
        self.cls_loss_func = nn.CrossEntropyLoss(reduction='mean')
        
        # Regression loss (smooth L1)
        self.reg_loss_func = nn.SmoothL1Loss(reduction='mean')
        
        # Direction (orientation) loss
        self.dir_loss_func = nn.CrossEntropyLoss(reduction='mean')
        
        # Loss weights
        self.cls_weight = self.config.get('cls_weight', 1.0)
        self.reg_weight = self.config.get('reg_weight', 1.0)
        self.dir_weight = self.config.get('dir_weight', 0.2)
    
    def compute_loss(self, batch_dict: Dict, outputs: Dict) -> Dict:
        """
        Compute detection losses.
        
        Args:
            batch_dict: Dictionary containing batch data
            outputs: Dictionary containing model outputs
            
        Returns:
            Dictionary containing computed losses
        """
        # Extract predictions and targets
        cls_preds = outputs.get('cls_preds')
        box_preds = outputs.get('box_preds')
        dir_preds = outputs.get('dir_preds')
        
        cls_targets = batch_dict.get('cls_targets')
        box_targets = batch_dict.get('box_targets')
        dir_targets = batch_dict.get('dir_targets')
        
        # Compute classification loss
        cls_loss = self.cls_loss_func(cls_preds, cls_targets) if cls_preds is not None else 0
        
        # Compute regression loss
        reg_loss = self.reg_loss_func(box_preds, box_targets) if box_preds is not None else 0
        
        # Compute direction loss
        dir_loss = self.dir_loss_func(dir_preds, dir_targets) if dir_preds is not None else 0
        
        # Compute total loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.dir_weight * dir_loss
        
        # Return loss dictionary
        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'dir_loss': dir_loss,
            'total_loss': total_loss
        }
        
        return loss_dict
    
    def generate_detections(self, outputs: Dict) -> List[Dict]:
        """
        Generate detection results from model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            
        Returns:
            List of detection dictionaries
        """
        # Implement in concrete subclasses
        raise NotImplementedError("Subclasses must implement generate_detections")


class FusionModel(BaseModel):
    """
    Base class for different fusion strategies.
    
    This class extends BaseModel with functionality specific to fusion
    models, such as fusing features or detections from vehicle and
    infrastructure sensors.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fusion model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__(config)
        self.fusion_strategy = config.get('fusion_strategy', 'late_fusion')
    
    def fuse(self, vehicle_data: Dict, infrastructure_data: Dict, vehicle_to_world: np.ndarray,
            infrastructure_to_world: np.ndarray, timestamp_diff: float = 0.0) -> Tuple[Dict, Optional[Dict]]:
        """
        Fuse vehicle and infrastructure data.
        
        Args:
            vehicle_data: Dictionary containing vehicle sensor data
            infrastructure_data: Dictionary containing infrastructure sensor data
            vehicle_to_world: Transformation from vehicle to world coordinates
            infrastructure_to_world: Transformation from infrastructure to world coordinates
            timestamp_diff: Time difference between vehicle and infrastructure frames
            
        Returns:
            Tuple of (fused_data, bandwidth_usage)
        """
        # Implement in concrete subclasses
        raise NotImplementedError("Subclasses must implement fuse")
    
    def compensate_latency(self, infrastructure_data: Dict, timestamp_diff: float) -> Dict:
        """
        Compensate for latency in infrastructure data.
        
        Args:
            infrastructure_data: Dictionary containing infrastructure sensor data
            timestamp_diff: Time difference between vehicle and infrastructure frames
            
        Returns:
            Dictionary containing compensated infrastructure data
        """
        # Implement in concrete subclasses
        raise NotImplementedError("Subclasses must implement compensate_latency")
    
    def compute_bandwidth_usage(self, data: Any) -> int:
        """
        Compute bandwidth usage for data transmission.
        
        Args:
            data: Data being transmitted
            
        Returns:
            Bandwidth usage in bytes
        """
        # Calculate size based on data type
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        elif isinstance(data, (list, tuple, dict)):
            import json
            return len(json.dumps(data).encode('utf-8'))
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        else:
            # Default case: convert to string and calculate bytes
            return len(str(data).encode('utf-8'))


class LossCalculator:
    """
    Utility class for calculating and combining losses.
    
    This class provides methods for calculating different types of losses
    and combining them with weights.
    """
    
    def __init__(self, loss_weights: Dict[str, float]):
        """
        Initialize loss calculator with loss weights.
        
        Args:
            loss_weights: Dictionary mapping loss names to weights
        """
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        self.cls_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss_func = nn.SmoothL1Loss(reduction='mean')
        self.dir_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.seg_loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    
    def compute_cls_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            pred: Prediction tensor
            target: Target tensor
            
        Returns:
            Classification loss
        """
        return self.cls_loss_func(pred, target)
    
    def compute_reg_loss(self, pred: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute regression loss.
        
        Args:
            pred: Prediction tensor
            target: Target tensor
            mask: Optional mask for valid regression targets
            
        Returns:
            Regression loss
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
            if pred.shape[0] == 0:
                return torch.tensor(0.0, device=pred.device)
        
        return self.reg_loss_func(pred, target)
    
    def compute_dir_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute direction loss.
        
        Args:
            pred: Prediction tensor
            target: Target tensor
            
        Returns:
            Direction loss
        """
        return self.dir_loss_func(pred, target)
    
    def compute_seg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute segmentation loss.
        
        Args:
            pred: Prediction tensor
            target: Target tensor
            
        Returns:
            Segmentation loss
        """
        return self.seg_loss_func(pred, target)
    
    def compute_combined_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute combined loss with weights.
        
        Args:
            losses: Dictionary mapping loss names to loss tensors
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights:
                total_loss += self.loss_weights[loss_name] * loss_value
            else:
                # If weight not specified, use 1.0
                total_loss += loss_value
        
        return total_loss


def create_model(model_type: str, config: Dict) -> BaseModel:
    """
    Factory function to create a model based on model type.
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Created model instance
    """
    if model_type == 'pointpillars':
        from models.detection.pointpillars import PointPillarsDetector
        return PointPillarsDetector(config)
    elif model_type == 'centerpoint':
        from models.detection.center_point import CenterPointDetector
        return CenterPointDetector(config)
    elif model_type == 'late_fusion':
        from fusion.late_fusion import LateFusion
        return LateFusion(config)
    elif model_type == 'early_fusion':
        from fusion.early_fusion import EarlyFusion
        return EarlyFusion(config)
    elif model_type == 'middle_fusion':
        from fusion.middle_fusion import MiddleFusion
        return MiddleFusion(config)
    elif model_type == 'ff_tracking':
        from fusion.ff_tracking import FFTracking
        return FFTracking(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")