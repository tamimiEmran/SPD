"""
fusion_base module for V2X-Seq project.

This module provides the abstract base class for all fusion strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class FusionBase(ABC):
    """
    Abstract base class for all fusion strategies in Vehicle-Infrastructure Cooperative 3D Tracking.
    
    This class defines the interface that all fusion implementations must follow,
    ensuring consistent behavior across different fusion strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fusion base with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for the fusion strategy
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def fuse(self, vehicle_data: Dict[str, Any], infrastructure_data: Dict[str, Any], 
             vehicle_to_world_transform: Any, infrastructure_to_world_transform: Any,
             timestamp_diff: float = 0.0) -> Dict[str, Any]:
        """
        Fuse data from vehicle and infrastructure sources.
        
        Args:
            vehicle_data: Dictionary containing vehicle sensor data or processed results
            infrastructure_data: Dictionary containing infrastructure sensor data or processed results
            vehicle_to_world_transform: Transformation from vehicle to world coordinate system
            infrastructure_to_world_transform: Transformation from infrastructure to world coordinate system
            timestamp_diff: Time difference between vehicle and infrastructure frames (seconds)
            
        Returns:
            Dictionary containing fused results
        """
        pass
    
    @abstractmethod
    def get_bandwidth_usage(self, data_size: int) -> int:
        """
        Calculate the bandwidth usage for the fusion strategy.
        
        Args:
            data_size: Size of the data (e.g., number of objects, size of features)
            
        Returns:
            Estimated bandwidth usage in bytes
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the fusion strategy.
        
        Returns:
            Name of the fusion strategy
        """
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the fusion strategy.
        
        Returns:
            Dictionary containing the configuration parameters
        """
        return self.config
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration parameters.
        
        Args:
            config: Dictionary containing new configuration parameters
        """
        self.config.update(config)