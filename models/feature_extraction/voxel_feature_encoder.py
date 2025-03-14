"""
voxel_feature_encoder module for V2X-Seq project.

This module provides functionality for extracting features from voxelized point clouds,
which serves as a key component for 3D object detection and tracking in the
vehicle-infrastructure cooperative perception system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class VFELayer(nn.Module):
    """
    Voxel Feature Encoding Layer.
    
    This layer applies pointwise feature transformation to points in each voxel
    and aggregates them using max pooling to generate the voxel-level features.
    """
    
    def __init__(self, in_channels: int, out_channels: int, max_points: int, use_norm: bool = True):
        """
        Initialize the VFE layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            max_points: Maximum number of points per voxel
            use_norm: Whether to use batch normalization
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_points = max_points
        self.use_norm = use_norm
        
        # Pointwise MLP
        self.linear = nn.Linear(in_channels, out_channels, bias=(not use_norm))
        
        # Batch normalization
        if use_norm:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
    
    def forward(self, features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VFE layer.
        
        Args:
            features: Input features of shape (B, num_voxels, max_points, in_channels)
            num_points: Number of points in each voxel of shape (B, num_voxels, 1)
            
        Returns:
            Processed features of shape (B, num_voxels, max_points, out_channels)
        """
        # Extract batch and voxel dimensions
        batch_size, num_voxels, max_points, _ = features.shape
        
        # Reshape for linear layer
        features = features.reshape(-1, self.in_channels)
        
        # Apply linear layer
        features = self.linear(features)
        
        # Apply batch normalization
        if self.use_norm:
            features = self.norm(features)
        
        # Apply ReLU
        features = F.relu(features, inplace=True)
        
        # Reshape back
        features = features.reshape(batch_size, num_voxels, max_points, self.out_channels)
        
        # Get max feature for each voxel
        max_features = torch.max(features, dim=2, keepdim=True)[0]
        
        # Repeat max feature for each point
        max_features = max_features.repeat(1, 1, max_points, 1)
        
        # Concatenate point feature with max feature
        combined_features = torch.cat([features, max_features], dim=-1)
        
        # Create a mask for valid points
        mask = torch.arange(max_points, device=features.device).reshape(1, 1, -1, 1) < num_points
        mask = mask.expand(-1, -1, -1, combined_features.shape[-1])
        
        # Apply mask
        masked_features = combined_features * mask.float()
        
        return masked_features


class PFNLayer(nn.Module):
    """
    Pillar Feature Network Layer.
    
    This layer applies pointwise feature transformation to points in each pillar
    and aggregates them using max pooling to generate the pillar-level features.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True, last_layer: bool = False):
        """
        Initialize the PFN layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_norm: Whether to use batch normalization
            last_layer: Whether this is the last layer in the PFN
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.last_layer = last_layer
        
        # Pointwise MLP
        self.linear = nn.Linear(in_channels, out_channels, bias=(not use_norm))
        
        # Batch normalization
        if use_norm:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PFN layer.
        
        Args:
            features: Input features of shape (B, num_pillars, max_points, in_channels)
            
        Returns:
            Processed features of shape (B, num_pillars, feature_dim) if last_layer is True,
            else (B, num_pillars, max_points, feature_dim)
        """
        # Extract batch and pillar dimensions
        batch_size, num_pillars, max_points, _ = features.shape
        
        # Reshape for linear layer
        features_flat = features.view(-1, self.in_channels)
        
        # Apply linear layer
        features_flat = self.linear(features_flat)
        
        # Apply batch normalization
        if self.use_norm:
            features_flat = features_flat.view(batch_size * num_pillars, max_points, self.out_channels)
            features_flat = features_flat.transpose(1, 2).contiguous()  # (B*N, C, P)
            features_flat = self.norm(features_flat)
            features_flat = features_flat.transpose(1, 2).contiguous()  # (B*N, P, C)
            features_flat = features_flat.view(-1, self.out_channels)
        
        # Apply ReLU
        features_flat = F.relu(features_flat, inplace=True)
        
        # Reshape back
        features = features_flat.view(batch_size, num_pillars, max_points, self.out_channels)
        
        # For last layer, perform max pooling to get one feature per pillar
        if self.last_layer:
            # Max pooling across points
            pooled_features = features.max(dim=2)[0]  # (B, N, C)
            return pooled_features
        
        return features


class VoxelFeatureEncoder(nn.Module):
    """
    Feature encoder for extracting features from voxelized point clouds.
    
    This class implements a voxel feature encoder that can be used to extract
    features from point cloud data for 3D object detection and tracking.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # x, y, z, intensity
        feature_dim: int = 64,
        max_points_per_voxel: int = 100,
        voxel_size: List[float] = [0.16, 0.16, 4],
        point_cloud_range: List[float] = [0, -39.68, -3, 100, 39.68, 1],
        use_height: bool = True
    ):
        """
        Initialize the voxel feature encoder.
        
        Args:
            in_channels: Number of input channels per point (x, y, z, intensity, ...)
            feature_dim: Dimension of output features
            max_points_per_voxel: Maximum number of points per voxel
            voxel_size: Size of voxels in [x, y, z]
            point_cloud_range: Range of point cloud in [x_min, y_min, z_min, x_max, y_max, z_max]
            use_height: Whether to use height as an additional feature
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_height = use_height
        
        # Calculate grid size
        grid_size = (
            np.round((np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])) / 
                     np.array(voxel_size)).astype(np.int32)
        )
        self.grid_size = grid_size
        
        # Number of channels for voxel features
        num_vfe_features = in_channels + 3  # Raw features + offset to voxel center
        if use_height:
            num_vfe_features += 1  # Add height feature
        
        # Create Voxel Feature Encoding (VFE) layers
        self.vfe1 = VFELayer(num_vfe_features, 32, max_points_per_voxel)
        self.vfe2 = VFELayer(32, feature_dim, max_points_per_voxel)
        
        # MLP for pointwise feature transform
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)
        self.norm = nn.BatchNorm1d(feature_dim)
    
    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass of the voxel feature encoder.
        
        Args:
            batch_dict: Dictionary containing:
                - 'voxels': (B, max_voxels, max_points_per_voxel, in_channels)
                - 'voxel_coords': (B, max_voxels, 4) [batch_idx, z, y, x]
                - 'voxel_num_points': (B, max_voxels)
            
        Returns:
            Updated batch_dict with:
                - 'voxel_features': (B, max_voxels, feature_dim)
                - 'voxel_coords': (B, max_voxels, 4) [batch_idx, z, y, x]
        """
        voxels = batch_dict['voxels']
        voxel_coords = batch_dict['voxel_coords']
        voxel_num_points = batch_dict['voxel_num_points']
        
        # Reshape for batch processing
        batch_size = voxels.shape[0]
        num_voxels = voxels.shape[1]
        
        # Extract point coordinates and features
        points_xyz = voxels[:, :, :, :3]
        points_features = voxels[:, :, :, 3:]
        
        # Calculate voxel centers
        voxel_centers = self._get_voxel_centers(
            voxel_coords[:, :, 1:4],
            batch_size,
            num_voxels
        )
        
        # Calculate points offset from voxel centers
        points_relative = points_xyz - voxel_centers.unsqueeze(2)
        
        # Combine features
        features = [points_xyz, points_relative, points_features]
        
        # Add height feature if requested
        if self.use_height:
            height_feature = points_xyz[:, :, :, 2:3]
            features.append(height_feature)
        
        # Concatenate all features
        features = torch.cat(features, dim=-1)
        
        # Apply VFE layers
        voxel_features = self.vfe1(features, voxel_num_points.unsqueeze(-1))
        voxel_features = self.vfe2(voxel_features, voxel_num_points.unsqueeze(-1))
        
        # Max pooling to get single feature per voxel
        voxel_features = F.max_pool2d(
            voxel_features.permute(0, 3, 1, 2),
            kernel_size=[1, voxel_features.shape[2]],
            stride=[1, voxel_features.shape[2]]
        ).squeeze(3).permute(0, 2, 1)
        
        # Apply final linear and batch norm
        voxel_features = self.linear(voxel_features)
        voxel_features = self.norm(voxel_features.transpose(1, 2)).transpose(1, 2)
        
        # Update batch_dict
        batch_dict['voxel_features'] = voxel_features
        
        return batch_dict
    
    def _get_voxel_centers(self, voxel_indices: torch.Tensor, batch_size: int, num_voxels: int) -> torch.Tensor:
        """
        Get the center coordinates of voxels.
        
        Args:
            voxel_indices: Voxel indices tensor of shape (B, num_voxels, 3) [z, y, x]
            batch_size: Batch size
            num_voxels: Number of voxels per batch item
            
        Returns:
            Center coordinates of voxels of shape (B, num_voxels, 3) [x, y, z]
        """
        # Convert indices to float and reorder to [x, y, z]
        indices = voxel_indices.float()
        indices = indices[:, :, [2, 1, 0]]  # [x, y, z]
        
        # Calculate voxel centers by adding 0.5 to indices and multiplying by voxel size
        voxel_centers = (indices + 0.5) * torch.tensor(
            self.voxel_size, device=voxel_indices.device
        )
        
        # Add point cloud range offset
        voxel_centers += torch.tensor(
            self.point_cloud_range[:3], device=voxel_indices.device
        )
        
        return voxel_centers


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network for PointPillars-style feature extraction.
    
    This network divides the point cloud into pillars (vertical columns)
    and extracts features from each pillar.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # x, y, z, intensity
        feature_dim: int = 64,
        max_points_per_pillar: int = 100,
        max_pillars: int = 10000,
        voxel_size: List[float] = [0.16, 0.16, 4],
        point_cloud_range: List[float] = [0, -39.68, -3, 100, 39.68, 1],
        use_height: bool = True
    ):
        """
        Initialize the Pillar Feature Network.
        
        Args:
            in_channels: Number of input channels per point (x, y, z, intensity, ...)
            feature_dim: Dimension of output features
            max_points_per_pillar: Maximum number of points per pillar
            max_pillars: Maximum number of pillars
            voxel_size: Size of voxels in [x, y, z]
            point_cloud_range: Range of point cloud in [x_min, y_min, z_min, x_max, y_max, z_max]
            use_height: Whether to use height as an additional feature
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_height = use_height
        
        # Calculate grid size
        grid_size = (
            np.round((np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])) / 
                     np.array(voxel_size)).astype(np.int32)
        )
        self.grid_size = grid_size
        
        # Number of channels for pillar features
        # x, y, z, intensity, x_center, y_center, z_center, x_offset, y_offset, z_offset
        num_input_features = in_channels + 6
        
        # Create PFN layers
        self.pfn_layers = nn.ModuleList()
        
        # First PFN layer
        self.pfn_layers.append(
            PFNLayer(
                in_channels=num_input_features,
                out_channels=feature_dim,
                use_norm=True,
                last_layer=False
            )
        )
        
        # Second PFN layer
        self.pfn_layers.append(
            PFNLayer(
                in_channels=feature_dim,
                out_channels=feature_dim,
                use_norm=True,
                last_layer=True
            )
        )
    
    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass of the Pillar Feature Network.
        
        Args:
            batch_dict: Dictionary containing:
                - 'voxels': (B, max_pillars, max_points_per_pillar, in_channels)
                - 'voxel_coords': (B, max_pillars, 4) [batch_idx, z, y, x]
                - 'voxel_num_points': (B, max_pillars)
            
        Returns:
            Updated batch_dict with:
                - 'pillar_features': (B, max_pillars, feature_dim)
                - 'voxel_coords': (B, max_pillars, 4) [batch_idx, z, y, x]
        """
        voxels = batch_dict['voxels']
        voxel_coords = batch_dict['voxel_coords']
        voxel_num_points = batch_dict['voxel_num_points']
        
        # Reshape for batch processing
        batch_size = voxels.shape[0]
        num_pillars = voxels.shape[1]
        
        # Extract point coordinates and features
        points_xyz = voxels[:, :, :, :3]
        points_features = voxels[:, :, :, 3:]
        
        # Calculate pillar centers
        pillar_centers = self._get_pillar_centers(
            voxel_coords[:, :, [3, 2]],  # x, y
            batch_size,
            num_pillars
        )
        
        # Calculate points offset from pillar centers
        points_offset = points_xyz[:, :, :, :2] - pillar_centers.unsqueeze(2)
        
        # Combine features
        # [x, y, z, intensity, x_offset, y_offset, x_center, y_center, z_center]
        pillar_features_list = [points_xyz, points_features, points_offset]
        
        # Add pillar center as feature
        pillar_centers_expanded = pillar_centers.unsqueeze(2).expand(-1, -1, self.max_points_per_pillar, -1)
        pillar_z_centers = points_xyz[:, :, :, 2:3].mean(dim=2, keepdim=True).expand(-1, -1, self.max_points_per_pillar, -1)
        pillar_centers_expanded = torch.cat([pillar_centers_expanded, pillar_z_centers], dim=-1)
        pillar_features_list.append(pillar_centers_expanded)
        
        # Concatenate all features
        pillar_features = torch.cat(pillar_features_list, dim=-1)
        
        # Create a mask for valid points
        mask = torch.arange(self.max_points_per_pillar, device=voxels.device).reshape(
            1, 1, -1, 1) < voxel_num_points.unsqueeze(-1).unsqueeze(-1)
        mask = mask.expand(-1, -1, -1, pillar_features.shape[-1])
        
        # Apply mask
        pillar_features = pillar_features * mask.float()
        
        # Forward through PFN layers
        for pfn in self.pfn_layers:
            pillar_features = pfn(pillar_features)
        
        # Update batch_dict
        batch_dict['pillar_features'] = pillar_features
        
        return batch_dict
    
    def _get_pillar_centers(self, pillar_indices: torch.Tensor, batch_size: int, num_pillars: int) -> torch.Tensor:
        """
        Get the center coordinates of pillars.
        
        Args:
            pillar_indices: Pillar indices tensor of shape (B, num_pillars, 2) [x, y]
            batch_size: Batch size
            num_pillars: Number of pillars per batch item
            
        Returns:
            Center coordinates of pillars of shape (B, num_pillars, 2) [x, y]
        """
        # Convert indices to float
        indices = pillar_indices.float()
        
        # Calculate pillar centers by adding 0.5 to indices and multiplying by voxel size
        pillar_centers = (indices + 0.5) * torch.tensor(
            self.voxel_size[:2], device=pillar_indices.device
        )
        
        # Add point cloud range offset
        pillar_centers += torch.tensor(
            self.point_cloud_range[:2], device=pillar_indices.device
        )
        
        return pillar_centers


class PointPillarScatter(nn.Module):
    """
    Point Pillar Scatter for converting pillar features to a pseudo image.
    
    This module takes pillar features and their coordinates and scatters them
    into a 2D pseudo image for further processing by a 2D CNN.
    """
    
    def __init__(
        self, 
        feature_dim: int,
        grid_size: List[int]
    ):
        """
        Initialize the Point Pillar Scatter.
        
        Args:
            feature_dim: Number of features per pillar
            grid_size: Size of the grid [x_size, y_size, z_size]
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.nx = grid_size[0]
        self.ny = grid_size[1]
    
    def forward(self, pillar_features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Point Pillar Scatter.
        
        Args:
            pillar_features: Pillar features of shape (B, num_pillars, feature_dim)
            coords: Pillar coordinates of shape (B, num_pillars, 4) [batch_idx, z, y, x]
            
        Returns:
            Pseudo image of shape (B, feature_dim, ny, nx)
        """
        batch_size = coords[:, 0, 0].max().int().item() + 1
        
        # Create canvas for each batch item
        batch_canvas = []
        for batch_idx in range(batch_size):
            # Create empty canvas
            canvas = torch.zeros(
                self.feature_dim,
                self.ny,
                self.nx,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            
            # Get batch mask
            batch_mask = coords[:, :, 0] == batch_idx
            
            # Get this batch's pillars and coordinates
            this_coords = coords[batch_mask, :][0]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :][0]
            
            # Scatter pillars to canvas
            canvas[:, this_coords[:, 2].type(torch.long), this_coords[:, 3].type(torch.long)] = pillars.t()
            
            batch_canvas.append(canvas)
        
        # Stack canvases into a batch
        batch_canvas = torch.stack(batch_canvas, dim=0)
        
        return batch_canvas


def build_voxel_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Build a voxel feature encoder based on configuration.
    
    Args:
        config: Configuration dictionary for the voxel encoder
        
    Returns:
        Voxel feature encoder module
    """
    encoder_type = config.get('type', 'vfe')
    
    if encoder_type == 'vfe':
        return VoxelFeatureEncoder(
            in_channels=config.get('in_channels', 4),
            feature_dim=config.get('feature_dim', 64),
            max_points_per_voxel=config.get('max_points_per_voxel', 100),
            voxel_size=config.get('voxel_size', [0.16, 0.16, 4]),
            point_cloud_range=config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1]),
            use_height=config.get('use_height', True)
        )
    elif encoder_type == 'pillar':
        return PillarFeatureNet(
            in_channels=config.get('in_channels', 4),
            feature_dim=config.get('feature_dim', 64),
            max_points_per_pillar=config.get('max_points_per_pillar', 100),
            max_pillars=config.get('max_pillars', 10000),
            voxel_size=config.get('voxel_size', [0.16, 0.16, 4]),
            point_cloud_range=config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1]),
            use_height=config.get('use_height', True)
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create a dummy batch dictionary
    batch_size = 2
    max_voxels = 1000
    max_points = 32
    in_channels = 4
    
    # Create random voxels
    voxels = torch.rand(batch_size, max_voxels, max_points, in_channels)
    
    # Create random coordinates (batch_idx, z, y, x)
    coords = torch.zeros(batch_size, max_voxels, 4)
    coords[:, :, 0] = torch.arange(batch_size).view(-1, 1).repeat(1, max_voxels)
    coords[:, :, 1] = torch.randint(0, 10, (batch_size, max_voxels))  # z
    coords[:, :, 2] = torch.randint(0, 100, (batch_size, max_voxels))  # y
    coords[:, :, 3] = torch.randint(0, 100, (batch_size, max_voxels))  # x
    
    # Create random number of points
    num_points = torch.randint(1, max_points, (batch_size, max_voxels))
    
    # Create batch dictionary
    batch_dict = {
        'voxels': voxels,
        'voxel_coords': coords,
        'voxel_num_points': num_points
    }
    
    # Create voxel feature encoder
    vfe = VoxelFeatureEncoder(
        in_channels=in_channels,
        feature_dim=64,
        max_points_per_voxel=max_points,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 100, 39.68, 1],
        use_height=True
    )
    
    # Forward pass
    output_dict = vfe(batch_dict)
    
    # Print output shape
    print(f"Input voxels shape: {voxels.shape}")
    print(f"Output voxel features shape: {output_dict['voxel_features'].shape}")
    
    # Create pillar feature encoder
    pfn = PillarFeatureNet(
        in_channels=in_channels,
        feature_dim=64,
        max_points_per_pillar=max_points,
        max_pillars=max_voxels,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 100, 39.68, 1],
        use_height=True
    )
    
    # Forward pass
    pillar_dict = pfn(batch_dict)
    
    # Print output shape
    print(f"Output pillar features shape: {pillar_dict['pillar_features'].shape}")
    
    # Create point pillar scatter
    scatter = PointPillarScatter(
        feature_dim=64,
        grid_size=[200, 200, 1]  # Assuming a 200x200 grid
    )
    
    # Forward pass
    pseudo_image = scatter(pillar_dict['pillar_features'], coords)
    
    # Print output shape
    print(f"Output pseudo image shape: {pseudo_image.shape}")