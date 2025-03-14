"""
center_point.py module for V2X-Seq project.

This module implements the CenterPoint detector for 3D object detection from point clouds.
CenterPoint is a center-based 3D object detection and tracking framework that first detects
centers of objects using a keypoint detector and then regresses to other attributes.

Reference:
Yin, Tianwei, Xingyi Zhou, and Philipp Krähenbühl. "Center-based 3D Object Detection and Tracking."
CVPR 2021. https://arxiv.org/abs/2006.11275
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union


class VoxelFeatureExtractor(nn.Module):
    """
    Voxel feature extractor that converts point cloud to voxel features.
    This is the first stage of CenterPoint.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the voxel feature extractor.
        
        Args:
            config: Configuration dictionary containing model parameters
                - voxel_size: Size of voxels in [x, y, z]
                - point_cloud_range: Range of point cloud in [x_min, y_min, z_min, x_max, y_max, z_max]
                - max_points_per_voxel: Maximum number of points per voxel
                - max_voxels: Maximum number of voxels
                - num_features: Number of features per point (default: 4 for x, y, z, intensity)
                - vfe_filters: List of filter dimensions for VFE layers
        """
        super().__init__()
        self.voxel_size = config.get('voxel_size', [0.1, 0.1, 0.15])
        self.point_cloud_range = config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1])
        self.max_points = config.get('max_points_per_voxel', 100)
        self.max_voxels = config.get('max_voxels', 40000)
        self.num_features = config.get('num_features', 4)
        
        # Calculate grid size based on point cloud range and voxel size
        grid_size = (
            (np.array(self.point_cloud_range[3:]) - np.array(self.point_cloud_range[:3])) / 
            np.array(self.voxel_size)
        ).astype(np.int64)
        self.grid_size = grid_size
        
        # Create VFE layers
        vfe_filters = config.get('vfe_filters', [64, 64])
        self.vfe_layers = nn.ModuleList()
        
        # First VFE layer
        self.vfe_layers.append(
            VoxelFeatureEncodingLayer(
                in_channels=self.num_features,
                out_channels=vfe_filters[0],
                max_points=self.max_points
            )
        )
        
        # Additional VFE layers
        for i in range(1, len(vfe_filters)):
            self.vfe_layers.append(
                VoxelFeatureEncodingLayer(
                    in_channels=vfe_filters[i-1],
                    out_channels=vfe_filters[i],
                    max_points=self.max_points
                )
            )
        
        # Output dimension
        self.num_vfe_features = vfe_filters[-1]
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of voxel feature extractor.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'points': List of (N, 4+) tensors of point clouds
                
        Returns:
            Updated batch_dict with:
                - 'voxel_features': (M, C) tensor of voxel features
                - 'voxel_coords': (M, 4) tensor of voxel coordinates (batch_idx, z, y, x)
                - 'voxel_num_points': (M,) tensor of number of points in each voxel
        """
        # Get points from batch
        points = batch_dict['points']
        batch_size = len(points)
        
        # Initialize outputs
        voxel_features_list = []
        voxel_coords_list = []
        voxel_num_points_list = []
        
        # Process each batch item
        for batch_idx, point_cloud in enumerate(points):
            # Skip empty point clouds
            if point_cloud.shape[0] == 0:
                continue
                
            # Voxelize point cloud
            voxels, coords, num_points = self.voxelize_points(
                points=point_cloud, 
                batch_idx=batch_idx
            )
            
            # Skip if no valid voxels
            if voxels is None:
                continue
                
            # Add to lists
            voxel_features_list.append(voxels)
            voxel_coords_list.append(coords)
            voxel_num_points_list.append(num_points)
        
        # Combine voxels from all batches
        if voxel_features_list:
            voxel_features = torch.cat(voxel_features_list, dim=0)
            voxel_coords = torch.cat(voxel_coords_list, dim=0)
            voxel_num_points = torch.cat(voxel_num_points_list, dim=0)
            
            # Apply VFE layers
            for vfe in self.vfe_layers:
                voxel_features = vfe(voxel_features, voxel_num_points)
            
            # Update batch_dict
            batch_dict.update({
                'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points
            })
        
        return batch_dict
    
    def voxelize_points(self, points: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert point cloud to voxels.
        
        Args:
            points: (N, 4+) tensor of points [x, y, z, intensity, ...]
            batch_idx: Batch index
            
        Returns:
            Tuple of:
                - voxel_features: (M, max_points, C) tensor of voxel features
                - voxel_coords: (M, 4) tensor of voxel coordinates (batch_idx, z, y, x)
                - voxel_num_points: (M,) tensor of number of points in each voxel
        """
        # Filter points outside the range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]
        
        # Return None if no points are in range
        if points.shape[0] == 0:
            return None, None, None
        
        # Calculate voxel coordinates
        voxel_coords = ((points[:, :3] - torch.tensor(self.point_cloud_range[:3], device=points.device)) / 
                       torch.tensor(self.voxel_size, device=points.device))
        voxel_coords = voxel_coords.int()
        
        # Create unique voxel coordinates
        voxel_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        
        # Limit to maximum number of voxels
        if voxel_coords.shape[0] > self.max_voxels:
            # Select random voxels
            indices = torch.randperm(voxel_coords.shape[0])[:self.max_voxels]
            voxel_coords = voxel_coords[indices]
            
            # Update mask for points
            mask = torch.zeros(inverse_indices.shape[0], dtype=torch.bool, device=points.device)
            for idx in indices:
                mask |= (inverse_indices == idx)
            
            # Filter points
            points = points[mask]
            inverse_indices = torch.zeros_like(mask.long())
            for i, idx in enumerate(indices):
                inverse_indices[mask & (inverse_indices == idx)] = i
        
        # Count points in each voxel
        num_voxels = voxel_coords.shape[0]
        point_to_voxel = inverse_indices
        
        # Create voxel features array
        voxel_features = torch.zeros(
            (num_voxels, self.max_points, points.shape[1]),
            dtype=points.dtype,
            device=points.device
        )
        
        # Create counter for points per voxel
        voxel_num_points = torch.zeros(
            num_voxels,
            dtype=torch.int,
            device=points.device
        )
        
        # Fill voxel features
        for i in range(points.shape[0]):
            voxel_idx = point_to_voxel[i]
            point_idx = voxel_num_points[voxel_idx]
            
            if point_idx < self.max_points:
                voxel_features[voxel_idx, point_idx] = points[i]
                voxel_num_points[voxel_idx] += 1
        
        # Add batch index to voxel coordinates
        batch_indices = torch.full(
            (voxel_coords.shape[0], 1),
            batch_idx,
            dtype=torch.int,
            device=points.device
        )
        voxel_coords = torch.cat([batch_indices, voxel_coords[:, [2, 1, 0]]], dim=1)
        
        return voxel_features, voxel_coords, voxel_num_points


class VoxelFeatureEncodingLayer(nn.Module):
    """
    Voxel Feature Encoding Layer for processing points in a voxel.
    """
    
    def __init__(self, in_channels: int, out_channels: int, max_points: int):
        """
        Initialize the VFE layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            max_points: Maximum number of points per voxel
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_points = max_points
        
        # Linear layer
        self.linear = nn.Linear(in_channels, out_channels)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, voxel_features: torch.Tensor, voxel_num_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VFE layer.
        
        Args:
            voxel_features: (B*N, P, C_in) tensor of voxel features
            voxel_num_points: (B*N,) tensor of number of points in each voxel
            
        Returns:
            (B*N, C_out) tensor of voxel features
        """
        # Check for empty voxels
        if voxel_features.shape[0] == 0:
            return voxel_features.new_zeros((0, self.out_channels))
        
        # Extract valid features only (for efficiency)
        mask = torch.arange(self.max_points, device=voxel_features.device).unsqueeze(0) < voxel_num_points.unsqueeze(1)
        valid_features = voxel_features[mask]
        
        # Apply linear layer
        features = self.linear(valid_features)
        
        # Apply batch normalization
        features = features.contiguous().view(-1, self.out_channels)
        features = self.bn(features)
        features = F.relu(features)
        
        # Scatter back to original shape
        voxel_count = mask.sum(dim=1)
        out_features = torch.zeros(voxel_features.shape[0], self.out_channels, device=voxel_features.device)
        
        # Compute mean feature for each voxel
        start = 0
        for i in range(voxel_features.shape[0]):
            count = voxel_count[i]
            if count > 0:
                end = start + count
                out_features[i] = features[start:end].mean(dim=0)
                start = end
        
        return out_features


class SparseBackbone3D(nn.Module):
    """
    Sparse 3D backbone for processing voxel features.
    This is the second stage of CenterPoint.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the sparse 3D backbone.
        
        Args:
            config: Configuration dictionary containing model parameters
                - input_channels: Number of input channels
                - conv_filters: List of filter dimensions for convolutional layers
                - norm_layer: Type of normalization (batch_norm, instance_norm)
        """
        super().__init__()
        input_channels = config.get('vfe_filters', [64, 64])[-1]
        conv_filters = config.get('conv_filters', [32, 64, 128])
        
        # Create sparse encoder layers
        self.sparse_layers = nn.ModuleList()
        
        # First sparse layer
        self.sparse_layers.append(
            SparseConv3D(
                in_channels=input_channels,
                out_channels=conv_filters[0],
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key="spconv1"
            )
        )
        
        # Additional sparse layers with downsampling
        for i in range(1, len(conv_filters)):
            self.sparse_layers.append(
                SparseConv3D(
                    in_channels=conv_filters[i-1],
                    out_channels=conv_filters[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=f"spconv{i+1}"
                )
            )
        
        # Output dimension
        self.num_features_out = conv_filters[-1]
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of sparse 3D backbone.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'voxel_features': (M, C) tensor of voxel features
                - 'voxel_coords': (M, 4) tensor of voxel coordinates (batch_idx, z, y, x)
                
        Returns:
            Updated batch_dict with:
                - 'encoded_spconv_tensor': Sparse tensor from backbone
                - 'encoded_spconv_features': Dense tensor of backbone features
        """
        # Create sparse input tensor
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Skip if no valid voxels
        if voxel_features.shape[0] == 0:
            # Return empty feature map
            return batch_dict
        
        # Create sparse tensor (placeholders for actual sparse tensor implementation)
        # In a real implementation, this would use a library like spconv or MinkowskiEngine
        sparse_features = {
            'features': voxel_features,
            'coordinates': voxel_coords,
            'batch_size': batch_size
        }
        
        # Apply sparse layers
        for layer in self.sparse_layers:
            sparse_features = layer(sparse_features)
        
        # Extract features
        batch_dict['encoded_spconv_tensor'] = sparse_features
        batch_dict['encoded_spconv_features'] = sparse_features['features']
        
        return batch_dict


class SparseConv3D(nn.Module):
    """
    Sparse 3D convolution layer.
    This is a simplified placeholder for actual sparse convolution implementation.
    In a real implementation, this would use spconv or MinkowskiEngine.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                stride: int = 1, padding: int = 0, indice_key: str = None):
        """
        Initialize sparse 3D convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            padding: Padding size
            indice_key: Key for sparse convolution indices
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.indice_key = indice_key
        
        # Create regular convolution and batchnorm layers
        # In real implementation, these would be replaced with sparse variants
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, sparse_tensor: Dict) -> Dict:
        """
        Forward pass for sparse convolution.
        This is a simplified placeholder implementation.
        
        Args:
            sparse_tensor: Dictionary representing sparse tensor with:
                - 'features': Dense features tensor
                - 'coordinates': Coordinates tensor
                - 'batch_size': Batch size
                
        Returns:
            Updated sparse tensor
        """
        # In a real implementation, this would use actual sparse convolution
        features = sparse_tensor['features']
        coordinates = sparse_tensor['coordinates']
        batch_size = sparse_tensor['batch_size']
        
        # Simple dense approximation (this is just a placeholder)
        # In real implementation, this would perform sparse convolution
        if self.stride > 1:
            # Downsample coordinates (simplified)
            coordinates = torch.floor(coordinates.float() / self.stride).int()
            
            # Create unique coordinates
            coordinates, inverse_indices = torch.unique(coordinates, dim=0, return_inverse=True)
            
            # Aggregate features by coordinates
            new_features = torch.zeros(
                (coordinates.shape[0], self.out_channels),
                device=features.device,
                dtype=features.dtype
            )
            
            # Simple feature pooling (very simplified)
            for i in range(features.shape[0]):
                new_idx = inverse_indices[i]
                new_features[new_idx] += features[i]
            
            # Apply ReLU
            new_features = F.relu(new_features)
            
            return {
                'features': new_features,
                'coordinates': coordinates,
                'batch_size': batch_size
            }
        else:
            # No downsampling - just transform features
            new_features = F.relu(features)
            
            # Conv 1x1 approximation
            new_features = torch.mm(
                new_features, 
                torch.randn(self.in_channels, self.out_channels, device=features.device)
            )
            
            return {
                'features': new_features,
                'coordinates': coordinates,
                'batch_size': batch_size
            }


class BEVFeatureExtractor(nn.Module):
    """
    Bird's Eye View feature extractor.
    This converts 3D sparse features to 2D BEV features.
    This is the third stage of CenterPoint.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the BEV feature extractor.
        
        Args:
            config: Configuration dictionary containing model parameters
                - bev_height: Height of BEV feature map
                - bev_width: Width of BEV feature map
                - num_channels: Number of channels in BEV feature map
                - use_height: Whether to use height information
        """
        super().__init__()
        # Get configuration
        self.use_height = config.get('use_height', True)
        self.num_bev_features = config.get('num_bev_features', 256)
        self.bev_height = config.get('bev_height', 400)
        self.bev_width = config.get('bev_width', 400)
        
        # Create BEV projection layers
        # In real implementation, this would be more sophisticated
        input_channels = config.get('conv_filters', [32, 64, 128])[-1]
        if self.use_height:
            input_channels += 1  # Add channel for height encoding
        
        # BEV projection network
        self.bev_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.num_bev_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU()
        )
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of BEV feature extractor.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'encoded_spconv_tensor': Sparse tensor from backbone
                - 'encoded_spconv_features': Dense tensor of backbone features
                
        Returns:
            Updated batch_dict with:
                - 'spatial_features': BEV feature map
        """
        # Extract sparse features
        sparse_tensor = batch_dict['encoded_spconv_tensor']
        features = sparse_tensor['features']
        coordinates = sparse_tensor['coordinates']
        batch_size = sparse_tensor['batch_size']
        
        if features.shape[0] == 0:
            # Handle empty case
            spatial_features = torch.zeros(
                (batch_size, self.num_bev_features, self.bev_height, self.bev_width),
                device=features.device
            )
            batch_dict['spatial_features'] = spatial_features
            return batch_dict
        
        # Get batch indices
        batch_indices = coordinates[:, 0]
        
        # Get height indices
        height_indices = coordinates[:, 1]
        
        # Get spatial indices
        spatial_indices = coordinates[:, 2:4]  # (y, x)
        
        # Create BEV grid indices
        bev_indices = torch.zeros_like(spatial_indices)
        bev_indices[:, 0] = spatial_indices[:, 0]  # y
        bev_indices[:, 1] = spatial_indices[:, 1]  # x
        
        # Create BEV feature tensor
        bev_features = torch.zeros(
            (batch_size, self.bev_height, self.bev_width, features.shape[1]),
            dtype=features.dtype,
            device=features.device
        )
        
        # Fill BEV grid with features
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            
            if batch_mask.sum() > 0:
                # Get spatial indices and features for this batch
                this_coords = bev_indices[batch_mask]
                this_features = features[batch_mask]
                
                # Create feature grid (simplified)
                # In real implementation, would handle overlapping indices properly
                valid_y = (this_coords[:, 0] >= 0) & (this_coords[:, 0] < self.bev_height)
                valid_x = (this_coords[:, 1] >= 0) & (this_coords[:, 1] < self.bev_width)
                valid_mask = valid_y & valid_x
                
                if valid_mask.sum() > 0:
                    valid_coords = this_coords[valid_mask]
                    valid_features = this_features[valid_mask]
                    
                    y_indices = valid_coords[:, 0].long()
                    x_indices = valid_coords[:, 1].long()
                    
                    # Assign features to BEV grid
                    # In case of overlapping indices, use max pooling
                    for i in range(len(y_indices)):
                        y, x = y_indices[i], x_indices[i]
                        curr_feat = valid_features[i]
                        existing_feat = bev_features[b, y, x]
                        
                        if self.use_height:
                            # Add encoded height as additional feature (simplified)
                            h = height_indices[batch_mask][valid_mask][i].float() / 10.0  # Normalize height
                            height_encoding = torch.tensor([h], device=features.device)
                            curr_feat = torch.cat([curr_feat, height_encoding])
                        
                        # Max pooling for overlapping voxels
                        if (existing_feat == 0).all():
                            bev_features[b, y, x] = curr_feat
                        else:
                            bev_features[b, y, x] = torch.max(existing_feat, curr_feat)
        
        # Convert to (B, C, H, W) format
        bev_features = bev_features.permute(0, 3, 1, 2).contiguous()
        
        # Apply BEV projection
        bev_features = self.bev_projection(bev_features)
        
        # Update batch_dict
        batch_dict['spatial_features'] = bev_features
        
        return batch_dict


class HeadNetwork(nn.Module):
    """
    Head network for CenterPoint detector.
    This generates center heatmap, box size, 3D offset, etc.
    This is the fourth and final stage of CenterPoint.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the head network.
        
        Args:
            config: Configuration dictionary containing model parameters
                - num_class: Number of object classes
                - num_bev_features: Number of BEV features
                - use_dcn: Whether to use deformable convolution
                - target_assigner_config: Configuration for target assignment
        """
        super().__init__()
        num_classes = config.get('num_classes', 4)
        num_bev_features = config.get('num_bev_features', 256)
        
        # Create heads
        self.class_head = self._make_head(num_bev_features, num_classes)
        self.offset_head = self._make_head(num_bev_features, 2)  # x, y offset
        self.z_head = self._make_head(num_bev_features, 1)  # z center
        self.size_head = self._make_head(num_bev_features, 3)  # w, l, h
        self.yaw_head = self._make_head(num_bev_features, 2)  # sin, cos
        self.vel_head = self._make_head(num_bev_features, 2)  # vx, vy
        
        # Initialize parameters to improve training stability
        self._init_weights()
        
        # Module to generate targets
        self.target_assigner = TargetAssigner(config.get('target_assigner_config', {}))
    
    def _make_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a head module for a specific task.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Head module
        """
        # Simple 2D convolution head
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        # Initialize class head with small weights
        for m in self.class_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of head network.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'spatial_features': BEV feature map
                
        Returns:
            Updated batch_dict with:
                - 'heatmap': Class center heatmap
                - 'offset': Center offset prediction
                - 'height': Height prediction
                - 'size': Size prediction
                - 'angle': Orientation prediction
                - 'velocity': Velocity prediction
        """
        # Extract BEV features
        spatial_features = batch_dict['spatial_features']
        
        # Apply heads
        heatmap = self.class_head(spatial_features)
        offset = self.offset_head(spatial_features)
        z = self.z_head(spatial_features)
        size = self.size_head(spatial_features)
        yaw = self.yaw_head(spatial_features)
        vel = self.vel_head(spatial_features)
        
        # Update batch_dict with predictions
        batch_dict.update({
            'heatmap': heatmap,
            'offset': offset,
            'height': z,
            'size': size,
            'angle': yaw,
            'velocity': vel
        })
        
        # Generate targets during training
        if self.training and 'gt_boxes' in batch_dict:
            targets = self.target_assigner(batch_dict)
            batch_dict.update(targets)
        
        # Generate detections during testing
        if not self.training:
            detections = self.generate_predicted_boxes(batch_dict)
            batch_dict['detections'] = detections
        
        return batch_dict
    
    def generate_predicted_boxes(self, batch_dict: Dict) -> List[Dict]:
        """
        Generate predicted 3D boxes from network outputs.
        
        Args:
            batch_dict: Input dictionary containing network outputs
                
        Returns:
            List of dictionaries with predicted boxes
        """
        # Apply sigmoid to heatmap for class probabilities
        batch_heatmap = torch.sigmoid(batch_dict['heatmap'])
        batch_offset = batch_dict['offset']
        batch_z = batch_dict['height']
        batch_size = batch_dict['size']
        batch_yaw = batch_dict['angle']
        batch_vel = batch_dict['velocity']
        
        # Get grid size and voxel size
        grid_size = torch.tensor(batch_dict.get('grid_size', [400, 400]), 
                               device=batch_heatmap.device)
        voxel_size = torch.tensor(batch_dict.get('voxel_size', [0.1, 0.1, 0.15]), 
                                device=batch_heatmap.device)
        point_cloud_range = torch.tensor(batch_dict.get('point_cloud_range', 
                                                      [0, -39.68, -3, 100, 39.68, 1]),
                                      device=batch_heatmap.device)
        
        batch_size = batch_heatmap.shape[0]
        batch_detections = []
        
        # Process each sample in batch
        for batch_idx in range(batch_size):
            # Get outputs for this sample
            heatmap = batch_heatmap[batch_idx]
            offset = batch_offset[batch_idx]
            z_pred = batch_z[batch_idx]
            size_pred = batch_size[batch_idx]
            yaw_pred = batch_yaw[batch_idx]
            vel_pred = batch_vel[batch_idx]
            
            # Detect peaks in heatmap (simple non-max suppression)
            # In a real implementation, use more sophisticated peak detection
            heatmap_np = heatmap.detach().cpu().numpy()
            peaks = self._simple_peak_detection(heatmap_np, threshold=0.3, window_size=3)
            
            # Prepare detection array
            detections = []
            
            # Process each peak (potential object center)
            for class_id, peak_indices in enumerate(peaks):
                for peak_idx in peak_indices:
                    # Convert peak index to grid coordinates
                    y_idx, x_idx = peak_idx
                    
                    # Get predictions at peak location
                    score = heatmap[class_id, y_idx, x_idx].item()
                    
                    # Get offset predictions
                    offset_x = offset[0, y_idx, x_idx].item()
                    offset_y = offset[1, y_idx, x_idx].item()
                    
                    # Get z prediction
                    z = z_pred[0, y_idx, x_idx].item()
                    
                    # Get size predictions
                    width = size_pred[0, y_idx, x_idx].item()
                    length = size_pred[1, y_idx, x_idx].item()
                    height = size_pred[2, y_idx, x_idx].item()
                    
                    # Get orientation prediction
                    sin_yaw = yaw_pred[0, y_idx, x_idx].item()
                    cos_yaw = yaw_pred[1, y_idx, x_idx].item()
                    yaw = np.arctan2(sin_yaw, cos_yaw)
                    
                    # Get velocity prediction
                    vx = vel_pred[0, y_idx, x_idx].item()
                    vy = vel_pred[1, y_idx, x_idx].item()
                    
                    # Convert center from grid coordinates to world coordinates
                    # Grid coordinates are centered at cell centers
                    center_x = (x_idx + offset_x) * voxel_size[0] + point_cloud_range[0]
                    center_y = (y_idx + offset_y) * voxel_size[1] + point_cloud_range[1]
                    
                    # Create detection object
                    detection = {
                        'score': score,
                        'label': class_id,
                        'box': [center_x, center_y, z, width, length, height, yaw],
                        'velocity': [vx, vy]
                    }
                    
                    detections.append(detection)
            
            # Sort detections by score and keep top K
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            detections = detections[:100]  # Keep top 100 detections
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def _simple_peak_detection(self, heatmap: np.ndarray, threshold: float = 0.3, 
                             window_size: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Simple peak detection in heatmap.
        
        Args:
            heatmap: Heatmap array of shape (num_classes, H, W)
            threshold: Detection threshold
            window_size: Size of NMS window
            
        Returns:
            List of lists containing peak indices for each class
        """
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure, binary_erosion
        
        num_classes, height, width = heatmap.shape
        all_peaks = []
        
        for c in range(num_classes):
            class_heatmap = heatmap[c]
            
            # Apply threshold
            binary_heatmap = (class_heatmap > threshold)
            
            # Define connectivity structure
            struct = generate_binary_structure(2, 2)
            
            # Find local maxima
            local_max = maximum_filter(class_heatmap, footprint=struct) == class_heatmap
            
            # Remove background from local maxima
            local_max &= binary_heatmap
            
            # Get peak coordinates
            peak_coords = np.argwhere(local_max)
            
            # Convert to list of (y, x) tuples
            peaks = [(y, x) for y, x in peak_coords]
            
            all_peaks.append(peaks)
        
        return all_peaks


class TargetAssigner(nn.Module):
    """
    Target assignment module for CenterPoint.
    Converts ground truth boxes to heatmap, offset, size, etc. targets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize target assigner.
        
        Args:
            config: Configuration dictionary for target assignment
                - gaussian_overlap: Gaussian overlap for heatmap generation
                - min_radius: Minimum radius for heatmap generation
                - target_assigner_config: Configuration for target assignment
        """
        super().__init__()
        self.gaussian_overlap = config.get('gaussian_overlap', 0.1)
        self.min_radius = config.get('min_radius', 2)
        self.target_assigner_config = config.get('target_assigner_config', {})
        
        # Set class-specific size ranges if provided
        self.class_size_ranges = config.get('class_size_ranges', None)
        
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Generate targets from ground truth.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'gt_boxes': Ground truth 3D boxes (B, N, 7+C)
                - 'spatial_features': BEV feature map to get output shape
                
        Returns:
            Dictionary of targets:
                - 'heatmap_targets': Heatmap targets
                - 'offset_targets': Offset targets
                - 'size_targets': Size targets
                - 'mask_targets': Mask for valid targets
                - 'z_targets': Height targets
                - 'yaw_targets': Orientation targets
                - 'vel_targets': Velocity targets
        """
        # Extract ground truth boxes
        gt_boxes = batch_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]
        
        # Get BEV feature map shape
        spatial_features = batch_dict['spatial_features']
        feature_shape = spatial_features.shape[-2:]  # (H, W)
        
        # Get grid size and voxel size
        grid_size = batch_dict.get('grid_size', [400, 400])
        voxel_size = batch_dict.get('voxel_size', [0.1, 0.1, 0.15])
        point_cloud_range = batch_dict.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1])
        
        # Initialize targets
        heatmap_targets = torch.zeros((batch_size, self.target_assigner_config.get('num_classes', 4),
                                      feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        offset_targets = torch.zeros((batch_size, 2, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        z_targets = torch.zeros((batch_size, 1, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        size_targets = torch.zeros((batch_size, 3, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        yaw_targets = torch.zeros((batch_size, 2, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        vel_targets = torch.zeros((batch_size, 2, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        mask_targets = torch.zeros((batch_size, 1, feature_shape[0], feature_shape[1]), device=gt_boxes.device)
        
        # Generate targets for each batch
        for batch_idx in range(batch_size):
            # Get ground truth boxes for this batch
            gt_boxes_batch = gt_boxes[batch_idx]
            
            # Remove empty boxes
            valid_mask = gt_boxes_batch[:, 0:3].abs().sum(dim=1) > 0
            gt_boxes_batch = gt_boxes_batch[valid_mask]
            
            # Skip if no valid boxes
            if gt_boxes_batch.shape[0] == 0:
                continue
            
            # Extract parameters from ground truth boxes
            center_x = gt_boxes_batch[:, 0]
            center_y = gt_boxes_batch[:, 1]
            center_z = gt_boxes_batch[:, 2]
            width = gt_boxes_batch[:, 3]
            length = gt_boxes_batch[:, 4]
            height = gt_boxes_batch[:, 5]
            yaw = gt_boxes_batch[:, 6]
            class_ids = gt_boxes_batch[:, 7].long()
            
            # Extract velocity if available (in extended boxes)
            velocity = torch.zeros((gt_boxes_batch.shape[0], 2), device=gt_boxes_batch.device)
            if gt_boxes_batch.shape[1] > 8:
                velocity = gt_boxes_batch[:, 8:10]
            
            # Convert center from world coordinates to grid coordinates
            # Grid coordinates are centered at cell centers
            grid_x = (center_x - point_cloud_range[0]) / voxel_size[0]
            grid_y = (center_y - point_cloud_range[1]) / voxel_size[1]
            
            # Calculate grid cell indices
            grid_x_idx = torch.floor(grid_x).long()
            grid_y_idx = torch.floor(grid_y).long()
            
            # Calculate offsets
            x_offset = grid_x - grid_x_idx.float()
            y_offset = grid_y - grid_y_idx.float()
            
            # Limit to feature map dimensions
            valid_mask = (
                (grid_x_idx >= 0) & (grid_x_idx < feature_shape[1]) &
                (grid_y_idx >= 0) & (grid_y_idx < feature_shape[0])
            )
            
            # Skip if no valid boxes
            if valid_mask.sum() == 0:
                continue
            
            # Filter valid boxes
            grid_x_idx = grid_x_idx[valid_mask]
            grid_y_idx = grid_y_idx[valid_mask]
            x_offset = x_offset[valid_mask]
            y_offset = y_offset[valid_mask]
            center_z = center_z[valid_mask]
            width = width[valid_mask]
            length = length[valid_mask]
            height = height[valid_mask]
            yaw = yaw[valid_mask]
            class_ids = class_ids[valid_mask]
            velocity = velocity[valid_mask]
            
            # Generate gaussian heatmap for each object
            for i in range(grid_x_idx.shape[0]):
                self._generate_gaussian_heatmap(
                    heatmap_targets[batch_idx, class_ids[i]],
                    grid_x_idx[i], grid_y_idx[i],
                    width[i], length[i]
                )
                
                # Set targets at center location
                offset_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = x_offset[i]
                offset_targets[batch_idx, 1, grid_y_idx[i], grid_x_idx[i]] = y_offset[i]
                
                z_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = center_z[i]
                
                size_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = width[i]
                size_targets[batch_idx, 1, grid_y_idx[i], grid_x_idx[i]] = length[i]
                size_targets[batch_idx, 2, grid_y_idx[i], grid_x_idx[i]] = height[i]
                
                # Encode yaw as sin/cos
                yaw_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = torch.sin(yaw[i])
                yaw_targets[batch_idx, 1, grid_y_idx[i], grid_x_idx[i]] = torch.cos(yaw[i])
                
                vel_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = velocity[i, 0]
                vel_targets[batch_idx, 1, grid_y_idx[i], grid_x_idx[i]] = velocity[i, 1]
                
                # Set mask
                mask_targets[batch_idx, 0, grid_y_idx[i], grid_x_idx[i]] = 1.0
        
        # Prepare targets dict
        targets_dict = {
            'heatmap_targets': heatmap_targets,
            'offset_targets': offset_targets,
            'z_targets': z_targets,
            'size_targets': size_targets,
            'yaw_targets': yaw_targets,
            'vel_targets': vel_targets,
            'mask_targets': mask_targets
        }
        
        return targets_dict
    
    def _generate_gaussian_heatmap(self, heatmap: torch.Tensor, center_x: torch.Tensor, 
                                 center_y: torch.Tensor, width: torch.Tensor, 
                                 length: torch.Tensor) -> None:
        """
        Generate gaussian heatmap for an object.
        
        Args:
            heatmap: Target heatmap tensor to modify (H, W)
            center_x: Center x coordinate in grid
            center_y: Center y coordinate in grid
            width: Object width
            length: Object length
            
        Returns:
            None (modifies heatmap in-place)
        """
        # Calculate gaussian radius based on object size
        object_size = torch.sqrt(width ** 2 + length ** 2)
        radius = torch.clamp(object_size / 2, min=self.min_radius)
        
        # Get gaussian parameters
        sigma = radius / 3  # 3 sigma rule
        
        # Get heatmap dimensions
        height, width = heatmap.shape
        
        # Create meshgrid
        x = torch.arange(0, width, device=heatmap.device)
        y = torch.arange(0, height, device=heatmap.device)
        y_grid, x_grid = torch.meshgrid(y, x)
        
        # Calculate squared distance to center
        squared_dist = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        
        # Calculate gaussian
        gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
        
        # Update heatmap (use max to handle overlapping objects)
        heatmap = torch.max(heatmap, gaussian)


class CenterPointDetector(nn.Module):
    """
    CenterPoint detector for 3D object detection from point clouds.
    
    CenterPoint is a center-based 3D object detection and tracking framework that first detects
    centers of objects using a keypoint detector and then regresses to other attributes.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize CenterPoint detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Create model components
        self.voxel_feature_extractor = VoxelFeatureExtractor(config)
        self.backbone_3d = SparseBackbone3D(config)
        self.bev_extractor = BEVFeatureExtractor(config)
        self.head = HeadNetwork(config)
        
        # Loss functions
        self.loss_weights = config.get('loss_weights', {
            'heatmap_loss': 1.0,
            'offset_loss': 1.0,
            'size_loss': 1.0,
            'angle_loss': 1.0,
            'height_loss': 1.0,
            'velocity_loss': 0.5
        })
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass of CenterPoint detector.
        
        Args:
            batch_dict: Input dictionary containing:
                - 'points': List of point clouds
                - 'batch_size': Batch size
                
        Returns:
            Dictionary with detection results
        """
        # Set batch size
        batch_dict['batch_size'] = len(batch_dict['points'])
        
        # Add voxel size and point cloud range to batch_dict
        batch_dict['voxel_size'] = self.config.get('voxel_size', [0.1, 0.1, 0.15])
        batch_dict['point_cloud_range'] = self.config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1])
        
        # Apply voxel feature extractor
        batch_dict = self.voxel_feature_extractor(batch_dict)
        
        # Apply 3D backbone
        batch_dict = self.backbone_3d(batch_dict)
        
        # Apply BEV extractor
        batch_dict = self.bev_extractor(batch_dict)
        
        # Apply head network
        batch_dict = self.head(batch_dict)
        
        # Calculate loss during training
        if self.training:
            loss, loss_dict = self.get_loss(batch_dict)
            batch_dict['loss'] = loss
            batch_dict['loss_dict'] = loss_dict
        
        return batch_dict
    
    def get_loss(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss for CenterPoint detector.
        
        Args:
            batch_dict: Input dictionary containing network outputs and targets
                
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Extract predictions and targets
        pred_heatmap = batch_dict['heatmap']
        pred_offset = batch_dict['offset']
        pred_z = batch_dict['height']
        pred_size = batch_dict['size']
        pred_angle = batch_dict['angle']
        pred_velocity = batch_dict['velocity']
        
        target_heatmap = batch_dict['heatmap_targets']
        target_offset = batch_dict['offset_targets']
        target_z = batch_dict['z_targets']
        target_size = batch_dict['size_targets']
        target_angle = batch_dict['yaw_targets']
        target_velocity = batch_dict['vel_targets']
        target_mask = batch_dict['mask_targets']
        
        # Calculate focal loss for heatmap
        heatmap_loss = self._focal_loss(pred_heatmap, target_heatmap)
        
        # Calculate L1 loss for other regression targets
        offset_loss = self._masked_l1_loss(pred_offset, target_offset, target_mask)
        z_loss = self._masked_l1_loss(pred_z, target_z, target_mask)
        size_loss = self._masked_l1_loss(pred_size, target_size, target_mask)
        angle_loss = self._masked_l1_loss(pred_angle, target_angle, target_mask)
        velocity_loss = self._masked_l1_loss(pred_velocity, target_velocity, target_mask)
        
        # Combine losses
        loss_dict = {
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
            'height_loss': z_loss,
            'size_loss': size_loss,
            'angle_loss': angle_loss,
            'velocity_loss': velocity_loss
        }
        
        # Apply weights and sum
        total_loss = 0
        for k, v in loss_dict.items():
            weight = self.loss_weights.get(k, 1.0)
            total_loss += weight * v
        
        return total_loss, loss_dict
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, 
                  beta: float = 4.0) -> torch.Tensor:
        """
        Focal loss for heatmap prediction.
        
        Args:
            pred: Predicted heatmap (B, C, H, W)
            target: Target heatmap (B, C, H, W)
            alpha: Focal loss alpha parameter
            beta: Focal loss beta parameter
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Calculate focal weights
        pos_mask = (target == 1)
        neg_mask = (target < 1)
        
        # Positive samples
        pos_weight = torch.pow(1 - pred, alpha)
        pos_loss = -pos_weight * torch.log(pred + 1e-6)
        
        # Negative samples
        neg_weight = torch.pow(1 - target, beta) * torch.pow(pred, alpha)
        neg_loss = -neg_weight * torch.log(1 - pred + 1e-6)
        
        # Combine and normalize
        num_pos = pos_mask.sum().float() + 1e-6
        loss = (pos_loss * pos_mask + neg_loss * neg_mask).sum() / num_pos
        
        return loss


def prepare_data_for_centerpoint(points_list: List[torch.Tensor], 
                               config: Optional[Dict] = None) -> Dict:
    """
    Prepare data for CenterPoint detector.
    
    Args:
        points_list: List of point clouds, each as (N, 4+) tensor [x, y, z, intensity, ...]
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with prepared data
    """
    batch_dict = {
        'points': points_list,
        'batch_size': len(points_list)
    }
    
    # Add configuration parameters if provided
    if config is not None:
        batch_dict.update({
            'voxel_size': config.get('voxel_size', [0.1, 0.1, 0.15]),
            'point_cloud_range': config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1]),
            'grid_size': config.get('grid_size', [400, 400])
        })
    
    return batch_dict


if __name__ == "__main__":
    # Example usage of CenterPoint detector
    import torch
    
    # Configuration
    config = {
        'voxel_size': [0.1, 0.1, 0.15],
        'point_cloud_range': [0, -39.68, -3, 100, 39.68, 1],
        'max_points_per_voxel': 100,
        'max_voxels': 40000,
        'num_classes': 4,
        'vfe_filters': [64, 64],
        'conv_filters': [32, 64, 128],
        'num_bev_features': 256,
        'use_height': True,
        'target_assigner_config': {
            'num_classes': 4,
            'gaussian_overlap': 0.1,
            'min_radius': 2
        },
        'loss_weights': {
            'heatmap_loss': 1.0,
            'offset_loss': 1.0,
            'size_loss': 1.0,
            'angle_loss': 1.0,
            'height_loss': 1.0,
            'velocity_loss': 0.5
        }
    }
    
    # Create detector
    detector = CenterPointDetector(config)
    detector.eval()  # Set to evaluation mode
    
    # Create dummy point clouds
    num_points = 1000
    point_cloud1 = torch.rand(num_points, 4)  # [x, y, z, intensity]
    point_cloud2 = torch.rand(num_points, 4)
    
    # Scale point clouds to be in range
    point_cloud1[:, 0] = point_cloud1[:, 0] * 100  # x in [0, 100]
    point_cloud1[:, 1] = point_cloud1[:, 1] * 79.36 - 39.68  # y in [-39.68, 39.68]
    point_cloud1[:, 2] = point_cloud1[:, 2] * 4 - 3  # z in [-3, 1]
    
    point_cloud2[:, 0] = point_cloud2[:, 0] * 100
    point_cloud2[:, 1] = point_cloud2[:, 1] * 79.36 - 39.68
    point_cloud2[:, 2] = point_cloud2[:, 2] * 4 - 3
    
    # Prepare input data
    points_list = [point_cloud1, point_cloud2]
    batch_dict = prepare_data_for_centerpoint(points_list, config)
    
    # Run detector
    with torch.no_grad():
        result_dict = detector(batch_dict)
    
    # Process results
    if 'detections' in result_dict:
        detections = result_dict['detections']
        print(f"Number of batches: {len(detections)}")
        for batch_idx, batch_dets in enumerate(detections):
            print(f"Batch {batch_idx}: {len(batch_dets)} detections")
            for det in batch_dets[:5]:  # Print first 5 detections
                print(f"  Score: {det['score']:.3f}, Class: {det['label']}, "
                     f"Box: {[f'{x:.2f}' for x in det['box']]}")
    else:
        print("No detections found.")
    
    def _masked_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Masked L1 loss for regression targets.
        
        Args:
            pred: Prediction tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Mask tensor (B, 1, H, W)
            
        Returns:
            Masked L1 loss value
        """
        # Expand mask to match prediction channels
        expanded_mask = mask.expand_as(pred)
        
        # Calculate L1 loss
        loss = torch.abs(pred - target) * expanded_mask
        
        # Normalize by number of positive locations
        num_pos = expanded_mask.sum().float() + 1e-6
        loss = loss.sum() / num_pos
        
        return loss