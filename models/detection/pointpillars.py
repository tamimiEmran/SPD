"""
pointpillars.py module for V2X-Seq project.

This module provides an implementation of PointPillars for 3D object detection
from point clouds, adapted for the V2X-Seq dataset.

Reference: https://arxiv.org/abs/1812.05784
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network to extract features from point clouds.
    """
    def __init__(
        self,
        num_input_features=4,
        use_norm=True,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.16, 0.16, 4),
        pc_range=(0, -39.68, -3, 100, 39.68, 1)
    ):
        """
        Args:
            num_input_features: Number of input features per point
            use_norm: Whether to use batch normalization
            num_filters: List of feature dimensions for each PFNLayer
            with_distance: Whether to include Euclidean distance to points
            voxel_size: Size of voxels (m) in [x, y, z]
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5  # [x, y, z, i] + [xc, yc, zc, xp, yp]
        if with_distance:
            num_input_features += 1
            
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        
        # Create PFNLayers
        self.pfn_layers = nn.ModuleList()
        for i in range(len(num_filters)):
            in_channels = num_input_features if i == 0 else num_filters[i-1]
            self.pfn_layers.append(
                PFNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters[i],
                    use_norm=use_norm,
                    last_layer=(i == len(num_filters) - 1)
                )
            )
        
        # Feature dimension after going through PFNLayers
        self.num_output_features = num_filters[-1]
    
    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar, coors):
        """
        Args:
            pillar_x: (num_pillars, max_points_per_pillar) x coordinates of points
            pillar_y: (num_pillars, max_points_per_pillar) y coordinates of points
            pillar_z: (num_pillars, max_points_per_pillar) z coordinates of points
            pillar_i: (num_pillars, max_points_per_pillar) intensity values
            num_points_per_pillar: (num_pillars) number of points in each pillar
            coors: (num_pillars, 4) [batch_id, z_idx, y_idx, x_idx]
            
        Returns:
            features: (num_pillars, num_output_features)
        """
        device = pillar_x.device
        num_pillars = pillar_x.shape[0]
        max_points = pillar_x.shape[1]
        
        # Find mean of points in each pillar
        # Use masked average pooling
        points_mean = torch.zeros_like(pillar_x[:, :1])
        points_mean_x = torch.sum(pillar_x, dim=1, keepdim=True) / torch.clamp(num_points_per_pillar.view(-1, 1), min=1)
        points_mean_y = torch.sum(pillar_y, dim=1, keepdim=True) / torch.clamp(num_points_per_pillar.view(-1, 1), min=1)
        points_mean_z = torch.sum(pillar_z, dim=1, keepdim=True) / torch.clamp(num_points_per_pillar.view(-1, 1), min=1)
        
        # Compute offsets from pillar centers
        x_offset = pillar_x - points_mean_x
        y_offset = pillar_y - points_mean_y
        z_offset = pillar_z - points_mean_z
        
        # Compute position of pillars in the pseudo-image
        x_bev = (coors[:, 3].float() * self.voxel_size[0] + self.voxel_size[0] / 2 + self.pc_range[0])
        y_bev = (coors[:, 2].float() * self.voxel_size[1] + self.voxel_size[1] / 2 + self.pc_range[1])
        
        # Repeat for each point
        x_bev = x_bev.view(-1, 1).repeat(1, max_points)
        y_bev = y_bev.view(-1, 1).repeat(1, max_points)
        
        # Create mask for valid points (non-zero)
        mask = torch.zeros_like(pillar_x)
        for i in range(num_pillars):
            mask[i, :num_points_per_pillar[i]] = 1
        
        # Combine features
        features = [pillar_x, pillar_y, pillar_z, pillar_i, x_offset, y_offset, z_offset, x_bev, y_bev]
        features = torch.stack(features, dim=-1)
        
        # Add distance feature if required
        if self.with_distance:
            dist = torch.sqrt(pillar_x ** 2 + pillar_y ** 2 + pillar_z ** 2)
            features = torch.cat([features, dist.unsqueeze(-1)], dim=-1)
        
        # Mask out invalid points
        mask = mask.unsqueeze(-1).repeat(1, 1, features.shape[-1])
        features = features * mask
        
        # Forward through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        return features


class PFNLayer(nn.Module):
    """
    Pillar Feature Network Layer to aggregate features using max pooling.
    """
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_norm: Whether to use batch normalization
            last_layer: Whether this is the last PFNLayer
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_layer = last_layer
        
        # Linear layer
        self.linear = nn.Linear(in_channels, out_channels, bias=not use_norm)
        
        # Batch normalization
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, inputs):
        """
        Args:
            inputs: (num_pillars, max_points_per_pillar, in_channels)
            
        Returns:
            x: (num_pillars, out_channels) if last_layer
               (num_pillars, max_points_per_pillar, out_channels) otherwise
        """
        x = self.linear(inputs)
        
        # Batch normalization
        if self.use_norm:
            # Transpose for BatchNorm1d
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        
        # ReLU
        x = F.relu(x)
        
        # Max pooling if last layer
        if self.last_layer:
            x = torch.max(x, dim=1)[0]
            
        return x


class PointPillarsScatter(nn.Module):
    """
    Scatter pillar features to a BEV pseudo-image.
    """
    def __init__(self, num_features, voxel_size, pc_range, output_shape):
        """
        Args:
            num_features: Number of features per pillar
            voxel_size: Size of voxels (m) in [x, y, z]
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            output_shape: Output BEV feature map shape [H, W]
        """
        super().__init__()
        self.name = 'PointPillarsScatter'
        self.num_features = num_features
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.output_shape = output_shape
        
    def forward(self, pillar_features, coors):
        """
        Args:
            pillar_features: (num_pillars, num_features) Pillar features
            coors: (num_pillars, 4) [batch_id, z_idx, y_idx, x_idx]
            
        Returns:
            batch_canvas: (batch_size, num_features, H, W) BEV feature map
        """
        batch_size = coors[:, 0].max().int().item() + 1
        canvas_h, canvas_w = self.output_shape
        
        # Create empty canvas for batch
        batch_canvas = []
        for i in range(batch_size):
            # Create empty canvas for each batch item
            canvas = torch.zeros(
                self.num_features,
                canvas_h,
                canvas_w,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            
            # Get indices of pillars for this batch item
            batch_mask = coors[:, 0] == i
            this_coors = coors[batch_mask, :]
            this_features = pillar_features[batch_mask, :]
            
            # Convert coordinate indices to indices in canvas
            indices = this_coors[:, 2] * canvas_w + this_coors[:, 3]
            indices = indices.type(torch.long)
            
            # Scatter pillar features to canvas
            # We scatter to (F, H*W) and then reshape
            canvas_flat = canvas.view(self.num_features, -1)
            canvas_flat.index_add_(1, indices, this_features.t())
            
            batch_canvas.append(canvas)
            
        batch_canvas = torch.stack(batch_canvas, dim=0)
        
        return batch_canvas


class BEVBackbone(nn.Module):
    """
    Backbone network for BEV feature map processing.
    """
    def __init__(
        self,
        in_channels,
        out_channels=256,
        layer_nums=(3, 5, 5),
        layer_strides=(2, 2, 2),
        num_filters=(64, 128, 256)
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            layer_nums: Number of layers in each stage
            layer_strides: Stride of the first layer in each stage
            num_filters: Number of filters in each stage
        """
        super().__init__()
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        # Current feature dimension
        cur_channels = in_channels
        
        # Construct backbone blocks
        for i in range(len(layer_nums)):
            block, num_out_channels = self._make_layer(
                cur_channels,
                num_filters[i],
                layer_nums[i],
                stride=layer_strides[i]
            )
            
            self.blocks.append(block)
            cur_channels = num_out_channels
            
            # Create deconvolution layers
            if i > 0:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_out_channels,
                            num_filters[0],
                            layer_strides[i],
                            stride=layer_strides[i],
                            bias=False
                        ),
                        nn.BatchNorm2d(num_filters[0]),
                        nn.ReLU()
                    )
                )
        
        # Final layer
        self.final_conv = nn.Conv2d(
            sum([num_filters[0]] * len(layer_nums)), 
            out_channels,
            kernel_size=1
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """
        Create a block of convolutional layers.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks
            stride: Stride of the first layer
            
        Returns:
            nn.Sequential: Block of layers
            int: Number of output channels
        """
        block = []
        
        # First layer with stride
        block.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU())
        
        # Rest of the layers
        for _ in range(num_blocks - 1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU())
            
        return nn.Sequential(*block), out_channels
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, H, W) Input BEV feature map
            
        Returns:
            x_concat: (batch_size, out_channels, H, W) Output feature map
        """
        ups = []
        
        # Forward through backbone blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # For the first block, add input directly
            if i == 0:
                ups.append(x)
            # For other blocks, apply deconvolution
            else:
                ups.append(self.deblocks[i-1](x))
        
        # Concatenate all scales
        x = torch.cat(ups, dim=1)
        
        # Apply final convolution
        x = self.final_conv(x)
        
        return x


class DetectionHead(nn.Module):
    """
    Detection head for classification and regression.
    """
    def __init__(
        self,
        in_channels,
        num_classes=4,  # Car, Van, Truck, Bus
        num_anchors_per_location=2,
        box_code_size=7  # [x, y, z, w, l, h, theta]
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of classes
            num_anchors_per_location: Number of anchors per location
            box_code_size: Size of box encoding
        """
        super().__init__()
        
        # Classification head
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors_per_location * num_classes, 
            kernel_size=1
        )
        
        # Regression head
        self.reg_head = nn.Conv2d(
            in_channels, num_anchors_per_location * box_code_size,
            kernel_size=1
        )
        
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        self.box_code_size = box_code_size
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, H, W) Input features
            
        Returns:
            cls_preds: (batch_size, num_anchors_per_location * num_classes, H, W)
            box_preds: (batch_size, num_anchors_per_location * box_code_size, H, W)
        """
        # Classification predictions
        cls_preds = self.cls_head(x)
        
        # Regression predictions
        box_preds = self.reg_head(x)
        
        return cls_preds, box_preds
    
    def predict(self, cls_preds, box_preds, anchors):
        """
        Generate detections from network outputs and anchors.
        
        Args:
            cls_preds: (batch_size, num_anchors_per_location * num_classes, H, W)
            box_preds: (batch_size, num_anchors_per_location * box_code_size, H, W)
            anchors: (num_anchors, box_code_size) Anchor boxes
            
        Returns:
            batch_detections: List of detections for each batch, each a dict with:
                              - 'boxes_3d': Tensor of shape (N, 7) with box parameters
                              - 'scores': Tensor of shape (N,) with confidence scores
                              - 'labels': Tensor of shape (N,) with predicted class labels
        """
        batch_size = cls_preds.shape[0]
        
        # Reshape predictions
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        # Compute softmax along the class dimension
        cls_preds = cls_preds.view(batch_size, -1, self.num_classes)
        cls_scores = F.softmax(cls_preds, dim=-1)
        
        # Get maximum class score and class index
        cls_scores, cls_labels = torch.max(cls_scores, dim=-1)
        
        # Reshape box predictions
        box_preds = box_preds.view(batch_size, -1, self.box_code_size)
        
        # Apply anchor decoding
        box_preds = self._decode_boxes(box_preds, anchors)
        
        batch_detections = []
        
        # Generate detections for each batch
        for batch_idx in range(batch_size):
            # Get predictions for this batch
            scores = cls_scores[batch_idx]
            labels = cls_labels[batch_idx]
            boxes = box_preds[batch_idx]
            
            # Filter boxes with low scores
            # This is just an example, you might want to use NMS here
            mask = scores > 0.5
            scores = scores[mask]
            labels = labels[mask]
            boxes = boxes[mask]
            
            detections = {
                'boxes_3d': boxes,
                'scores': scores,
                'labels': labels
            }
            
            batch_detections.append(detections)
            
        return batch_detections
    
    def _decode_boxes(self, box_preds, anchors):
        """
        Decode box predictions from anchor boxes.
        
        Args:
            box_preds: (batch_size, num_anchors, box_code_size)
            anchors: (num_anchors, box_code_size)
            
        Returns:
            decoded_boxes: (batch_size, num_anchors, box_code_size)
        """
        # This is a simplified version of anchor-based decoding
        # In a real implementation, you would use a proper box decoding method
        # which considers anchor parameters and residual predictions
        
        # Repeat anchors for each batch item
        batch_size = box_preds.shape[0]
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply anchor-based decoding
        # This is a simplified example, should be replaced with proper decoding
        decoded_boxes = anchors + box_preds
        
        return decoded_boxes


class PointPillarsDetector(nn.Module):
    """
    Full PointPillars detector implementation.
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Extract parameters from config
        voxel_size = config.get('voxel_size', [0.16, 0.16, 4])
        pc_range = config.get('point_cloud_range', [0, -39.68, -3, 100, 39.68, 1])
        max_points_per_voxel = config.get('max_points_per_voxel', 32)
        max_voxels = config.get('max_voxels', 40000)
        
        # Calculate output feature map size
        grid_size = np.round((np.array(pc_range[3:]) - np.array(pc_range[:3])) / 
                             np.array(voxel_size)).astype(np.int64)
        self.grid_size = grid_size
        
        # Feature sizes
        pillar_feature_size = config.get('pillar_feature_size', 64)
        
        # BEV feature map size
        bev_h, bev_w = grid_size[1], grid_size[0]
        
        # Create model components
        self.pillar_feature_net = PillarFeatureNet(
            num_input_features=4,  # x, y, z, intensity
            num_filters=[pillar_feature_size],
            voxel_size=voxel_size,
            pc_range=pc_range
        )
        
        self.scatter = PointPillarsScatter(
            num_features=pillar_feature_size,
            voxel_size=voxel_size,
            pc_range=pc_range,
            output_shape=[bev_h, bev_w]
        )
        
        self.backbone = BEVBackbone(
            in_channels=pillar_feature_size,
            out_channels=config.get('backbone_out_channels', 256)
        )
        
        self.detection_head = DetectionHead(
            in_channels=config.get('backbone_out_channels', 256),
            num_classes=config.get('num_classes', 4),
            num_anchors_per_location=config.get('num_anchors_per_location', 2)
        )
        
        # Parameters for point cloud voxelization
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict: Dictionary with point cloud data
                - 'points': List of point clouds, each with shape (N, 4+) [x, y, z, intensity, ...]
                - 'anchors': Anchor boxes (optional)
            
        Returns:
            batch_dict: Updated with detection results
        """
        # Assuming batch_dict contains pre-processed voxels, coords, and num_points
        # In a real implementation, you would do voxelization here
        voxels = batch_dict['voxels']  # (num_voxels, max_points_per_voxel, features)
        coords = batch_dict['voxel_coords']  # (num_voxels, 4) [batch_idx, z_idx, y_idx, x_idx]
        num_points = batch_dict['voxel_num_points']  # (num_voxels,)
        
        # Extract features to feed to PFN
        pillar_x = voxels[..., 0]
        pillar_y = voxels[..., 1]
        pillar_z = voxels[..., 2]
        pillar_i = voxels[..., 3]
        
        # Get pillar features
        pillar_features = self.pillar_feature_net(
            pillar_x, pillar_y, pillar_z, pillar_i, num_points, coords
        )
        
        # Scatter to BEV feature map
        spatial_features = self.scatter(pillar_features, coords)
        
        # Process with backbone
        spatial_features_2d = self.backbone(spatial_features)
        
        # Generate detections
        cls_preds, box_preds = self.detection_head(spatial_features_2d)
        
        # Store predictions
        batch_dict.update({
            'cls_preds': cls_preds,
            'box_preds': box_preds
        })
        
        # If anchors are provided, generate detections
        if 'anchors' in batch_dict:
            batch_dict['detections'] = self.detection_head.predict(
                cls_preds, box_preds, batch_dict['anchors']
            )
        
        return batch_dict

    def voxelize(self, points):
        """
        Convert point cloud to voxels.
        
        Args:
            points: (N, 4+) [x, y, z, intensity, ...]
            
        Returns:
            voxels: (num_voxels, max_points_per_voxel, features)
            coords: (num_voxels, 4) [batch_idx, z_idx, y_idx, x_idx]
            num_points_per_voxel: (num_voxels,)
        """
        # This is a placeholder function for voxelization
        # In a real implementation, you would use a voxelization function
        # or library (e.g., spconv)
        
        # Filter points outside the range
        mask = (
            (points[:, 0] >= self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) &
            (points[:, 1] >= self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) &
            (points[:, 2] >= self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
        )
        points = points[mask]
        
        # Convert points to voxel indices
        voxel_indices = (
            (points[:, :3] - torch.tensor(self.pc_range[:3], device=points.device)) / 
            torch.tensor(self.voxel_size, device=points.device)
        ).int()
        
        # Group points by voxel (simplified)
        # In practice, you would use a more efficient method
        voxel_indices_unique = torch.unique(voxel_indices, dim=0)
        num_voxels = min(len(voxel_indices_unique), self.max_voxels)
        
        # Initialize outputs
        voxels = torch.zeros(
            (num_voxels, self.max_points_per_voxel, points.shape[1]),
            dtype=points.dtype, device=points.device
        )
        coords = torch.zeros(
            (num_voxels, 4), dtype=torch.int32, device=points.device
        )
        num_points_per_voxel = torch.zeros(
            (num_voxels,), dtype=torch.int32, device=points.device
        )
        
        # Fill voxels (simplified implementation)
        for i in range(num_voxels):
            v_idx = voxel_indices_unique[i]
            points_in_voxel = points[
                (voxel_indices[:, 0] == v_idx[0]) &
                (voxel_indices[:, 1] == v_idx[1]) &
                (voxel_indices[:, 2] == v_idx[2])
            ]
            
            num_points = min(len(points_in_voxel), self.max_points_per_voxel)
            voxels[i, :num_points] = points_in_voxel[:num_points]
            coords[i] = torch.tensor([0, v_idx[2], v_idx[1], v_idx[0]], device=points.device)
            num_points_per_voxel[i] = num_points
            
        return voxels, coords, num_points_per_voxel