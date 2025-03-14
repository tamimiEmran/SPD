"""
image_feature_encoder module for V2X-Seq project.

This module provides functionality for extracting features from camera images,
which can be used for fusion with LiDAR data in the vehicle-infrastructure 
cooperative 3D tracking task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any


class ImageFeatureEncoder(nn.Module):
    """
    Feature encoder for extracting features from images.
    
    This class implements a CNN-based feature extractor that can be used
    to extract features from camera images for fusion with LiDAR features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        use_fpn: bool = True,
        fpn_size: int = 256,
        freeze_backbone: bool = False
    ):
        """
        Initialize the image feature encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            feature_dim: Dimension of output features
            backbone: Name of the backbone CNN to use (resnet18, resnet34, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            use_fpn: Whether to use Feature Pyramid Network
            fpn_size: Size of FPN features
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.backbone_name = backbone
        self.use_fpn = use_fpn
        self.fpn_size = fpn_size
        
        # Initialize backbone network
        self.backbone = self._build_backbone(backbone, pretrained)
        
        # Get the backbone feature dimensions
        self.backbone_channels = self._get_backbone_channels()
        
        # Initialize FPN if needed
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=self.backbone_channels,
                out_channels=fpn_size
            )
        
        # Output adaptation layer
        if use_fpn:
            self.output_layer = nn.Conv2d(fpn_size, feature_dim, kernel_size=1)
        else:
            self.output_layer = nn.Conv2d(self.backbone_channels[-1], feature_dim, kernel_size=1)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _build_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """
        Build the backbone CNN.
        
        Args:
            backbone_name: Name of the backbone CNN to use
            pretrained: Whether to use pretrained weights
            
        Returns:
            Backbone CNN module
        """
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove the classification head
        if backbone_name.startswith('resnet'):
            layers = list(backbone.children())[:-2]  # Remove avg pool and fc
            backbone = nn.Sequential(*layers)
        elif backbone_name == 'mobilenet_v2':
            backbone = backbone.features  # Just the feature extractor
        
        return backbone
    
    def _get_backbone_channels(self) -> List[int]:
        """
        Get the output channels for each stage of the backbone.
        
        Returns:
            List of output channels for each stage
        """
        if self.backbone_name.startswith('resnet'):
            if self.backbone_name == 'resnet18' or self.backbone_name == 'resnet34':
                return [64, 128, 256, 512]
            else:  # resnet50, resnet101, resnet152
                return [256, 512, 1024, 2048]
        elif self.backbone_name == 'mobilenet_v2':
            return [24, 32, 96, 320]  # Approximate values, may need adjustment
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def _extract_resnet_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from each stage of ResNet.
        
        Args:
            x: Input tensor
            
        Returns:
            List of features from each stage
        """
        features = []
        
        # Stage 1
        x = self.backbone[0](x)  # Conv1
        x = self.backbone[1](x)  # BN1
        x = self.backbone[2](x)  # ReLU
        x = self.backbone[3](x)  # MaxPool
        
        # Stage 2
        x = self.backbone[4](x)  # Layer1
        features.append(x)
        
        # Stage 3
        x = self.backbone[5](x)  # Layer2
        features.append(x)
        
        # Stage 4
        x = self.backbone[6](x)  # Layer3
        features.append(x)
        
        # Stage 5
        x = self.backbone[7](x)  # Layer4
        features.append(x)
        
        return features
    
    def _extract_mobilenet_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from each stage of MobileNetV2.
        
        Args:
            x: Input tensor
            
        Returns:
            List of features from each stage
        """
        features = []
        
        # Extract features at specified layers
        feature_layers = [3, 6, 13, 18]  # Approximate layer indices for MobileNetV2
        
        prev_layer = 0
        for idx in feature_layers:
            for i in range(prev_layer, idx):
                x = self.backbone[i](x)
            prev_layer = idx
            features.append(x)
        
        # Finish the forward pass
        for i in range(prev_layer, len(self.backbone)):
            x = self.backbone[i](x)
        
        features.append(x)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the image feature encoder.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            If use_fpn is True, returns a dictionary of multi-scale features
            Otherwise, returns a tensor of shape (B, feature_dim, H', W')
        """
        # Extract features from backbone
        if self.backbone_name.startswith('resnet'):
            features = self._extract_resnet_features(x)
        elif self.backbone_name == 'mobilenet_v2':
            features = self._extract_mobilenet_features(x)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Apply FPN if specified
        if self.use_fpn:
            fpn_features = self.fpn(features)
            
            # Apply output layer to the highest resolution feature map
            output = self.output_layer(fpn_features["0"])
            
            # Return multi-scale features and final output
            return {
                "multi_scale_features": fpn_features,
                "features": output
            }
        else:
            # Apply output layer to the highest level feature
            output = self.output_layer(features[-1])
            return output


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    
    This is a simplified implementation of FPN that aggregates features
    from different scales of the backbone network.
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        """
        Initialize the Feature Pyramid Network.
        
        Args:
            in_channels_list: List of input channels for each level
            out_channels: Number of output channels for each level
        """
        super().__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        # Create lateral connections and output convolutions
        for in_channels in in_channels_list:
            self.inner_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.layer_blocks.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Feature Pyramid Network.
        
        Args:
            features: List of feature maps from the backbone
            
        Returns:
            Dictionary of processed feature maps
        """
        # Create inner features (1x1 convolutions)
        inner_features = []
        for i, feature in enumerate(features):
            inner_features.append(self.inner_blocks[i](feature))
        
        # Create top-down pathway with upsampling
        fpn_features = {}
        last_inner = inner_features[-1]
        fpn_features[str(len(features) - 1)] = self.layer_blocks[-1](last_inner)
        
        for idx in range(len(features) - 2, -1, -1):
            # Upsample higher level feature
            upsample = F.interpolate(
                last_inner, 
                size=inner_features[idx].shape[-2:],
                mode='nearest'
            )
            
            # Add lateral connection
            last_inner = upsample + inner_features[idx]
            
            # Apply 3x3 convolution
            fpn_features[str(idx)] = self.layer_blocks[idx](last_inner)
        
        return fpn_features


class DeepLabV3Encoder(nn.Module):
    """
    Image feature encoder based on DeepLabV3.
    
    This class implements a feature extractor based on DeepLabV3,
    which is especially useful for semantic segmentation tasks.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the DeepLabV3 feature encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            feature_dim: Dimension of output features
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Load pretrained DeepLabV3 with ResNet50 backbone
        self.backbone = models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained
        )
        
        # Remove the classification head
        self.backbone = self.backbone.backbone
        
        # Add an output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepLabV3 feature encoder.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim, H', W')
        """
        # Extract features from backbone
        features = self.backbone(x)["out"]
        
        # Apply output layer
        output = self.output_layer(features)
        
        return output


def build_image_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Build an image feature encoder based on configuration.
    
    Args:
        config: Configuration dictionary for the image encoder
        
    Returns:
        Image feature encoder module
    """
    encoder_type = config.get('type', 'resnet')
    
    if encoder_type == 'resnet':
        return ImageFeatureEncoder(
            in_channels=config.get('in_channels', 3),
            feature_dim=config.get('feature_dim', 256),
            backbone=config.get('backbone', 'resnet18'),
            pretrained=config.get('pretrained', True),
            use_fpn=config.get('use_fpn', True),
            fpn_size=config.get('fpn_size', 256),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    elif encoder_type == 'deeplabv3':
        return DeepLabV3Encoder(
            in_channels=config.get('in_channels', 3),
            feature_dim=config.get('feature_dim', 256),
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create a dummy input
    batch_size = 2
    in_channels = 3
    height = 360
    width = 640
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create the encoder
    encoder = ImageFeatureEncoder(
        in_channels=in_channels,
        feature_dim=256,
        backbone='resnet18',
        pretrained=False,
        use_fpn=True
    )
    
    # Forward pass
    output = encoder(x)
    
    # Print output shape
    if isinstance(output, dict):
        print("Multi-scale features:")
        for k, v in output["multi_scale_features"].items():
            print(f"  Level {k}: {v.shape}")
        print(f"Output features: {output['features'].shape}")
    else:
        print(f"Output features: {output.shape}")