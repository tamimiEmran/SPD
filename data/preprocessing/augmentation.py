"""
augmentation module for V2X-Seq project.

This module provides functionality for augmenting point cloud data to improve
the robustness and generalization capability of 3D object detection and tracking models.
Augmentations include random rotation, scaling, translation, flipping, and point dropout.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class PointCloudAugmentor:
    """
    Class for applying various augmentations to point cloud data.
    
    This class implements common augmentation techniques specifically designed
    for point cloud data in autonomous driving scenarios.
    """
    
    def __init__(self, 
                 rotation_range: Optional[Tuple[float, float]] = (-np.pi/10, np.pi/10),
                 scaling_range: Optional[Tuple[float, float]] = (0.95, 1.05),
                 translation_range: Optional[Tuple[float, float, float]] = (-0.2, 0.2, -0.2, 0.2, -0.1, 0.1),
                 flip_probability: Optional[float] = 0.5,
                 dropout_probability: Optional[float] = 0.05,
                 random_seed: Optional[int] = None):
        """
        Initialize the augmentor with augmentation parameters.
        
        Args:
            rotation_range: Range for random rotation around z-axis in radians (min, max)
            scaling_range: Range for random scaling (min, max)
            translation_range: Range for random translation (x_min, x_max, y_min, y_max, z_min, z_max)
            flip_probability: Probability of flipping the point cloud horizontally
            dropout_probability: Probability of dropping each point
            random_seed: Random seed for reproducibility
        """
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.translation_range = translation_range
        self.flip_probability = flip_probability
        self.dropout_probability = dropout_probability
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply a series of random augmentations to the point cloud.
        
        Args:
            points: Nx4+ array of points (x, y, z, intensity, ...)
            boxes: Optional Mx7 array of 3D boxes (x, y, z, w, l, h, yaw)
            
        Returns:
            Tuple of (augmented_points, augmented_boxes)
        """
        # Apply random rotation
        if self.rotation_range is not None:
            points, boxes = self._random_rotation(points, boxes)
        
        # Apply random scaling
        if self.scaling_range is not None:
            points, boxes = self._random_scaling(points, boxes)
        
        # Apply random translation
        if self.translation_range is not None:
            points, boxes = self._random_translation(points, boxes)
        
        # Apply random horizontal flipping
        if self.flip_probability > 0:
            points, boxes = self._random_flip(points, boxes)
        
        # Apply random point dropout
        if self.dropout_probability > 0:
            points = self._random_dropout(points)
        
        return points, boxes
    
    def _random_rotation(self, points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random rotation around z-axis.
        
        Args:
            points: Nx4+ array of points
            boxes: Optional Mx7 array of 3D boxes
            
        Returns:
            Tuple of (rotated_points, rotated_boxes)
        """
        # Sample rotation angle
        angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        # Rotation matrix
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Rotate points (only x, y, z coordinates)
        rotated_points = points.copy()
        rotated_points[:, :3] = np.dot(rotated_points[:, :3], rot_mat.T)
        
        # Rotate boxes if provided
        if boxes is not None:
            rotated_boxes = boxes.copy()
            # Rotate box centers
            rotated_boxes[:, :3] = np.dot(rotated_boxes[:, :3], rot_mat.T)
            # Update box orientations (yaw angle)
            rotated_boxes[:, 6] = (rotated_boxes[:, 6] + angle) % (2 * np.pi)
            return rotated_points, rotated_boxes
        
        return rotated_points, None
    
    def _random_scaling(self, points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random scaling to point cloud.
        
        Args:
            points: Nx4+ array of points
            boxes: Optional Mx7 array of 3D boxes
            
        Returns:
            Tuple of (scaled_points, scaled_boxes)
        """
        # Sample scaling factor
        scale = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
        
        # Scale points (only x, y, z coordinates)
        scaled_points = points.copy()
        scaled_points[:, :3] *= scale
        
        # Scale boxes if provided
        if boxes is not None:
            scaled_boxes = boxes.copy()
            # Scale box centers
            scaled_boxes[:, :3] *= scale
            # Scale box dimensions (width, length, height)
            scaled_boxes[:, 3:6] *= scale
            return scaled_points, scaled_boxes
        
        return scaled_points, None
    
    def _random_translation(self, points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random translation to point cloud.
        
        Args:
            points: Nx4+ array of points
            boxes: Optional Mx7 array of 3D boxes
            
        Returns:
            Tuple of (translated_points, translated_boxes)
        """
        # Sample translation offsets
        x_offset = np.random.uniform(self.translation_range[0], self.translation_range[1])
        y_offset = np.random.uniform(self.translation_range[2], self.translation_range[3])
        z_offset = np.random.uniform(self.translation_range[4], self.translation_range[5])
        
        # Apply translation to points
        translated_points = points.copy()
        translated_points[:, 0] += x_offset
        translated_points[:, 1] += y_offset
        translated_points[:, 2] += z_offset
        
        # Translate boxes if provided
        if boxes is not None:
            translated_boxes = boxes.copy()
            translated_boxes[:, 0] += x_offset
            translated_boxes[:, 1] += y_offset
            translated_boxes[:, 2] += z_offset
            return translated_points, translated_boxes
        
        return translated_points, None
    
    def _random_flip(self, points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Randomly flip point cloud horizontally (along x-axis).
        
        Args:
            points: Nx4+ array of points
            boxes: Optional Mx7 array of 3D boxes
            
        Returns:
            Tuple of (flipped_points, flipped_boxes)
        """
        if np.random.random() < self.flip_probability:
            # Flip points
            flipped_points = points.copy()
            flipped_points[:, 1] = -flipped_points[:, 1]  # Flip y-coordinate
            
            # Flip boxes if provided
            if boxes is not None:
                flipped_boxes = boxes.copy()
                flipped_boxes[:, 1] = -flipped_boxes[:, 1]  # Flip y-coordinate of center
                flipped_boxes[:, 6] = -flipped_boxes[:, 6]  # Flip yaw angle
                # Normalize yaw angle to [-pi, pi]
                flipped_boxes[:, 6] = ((flipped_boxes[:, 6] + np.pi) % (2 * np.pi)) - np.pi
                return flipped_points, flipped_boxes
            
            return flipped_points, None
        
        return points, boxes
    
    def _random_dropout(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop points from the point cloud.
        
        Args:
            points: Nx4+ array of points
            
        Returns:
            Points with some randomly dropped
        """
        # Create dropout mask
        mask = np.random.random(points.shape[0]) > self.dropout_probability
        return points[mask]


def augment_point_cloud(points: np.ndarray, boxes: Optional[np.ndarray] = None, 
                       **augmentor_kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply random augmentations to a point cloud.
    
    Args:
        points: Nx4+ array of points (x, y, z, intensity, ...)
        boxes: Optional Mx7 array of 3D boxes (x, y, z, w, l, h, yaw)
        **augmentor_kwargs: Keyword arguments for the PointCloudAugmentor
        
    Returns:
        Tuple of (augmented_points, augmented_boxes)
    """
    augmentor = PointCloudAugmentor(**augmentor_kwargs)
    return augmentor.augment(points, boxes)


def global_rotation(points: np.ndarray, angle: float, 
                   boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a fixed rotation around z-axis.
    
    Args:
        points: Nx4+ array of points
        angle: Rotation angle in radians
        boxes: Optional Mx7 array of 3D boxes
        
    Returns:
        Tuple of (rotated_points, rotated_boxes)
    """
    # Create rotation matrix
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Rotate points (only x, y, z coordinates)
    rotated_points = points.copy()
    rotated_points[:, :3] = np.dot(rotated_points[:, :3], rot_mat.T)
    
    # Rotate boxes if provided
    if boxes is not None:
        rotated_boxes = boxes.copy()
        # Rotate box centers
        rotated_boxes[:, :3] = np.dot(rotated_boxes[:, :3], rot_mat.T)
        # Update box orientations (yaw angle)
        rotated_boxes[:, 6] = (rotated_boxes[:, 6] + angle) % (2 * np.pi)
        return rotated_points, rotated_boxes
    
    return rotated_points, None


def global_scaling(points: np.ndarray, scale: float, 
                  boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a fixed scaling to point cloud.
    
    Args:
        points: Nx4+ array of points
        scale: Scaling factor
        boxes: Optional Mx7 array of 3D boxes
        
    Returns:
        Tuple of (scaled_points, scaled_boxes)
    """
    # Scale points (only x, y, z coordinates)
    scaled_points = points.copy()
    scaled_points[:, :3] *= scale
    
    # Scale boxes if provided
    if boxes is not None:
        scaled_boxes = boxes.copy()
        # Scale box centers
        scaled_boxes[:, :3] *= scale
        # Scale box dimensions (width, length, height)
        scaled_boxes[:, 3:6] *= scale
        return scaled_points, scaled_boxes
    
    return scaled_points, None


def global_translation(points: np.ndarray, offset: Tuple[float, float, float], 
                      boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a fixed translation to point cloud.
    
    Args:
        points: Nx4+ array of points
        offset: Translation offset as (x_offset, y_offset, z_offset)
        boxes: Optional Mx7 array of 3D boxes
        
    Returns:
        Tuple of (translated_points, translated_boxes)
    """
    # Apply translation to points
    translated_points = points.copy()
    translated_points[:, 0] += offset[0]
    translated_points[:, 1] += offset[1]
    translated_points[:, 2] += offset[2]
    
    # Translate boxes if provided
    if boxes is not None:
        translated_boxes = boxes.copy()
        translated_boxes[:, 0] += offset[0]
        translated_boxes[:, 1] += offset[1]
        translated_boxes[:, 2] += offset[2]
        return translated_points, translated_boxes
    
    return translated_points, None


def horizontal_flip(points: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Flip point cloud horizontally (along x-axis).
    
    Args:
        points: Nx4+ array of points
        boxes: Optional Mx7 array of 3D boxes
        
    Returns:
        Tuple of (flipped_points, flipped_boxes)
    """
    # Flip points
    flipped_points = points.copy()
    flipped_points[:, 1] = -flipped_points[:, 1]  # Flip y-coordinate
    
    # Flip boxes if provided
    if boxes is not None:
        flipped_boxes = boxes.copy()
        flipped_boxes[:, 1] = -flipped_boxes[:, 1]  # Flip y-coordinate of center
        flipped_boxes[:, 6] = -flipped_boxes[:, 6]  # Flip yaw angle
        # Normalize yaw angle to [-pi, pi]
        flipped_boxes[:, 6] = ((flipped_boxes[:, 6] + np.pi) % (2 * np.pi)) - np.pi
        return flipped_points, flipped_boxes
    
    return flipped_points, None


def random_dropout(points: np.ndarray, dropout_probability: float = 0.05) -> np.ndarray:
    """
    Randomly drop points from the point cloud.
    
    Args:
        points: Nx4+ array of points
        dropout_probability: Probability of dropping each point
        
    Returns:
        Points with some randomly dropped
    """
    # Create dropout mask
    mask = np.random.random(points.shape[0]) > dropout_probability
    return points[mask]


def jitter_point_cloud(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """
    Add random jitter to point cloud.
    
    Args:
        points: Nx4+ array of points
        sigma: Standard deviation of Gaussian noise
        clip: Maximum absolute value of noise
        
    Returns:
        Points with jitter added
    """
    jittered_points = points.copy()
    
    # Only add jitter to x, y, z coordinates
    noise = np.clip(sigma * np.random.randn(*jittered_points[:, :3].shape), -clip, clip)
    jittered_points[:, :3] += noise
    
    return jittered_points


def mix_point_clouds(points1: np.ndarray, points2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Mix two point clouds by randomly selecting points from each based on alpha.
    
    Args:
        points1: Nx4+ array of points from first point cloud
        points2: Mx4+ array of points from second point cloud
        alpha: Proportion of points to take from points1 (0 to 1)
        
    Returns:
        Mixed point cloud
    """
    # Ensure both point clouds have the same number of features
    assert points1.shape[1] == points2.shape[1], "Point clouds must have the same number of features"
    
    # Determine how many points to take from each cloud
    n1 = points1.shape[0]
    n2 = points2.shape[0]
    
    # Target size (average of input sizes)
    target_size = int((n1 + n2) / 2)
    
    # Number of points to take from points1
    num_from_points1 = int(alpha * target_size)
    # Number of points to take from points2
    num_from_points2 = target_size - num_from_points1
    
    # Randomly select points from each cloud
    if num_from_points1 > 0:
        indices1 = np.random.choice(n1, num_from_points1, replace=(num_from_points1 > n1))
        selected1 = points1[indices1]
    else:
        selected1 = np.empty((0, points1.shape[1]), dtype=points1.dtype)
    
    if num_from_points2 > 0:
        indices2 = np.random.choice(n2, num_from_points2, replace=(num_from_points2 > n2))
        selected2 = points2[indices2]
    else:
        selected2 = np.empty((0, points2.shape[1]), dtype=points2.dtype)
    
    # Combine the selected points
    mixed_points = np.vstack([selected1, selected2])
    
    return mixed_points


def ground_removal(points: np.ndarray, height_threshold: float = 0.2, 
                  ground_height: float = -1.5) -> np.ndarray:
    """
    Remove ground points based on a simple height threshold.
    
    Args:
        points: Nx4+ array of points
        height_threshold: Height threshold above estimated ground plane
        ground_height: Estimated ground height
        
    Returns:
        Points with ground removed
    """
    # Filter points based on height threshold
    mask = points[:, 2] > (ground_height + height_threshold)
    return points[mask]


def random_jitter_boxes(boxes: np.ndarray, 
                       position_jitter: float = 0.1, 
                       dimension_jitter: float = 0.05, 
                       angle_jitter: float = 0.05) -> np.ndarray:
    """
    Add random jitter to box parameters for data augmentation.
    
    Args:
        boxes: Mx7 array of 3D boxes (x, y, z, w, l, h, yaw)
        position_jitter: Maximum position jitter in meters
        dimension_jitter: Maximum dimension jitter as a fraction
        angle_jitter: Maximum angle jitter in radians
        
    Returns:
        Boxes with jitter added
    """
    jittered_boxes = boxes.copy()
    
    # Add position jitter
    position_noise = position_jitter * np.random.uniform(-1, 1, size=(len(boxes), 3))
    jittered_boxes[:, :3] += position_noise
    
    # Add dimension jitter
    dimension_noise = 1.0 + dimension_jitter * np.random.uniform(-1, 1, size=(len(boxes), 3))
    jittered_boxes[:, 3:6] *= dimension_noise
    
    # Add angle jitter
    angle_noise = angle_jitter * np.random.uniform(-1, 1, size=len(boxes))
    jittered_boxes[:, 6] += angle_noise
    
    # Normalize angles to [-pi, pi]
    jittered_boxes[:, 6] = ((jittered_boxes[:, 6] + np.pi) % (2 * np.pi)) - np.pi
    
    return jittered_boxes


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a sample point cloud
    num_points = 1000
    x = np.random.uniform(-5, 5, num_points)
    y = np.random.uniform(-5, 5, num_points)
    z = np.random.uniform(-1, 1, num_points)
    intensity = np.random.uniform(0, 1, num_points)
    
    points = np.column_stack([x, y, z, intensity])
    
    # Create a sample bounding box
    box = np.array([[0, 0, 0, 2, 3, 1.5, 0]])  # [x, y, z, w, l, h, yaw]
    
    # Apply augmentation
    augmented_points, augmented_box = augment_point_cloud(points, box)
    
    # Visualize original and augmented point clouds
    fig = plt.figure(figsize=(12, 5))
    
    # Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Augmented point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(augmented_points[:, 0], augmented_points[:, 1], augmented_points[:, 2], 
               c=augmented_points[:, 3], s=1)
    ax2.set_title('Augmented Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()