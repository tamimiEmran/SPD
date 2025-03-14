"""
transform module for V2X-Seq project.

This module provides functionality for transforming point clouds and bounding boxes
between different coordinate systems, which is essential for fusing data from
vehicle and infrastructure sensors in the V2X-Seq dataset.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def transform_points_to_veh_coordinate(points: np.ndarray,
                                      transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points from one coordinate system to another using a transformation matrix.
    
    Args:
        points: Nx3 or Nx4 array of points (x, y, z, intensity)
        transformation_matrix: 4x4 transformation matrix
        
    Returns:
        Transformed points
    """
    # Check if points include intensity
    has_intensity = points.shape[1] >= 4
    
    # Extract xyz coordinates
    xyz = points[:, :3]
    
    # Convert to homogeneous coordinates
    num_points = xyz.shape[0]
    homogeneous_points = np.ones((num_points, 4))
    homogeneous_points[:, :3] = xyz
    
    # Apply transformation
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)
    
    # Convert back to original format
    if has_intensity:
        result = np.zeros((num_points, points.shape[1]))
        result[:, :3] = transformed_points[:, :3]
        result[:, 3:] = points[:, 3:]
        return result
    else:
        return transformed_points[:, :3]


def transform_boxes_to_veh_coordinate(boxes: np.ndarray,
                                     transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform 3D bounding boxes from one coordinate system to another.
    
    Args:
        boxes: Nx7 array of boxes (x, y, z, w, l, h, yaw)
        transformation_matrix: 4x4 transformation matrix
        
    Returns:
        Transformed boxes
    """
    # Make a copy to avoid modifying the original
    transformed_boxes = boxes.copy()
    num_boxes = boxes.shape[0]
    
    # Transform center points
    centers = np.ones((num_boxes, 4))
    centers[:, :3] = boxes[:, :3]
    transformed_centers = np.dot(centers, transformation_matrix.T)
    transformed_boxes[:, :3] = transformed_centers[:, :3]
    
    # Extract rotation part of the transformation
    rotation_matrix = transformation_matrix[:3, :3]
    
    # Transform orientation (yaw angle)
    for i in range(num_boxes):
        # Original orientation in the source coordinate
        original_yaw = boxes[i, 6]
        
        # Create rotation matrix from yaw
        original_rotation = np.array([
            [np.cos(original_yaw), -np.sin(original_yaw), 0],
            [np.sin(original_yaw), np.cos(original_yaw), 0],
            [0, 0, 1]
        ])
        
        # Apply transformation rotation to the box rotation
        transformed_rotation = np.dot(rotation_matrix, original_rotation)
        
        # Extract yaw from transformed rotation
        transformed_yaw = np.arctan2(transformed_rotation[1, 0], transformed_rotation[0, 0])
        transformed_boxes[i, 6] = transformed_yaw
    
    return transformed_boxes


def get_transformation_matrix(rotation: Union[np.ndarray, List],
                            translation: Union[np.ndarray, List]) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from rotation and translation.
    
    Args:
        rotation: 3x3 rotation matrix or 9-element list/array
        translation: 3-element translation vector
        
    Returns:
        4x4 transformation matrix
    """
    # Initialize transformation matrix
    transformation_matrix = np.eye(4)
    
    # Set rotation part
    if isinstance(rotation, list):
        rotation = np.array(rotation).reshape(3, 3)
    transformation_matrix[:3, :3] = rotation
    
    # Set translation part
    if isinstance(translation, list):
        translation = np.array(translation)
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix


def load_transformation_from_json(json_data: Dict) -> np.ndarray:
    """
    Load transformation matrix from a JSON dictionary.
    
    Args:
        json_data: Dictionary containing 'rotation' and 'translation' keys
        
    Returns:
        4x4 transformation matrix
    """
    rotation = np.array(json_data['rotation']).reshape(3, 3)
    translation = np.array(json_data['translation'])
    
    return get_transformation_matrix(rotation, translation)


def inverse_transform_matrix(transform_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a transformation matrix.
    
    Args:
        transform_matrix: 4x4 transformation matrix
        
    Returns:
        Inverse transformation matrix
    """
    # Initialize inverse matrix
    inverse_matrix = np.eye(4)
    
    # Extract rotation and translation
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    
    # Invert rotation (transpose for orthogonal matrices)
    inverse_rotation = rotation.T
    
    # Invert translation
    inverse_translation = -np.dot(inverse_rotation, translation)
    
    # Set rotation and translation in inverse matrix
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation
    
    return inverse_matrix


def compose_transforms(transform1: np.ndarray, transform2: np.ndarray) -> np.ndarray:
    """
    Compose two transformation matrices.
    
    Args:
        transform1: First 4x4 transformation matrix
        transform2: Second 4x4 transformation matrix
        
    Returns:
        Composed transformation matrix (transform2 * transform1)
    """
    return np.dot(transform2, transform1)


def transform_points_to_world(points: np.ndarray,
                            lidar_to_novatel: np.ndarray,
                            novatel_to_world: np.ndarray) -> np.ndarray:
    """
    Transform points from LiDAR coordinate system to world coordinate system.
    
    Args:
        points: Nx3 or Nx4 array of points
        lidar_to_novatel: 4x4 transformation from LiDAR to Novatel
        novatel_to_world: 4x4 transformation from Novatel to world
        
    Returns:
        Points in world coordinate system
    """
    # Compose transformations
    lidar_to_world = compose_transforms(lidar_to_novatel, novatel_to_world)
    
    # Transform points
    return transform_points_to_veh_coordinate(points, lidar_to_world)


def transform_inf_points_to_veh(inf_points: np.ndarray,
                             inf_to_world: np.ndarray,
                             veh_to_world: np.ndarray) -> np.ndarray:
    """
    Transform infrastructure points to vehicle coordinate system.
    
    Args:
        inf_points: Nx3 or Nx4 array of infrastructure points
        inf_to_world: 4x4 transformation from infrastructure to world
        veh_to_world: 4x4 transformation from vehicle to world
        
    Returns:
        Infrastructure points in vehicle coordinate system
    """
    # Get world to vehicle transformation (inverse of vehicle to world)
    world_to_veh = inverse_transform_matrix(veh_to_world)
    
    # Compose transformations: inf -> world -> vehicle
    inf_to_veh = compose_transforms(inf_to_world, world_to_veh)
    
    # Transform points
    return transform_points_to_veh_coordinate(inf_points, inf_to_veh)


def transform_veh_points_to_inf(veh_points: np.ndarray,
                             veh_to_world: np.ndarray,
                             inf_to_world: np.ndarray) -> np.ndarray:
    """
    Transform vehicle points to infrastructure coordinate system.
    
    Args:
        veh_points: Nx3 or Nx4 array of vehicle points
        veh_to_world: 4x4 transformation from vehicle to world
        inf_to_world: 4x4 transformation from infrastructure to world
        
    Returns:
        Vehicle points in infrastructure coordinate system
    """
    # Get world to infrastructure transformation (inverse of infrastructure to world)
    world_to_inf = inverse_transform_matrix(inf_to_world)
    
    # Compose transformations: vehicle -> world -> infrastructure
    veh_to_inf = compose_transforms(veh_to_world, world_to_inf)
    
    # Transform points
    return transform_points_to_veh_coordinate(veh_points, veh_to_inf)


def transform_boxes_to_world(boxes: np.ndarray,
                           lidar_to_novatel: np.ndarray,
                           novatel_to_world: np.ndarray) -> np.ndarray:
    """
    Transform boxes from LiDAR coordinate system to world coordinate system.
    
    Args:
        boxes: Nx7 array of boxes (x, y, z, w, l, h, yaw)
        lidar_to_novatel: 4x4 transformation from LiDAR to Novatel
        novatel_to_world: 4x4 transformation from Novatel to world
        
    Returns:
        Boxes in world coordinate system
    """
    # Compose transformations
    lidar_to_world = compose_transforms(lidar_to_novatel, novatel_to_world)
    
    # Transform boxes
    return transform_boxes_to_veh_coordinate(boxes, lidar_to_world)


def transform_inf_boxes_to_veh(inf_boxes: np.ndarray,
                            inf_to_world: np.ndarray,
                            veh_to_world: np.ndarray) -> np.ndarray:
    """
    Transform infrastructure boxes to vehicle coordinate system.
    
    Args:
        inf_boxes: Nx7 array of infrastructure boxes
        inf_to_world: 4x4 transformation from infrastructure to world
        veh_to_world: 4x4 transformation from vehicle to world
        
    Returns:
        Infrastructure boxes in vehicle coordinate system
    """
    # Get world to vehicle transformation (inverse of vehicle to world)
    world_to_veh = inverse_transform_matrix(veh_to_world)
    
    # Compose transformations: inf -> world -> vehicle
    inf_to_veh = compose_transforms(inf_to_world, world_to_veh)
    
    # Transform boxes
    return transform_boxes_to_veh_coordinate(inf_boxes, inf_to_veh)


def project_points_to_image(points: np.ndarray,
                          intrinsic_matrix: np.ndarray,
                          extrinsic_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points: Nx3 array of 3D points
        intrinsic_matrix: 3x3 camera intrinsic matrix
        extrinsic_matrix: 4x4 extrinsic matrix (LiDAR to camera)
        
    Returns:
        Tuple of (projected points as Nx2 array, mask of visible points)
    """
    # Transform points to camera coordinate
    points_homogeneous = np.ones((points.shape[0], 4))
    points_homogeneous[:, :3] = points
    points_camera = np.dot(points_homogeneous, extrinsic_matrix.T)
    
    # Filter points in front of the camera
    mask = points_camera[:, 2] > 0
    
    # Project to image plane
    points_image = np.zeros((points.shape[0], 2))
    
    if np.any(mask):
        # Normalize by z-coordinate
        points_normalized = points_camera[mask, :3] / points_camera[mask, 2:]
        
        # Apply camera intrinsic matrix
        points_image_homogeneous = np.dot(points_normalized, intrinsic_matrix.T)
        
        # Get image coordinates
        points_image[mask, 0] = points_image_homogeneous[:, 0]
        points_image[mask, 1] = points_image_homogeneous[:, 1]
    
    return points_image, mask


def get_box_corners(box: np.ndarray) -> np.ndarray:
    """
    Convert a 3D bounding box to 8 corner points.
    
    Args:
        box: 7-element array [x, y, z, w, l, h, yaw]
        
    Returns:
        8x3 array of corner coordinates
    """
    # Extract box parameters
    x, y, z, w, l, h, yaw = box
    
    # Create corners in object frame
    corners = np.array([
        [l/2, w/2, h/2],
        [l/2, w/2, -h/2],
        [l/2, -w/2, h/2],
        [l/2, -w/2, -h/2],
        [-l/2, w/2, h/2],
        [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],
        [-l/2, -w/2, -h/2]
    ])
    
    # Rotation matrix
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotate corners
    corners = np.dot(corners, rot.T)
    
    # Translate corners
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    
    return corners


def box_to_image(box: np.ndarray,
                intrinsic_matrix: np.ndarray,
                extrinsic_matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Project a 3D bounding box to 2D image coordinates.
    
    Args:
        box: 7-element array [x, y, z, w, l, h, yaw]
        intrinsic_matrix: 3x3 camera intrinsic matrix
        extrinsic_matrix: 4x4 extrinsic matrix (LiDAR to camera)
        
    Returns:
        Tuple of (projected corners as 8x2 array, visibility flag)
    """
    # Get box corners
    corners_3d = get_box_corners(box)
    
    # Project corners to image
    corners_2d, mask = project_points_to_image(corners_3d, intrinsic_matrix, extrinsic_matrix)
    
    # Check if any corner is visible
    is_visible = np.any(mask)
    
    return corners_2d, is_visible


def merge_point_clouds(pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    """
    Merge two point clouds.
    
    Args:
        pc1: First point cloud as Nx4 array (x, y, z, intensity)
        pc2: Second point cloud as Mx4 array (x, y, z, intensity)
        
    Returns:
        Merged point cloud as (N+M)x4 array
    """
    # Ensure both point clouds have the same number of features
    if pc1.shape[1] != pc2.shape[1]:
        raise ValueError(f"Point clouds must have the same number of features, "
                        f"got {pc1.shape[1]} and {pc2.shape[1]}")
    
    # Concatenate point clouds
    return np.vstack([pc1, pc2])


def filter_points_in_range(points: np.ndarray, 
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float],
                        z_range: Tuple[float, float]) -> np.ndarray:
    """
    Filter points within a specified range.
    
    Args:
        points: Nx3+ array of points
        x_range: (min_x, max_x) range
        y_range: (min_y, max_y) range
        z_range: (min_z, max_z) range
        
    Returns:
        Filtered points
    """
    # Create mask for points within range
    mask = ((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
           (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1]))
    
    # Apply mask
    return points[mask]


def filter_points_in_frustum(points: np.ndarray,
                          intrinsic_matrix: np.ndarray,
                          extrinsic_matrix: np.ndarray,
                          image_size: Tuple[int, int],
                          margin: int = 0) -> np.ndarray:
    """
    Filter points that project within camera frustum.
    
    Args:
        points: Nx3+ array of points
        intrinsic_matrix: 3x3 camera intrinsic matrix
        extrinsic_matrix: 4x4 extrinsic matrix (LiDAR to camera)
        image_size: (width, height) of the image
        margin: Optional margin to extend the image boundary
        
    Returns:
        Filtered points
    """
    # Project points to image
    image_points, mask = project_points_to_image(points[:, :3], intrinsic_matrix, extrinsic_matrix)
    
    # Filter points that are in front of camera
    points_in_front = points[mask]
    image_points = image_points[mask]
    
    # Filter points within image bounds (with margin)
    width, height = image_size
    img_mask = ((image_points[:, 0] >= -margin) & 
               (image_points[:, 0] < width + margin) &
               (image_points[:, 1] >= -margin) & 
               (image_points[:, 1] < height + margin))
    
    return points_in_front[img_mask]


def downsample_point_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Nx3+ array of points
        voxel_size: Size of voxel grid for downsampling
        
    Returns:
        Downsampled point cloud
    """
    # Create dictionary to store points in each voxel
    voxel_dict = {}
    
    # Quantize points to voxel coordinates
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)
    
    # Create hashable voxel index
    voxel_hash = np.array([
        voxel_indices[:, 0] * 100000 + voxel_indices[:, 1] * 100 + voxel_indices[:, 2]
    ]).T
    
    # Group points by voxel
    for i, hash_value in enumerate(voxel_hash):
        h = hash_value[0]
        if h in voxel_dict:
            voxel_dict[h].append(i)
        else:
            voxel_dict[h] = [i]
    
    # Take centroid of points in each voxel
    downsampled_points = []
    for indices in voxel_dict.values():
        voxel_points = points[indices]
        centroid = np.mean(voxel_points, axis=0)
        downsampled_points.append(centroid)
    
    return np.array(downsampled_points)


if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=3, suppress=True)
    
    # Create example points
    points = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8],
        [7.0, 8.0, 9.0, 0.3]
    ])
    
    # Create example transformation matrix
    rotation = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    translation = np.array([10, 20, 5])
    transform = get_transformation_matrix(rotation, translation)
    
    print("Original points:")
    print(points)
    
    # Transform points
    transformed_points = transform_points_to_veh_coordinate(points, transform)
    print("\nTransformed points:")
    print(transformed_points)
    
    # Create example 3D box
    box = np.array([5.0, 5.0, 0.0, 3.0, 6.0, 2.0, np.pi/4])
    
    print("\nOriginal box:")
    print(box)
    
    # Transform box
    transformed_box = transform_boxes_to_veh_coordinate(box.reshape(1, 7), transform)
    print("\nTransformed box:")
    print(transformed_box[0])