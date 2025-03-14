"""
coordinate_transform module for V2X-Seq project.

This module provides functionality for coordinate transform.
"""

# Add your code here
"""
coordinate_transform module for V2X-Seq project.

This module provides functionality for transforming between different coordinate
systems used in the V2X-Seq dataset, including infrastructure-to-vehicle transformations, 
LiDAR-to-camera transformations, and other coordinate system conversions.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional
import json
import os


def load_transformation_matrix(json_path: str) -> np.ndarray:
    """
    Load transformation matrix from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing transformation parameters
        
    Returns:
        4x4 transformation matrix as numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract rotation and translation
    if 'rotation' in data and 'translation' in data:
        # Direct rotation matrix and translation vector format
        rotation = np.array(data['rotation']).reshape(3, 3)
        translation = np.array(data['translation']).reshape(3, 1)
    elif 'R' in data and 't' in data:
        # Alternative naming in some files
        rotation = np.array(data['R']).reshape(3, 3)
        translation = np.array(data['t']).reshape(3, 1)
    else:
        raise ValueError(f"Unexpected format in {json_path}. Missing rotation and translation.")
    
    # Construct 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3:4] = translation
    
    return transformation
# Add this at the end of coordinate_transform.py
get_transformation_matrix = load_transformation_matrix  # Alias for compatibility

def inverse_transform_matrix(transform: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 transformation matrix.
    
    Args:
        transform: 4x4 transformation matrix
        
    Returns:
        4x4 inverse transformation matrix
    """
    inverse = np.eye(4)
    
    # Rotation part is transposed (for orthogonal rotation matrices)
    inverse[:3, :3] = transform[:3, :3].T
    
    # Translation part is negated and rotated
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    
    return inverse


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using a transformation matrix.
    
    Args:
        points: Array of shape (N, 3) where N is the number of points
        transform: 4x4 transformation matrix
        
    Returns:
        Transformed points as an array of shape (N, 3)
    """
    # Convert to homogeneous coordinates
    if points.shape[1] == 3:
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        raise ValueError(f"Expected points with 3 coordinates, got {points.shape[1]}")
    
    # Apply transformation
    transformed_points = homogeneous_points @ transform.T
    
    # Convert back from homogeneous coordinates
    return transformed_points[:, :3]


def transform_boxes(boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform 3D bounding boxes using a transformation matrix.
    
    Args:
        boxes: Array of shape (N, 7) where each row is [x, y, z, w, l, h, yaw]
               representing the center, dimensions, and orientation of boxes
        transform: 4x4 transformation matrix
        
    Returns:
        Transformed boxes as an array of shape (N, 7)
    """
    transformed_boxes = boxes.copy()
    
    for i in range(len(boxes)):
        # Transform center points
        center = boxes[i, :3].reshape(1, 3)
        transformed_center = transform_points(center, transform)
        transformed_boxes[i, :3] = transformed_center.flatten()
        
        # Transform orientation (yaw angle)
        # Extract rotation around z-axis from transformation matrix
        rotation_z = np.arctan2(transform[1, 0], transform[0, 0])
        transformed_boxes[i, 6] = (boxes[i, 6] + rotation_z) % (2 * np.pi)
    
    return transformed_boxes


def infrastructure_to_vehicle_transform(calib_data: Dict) -> np.ndarray:
    """
    Compute transformation matrix from infrastructure to vehicle coordinate system.
    
    The transformation chain is:
    infrastructure_virtuallidar -> world -> vehicle_novatel -> vehicle_lidar
    
    Args:
        calib_data: Dictionary containing paths to calibration files
            - 'virtuallidar_to_world': Path to infrastructure LiDAR to world transform
            - 'novatel_to_world': Path to vehicle Novatel to world transform
            - 'lidar_to_novatel': Path to vehicle LiDAR to Novatel transform
            
    Returns:
        4x4 transformation matrix from infrastructure LiDAR to vehicle LiDAR
    """
    # Load individual transformation matrices
    infrastructure_to_world = load_transformation_matrix(calib_data['virtuallidar_to_world'])
    novatel_to_world = load_transformation_matrix(calib_data['novatel_to_world'])
    lidar_to_novatel = load_transformation_matrix(calib_data['lidar_to_novatel'])
    
    # Compute inverse transforms for the world to vehicle chain
    world_to_novatel = inverse_transform_matrix(novatel_to_world)
    
    # Combine transformations
    # infrastructure -> world -> novatel -> lidar
    infrastructure_to_novatel = world_to_novatel @ infrastructure_to_world
    infrastructure_to_vehicle = inverse_transform_matrix(lidar_to_novatel) @ infrastructure_to_novatel
    
    return infrastructure_to_vehicle


def lidar_to_camera_transform(calib_data: Dict, is_infrastructure: bool = False) -> np.ndarray:
    """
    Compute transformation matrix from LiDAR to camera coordinate system.
    
    Args:
        calib_data: Dictionary containing paths to calibration files
            - 'lidar_to_camera' or 'virtuallidar_to_camera': Path to LiDAR to camera transform
        is_infrastructure: Whether the transformation is for infrastructure side
            
    Returns:
        4x4 transformation matrix from LiDAR to camera
    """
    # Select appropriate calibration file
    if is_infrastructure:
        lidar_to_camera_path = calib_data['virtuallidar_to_camera']
    else:
        lidar_to_camera_path = calib_data['lidar_to_camera']
    
    # Load transformation matrix
    lidar_to_camera = load_transformation_matrix(lidar_to_camera_path)
    
    return lidar_to_camera


def project_points_to_image(points: np.ndarray, transform: np.ndarray, 
                           intrinsic_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points: Array of shape (N, 3) containing 3D points in LiDAR coordinate system
        transform: 4x4 transformation matrix from LiDAR to camera
        intrinsic_path: Path to camera intrinsic parameter file
        
    Returns:
        Tuple of:
            - Array of shape (N, 2) containing projected 2D points
            - Boolean mask of points that are visible in the image
    """
    # Load camera intrinsic parameters
    with open(intrinsic_path, 'r') as f:
        intrinsic_data = json.load(f)
    
    # Extract camera matrix
    if 'cam_K' in intrinsic_data:
        camera_matrix = np.array(intrinsic_data['cam_K']).reshape(3, 3)
    elif 'P' in intrinsic_data:  # Alternative format
        P = np.array(intrinsic_data['P']).reshape(3, 4)
        camera_matrix = P[:, :3]
    else:
        raise ValueError(f"Unexpected format in {intrinsic_path}. Missing camera matrix.")
    
    # Transform points to camera coordinate system
    points_camera = transform_points(points, transform)
    
    # Filter points in front of the camera
    mask = points_camera[:, 2] > 0
    
    # Project to image coordinates using camera matrix
    points_2d_homogeneous = np.zeros((points_camera.shape[0], 3))
    points_2d_homogeneous[mask] = points_camera[mask] @ camera_matrix.T
    
    # Convert from homogeneous coordinates
    points_2d = np.zeros((points_camera.shape[0], 2))
    points_2d[mask, 0] = points_2d_homogeneous[mask, 0] / points_2d_homogeneous[mask, 2]
    points_2d[mask, 1] = points_2d_homogeneous[mask, 1] / points_2d_homogeneous[mask, 2]
    
    return points_2d, mask


def compensate_latency(infra_boxes: np.ndarray, vehicle_boxes: np.ndarray, 
                       latency: float, infra_velocities: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compensate for latency in infrastructure data by predicting forward.
    
    Args:
        infra_boxes: Array of shape (N, 7) containing infrastructure boxes
        vehicle_boxes: Array of shape (M, 7) containing vehicle boxes
        latency: Time delay in seconds between infrastructure and vehicle data
        infra_velocities: Optional array of shape (N, 3) containing velocity estimates
                         for infrastructure objects
        
    Returns:
        Latency-compensated infrastructure boxes as an array of shape (N, 7)
    """
    if infra_velocities is None:
        # If no velocity estimates provided, try to match with vehicle boxes to estimate motion
        # This is a simple implementation - in practice, more sophisticated matching and
        # motion estimation would be needed
        compensated_boxes = infra_boxes.copy()
        
        # Basic forward prediction assuming constant velocity
        # In a real implementation, you would use object matching and tracking
        # to get better velocity estimates
        for i in range(len(infra_boxes)):
            # Simple forward prediction using a default speed estimate
            # In real implementation, you would use actual tracked velocities
            # This is just a placeholder example
            # Move forward in the direction of orientation at an estimated speed
            yaw = infra_boxes[i, 6]
            estimated_speed = 5.0  # meters per second, just a placeholder
            
            dx = np.cos(yaw) * estimated_speed * latency
            dy = np.sin(yaw) * estimated_speed * latency
            
            compensated_boxes[i, 0] += dx
            compensated_boxes[i, 1] += dy
    else:
        # Use provided velocity estimates for more accurate prediction
        compensated_boxes = infra_boxes.copy()
        
        for i in range(len(infra_boxes)):
            # Move each box according to its velocity
            compensated_boxes[i, :3] += infra_velocities[i] * latency
    
    return compensated_boxes


def transform_ego_motion(transform1: np.ndarray, transform2: np.ndarray) -> np.ndarray:
    """
    Compute the relative transformation between two ego poses.
    
    Args:
        transform1: 4x4 transformation matrix at time 1
        transform2: 4x4 transformation matrix at time 2
        
    Returns:
        4x4 relative transformation matrix from pose 1 to pose 2
    """
    # T_2^-1 * T_1 gives the transformation from pose 1 to pose 2
    return inverse_transform_matrix(transform2) @ transform1




#%% 
import numpy as np
import os
import json

# Create a directory for test files
os.makedirs("test_files", exist_ok=True)

def test_load_transformation_matrix():
    """Test loading transformation matrix from JSON file."""
    # Create a test calibration file
    test_calib = {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [10, 20, 30]
    }
    test_file = "test_files/test_transform.json"
    with open(test_file, "w") as f:
        json.dump(test_calib, f)
    
    # Load the transformation matrix
    transform = load_transformation_matrix(test_file)
    
    # Expected result
    expected = np.eye(4)
    expected[:3, 3] = [10, 20, 30]
    
    # Check result
    np.testing.assert_array_almost_equal(transform, expected)
    print("Test load_transformation_matrix passed!")


def test_inverse_transform_matrix():
    """Test inverting a transformation matrix."""
    # Create a test transformation matrix
    test_transform = np.array([
        [0, -1, 0, 5],
        [1, 0, 0, 10],
        [0, 0, 1, 15],
        [0, 0, 0, 1]
    ])
    
    # Compute the inverse
    inverse = inverse_transform_matrix(test_transform)
    
    # Check if T * T^-1 = I
    result = test_transform @ inverse
    np.testing.assert_array_almost_equal(result, np.eye(4))
    print("Test inverse_transform_matrix passed!")


def test_transform_points():
    """Test transforming 3D points."""
    # Create test points (unit cube vertices)
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    
    # Create a transformation matrix (90-degree rotation around Z + translation)
    transform = np.array([
        [0, -1, 0, 5],
        [1, 0, 0, 10],
        [0, 0, 1, 15],
        [0, 0, 0, 1]
    ])
    
    # Transform the points
    transformed = transform_points(points, transform)
    
    # Check a few transformed points
    # [0,0,0] should transform to [5,10,15]
    np.testing.assert_array_almost_equal(transformed[0], [5, 10, 15])
    # [1,0,0] should transform to [5,11,15]
    np.testing.assert_array_almost_equal(transformed[1], [5, 11, 15])
    print("Test transform_points passed!")


def test_transform_boxes():
    """Test transforming 3D bounding boxes."""
    # Create test boxes [x, y, z, width, length, height, yaw]
    boxes = np.array([
        [0, 0, 0, 2, 4, 1.5, 0],
        [10, 5, 0, 2, 5, 1.8, np.pi/4]
    ])
    
    # Create a transformation matrix (90-degree rotation around Z + translation)
    transform = np.array([
        [0, -1, 0, 5],
        [1, 0, 0, 10],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Transform the boxes
    transformed = transform_boxes(boxes, transform)
    
    # Check center of first box: [0,0,0] -> [5,10,0]
    np.testing.assert_array_almost_equal(transformed[0, :3], [5, 10, 0])
    # Check yaw of first box: 0 -> π/2
    np.testing.assert_almost_equal(transformed[0, 6], np.pi/2 % (2*np.pi))
    print("Test transform_boxes passed!")


def test_infrastructure_to_vehicle_transform():
    """Test computing transformation from infrastructure to vehicle coordinate system."""
    # Create test calibration files
    infra_to_world = {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [100, 200, 0]
    }
    novatel_to_world = {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [50, 100, 0]
    }
    lidar_to_novatel = {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [0, 0, 2]
    }
    
    test_files = {
        "virtuallidar_to_world": "test_files/infra_to_world.json",
        "novatel_to_world": "test_files/novatel_to_world.json",
        "lidar_to_novatel": "test_files/lidar_to_novatel.json"
    }
    
    with open(test_files["virtuallidar_to_world"], "w") as f:
        json.dump(infra_to_world, f)
    with open(test_files["novatel_to_world"], "w") as f:
        json.dump(novatel_to_world, f)
    with open(test_files["lidar_to_novatel"], "w") as f:
        json.dump(lidar_to_novatel, f)
    
    # Compute the transformation
    transform = infrastructure_to_vehicle_transform(test_files)
    
    # Expected transformation: infra -> world -> novatel -> lidar
    # Translation should be (100-50, 200-100, -2) = (50, 100, -2)
    expected = np.eye(4)
    expected[:3, 3] = [50, 100, -2]
    
    np.testing.assert_array_almost_equal(transform, expected)
    print("Test infrastructure_to_vehicle_transform passed!")


def test_lidar_to_camera_transform():
    """Test computing transformation from LiDAR to camera coordinate system."""
    # Create test calibration file
    lidar_to_camera = {
        "rotation": [0, -1, 0, 0, 0, -1, 1, 0, 0],  # LiDAR to camera coordinate change
        "translation": [0, 0.2, 0.5]
    }
    
    # Save to file
    test_file = "test_files/lidar_to_camera.json"
    with open(test_file, "w") as f:
        json.dump(lidar_to_camera, f)
    
    # Test for vehicle side
    calib_data = {"lidar_to_camera": test_file}
    transform = lidar_to_camera_transform(calib_data, is_infrastructure=False)
    
    # Check result
    expected = np.zeros((4, 4))
    expected[:3, :3] = np.array([0, -1, 0, 0, 0, -1, 1, 0, 0]).reshape(3, 3)
    expected[:3, 3] = [0, 0.2, 0.5]
    expected[3, 3] = 1
    
    np.testing.assert_array_almost_equal(transform, expected)
    
    # Test for infrastructure side
    calib_data = {"virtuallidar_to_camera": test_file}
    transform = lidar_to_camera_transform(calib_data, is_infrastructure=True)
    np.testing.assert_array_almost_equal(transform, expected)
    
    print("Test lidar_to_camera_transform passed!")

def test_project_points_to_image():
    """Test projecting 3D points to 2D image coordinates."""
    # Create test camera intrinsic file
    camera_intrinsic = {
        "cam_K": [800, 0, 640, 0, 800, 360, 0, 0, 1]  # Simple pinhole camera model
    }
    
    test_file = "test_files/camera_intrinsic.json"
    with open(test_file, "w") as f:
        json.dump(camera_intrinsic, f)
    
    # Create test points in LiDAR frame
    points = np.array([
        [10, 0, 0],   # Point in front of camera
        [0, 0, -10],  # Point behind camera
        [5, 1, 2]     # Another point in front
    ])
    
    # Create LiDAR to camera transform (swap axes for typical LiDAR->camera transform)
    transform = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Project points
    points_2d, mask = project_points_to_image(points, transform, test_file)
    
    # Check results
    # First point [10,0,0] in LiDAR becomes [0,0,10] in camera, projecting to [640,360]
    # Third point [5,1,2] in LiDAR becomes [-1,-2,5] in camera, projecting to [480,40]
    expected_points = np.array([
        [640, 360],
        [0, 0],  # Ignored point (behind camera)
        [480, 40]
    ])
    expected_mask = np.array([True, False, True])
    
    np.testing.assert_array_almost_equal(points_2d[mask], expected_points[expected_mask])
    np.testing.assert_array_equal(mask, expected_mask)
    print("Test project_points_to_image passed!")


def test_compensate_latency():
    """Test compensating for latency in infrastructure data."""
    # Create test boxes
    infra_boxes = np.array([
        [10, 20, 0, 2, 4, 1.5, 0],  # Box at (10,20) with 0 yaw (facing along x-axis)
        [30, 40, 0, 2, 5, 1.8, np.pi/2]  # Box at (30,40) with π/2 yaw (facing along y-axis)
    ])
    
    vehicle_boxes = np.array([])  # Not used in this example
    latency = 0.2  # 200ms
    
    # Test with velocity estimates
    infra_velocities = np.array([
        [5, 0, 0],  # Moving at 5 m/s along x-axis
        [0, 10, 0]  # Moving at 10 m/s along y-axis
    ])
    
    compensated = compensate_latency(infra_boxes, vehicle_boxes, latency, infra_velocities)
    
    # Expected: first box moves 5*0.2=1 meter along x, second box moves 10*0.2=2 meters along y
    expected = np.array([
        [11, 20, 0, 2, 4, 1.5, 0],
        [30, 42, 0, 2, 5, 1.8, np.pi/2]
    ])
    
    np.testing.assert_array_almost_equal(compensated, expected)
    print("Test compensate_latency passed!")


# Run all tests
if __name__ == "__main__":
    print("Running tests for coordinate_transform functions...")
    test_load_transformation_matrix()
    test_inverse_transform_matrix()
    test_transform_points()
    test_transform_boxes()
    test_infrastructure_to_vehicle_transform()
    test_lidar_to_camera_transform()
    test_project_points_to_image()
    test_compensate_latency()
    print("All tests passed!")