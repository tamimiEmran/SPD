"""
dataset module for V2X-Seq project.

This module provides functionality for loading and processing the V2X-Seq dataset
for Vehicle-Infrastructure Cooperative 3D Tracking (VIC3D) tasks. It includes:
- Base dataset class for loading V2X-Seq data
- Specialized dataset classes for different fusion strategies
- Data transformation and augmentation utilities
- Batch collation and sampling methods
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import copy
import random

import sys
sys.path.append("M:/Documents/Mwasalat/dataset/Full Dataset (train & val)-20250313T155844Z/Full Dataset (train & val)/V2X-Seq-SPD/V2X-Seq-SPD/v2x_tracking/data")


from preprocessing.transform import (
    transform_points_to_veh_coordinate,
    transform_inf_points_to_veh,
    transform_inf_boxes_to_veh,
    transform_boxes_to_veh_coordinate
)
from preprocessing.augmentation import augment_point_cloud
from calibration.coordinate_transform import (
    get_transformation_matrix,
    transform_points,
    inverse_transform_matrix
)

logger = logging.getLogger(__name__)


class V2XBaseDataset(Dataset):
    """Base dataset class for V2X-Seq data."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        segment_length: int = 10,  # Number of consecutive frames to load as sequence
        use_infrastructure: bool = True,
        use_image: bool = False,
        transform=None,
        augment: bool = False,
        point_cloud_range: List[float] = [0, -39.68, -3, 100, 39.68, 1],
        max_points: int = 100000,
    ):
        """
        Initialize V2X-Seq base dataset.
        
        Args:
            dataset_path: Path to the V2X-Seq-SPD dataset
            split: Data split ('train', 'val', or 'test')
            segment_length: Number of consecutive frames to load
            use_infrastructure: Whether to use infrastructure data
            use_image: Whether to use camera images (otherwise just LiDAR)
            transform: Optional transform to apply to samples
            augment: Whether to apply augmentation
            point_cloud_range: Range for point cloud [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points: Maximum number of points to keep in point clouds
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.segment_length = segment_length
        self.use_infrastructure = use_infrastructure
        self.use_image = use_image
        self.transform = transform
        self.augment = augment
        self.point_cloud_range = point_cloud_range
        self.max_points = max_points
        
        # Load split configuration
        split_file = self.dataset_path / f"{split}_split.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.split_data = json.load(f)
        
        # Create mapping from cooperative frames to their sequences
        self.frame_to_sequence_mapping = {}
        self.sequences = []
        self.sequence_frames = {}
        
        self._build_sequence_mapping()
        logger.info(f"Loaded {len(self.sequences)} sequences with {len(self.frame_to_sequence_mapping)} frames")
    
    def _build_sequence_mapping(self):
        """Build mapping from frames to sequences and create sequence list."""
        # This will vary based on the exact structure of the split file
        # Assuming split_data contains sequence IDs and their frames
        
        for seq_id, seq_info in self.split_data.items():
            self.sequences.append(seq_id)
            self.sequence_frames[seq_id] = []
            
            # Get all frames in this sequence
            cooperative_path = self.dataset_path / "cooperative" / "data_info.json"
            with open(cooperative_path, 'r') as f:
                coop_data = json.load(f)
            
            # Filter frames belonging to this sequence
            #clear the print statements in the terminal

            for frame_info in coop_data:
                if frame_info["vehicle_sequence"] == seq_id:
                    frame_id = frame_info["vehicle_frame"]
                    self.sequence_frames[seq_id].append(frame_id)
                    self.frame_to_sequence_mapping[frame_id] = seq_id
            
            # Sort frames by timestamp to ensure temporal order
            self.sequence_frames[seq_id].sort(key=lambda x: int(x.split('_')[-1]))
    
    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        count = 0
        for seq_id in self.sequences:
            # Number of possible segments = number of frames - segment length + 1
            num_frames = len(self.sequence_frames[seq_id])
            if num_frames >= self.segment_length:
                count += num_frames - self.segment_length + 1
        return count
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data segment by index.
        
        Returns a dictionary containing:
            - vehicle_points: List of vehicle point clouds for the segment
            - infrastructure_points: List of infrastructure point clouds
            - vehicle_labels: List of vehicle 3D tracking annotations
            - infrastructure_labels: List of infrastructure 3D tracking annotations
            - cooperative_labels: List of cooperative 3D tracking annotations
            - transformation_matrices: Matrices for coordinate transformations
            - metadata: Additional information about the sequence
        """
        # Find which sequence and starting frame this index corresponds to
        seq_idx, start_frame_idx = self._idx_to_sequence_frame(idx)
        seq_id = self.sequences[seq_idx]
        
        # Get frames for this segment
        frames = self.sequence_frames[seq_id][start_frame_idx:start_frame_idx+self.segment_length]
        
        # Initialize containers
        sample = {
            "vehicle_points": [],
            "infrastructure_points": [],
            "vehicle_labels": [],
            "infrastructure_labels": [],
            "cooperative_labels": [],
            "transformation_matrices": [],
            "metadata": {
                "sequence_id": seq_id,
                "frames": frames,
                "timestamps": []
            }
        }
        
        if self.use_image:
            sample["vehicle_images"] = []
            sample["infrastructure_images"] = []
        
        # Load data for each frame in the segment
        for frame_id in frames:
            frame_data = self._load_frame(frame_id)
            
            # Add data to the sample
            for key in sample.keys():
                if key in frame_data and key != "metadata":
                    sample[key].append(frame_data[key])
            
            # Add timestamp to metadata
            sample["metadata"]["timestamps"].append(frame_data["metadata"]["timestamp"])
        
        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _idx_to_sequence_frame(self, idx: int) -> Tuple[int, int]:
        """
        Convert flat index to (sequence_idx, start_frame_idx).
        
        Args:
            idx: Flat index into the dataset
            
        Returns:
            Tuple of (sequence index, start frame index within sequence)
        """
        count = 0
        for seq_idx, seq_id in enumerate(self.sequences):
            num_frames = len(self.sequence_frames[seq_id])
            if num_frames >= self.segment_length:
                seq_segments = num_frames - self.segment_length + 1
                if count + seq_segments > idx:
                    # This is the sequence we want
                    start_frame_idx = idx - count
                    return seq_idx, start_frame_idx
                count += seq_segments
        
        raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
    
    def _load_frame(self, frame_id: str) -> Dict[str, Any]:
        """
        Load all data for a single frame.
        
        Args:
            frame_id: ID of the frame to load
            
        Returns:
            Dictionary containing all data for this frame
        """
        # Initialize frame data container
        frame_data = {
            "metadata": {"frame_id": frame_id}
        }
        
        # Load cooperative data info
        coop_info_path = self.dataset_path / "cooperative" / "data_info.json"
        with open(coop_info_path, 'r') as f:
            coop_info = json.load(f)
        
        # Find the frame info in the list
        frame_info = None
        for info in coop_info:
            if info.get("vehicle_frame") == frame_id:
                frame_info = info
                break
                
        if not frame_info:
            logger.warning(f"Missing frame info for {frame_id}")
            return frame_data
        

        vehicle_frame = frame_info.get("vehicle_frame")
        infrastructure_frame = frame_info.get("infrastructure_frame")
        
        # If we can't find this frame, return empty data
        if not vehicle_frame or not infrastructure_frame:
            logger.warning(f"Missing frame info for {frame_id}")
            return frame_data
        
        # Load vehicle data
        vehicle_data = self._load_vehicle_data(vehicle_frame)


        frame_data["vehicle_points"] = vehicle_data["points"]
        frame_data["vehicle_labels"] = vehicle_data["labels"]
        frame_data["metadata"]["timestamp"] = vehicle_data["timestamp"]
        
        if self.use_image and "image" in vehicle_data:
            frame_data["vehicle_images"] = vehicle_data["image"]
        
        # Load infrastructure data if requested
        if self.use_infrastructure:
            infra_data = self._load_infrastructure_data(infrastructure_frame)
            frame_data["infrastructure_points"] = infra_data["points"]
            frame_data["infrastructure_labels"] = infra_data["labels"]
            
            if self.use_image and "image" in infra_data:
                frame_data["infrastructure_images"] = infra_data["image"]
        
        # Load cooperative labels
        coop_label_path = self.dataset_path / "cooperative" / "label" / f"{frame_id}.json"
        if coop_label_path.exists():
            with open(coop_label_path, 'r') as f:
                frame_data["cooperative_labels"] = json.load(f)
        
        # Load transformation matrices
        frame_data["transformation_matrices"] = self._load_transformation_matrices(
            vehicle_frame, infrastructure_frame
        )
        
        return frame_data


    def _load_vehicle_data(self, frame_id: str) -> Dict[str, Any]:
        """
        Load vehicle data for a specific frame.
        
        Args:
            frame_id: ID of the vehicle frame to load
        
        Returns:
            Dictionary containing vehicle data
        """
        vehicle_data = {}
        
        # Load vehicle info
        veh_info_path = self.dataset_path / "vehicle-side" / "data_info.json"
        with open(veh_info_path, 'r') as f:
            veh_info = json.load(f)
        
        # Find the frame info in the list
        frame_info = None
        for info in veh_info:
            if info.get("frame_id") == frame_id:
                frame_info = info
                break
        
        if not frame_info:
            logger.warning(f"Missing vehicle frame info for {frame_id}")
            return vehicle_data
        
        # Get the point cloud path and check if it exists
        pointcloud_path = self.dataset_path / "vehicle-side" / frame_info.get("pointcloud_path", "")
        
        # Get label path and check if it exists
        label_path = self.dataset_path / "vehicle-side" / frame_info.get("label_lidar_std_path", "")
        
        # Get timestamp
        timestamp = frame_info.get("pointcloud_timestamp", "")
        vehicle_data["timestamp"] = timestamp
        
        # Try to load point cloud with detailed error reporting
        if pointcloud_path.exists():
            try:
                # Check how _load_point_cloud is implemented
                points = self._load_point_cloud(pointcloud_path)
                vehicle_data["points"] = points
            except Exception as e:
                import traceback
                print(f"Error loading point cloud: {e}")
                print(traceback.format_exc())
                # Initialize with empty array as fallback
                vehicle_data["points"] = np.zeros((0, 4), dtype=np.float32)  # Empty point cloud
        else:
            print(f"Point cloud file not found: {pointcloud_path}")
            vehicle_data["points"] = np.zeros((0, 4), dtype=np.float32)  # Empty point cloud
        
        # Try to load labels with detailed error reporting
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    labels = json.load(f)
                vehicle_data["labels"] = labels
            except Exception as e:
                import traceback
                print(f"Error loading labels: {e}")
                print(traceback.format_exc())
                vehicle_data["labels"] = []  # Empty labels as fallback
        else:
            print(f"Label file not found: {label_path}")
            vehicle_data["labels"] = []  # Empty labels
        
        return vehicle_data

    def _load_infrastructure_data(self, frame_id: str) -> Dict:
        """
        Load infrastructure-side data for a frame.
        
        Args:
            frame_id: Infrastructure frame ID
            
        Returns:
            Dictionary with infrastructure point cloud, labels, and metadata
        """
        data = {}
        
        # Load infrastructure data info
        inf_info_path = self.dataset_path / "infrastructure-side" / "data_info.json"
        with open(inf_info_path, 'r') as f:
            inf_info = json.load(f)
        
        # Find the frame info in the list
        frame_info = None
        for info in inf_info:
            if info.get("frame_id") == frame_id:
                frame_info = info
                break
        
        if not frame_info:
            logger.warning(f"Missing infrastructure info for {frame_id}")
            return data
        
        
        # Load point cloud
        pc_path = self.dataset_path / "infrastructure-side" / frame_info.get("pointcloud_path", "")
        if pc_path.exists():
            try:
                points = self._load_point_cloud(pc_path)
                data["points"] = points
            except Exception as e:
                logger.error(f"Error loading infrastructure point cloud: {e}")
                data["points"] = np.zeros((0, 4), dtype=np.float32)  # Empty fallback
        else:
            logger.warning(f"Infrastructure point cloud file not found: {pc_path}")
            data["points"] = np.zeros((0, 4), dtype=np.float32)  # Empty fallback
        
        # Load labels
        label_path = self.dataset_path / "infrastructure-side" / frame_info.get("label_lidar_std_path", "")
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    data["labels"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading infrastructure labels: {e}")
                data["labels"] = []  # Empty fallback
        else:
            logger.warning(f"Infrastructure label file not found: {label_path}")
            data["labels"] = []  # Empty fallback
        
        # Load image if required
        if self.use_image:
            img_path = self.dataset_path / "infrastructure-side" / frame_info.get("image_path", "")
            if img_path.exists():
                try:
                    data["image"] = self._load_image(img_path)
                except Exception as e:
                    logger.error(f"Error loading infrastructure image: {e}")
        
        # Extract timestamp
        data["timestamp"] = frame_info.get("pointcloud_timestamp")
        
        return data
    
    def _load_point_cloud(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess a point cloud file.
        
        Args:
            file_path: Path to the point cloud file (.pcd)
            
        Returns:
            Numpy array of shape (N, 4) containing x, y, z, intensity
        """
        # This implementation depends on your point cloud format
        # For .pcd files, you might use Open3D or a custom parser
        
        try:
            # Open3D is a common library for point cloud processing
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            
            # If intensity is available as a separate attribute
            if pcd.has_colors():
                # Convert RGB to intensity (average of channels)
                colors = np.asarray(pcd.colors)
                intensity = np.mean(colors, axis=1, keepdims=True)
            else:
                # If no intensity, use ones
                intensity = np.ones((points.shape[0], 1))
            
            # Combine points and intensity
            points_with_intensity = np.hstack([points, intensity])
            
            # Filter points by range
            if self.point_cloud_range is not None:
                x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
                mask = (
                    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                )
                points_with_intensity = points_with_intensity[mask]
            
            # Subsample if too many points
            if points_with_intensity.shape[0] > self.max_points:
                indices = np.random.choice(
                    points_with_intensity.shape[0], 
                    self.max_points, 
                    replace=False
                )
                points_with_intensity = points_with_intensity[indices]
                
            return points_with_intensity
            
        except ImportError:
            # Fallback implementation if Open3D is not available
            logger.warning("Open3D not available. Using numpy to load PCD file.")
            
            try:
                # Using numpy to load a binary file
                # This is a simplified implementation and may need to be adapted
                # based on your specific PCD file format
                points = np.fromfile(file_path, dtype=np.float32)
                points = points.reshape(-1, 4)  # x, y, z, intensity
                
                # Filter points by range
                if self.point_cloud_range is not None:
                    x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
                    mask = (
                        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                    )
                    points = points[mask]
                
                # Subsample if too many points
                if points.shape[0] > self.max_points:
                    indices = np.random.choice(points.shape[0], self.max_points, replace=False)
                    points = points[indices]
                    
                return points
            except Exception as e:
                logger.error(f"Error loading point cloud {file_path}: {e}")
                # Return empty point cloud
                return np.zeros((0, 4), dtype=np.float32)
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Numpy array containing the image
        """
        try:
            import cv2
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            # Return empty image
            return np.zeros((300, 400, 3), dtype=np.uint8)
    
    def _load_transformation_matrices(self, vehicle_frame: str, 
                                     infrastructure_frame: str) -> Dict[str, np.ndarray]:
        """
        Load and compute transformation matrices between coordinate systems.
        
        Args:
            vehicle_frame: Vehicle frame ID
            infrastructure_frame: Infrastructure frame ID
            
        Returns:
            Dictionary of transformation matrices
        """
        matrices = {}
        
        # Load vehicle calibration data
        veh_calib_path = self.dataset_path / "vehicle-side" / "calib"
        
        # Load infrastructure calibration data
        inf_calib_path = self.dataset_path / "infrastructure-side" / "calib"
        
        try:
            # Vehicle LiDAR to world
            lidar_to_novatel_path = veh_calib_path / "lidar_to_novatel" / f"{vehicle_frame}.json"
            novatel_to_world_path = veh_calib_path / "novatel_to_world" / f"{vehicle_frame}.json"
            
            with open(lidar_to_novatel_path, 'r') as f:
                lidar_to_novatel = json.load(f)
            
            with open(novatel_to_world_path, 'r') as f:
                novatel_to_world = json.load(f)
            
            # Compute vehicle lidar to world matrix
            matrices["veh_lidar_to_novatel"] = self._compute_transformation_matrix(lidar_to_novatel)
            matrices["novatel_to_world"] = self._compute_transformation_matrix(novatel_to_world)
            matrices["veh_lidar_to_world"] = np.matmul(
                matrices["novatel_to_world"], matrices["veh_lidar_to_novatel"]
            )
            
            # Infrastructure virtualLiDAR to world
            vlidar_to_world_path = inf_calib_path / "virtuallidar_to_world" / f"{infrastructure_frame}.json"
            
            with open(vlidar_to_world_path, 'r') as f:
                vlidar_to_world = json.load(f)
            
            matrices["inf_lidar_to_world"] = self._compute_transformation_matrix(vlidar_to_world)
            
            # Compute world to vehicle lidar matrix (inverse of veh_lidar_to_world)
            matrices["world_to_veh_lidar"] = np.linalg.inv(matrices["veh_lidar_to_world"])
            
            # Compute infrastructure lidar to vehicle lidar
            matrices["inf_lidar_to_veh_lidar"] = np.matmul(
                matrices["world_to_veh_lidar"], matrices["inf_lidar_to_world"]
            )
            
            # If using images, also load camera calibration
            if self.use_image:
                veh_cam_intrinsic_path = veh_calib_path / "camera_intrinsic" / f"{vehicle_frame}.json"
                with open(veh_cam_intrinsic_path, 'r') as f:
                    veh_cam_intrinsic = json.load(f)
                matrices["veh_cam_intrinsic"] = np.array(veh_cam_intrinsic["cam_K"]).reshape(3, 3)
                
                veh_lidar_to_cam_path = veh_calib_path / "lidar_to_camera" / f"{vehicle_frame}.json"
                with open(veh_lidar_to_cam_path, 'r') as f:
                    veh_lidar_to_cam = json.load(f)
                matrices["veh_lidar_to_cam"] = self._compute_transformation_matrix(veh_lidar_to_cam)
                
                inf_cam_intrinsic_path = inf_calib_path / "camera_intrinsic" / f"{infrastructure_frame}.json"
                with open(inf_cam_intrinsic_path, 'r') as f:
                    inf_cam_intrinsic = json.load(f)
                matrices["inf_cam_intrinsic"] = np.array(inf_cam_intrinsic["cam_K"]).reshape(3, 3)
                
                inf_lidar_to_cam_path = inf_calib_path / "virtuallidar_to_camera" / f"{infrastructure_frame}.json"
                with open(inf_lidar_to_cam_path, 'r') as f:
                    inf_lidar_to_cam = json.load(f)
                matrices["inf_lidar_to_cam"] = self._compute_transformation_matrix(inf_lidar_to_cam)
            
        except Exception as e:
            logger.error(f"Error computing transformation matrices: {e}")
            # Return identity matrices as fallback
            matrices["veh_lidar_to_world"] = np.eye(4)
            matrices["inf_lidar_to_world"] = np.eye(4)
            matrices["world_to_veh_lidar"] = np.eye(4)
            matrices["inf_lidar_to_veh_lidar"] = np.eye(4)
        
        return matrices
    
    def _compute_transformation_matrix(self, calib_data):
        """
        Compute 4x4 transformation matrix from calibration data.
        
        Args:
            calib_data: Calibration data dictionary
        
        Returns:
            4x4 transformation matrix
        """
        try:
            # Check if data has a nested transform structure
            if 'transform' in calib_data and isinstance(calib_data['transform'], dict):
                transform_data = calib_data['transform']
                if 'rotation' in transform_data and 'translation' in transform_data:
                    rotation = np.array(transform_data['rotation']).reshape(3, 3)
                    translation = np.array(transform_data['translation']).reshape(3, 1)
                else:
                    print(f"Missing rotation or translation in transform data: {transform_data.keys()}")
                    return np.eye(4)
            # Check if rotation and translation are at the top level
            elif 'rotation' in calib_data and 'translation' in calib_data:
                rotation = np.array(calib_data['rotation']).reshape(3, 3)
                translation = np.array(calib_data['translation']).reshape(3, 1)
            else:
                print(f"Could not find rotation and translation in calibration data")
                return np.eye(4)
            
            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation.flatten()
            
            return transform
        except Exception as e:
            print(f"Error in _compute_transformation_matrix: {e}")
            # Return identity matrix as fallback
            return np.eye(4)
            
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching samples together.
        
        Args:
            batch: List of samples to collate
            
        Returns:
            Collated batch dictionary
        """
        batch_size = len(batch)
        
        # Initialize return dictionary
        collated_batch = {
            "vehicle_points": [],
            "infrastructure_points": [],
            "vehicle_labels": [],
            "infrastructure_labels": [],
            "cooperative_labels": [],
            "transformation_matrices": [],
            "metadata": []
        }
        
        # Add images if present
        if "vehicle_images" in batch[0]:
            collated_batch["vehicle_images"] = []
            
        if "infrastructure_images" in batch[0]:
            collated_batch["infrastructure_images"] = []
        
        # Collate each key
        for key in collated_batch.keys():
            if key == "metadata":
                collated_batch[key] = [sample[key] for sample in batch]
            else:
                # Check if this is a list of items (e.g., points from multiple frames)
                if isinstance(batch[0][key], list):
                    # For each frame position
                    for frame_idx in range(len(batch[0][key])):
                        frame_items = [sample[key][frame_idx] for sample in batch]
                        collated_batch[key].append(frame_items)
                else:
                    # Single item per batch
                    collated_batch[key] = [sample[key] for sample in batch]
        
        return collated_batch


class EarlyFusionDataset(V2XBaseDataset):
    """Dataset class for early fusion strategy."""
    
    def __init__(self, **kwargs):
        """Initialize the early fusion dataset."""
        super().__init__(**kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample with early fusion applied.
        
        Early fusion combines raw sensor data from vehicle and infrastructure
        before processing.
        """
        # Get the base sample
        sample = super().__getitem__(idx)
        
        # Initialize containers for early fusion
        early_fusion_sample = {
            "fused_points": [],
            "fused_labels": [],
            "metadata": sample["metadata"],
            "transformation_matrices": sample["transformation_matrices"]
        }
        
        # Perform early fusion for each frame
        for frame_idx in range(len(sample["vehicle_points"])):
            # Get vehicle and infrastructure data
            veh_points = sample["vehicle_points"][frame_idx]
            inf_points = sample["infrastructure_points"][frame_idx]
            veh_labels = sample["vehicle_labels"][frame_idx]
            inf_labels = sample["infrastructure_labels"][frame_idx]
            trans_matrices = sample["transformation_matrices"][frame_idx]
            
            # Transform infrastructure points to vehicle coordinate system
            inf_points_veh = transform_points_to_veh_coordinate(
                inf_points, trans_matrices["inf_lidar_to_veh_lidar"]
            )
            
            # Merge point clouds
            fused_points = np.vstack([veh_points, inf_points_veh])
            
            # Transform infrastructure labels to vehicle coordinate system
            inf_boxes = np.array([
                [obj["3d_location"]["x"], obj["3d_location"]["y"], obj["3d_location"]["z"],
                 obj["3d_dimensions"]["w"], obj["3d_dimensions"]["l"], obj["3d_dimensions"]["h"],
                 obj["rotation"]]
                for obj in inf_labels
            ])
            
            inf_boxes_veh = transform_boxes_to_veh_coordinate(
                inf_boxes, trans_matrices["inf_lidar_to_veh_lidar"]
            )
            
            # Update infrastructure labels with transformed boxes
            for i, obj in enumerate(inf_labels):
                obj["3d_location"]["x"] = float(inf_boxes_veh[i, 0])
                obj["3d_location"]["y"] = float(inf_boxes_veh[i, 1])
                obj["3d_location"]["z"] = float(inf_boxes_veh[i, 2])
                obj["3d_dimensions"]["w"] = float(inf_boxes_veh[i, 3])
                obj["3d_dimensions"]["l"] = float(inf_boxes_veh[i, 4])
                obj["3d_dimensions"]["h"] = float(inf_boxes_veh[i, 5])
                obj["rotation"] = float(inf_boxes_veh[i, 6])
                
                # Mark as infrastructure source
                obj["source"] = "infrastructure"
            
            # Mark vehicle labels as vehicle source
            for obj in veh_labels:
                obj["source"] = "vehicle"
            
            # Combine labels
            fused_labels = veh_labels + inf_labels
            
            # Store in early fusion sample
            early_fusion_sample["fused_points"].append(fused_points)
            early_fusion_sample["fused_labels"].append(fused_labels)
        
        return early_fusion_sample


class LateFusionDataset(V2XBaseDataset):
    """Dataset class for late fusion strategy."""
    
    def __init__(self, **kwargs):
        """Initialize the late fusion dataset."""
        super().__init__(**kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample for late fusion.
        
        Late fusion keeps vehicle and infrastructure data separate for processing,
        and fusion happens at detection/tracking result level.
        """
        # Get the base sample
        return super().__getitem__(idx)


class MiddleFusionDataset(V2XBaseDataset):
    """Dataset class for middle fusion strategy."""
    
    def __init__(self, feature_size: int = 64, **kwargs):
        """
        Initialize the middle fusion dataset.
        
        Args:
            feature_size: Size of feature vectors
            **kwargs: Additional arguments for the base dataset
        """
        super().__init__(**kwargs)
        self.feature_size = feature_size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample for middle fusion.
        
        Middle fusion keeps data separate but extracts features 
        that will be fused later in the network.
        """
        # Get the base sample
        sample = super().__getitem__(idx)
        
        # For middle fusion, we keep the raw data and add placeholder for features
        # that will be computed by the network
        sample["veh_features"] = [np.zeros((self.feature_size,)) for _ in range(len(sample["vehicle_points"]))]
        sample["inf_features"] = [np.zeros((self.feature_size,)) for _ in range(len(sample["infrastructure_points"]))]
        
        return sample


class V2XDataset(V2XBaseDataset):
    """Main dataset class for V2X-Seq with configurable fusion strategy."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        fusion_strategy: str = "late_fusion",
        simulate_latency: bool = False,
        latency_ms: int = 200,
        **kwargs
    ):
        """
        Initialize V2X-Seq dataset with configurable fusion strategy.
        
        Args:
            dataset_path: Path to the V2X-Seq-SPD dataset
            split: Data split ('train', 'val', or 'test')
            fusion_strategy: Fusion strategy ('early_fusion', 'middle_fusion', 'late_fusion', 'ff_tracking')
            simulate_latency: Whether to simulate communication latency
            latency_ms: Simulated latency in milliseconds
            **kwargs: Additional arguments for the base dataset
        """
        super().__init__(dataset_path=dataset_path, split=split, **kwargs)
        self.fusion_strategy = fusion_strategy
        self.simulate_latency = simulate_latency
        self.latency_ms = latency_ms
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample using the configured fusion strategy.
        
        Returns:
            Dictionary with data prepared according to the selected fusion strategy
        """
        # Get the base sample
        sample = super().__getitem__(idx)
        
        # Apply latency simulation if requested
        if self.simulate_latency:
            sample = self._apply_latency_simulation(sample)
        
        # Process according to fusion strategy
        if self.fusion_strategy == "early_fusion":
            return self._prepare_early_fusion(sample)
        elif self.fusion_strategy == "middle_fusion":
            return self._prepare_middle_fusion(sample)
        elif self.fusion_strategy == "ff_tracking":
            return self._prepare_ff_tracking(sample)
        else:  # late_fusion is default
            return sample
    
    def _apply_latency_simulation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply latency simulation to infrastructure data.
        
        Args:
            sample: Original data sample
            
        Returns:
            Sample with latency applied to infrastructure data
        """
        latency_sample = copy.deepcopy(sample)
        
        # Apply latency to infrastructure timestamps
        for i, (inf_points, inf_labels, matrices) in enumerate(zip(
            sample["infrastructure_points"],
            sample["infrastructure_labels"],
            sample["transformation_matrices"]
        )):
            # Calculate timestamp difference due to latency
            timestamp_diff = self.latency_ms / 1000.0  # Convert to seconds
            
            # Find an earlier infrastructure frame based on latency
            earlier_frame_idx = self._find_earlier_frame_idx(
                sample["metadata"]["timestamps"], i, timestamp_diff
            )
            
            if earlier_frame_idx >= 0:
                # Replace current infrastructure data with earlier data
                latency_sample["infrastructure_points"][i] = sample["infrastructure_points"][earlier_frame_idx]
                latency_sample["infrastructure_labels"][i] = sample["infrastructure_labels"][earlier_frame_idx]
                
                # Update transformation matrix
                earlier_matrices = sample["transformation_matrices"][earlier_frame_idx]
                latency_sample["transformation_matrices"][i]["inf_lidar_to_world"] = earlier_matrices["inf_lidar_to_world"]
                
                # Recompute inf_lidar_to_veh_lidar matrix
                latency_sample["transformation_matrices"][i]["inf_lidar_to_veh_lidar"] = np.matmul(
                    matrices["world_to_veh_lidar"],
                    earlier_matrices["inf_lidar_to_world"]
                )
                
                if self.use_image and "infrastructure_images" in sample:
                    latency_sample["infrastructure_images"][i] = sample["infrastructure_images"][earlier_frame_idx]
        
        return latency_sample
    
    def _find_earlier_frame_idx(self, timestamps: List[float], current_idx: int, time_diff: float) -> int:
        """
        Find index of an earlier frame that would be available with latency.
        
        Args:
            timestamps: List of frame timestamps
            current_idx: Index of current frame
            time_diff: Time difference to look backward
            
        Returns:
            Index of earlier frame or -1 if none available
        """
        current_time = timestamps[current_idx]
        target_time = current_time - time_diff
        
        # Look for the closest frame earlier than target_time
        best_idx = -1
        min_diff = float('inf')
        
        for i in range(current_idx):
            time_i = timestamps[i]
            if time_i <= target_time:
                diff = target_time - time_i
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
        
        return best_idx
    
    def _prepare_early_fusion(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare sample for early fusion.
        
        Args:
            sample: Original data sample
            
        Returns:
            Sample prepared for early fusion
        """
        early_fusion_sample = {
            "fused_points": [],
            "fused_labels": [],
            "metadata": sample["metadata"],
            "transformation_matrices": sample["transformation_matrices"]
        }
        
        # Perform early fusion for each frame
        for frame_idx in range(len(sample["vehicle_points"])):
            # Get vehicle and infrastructure data
            veh_points = sample["vehicle_points"][frame_idx]
            inf_points = sample["infrastructure_points"][frame_idx]
            veh_labels = sample["vehicle_labels"][frame_idx]
            inf_labels = sample["infrastructure_labels"][frame_idx]
            trans_matrices = sample["transformation_matrices"][frame_idx]
            
            # Transform infrastructure points to vehicle coordinate system
            inf_points_veh = transform_points_to_veh_coordinate(
                inf_points, trans_matrices["inf_lidar_to_veh_lidar"]
            )
            
            # Merge point clouds
            fused_points = np.vstack([veh_points, inf_points_veh])
            
            # Transform infrastructure labels to vehicle coordinate system
            inf_boxes = np.array([
                [obj["3d_location"]["x"], obj["3d_location"]["y"], obj["3d_location"]["z"],
                 obj["3d_dimensions"]["w"], obj["3d_dimensions"]["l"], obj["3d_dimensions"]["h"],
                 obj["rotation"]]
                for obj in inf_labels
            ])
            
            inf_boxes_veh = transform_boxes_to_veh_coordinate(
                inf_boxes, trans_matrices["inf_lidar_to_veh_lidar"]
            )
            
            # Update infrastructure labels with transformed boxes
            for i, obj in enumerate(inf_labels):
                obj["3d_location"]["x"] = float(inf_boxes_veh[i, 0])
                obj["3d_location"]["y"] = float(inf_boxes_veh[i, 1])
                obj["3d_location"]["z"] = float(inf_boxes_veh[i, 2])
                obj["3d_dimensions"]["w"] = float(inf_boxes_veh[i, 3])
                obj["3d_dimensions"]["l"] = float(inf_boxes_veh[i, 4])
                obj["3d_dimensions"]["h"] = float(inf_boxes_veh[i, 5])
                obj["rotation"] = float(inf_boxes_veh[i, 6])
                
                # Mark as infrastructure source
                obj["source"] = "infrastructure"
            
            # Mark vehicle labels as vehicle source
            for obj in veh_labels:
                obj["source"] = "vehicle"
            
            # Combine labels
            fused_labels = veh_labels + inf_labels
            
            # Store in early fusion sample
            early_fusion_sample["fused_points"].append(fused_points)
            early_fusion_sample["fused_labels"].append(fused_labels)
        
        return early_fusion_sample
    
    def _prepare_middle_fusion(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare sample for middle fusion.
        
        Args:
            sample: Original data sample
            
        Returns:
            Sample prepared for middle fusion
        """
        # For middle fusion, we keep the raw data separate
        # Features will be computed by the network
        return sample
    
    def _prepare_ff_tracking(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare sample for feature flow tracking fusion.
        
        Args:
            sample: Original data sample
            
        Returns:
            Sample prepared for feature flow tracking
        """
        # FF-Tracking needs both current and previous frames
        # to compute feature flow
        ff_sample = copy.deepcopy(sample)
        
        # Add additional key for feature flow (will be computed by network)
        ff_sample["infrastructure_flow"] = []
        
        # If we have more than one frame, calculate timestamps for flow computation
        if len(sample["metadata"]["timestamps"]) > 1:
            timesteps = []
            for i in range(1, len(sample["metadata"]["timestamps"])):
                current_ts = sample["metadata"]["timestamps"][i]
                prev_ts = sample["metadata"]["timestamps"][i-1]
                timesteps.append(current_ts - prev_ts)
            ff_sample["metadata"]["timesteps"] = timesteps
        
        return ff_sample


def build_dataloader(
    dataset_path: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    fusion_strategy: str = "late_fusion",
    simulate_latency: bool = False,
    latency_ms: int = 200,
    segment_length: int = 10,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    use_image: bool = False,
    augment: Optional[bool] = None,
    distributed: bool = False,
    **dataset_kwargs
) -> Tuple[Dataset, DataLoader]:
    """
    Build dataset and dataloader for V2X-Seq.
    
    Args:
        dataset_path: Path to the V2X-Seq-SPD dataset
        split: Data split ('train', 'val', or 'test')
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes for dataloader
        fusion_strategy: Fusion strategy ('early_fusion', 'middle_fusion', 'late_fusion', 'ff_tracking')
        simulate_latency: Whether to simulate communication latency
        latency_ms: Simulated latency in milliseconds
        segment_length: Number of consecutive frames to load as sequence
        shuffle: Whether to shuffle the data (defaults to True for train, False otherwise)
        drop_last: Whether to drop the last incomplete batch (defaults to True for train, False otherwise)
        use_image: Whether to use camera images
        augment: Whether to apply augmentation (defaults to True for train, False otherwise)
        distributed: Whether to use DistributedSampler for distributed training
        **dataset_kwargs: Additional arguments for the dataset
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    # Set defaults based on split if not explicitly provided
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train")
    if augment is None:
        augment = (split == "train")
    
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
        **dataset_kwargs
    )
    
    # Create sampler for distributed training
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler will handle shuffling
    
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