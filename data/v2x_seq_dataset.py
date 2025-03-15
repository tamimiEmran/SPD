"""
v2x_seq_dataset module for V2X-Seq project.

This module provides functionality for loading and processing the V2X-Seq dataset
for Vehicle-Infrastructure Cooperative 3D Tracking (VIC3D) tasks.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any


# Assuming these imports are available in your project structure

# add the path to the sys.path
import sys
sys.path.append(r'M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\v2x_tracking\data')



from preprocessing.transform import transform_points_to_veh_coordinate
from preprocessing.augmentation import augment_point_cloud
from calibration.coordinate_transform import (
    get_transformation_matrix,
    transform_points,
)

logger = logging.getLogger(__name__)

class V2XSeqDataset(Dataset):
    """V2X-Seq dataset for vehicle-infrastructure cooperative 3D tracking."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        segment_length: int = 10,  # Number of consecutive frames to load as sequence
        use_infrastructure: bool = True,
        use_image: bool = False,
        simulate_latency: bool = False,
        latency_ms: int = 200,
        transform=None,
        augment: bool = False,
        max_points: int = 100000,
    ):
        """
        Initialize V2X-Seq dataset.
        
        Args:
            dataset_path: Path to the V2X-Seq-SPD dataset
            split: Data split ('train', 'val', or 'test')
            segment_length: Number of consecutive frames to load
            use_infrastructure: Whether to use infrastructure data
            use_image: Whether to use camera images (otherwise just LiDAR)
            simulate_latency: Whether to simulate communication latency
            latency_ms: Simulated latency in milliseconds
            transform: Optional transform to apply to samples
            augment: Whether to apply augmentation
            max_points: Maximum number of points to keep in point clouds
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.segment_length = segment_length
        self.use_infrastructure = use_infrastructure
        self.use_image = use_image
        self.simulate_latency = simulate_latency
        self.latency_ms = latency_ms
        self.transform = transform
        self.augment = augment
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
            
            # Handle both list and dictionary formats
            if isinstance(coop_data, list):
                # List format
                for i, item in enumerate(coop_data):
                    if item.get("vehicle_sequence") == seq_id:
                        # Use existing frame_id if available, otherwise create one
                        frame_id = item.get("frame_id", f"frame_{i:06d}")
                        self.sequence_frames[seq_id].append(frame_id)
                        self.frame_to_sequence_mapping[frame_id] = seq_id
            else:
                # Dictionary format (original code)
                for frame_id, frame_info in coop_data.items():
                    if frame_info.get("vehicle_sequence") == seq_id:
                        self.sequence_frames[seq_id].append(frame_id)
                        self.frame_to_sequence_mapping[frame_id] = seq_id
            
            # Sort frames by timestamp to ensure temporal order
            try:
                self.sequence_frames[seq_id].sort(key=lambda x: int(x.split('_')[-1]))
            except (ValueError, IndexError):
                # Fallback to simple string sorting if numerical sorting fails
                self.sequence_frames[seq_id].sort()
    
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
        
        # Get frame info based on the format
        vehicle_frame = None
        infrastructure_frame = None
        
        if isinstance(coop_info, list):
            # List format - find the entry matching frame_id
            frame_info = None
            frame_idx = -1
            
            # Try to parse frame_id as index if it follows our naming convention
            try:
                if frame_id.startswith("frame_"):
                    frame_idx = int(frame_id.split('_')[-1])
                    if 0 <= frame_idx < len(coop_info):
                        frame_info = coop_info[frame_idx]
            except (ValueError, IndexError):
                pass
            
            # If we couldn't use index, search the list
            if frame_info is None:
                for item in coop_info:
                    if item.get("frame_id") == frame_id:
                        frame_info = item
                        break
            
            if frame_info:
                vehicle_frame = frame_info.get("vehicle_frame")
                infrastructure_frame = frame_info.get("infrastructure_frame")
        else:
            # Dictionary format
            frame_info = coop_info.get(frame_id, {})
            vehicle_frame = frame_info.get("vehicle_frame")
            infrastructure_frame = frame_info.get("infrastructure_frame")
        
        # If we can't find this frame, return empty data
        if not vehicle_frame or not infrastructure_frame:
            logger.warning(f"Missing frame info for {frame_id}")
            return frame_data
        
        # Load vehicle data
        vehicle_data = self._load_vehicle_data(vehicle_frame)
        frame_data["vehicle_points"] = vehicle_data.get("points", np.zeros((0, 4), dtype=np.float32))
        frame_data["vehicle_labels"] = vehicle_data.get("labels", [])
        frame_data["metadata"]["timestamp"] = vehicle_data.get("timestamp")
        
        if self.use_image and "image" in vehicle_data:
            frame_data["vehicle_images"] = vehicle_data["image"]
        
        # Load infrastructure data if requested
        if self.use_infrastructure:
            # Simulate latency if requested by using an earlier infrastructure frame
            if self.simulate_latency:
                infrastructure_frame = self._get_delayed_frame(
                    infrastructure_frame, self.latency_ms
                )
            
            infra_data = self._load_infrastructure_data(infrastructure_frame)
            frame_data["infrastructure_points"] = infra_data.get("points", np.zeros((0, 4), dtype=np.float32))
            frame_data["infrastructure_labels"] = infra_data.get("labels", [])
            
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
    
    def _load_vehicle_data(self, frame_id: str) -> Dict:
        """
        Load vehicle-side data for a frame.
        
        Args:
            frame_id: Vehicle frame ID
            
        Returns:
            Dictionary with vehicle point cloud, labels, and metadata
        """
        data = {}
        
        # Load vehicle data info
        veh_info_path = self.dataset_path / "vehicle-side" / "data_info.json"
        with open(veh_info_path, 'r') as f:
            veh_info = json.load(f)
        
        # Get frame info based on the format
        frame_info = None
        
        if isinstance(veh_info, list):
            # List format
            for item in veh_info:
                if str(item.get("frame_id")) == str(frame_id):
                    frame_info = item
                    break
        else:
            # Dictionary format
            frame_info = veh_info.get(frame_id, {})
        
        if not frame_info:
            logger.warning(f"Missing vehicle info for {frame_id}")
            return data
        
        # Load point cloud
        pointcloud_path = frame_info.get("pointcloud_path")
        if pointcloud_path:
            # Try multiple possible paths
            pc_paths_to_try = [
                self.dataset_path / pointcloud_path,  # Original path
                self.dataset_path / "vehicle-side" / "velodyne" / f"{frame_id}.pcd",  # Alternative with frame ID
                self.dataset_path / "vehicle-side" / "velodyne" / Path(pointcloud_path).name  # Alternative with original filename
            ]
            
            for pc_path in pc_paths_to_try:
                if pc_path.exists():
                    points = self._load_point_cloud(pc_path)
                    # Apply augmentation if specified
                    if self.augment and points.shape[0] > 0:
                        points = augment_point_cloud(points)
                    data["points"] = points
                    break
        
        # Load labels
        label_path_str = frame_info.get("label_lidar_std_path")
        if label_path_str:
            label_path = self.dataset_path / label_path_str
            if label_path.exists():
                with open(label_path, 'r') as f:
                    data["labels"] = json.load(f)
        
        # Load image if required
        if self.use_image:
            image_path_str = frame_info.get("image_path")
            if image_path_str:
                img_path = self.dataset_path / image_path_str
                if img_path.exists():
                    data["image"] = self._load_image(img_path)
        
        # Extract timestamp
        data["timestamp"] = frame_info.get("pointcloud_timestamp") or frame_info.get("timestamp")
        
        return data

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
        
        # Get frame info based on the format
        frame_info = None
        
        if isinstance(inf_info, list):
            # List format
            for item in inf_info:
                if str(item.get("frame_id")) == str(frame_id):
                    frame_info = item
                    break
        else:
            # Dictionary format
            frame_info = inf_info.get(frame_id, {})
        
        if not frame_info:
            logger.warning(f"Missing infrastructure info for {frame_id}")
            return data
        
        # Load point cloud
        pointcloud_path = frame_info.get("pointcloud_path")
        if pointcloud_path:
            # Try multiple possible paths
            pc_paths_to_try = [
                self.dataset_path / pointcloud_path,  # Original path
                self.dataset_path / "infrastructure-side" / "velodyne" / f"{frame_id}.pcd",  # Alternative with frame ID
                self.dataset_path / "infrastructure-side" / "velodyne" / Path(pointcloud_path).name  # Alternative with original filename
            ]
            
            for pc_path in pc_paths_to_try:
                if pc_path.exists():
                    points = self._load_point_cloud(pc_path)
                    data["points"] = points
                    break
        
        # Load labels
        label_path_str = frame_info.get("label_lidar_std_path")
        if label_path_str:
            label_path = self.dataset_path / label_path_str
            if label_path.exists():
                with open(label_path, 'r') as f:
                    data["labels"] = json.load(f)
        
        # Load image if required
        if self.use_image:
            image_path_str = frame_info.get("image_path")
            if image_path_str:
                img_path = self.dataset_path / image_path_str
                if img_path.exists():
                    data["image"] = self._load_image(img_path)
        
        # Extract timestamp
        data["timestamp"] = frame_info.get("pointcloud_timestamp") or frame_info.get("timestamp")
        
        return data

    def _load_point_cloud(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess a point cloud file.
        
        Args:
            file_path: Path to the point cloud file (.pcd)
            
        Returns:
            Numpy array of shape (N, 4) containing x, y, z, intensity
        """
        try:
            # For compressed binary PCD files, use Open3D
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            
            # Check if points were loaded
            if points.shape[0] == 0:
                logger.warning(f"No points loaded from {file_path} using Open3D")
                return np.zeros((0, 4), dtype=np.float32)
            
            # If intensity is available as colors, use it
            if pcd.has_colors():
                # Convert RGB to intensity (average of channels)
                intensity = np.mean(np.asarray(pcd.colors), axis=1, keepdims=True)
            else:
                # If no intensity, use ones
                intensity = np.ones((points.shape[0], 1))
            
            # Combine points and intensity
            points_with_intensity = np.hstack([points, intensity])
            
            # Subsample if too many points
            if points_with_intensity.shape[0] > self.max_points:
                indices = np.random.choice(points_with_intensity.shape[0], self.max_points, replace=False)
                points_with_intensity = points_with_intensity[indices]
            
            return points_with_intensity
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
        # This implementation depends on your image processing library
        # You might use OpenCV, PIL, or torchvision
        
        # Placeholder implementation - replace with actual loading code
        try:
            import cv2
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            # Return empty image
            return np.zeros((300, 400, 3), dtype=np.uint8)
    
    def _load_transformation_matrices(
        self, vehicle_frame: str, infrastructure_frame: str
    ) -> Dict[str, np.ndarray]:
        """
        Load and compute transformation matrices between coordinate systems.
        Adapted for the specific JSON format in your dataset.
        
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
            
            if lidar_to_novatel_path.exists() and novatel_to_world_path.exists():
                with open(lidar_to_novatel_path, 'r') as f:
                    lidar_to_novatel = json.load(f)
                
                with open(novatel_to_world_path, 'r') as f:
                    novatel_to_world = json.load(f)
                
                # Create transformation matrices based on available keys
                lidar_to_novatel_matrix = self._create_transformation_matrix(lidar_to_novatel)
                novatel_to_world_matrix = self._create_transformation_matrix(novatel_to_world)
                
                # Compute vehicle lidar to world matrix
                matrices["veh_lidar_to_world"] = novatel_to_world_matrix @ lidar_to_novatel_matrix
            else:
                # Use identity matrix if files don't exist
                matrices["veh_lidar_to_world"] = np.eye(4)
                logger.warning(f"Missing calibration files for vehicle frame {vehicle_frame}")
            
            # Infrastructure virtualLiDAR to world
            vlidar_to_world_path = inf_calib_path / "virtuallidar_to_world" / f"{infrastructure_frame}.json"
            
            if vlidar_to_world_path.exists():
                with open(vlidar_to_world_path, 'r') as f:
                    vlidar_to_world = json.load(f)
                
                # Create transformation matrix from available keys
                matrices["inf_lidar_to_world"] = self._create_transformation_matrix(vlidar_to_world)
            else:
                # Use identity matrix if file doesn't exist
                matrices["inf_lidar_to_world"] = np.eye(4)
                logger.warning(f"Missing calibration file for infrastructure frame {infrastructure_frame}")
            
            # Compute world to vehicle lidar matrix (inverse of veh_lidar_to_world)
            matrices["world_to_veh_lidar"] = np.linalg.inv(matrices["veh_lidar_to_world"])
            
            # Compute infrastructure lidar to vehicle lidar
            matrices["inf_lidar_to_veh_lidar"] = matrices["world_to_veh_lidar"] @ matrices["inf_lidar_to_world"]
            
        except Exception as e:
            logger.error(f"Error computing transformation matrices: {e}")
            # Return identity matrices as fallback
            matrices["veh_lidar_to_world"] = np.eye(4)
            matrices["inf_lidar_to_world"] = np.eye(4)
            matrices["world_to_veh_lidar"] = np.eye(4)
            matrices["inf_lidar_to_veh_lidar"] = np.eye(4)
        
        return matrices


    def _create_transformation_matrix(self, calib_data: Dict) -> np.ndarray:
        """
        Create a transformation matrix from calibration data, handling different key formats.
        
        Args:
            calib_data: Calibration data dictionary
            
        Returns:
            4x4 transformation matrix
        """
        # Start with identity matrix
        transform_matrix = np.eye(4)
        
        # Check different possible formats
        if "transform_matrix" in calib_data:
            # Direct transform matrix format
            return np.array(calib_data["transform_matrix"])
        
        if "rotation" in calib_data and "translation" in calib_data:
            # Rotation and translation format
            rotation = np.array(calib_data["rotation"]).reshape(3, 3)
            
            # Handle translation - ensure it's a 1D array of length 3
            translation = np.array(calib_data["translation"])
            if translation.ndim > 1:
                translation = translation.flatten()[:3]
            
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = translation
            return transform_matrix
        
        if "R" in calib_data:
            # R format (rotation matrix)
            rotation = np.array(calib_data["R"]).reshape(3, 3)
            transform_matrix[:3, :3] = rotation
        
        if "P" in calib_data:
            # P format (projection or pose matrix)
            # Assuming P is a 3x4 projection matrix
            P = np.array(calib_data["P"]).reshape(3, 4)
            transform_matrix[:3, :] = P
        
        if "t" in calib_data:
            # Translation vector - handle potential shape issues
            translation = np.array(calib_data["t"])
            if translation.ndim > 1:
                translation = translation.flatten()[:3]
            transform_matrix[:3, 3] = translation
        
        if "header" in calib_data:
            # Nothing to do with the header for the transformation matrix
            pass
        
        # Create a rotation matrix from quaternion if available
        if "orientation" in calib_data:
            try:
                from scipy.spatial.transform import Rotation
                quat = calib_data["orientation"]
                rotation = Rotation.from_quat(quat).as_matrix()
                transform_matrix[:3, :3] = rotation
            except (ImportError, ValueError):
                pass
        
        return transform_matrix












    def _get_delayed_frame(self, frame_id: str, latency_ms: int) -> str:
        """
        Get an earlier frame to simulate communication latency.
        
        Args:
            frame_id: Current frame ID
            latency_ms: Latency in milliseconds
            
        Returns:
            Earlier frame ID based on the latency
        """
        # This implementation depends on your frame naming convention
        # Assuming frames are named with timestamps
        
        # Load infrastructure data info to get timestamps
        inf_info_path = self.dataset_path / "infrastructure-side" / "data_info.json"
        with open(inf_info_path, 'r') as f:
            inf_info = json.load(f)
        
        # Get current frame info based on the format
        current_timestamp = 0
        
        if isinstance(inf_info, list):
            # List format
            current_frame_info = None
            for item in inf_info:
                if str(item.get("frame_id")) == str(frame_id):
                    current_frame_info = item
                    break
            
            if current_frame_info:
                current_timestamp = float(current_frame_info.get("pointcloud_timestamp", 0))
        else:
            # Dictionary format
            current_frame_info = inf_info.get(frame_id, {})
            if current_frame_info:
                current_timestamp = float(current_frame_info.get("pointcloud_timestamp", 0))
        
        if current_timestamp == 0:
            return frame_id  # Fallback to current frame
        
        target_timestamp = current_timestamp - (latency_ms / 1000.0)
        
        # Find the closest earlier frame
        best_frame = frame_id
        min_diff = float('inf')
        
        if isinstance(inf_info, list):
            # List format
            for item in inf_info:
                frame_timestamp = float(item.get("pointcloud_timestamp", 0))
                
                # Only consider earlier frames
                if frame_timestamp <= target_timestamp:
                    diff = abs(frame_timestamp - target_timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        best_frame = item.get("frame_id", frame_id)
        else:
            # Dictionary format
            for frame, info in inf_info.items():
                frame_timestamp = float(info.get("pointcloud_timestamp", 0))
                
                # Only consider earlier frames
                if frame_timestamp <= target_timestamp:
                    diff = abs(frame_timestamp - target_timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        best_frame = frame
        
        return best_frame
    

def collate_fn(batch):
    """
    Custom collate function for V2X-Seq dataset batches.
    
    Handles variable-sized point clouds and creates padded tensors.
    """
    batch_size = len(batch)
    
    # Find max number of points in this batch for each frame
    max_vehicle_points = [0] * len(batch[0]["vehicle_points"])
    max_infra_points = [0] * len(batch[0]["infrastructure_points"])
    
    for sample in batch:
        for i, points in enumerate(sample["vehicle_points"]):
            max_vehicle_points[i] = max(max_vehicle_points[i], points.shape[0])
        
        for i, points in enumerate(sample["infrastructure_points"]):
            max_infra_points[i] = max(max_infra_points[i], points.shape[0])
    
    # Initialize batch containers
    batched = {
        "vehicle_points": [],
        "infrastructure_points": [],
        "vehicle_labels": [],
        "infrastructure_labels": [],
        "cooperative_labels": [],
        "transformation_matrices": [],
        "point_masks": {
            "vehicle": [],
            "infrastructure": []
        },
        "metadata": []
    }
    
    # Add images if present
    if "vehicle_images" in batch[0]:
        batched["vehicle_images"] = []
        batched["infrastructure_images"] = []
    
    # Process each frame in the sequence
    for frame_idx in range(len(batch[0]["vehicle_points"])):
        # Vehicle points
        vehicle_points_batch = []
        vehicle_masks_batch = []
        
        for sample_idx in range(batch_size):
            points = batch[sample_idx]["vehicle_points"][frame_idx]
            num_points = points.shape[0]
            
            # Create padded point cloud
            padded = np.zeros((max_vehicle_points[frame_idx], points.shape[1]), dtype=points.dtype)
            padded[:num_points] = points
            
            # Create mask (1 for real points, 0 for padding)
            mask = np.zeros(max_vehicle_points[frame_idx], dtype=bool)
            mask[:num_points] = True
            
            vehicle_points_batch.append(padded)
            vehicle_masks_batch.append(mask)
        
        batched["vehicle_points"].append(torch.from_numpy(np.stack(vehicle_points_batch)))
        batched["point_masks"]["vehicle"].append(torch.from_numpy(np.stack(vehicle_masks_batch)))
        
        # Infrastructure points
        infra_points_batch = []
        infra_masks_batch = []
        
        for sample_idx in range(batch_size):
            points = batch[sample_idx]["infrastructure_points"][frame_idx]
            num_points = points.shape[0]
            
            # Create padded point cloud
            padded = np.zeros((max_infra_points[frame_idx], points.shape[1]), dtype=points.dtype)
            padded[:num_points] = points
            
            # Create mask (1 for real points, 0 for padding)
            mask = np.zeros(max_infra_points[frame_idx], dtype=bool)
            mask[:num_points] = True
            
            infra_points_batch.append(padded)
            infra_masks_batch.append(mask)
        
        batched["infrastructure_points"].append(torch.from_numpy(np.stack(infra_points_batch)))
        batched["point_masks"]["infrastructure"].append(torch.from_numpy(np.stack(infra_masks_batch)))
        
        # Labels and matrices (these are just lists)
        vehicle_labels = [sample["vehicle_labels"][frame_idx] for sample in batch]
        infra_labels = [sample["infrastructure_labels"][frame_idx] for sample in batch]
        coop_labels = [sample["cooperative_labels"][frame_idx] for sample in batch]
        trans_matrices = [sample["transformation_matrices"][frame_idx] for sample in batch]
        
        batched["vehicle_labels"].append(vehicle_labels)
        batched["infrastructure_labels"].append(infra_labels)
        batched["cooperative_labels"].append(coop_labels)
        batched["transformation_matrices"].append(trans_matrices)
        
        # Images if present
        if "vehicle_images" in batch[0]:
            vehicle_images = [sample["vehicle_images"][frame_idx] for sample in batch]
            infra_images = [sample["infrastructure_images"][frame_idx] for sample in batch]
            
            # Convert to torch tensors (assuming images are numpy arrays)
            vehicle_images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in vehicle_images])
            infra_images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in infra_images])
            
            batched["vehicle_images"].append(vehicle_images)
            batched["infrastructure_images"].append(infra_images)
    
    # Metadata
    batched["metadata"] = [sample["metadata"] for sample in batch]
    
    return batched


def create_dataloader(
    dataset_path: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle=None,  # Default to None, will set based on split
    drop_last=None,  # Default to None, will set based on split
    **dataset_kwargs
) -> Tuple[V2XSeqDataset, DataLoader]:
    """
    Create a dataset and dataloader for V2X-Seq.
    """
    dataset = V2XSeqDataset(dataset_path=dataset_path, split=split, **dataset_kwargs)
    
    # Set defaults based on split if not explicitly provided
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last
    )
    
    return dataset, dataloader






