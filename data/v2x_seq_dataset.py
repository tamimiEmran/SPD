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
            
            # Filter frames belonging to this sequence
            for frame_id, frame_info in coop_data.items():
                if frame_info["vehicle_sequence"] == seq_id:
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
        
        frame_info = coop_info.get(frame_id, {})
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
            # Simulate latency if requested by using an earlier infrastructure frame
            if self.simulate_latency:
                infrastructure_frame = self._get_delayed_frame(
                    infrastructure_frame, self.latency_ms
                )
            
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
        
        frame_info = veh_info.get(frame_id, {})
        if not frame_info:
            logger.warning(f"Missing vehicle info for {frame_id}")
            return data
        
        # Load point cloud
        pc_path = self.dataset_path / frame_info["pointcloud_path"]
        if pc_path.exists():
            points = self._load_point_cloud(pc_path)
            # Apply augmentation if specified
            if self.augment:
                points = augment_point_cloud(points)
            data["points"] = points
        
        # Load labels
        label_path = self.dataset_path / frame_info["label_lidar_std_path"]
        if label_path.exists():
            with open(label_path, 'r') as f:
                data["labels"] = json.load(f)
        
        # Load image if required
        if self.use_image:
            img_path = self.dataset_path / frame_info["image_path"]
            if img_path.exists():
                data["image"] = self._load_image(img_path)
        
        # Extract timestamp
        data["timestamp"] = frame_info.get("pointcloud_timestamp")
        
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
        
        frame_info = inf_info.get(frame_id, {})
        if not frame_info:
            logger.warning(f"Missing infrastructure info for {frame_id}")
            return data
        
        # Load point cloud
        pc_path = self.dataset_path / frame_info["pointcloud_path"]
        if pc_path.exists():
            points = self._load_point_cloud(pc_path)
            data["points"] = points
        
        # Load labels
        label_path = self.dataset_path / frame_info["label_lidar_std_path"]
        if label_path.exists():
            with open(label_path, 'r') as f:
                data["labels"] = json.load(f)
        
        # Load image if required
        if self.use_image:
            img_path = self.dataset_path / frame_info["image_path"]
            if img_path.exists():
                data["image"] = self._load_image(img_path)
        
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
        
        # Placeholder implementation - replace with actual loading code
        try:
            # Using numpy to load a binary file for this example
            # In practice, you'd use a proper PCD loader
            points = np.fromfile(file_path, dtype=np.float32)
            points = points.reshape(-1, 4)  # x, y, z, intensity
            
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
            matrices["veh_lidar_to_world"] = self._compute_transformation_matrix(
                lidar_to_novatel, novatel_to_world
            )
            
            # Infrastructure virtualLiDAR to world
            vlidar_to_world_path = inf_calib_path / "virtuallidar_to_world" / f"{infrastructure_frame}.json"
            
            with open(vlidar_to_world_path, 'r') as f:
                vlidar_to_world = json.load(f)
            
            matrices["inf_lidar_to_world"] = np.array(vlidar_to_world["transform_matrix"])
            
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
    
    def _compute_transformation_matrix(self, first: Dict, second: Dict) -> np.ndarray:
        """
        Compute combined transformation matrix from two transformations.
        
        Args:
            first: First transformation (e.g., lidar_to_novatel)
            second: Second transformation (e.g., novatel_to_world)
            
        Returns:
            Combined 4x4 transformation matrix
        """
        # Extract transformation parameters
        transform_matrix1 = np.array(first.get("transform_matrix", np.eye(4)))
        transform_matrix2 = np.array(second.get("transform_matrix", np.eye(4)))
        
        # Combine transformations (matrix multiplication)
        combined = transform_matrix2 @ transform_matrix1
        
        return combined
    
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
        
        current_frame_info = inf_info.get(frame_id, {})
        if not current_frame_info:
            return frame_id  # Fallback to current frame
        
        current_timestamp = float(current_frame_info.get("pointcloud_timestamp", 0))
        target_timestamp = current_timestamp - (latency_ms / 1000.0)
        
        # Find the closest earlier frame
        best_frame = frame_id
        min_diff = float('inf')
        
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



import unittest
import numpy as np
import os
import sys
import tempfile
import json
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add current directory to path if needed
sys.path.append('.')

# Import the module to test
# If this fails, adjust the path as needed
try:
    from v2x_seq_dataset import V2XSeqDataset, collate_fn, create_dataloader
except ImportError:
    print("Could not import V2XSeqDataset. Make sure v2x_seq_dataset.py is in the current directory or PYTHONPATH.")
    sys.exit(1)


class TestV2XSeqMethods(unittest.TestCase):
    """Tests for V2X-Seq dataset methods."""
    
    def setUp(self):
        """Set up a minimal dataset structure for testing."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        os.makedirs(os.path.join(self.test_dir, "cooperative"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "cooperative/label"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "vehicle-side"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "vehicle-side/velodyne"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "vehicle-side/label/lidar"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "vehicle-side/calib/lidar_to_novatel"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "vehicle-side/calib/novatel_to_world"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "infrastructure-side"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "infrastructure-side/velodyne"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "infrastructure-side/label/virtuallidar"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "infrastructure-side/calib/virtuallidar_to_world"), exist_ok=True)
        
        # Create sample split file
        split_data = {
            "seq_001": {"frames": ["frame_001", "frame_002", "frame_003"]},
            "seq_002": {"frames": ["frame_004", "frame_005"]}
        }
        with open(os.path.join(self.test_dir, "train_split.json"), 'w') as f:
            json.dump(split_data, f)
        
        # Create sample cooperative data info
        coop_data = {
            "frame_001": {
                "vehicle_frame": "v_frame_001",
                "infrastructure_frame": "i_frame_001",
                "vehicle_sequence": "seq_001",
                "infrastructure_sequence": "seq_001"
            },
            "frame_002": {
                "vehicle_frame": "v_frame_002",
                "infrastructure_frame": "i_frame_002",
                "vehicle_sequence": "seq_001",
                "infrastructure_sequence": "seq_001"
            },
            "frame_003": {
                "vehicle_frame": "v_frame_003",
                "infrastructure_frame": "i_frame_003",
                "vehicle_sequence": "seq_001",
                "infrastructure_sequence": "seq_001"
            },
            "frame_004": {
                "vehicle_frame": "v_frame_004",
                "infrastructure_frame": "i_frame_004",
                "vehicle_sequence": "seq_002",
                "infrastructure_sequence": "seq_002"
            },
            "frame_005": {
                "vehicle_frame": "v_frame_005",
                "infrastructure_frame": "i_frame_005",
                "vehicle_sequence": "seq_002",
                "infrastructure_sequence": "seq_002"
            }
        }
        with open(os.path.join(self.test_dir, "cooperative/data_info.json"), 'w') as f:
            json.dump(coop_data, f)
        
        # Create sample vehicle data info
        vehicle_data = {
            "v_frame_001": {
                "pointcloud_path": "vehicle-side/velodyne/v_frame_001.pcd",
                "label_lidar_std_path": "vehicle-side/label/lidar/v_frame_001.json",
                "pointcloud_timestamp": "1616000000.000000"
            },
            "v_frame_002": {
                "pointcloud_path": "vehicle-side/velodyne/v_frame_002.pcd",
                "label_lidar_std_path": "vehicle-side/label/lidar/v_frame_002.json",
                "pointcloud_timestamp": "1616000100.000000"
            },
            "v_frame_003": {
                "pointcloud_path": "vehicle-side/velodyne/v_frame_003.pcd",
                "label_lidar_std_path": "vehicle-side/label/lidar/v_frame_003.json",
                "pointcloud_timestamp": "1616000200.000000"
            },
            "v_frame_004": {
                "pointcloud_path": "vehicle-side/velodyne/v_frame_004.pcd",
                "label_lidar_std_path": "vehicle-side/label/lidar/v_frame_004.json",
                "pointcloud_timestamp": "1616000300.000000"
            },
            "v_frame_005": {
                "pointcloud_path": "vehicle-side/velodyne/v_frame_005.pcd",
                "label_lidar_std_path": "vehicle-side/label/lidar/v_frame_005.json",
                "pointcloud_timestamp": "1616000400.000000"
            }
        }
        with open(os.path.join(self.test_dir, "vehicle-side/data_info.json"), 'w') as f:
            json.dump(vehicle_data, f)
        
        # Create sample infrastructure data info
        infra_data = {
            "i_frame_001": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_001.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_001.json",
                "pointcloud_timestamp": "1616000000.000000"
            },
            "i_frame_002": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_002.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_002.json",
                "pointcloud_timestamp": "1616000100.000000"
            },
            "i_frame_003": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_003.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_003.json",
                "pointcloud_timestamp": "1616000200.000000"
            },
            "i_frame_004": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_004.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_004.json",
                "pointcloud_timestamp": "1616000300.000000"
            },
            "i_frame_005": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_005.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_005.json",
                "pointcloud_timestamp": "1616000400.000000"
            },
            # Extra frames with different timestamps for latency testing
            "i_frame_000": {
                "pointcloud_path": "infrastructure-side/velodyne/i_frame_000.pcd",
                "label_lidar_std_path": "infrastructure-side/label/virtuallidar/i_frame_000.json",
                "pointcloud_timestamp": "1615999800.000000"  # 200ms earlier than frame_001
            }
        }
        with open(os.path.join(self.test_dir, "infrastructure-side/data_info.json"), 'w') as f:
            json.dump(infra_data, f)
            
        # Create point cloud files
        for i in range(1, 6):
            # Create vehicle point cloud file
            v_points = np.random.rand(10, 4).astype(np.float32)
            v_points_path = os.path.join(self.test_dir, f"vehicle-side/velodyne/v_frame_00{i}.pcd")
            v_points.tofile(v_points_path)
            
            # Create infrastructure point cloud file
            i_points = np.random.rand(10, 4).astype(np.float32)
            i_points_path = os.path.join(self.test_dir, f"infrastructure-side/velodyne/i_frame_00{i}.pcd")
            i_points.tofile(i_points_path)
            
            # Create vehicle labels file
            v_labels = [{"token": f"v_token_{i}", "type": "Car", "track_id": f"track_{i}"}]
            v_labels_path = os.path.join(self.test_dir, f"vehicle-side/label/lidar/v_frame_00{i}.json")
            with open(v_labels_path, 'w') as f:
                json.dump(v_labels, f)
                
            # Create infrastructure labels file
            i_labels = [{"token": f"i_token_{i}", "type": "Car", "track_id": f"track_{i}"}]
            i_labels_path = os.path.join(self.test_dir, f"infrastructure-side/label/virtuallidar/i_frame_00{i}.json")
            with open(i_labels_path, 'w') as f:
                json.dump(i_labels, f)
                
            # Create cooperative labels file
            coop_labels = [{"token": f"coop_token_{i}", "type": "Car", "track_id": f"track_{i}"}]
            coop_labels_path = os.path.join(self.test_dir, f"cooperative/label/frame_00{i}.json")
            with open(coop_labels_path, 'w') as f:
                json.dump(coop_labels, f)
                
            # Create calibration files
            lidar_to_novatel = {"transform_matrix": np.eye(4).tolist()}
            lidar_to_novatel_path = os.path.join(self.test_dir, f"vehicle-side/calib/lidar_to_novatel/v_frame_00{i}.json")
            with open(lidar_to_novatel_path, 'w') as f:
                json.dump(lidar_to_novatel, f)
                
            novatel_to_world = {"transform_matrix": np.eye(4).tolist()}
            novatel_to_world_path = os.path.join(self.test_dir, f"vehicle-side/calib/novatel_to_world/v_frame_00{i}.json")
            with open(novatel_to_world_path, 'w') as f:
                json.dump(novatel_to_world, f)
                
            vlidar_to_world = {"transform_matrix": np.eye(4).tolist()}
            vlidar_to_world_path = os.path.join(self.test_dir, f"infrastructure-side/calib/virtuallidar_to_world/i_frame_00{i}.json")
            with open(vlidar_to_world_path, 'w') as f:
                json.dump(vlidar_to_world, f)
        
        # Create the extra frame for latency testing
        i_points = np.random.rand(10, 4).astype(np.float32)
        i_points_path = os.path.join(self.test_dir, "infrastructure-side/velodyne/i_frame_000.pcd")
        i_points.tofile(i_points_path)
        
        i_labels = [{"token": "i_token_0", "type": "Car", "track_id": "track_0"}]
        i_labels_path = os.path.join(self.test_dir, "infrastructure-side/label/virtuallidar/i_frame_000.json")
        with open(i_labels_path, 'w') as f:
            json.dump(i_labels, f)
            
        vlidar_to_world = {"transform_matrix": np.eye(4).tolist()}
        vlidar_to_world_path = os.path.join(self.test_dir, "infrastructure-side/calib/virtuallidar_to_world/i_frame_000.json")
        with open(vlidar_to_world_path, 'w') as f:
            json.dump(vlidar_to_world, f)
    
    def tearDown(self):
        """Clean up the temporary directory after tests."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test dataset initialization."""
        # Create dataset with minimal parameters
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            segment_length=2
        )
        
        # Check that dataset attributes are set correctly
        self.assertEqual(dataset.dataset_path, Path(self.test_dir))
        self.assertEqual(dataset.split, "train")
        self.assertEqual(dataset.segment_length, 2)
        self.assertTrue(dataset.use_infrastructure)
        self.assertFalse(dataset.use_image)
        
        # Check that sequences were loaded correctly
        self.assertEqual(len(dataset.sequences), 2)  # Two sequences in the split file
        self.assertIn("seq_001", dataset.sequences)
        self.assertIn("seq_002", dataset.sequences)
        
        # Check that frames were mapped correctly
        self.assertEqual(len(dataset.sequence_frames["seq_001"]), 3)  # Three frames in seq_001
        self.assertEqual(len(dataset.sequence_frames["seq_002"]), 2)  # Two frames in seq_002
        
        # Check the mapping from frames to sequences
        for frame_id in ["frame_001", "frame_002", "frame_003"]:
            self.assertEqual(dataset.frame_to_sequence_mapping[frame_id], "seq_001")
        
        for frame_id in ["frame_004", "frame_005"]:
            self.assertEqual(dataset.frame_to_sequence_mapping[frame_id], "seq_002")
    
    def test_idx_to_sequence_frame(self):
        """Test _idx_to_sequence_frame method."""
        # Create dataset with specific segment length
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            segment_length=2  # Two frames per segment
        )
        
        # Test conversion from flat index to sequence and frame indices
        seq_idx, frame_idx = dataset._idx_to_sequence_frame(0)
        self.assertEqual(seq_idx, 0)  # First sequence (seq_001)
        self.assertEqual(frame_idx, 0)  # First frame in sequence
        
        seq_idx, frame_idx = dataset._idx_to_sequence_frame(1)
        self.assertEqual(seq_idx, 0)  # First sequence (seq_001)
        self.assertEqual(frame_idx, 1)  # Second frame in sequence
        
        seq_idx, frame_idx = dataset._idx_to_sequence_frame(2)
        self.assertEqual(seq_idx, 1)  # Second sequence (seq_002)
        self.assertEqual(frame_idx, 0)  # First frame in second sequence
        
        # Test out of bounds index
        with self.assertRaises(IndexError):
            dataset._idx_to_sequence_frame(10)  # Out of bounds index
    
    def test_len(self):
        """Test __len__ method."""
        # Create dataset with segment length of 2
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            segment_length=2
        )
        
        # Expected length: 
        # seq_001 has 3 frames -> 2 segments (3 - 2 + 1)
        # seq_002 has 2 frames -> 1 segment (2 - 2 + 1)
        # Total: 3 segments
        self.assertEqual(len(dataset), 3)
        
        # Create dataset with segment length of 3
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            segment_length=3
        )
        
        # Expected length:
        # seq_001 has 3 frames -> 1 segment (3 - 3 + 1)
        # seq_002 has 2 frames -> 0 segments (2 < 3)
        # Total: 1 segment
        self.assertEqual(len(dataset), 1)
    
    def test_load_point_cloud(self):
        """Test _load_point_cloud method."""
        # Create dataset
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train"
        )
        
        # Test loading a vehicle point cloud
        v_pc_path = Path(os.path.join(self.test_dir, "vehicle-side/velodyne/v_frame_001.pcd"))
        v_points = dataset._load_point_cloud(v_pc_path)
        
        # Check the shape and type
        self.assertEqual(v_points.shape, (10, 4))  # 10 points with 4 values each
        self.assertEqual(v_points.dtype, np.float32)
        
        # Test loading an infrastructure point cloud
        i_pc_path = Path(os.path.join(self.test_dir, "infrastructure-side/velodyne/i_frame_001.pcd"))
        i_points = dataset._load_point_cloud(i_pc_path)
        
        # Check the shape and type
        self.assertEqual(i_points.shape, (10, 4))
        self.assertEqual(i_points.dtype, np.float32)
        
        # Test max_points parameter
        dataset.max_points = 5
        limited_points = dataset._load_point_cloud(v_pc_path)
        self.assertEqual(limited_points.shape[0], 5)  # Should have 5 points max
    
    def test_get_delayed_frame(self):
        """Test _get_delayed_frame method."""
        # Create dataset with latency simulation
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            simulate_latency=True,
            latency_ms=200
        )
        
        # Test getting a delayed frame
        # i_frame_001 has timestamp 1616000000.000000
        # i_frame_000 has timestamp 1615999800.000000 (200ms earlier)
        delayed_frame = dataset._get_delayed_frame("i_frame_001", 200)
        
        # Should find i_frame_000 as the delayed frame
        self.assertEqual(delayed_frame, "i_frame_000")
        
        # Test with a frame that has no earlier options
        delayed_frame = dataset._get_delayed_frame("i_frame_000", 200)
        self.assertEqual(delayed_frame, "i_frame_000")  # Should return the same frame
        
        # Test with a non-existent frame
        delayed_frame = dataset._get_delayed_frame("non_existent_frame", 200)
        self.assertEqual(delayed_frame, "non_existent_frame")  # Should return the same frame
    
    def test_compute_transformation_matrix(self):
        """Test _compute_transformation_matrix method."""
        # Create dataset
        dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train"
        )
        
        # Create test transformation matrices
        transform1 = {
            "transform_matrix": [
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 0, 1]
            ]
        }
        
        transform2 = {
            "transform_matrix": [
                [0, -1, 0, 4],
                [1, 0, 0, 5],
                [0, 0, 1, 6],
                [0, 0, 0, 1]
            ]
        }
        
        # Compute combined transformation
        combined = dataset._compute_transformation_matrix(transform1, transform2)
        
        # Expected result: transform2 @ transform1
        expected = np.array([
            [0, -1, 0, 4 - 2],  # -1*2 = -2
            [1, 0, 0, 5 + 1],   # 1*1 = 1
            [0, 0, 1, 6 + 3],   # 1*3 = 3
            [0, 0, 0, 1]
        ])
        
        np.testing.assert_array_equal(combined, expected)
        
        # Test with missing transformation matrix
        transform3 = {}
        combined = dataset._compute_transformation_matrix(transform3, transform2)
        
        # Should use identity matrix for transform3
        expected = np.array(transform2["transform_matrix"])
        np.testing.assert_array_equal(combined, expected)
    
    def test_collate_fn(self):
        """Test collate_fn function."""
        # Create sample batch data
        batch = [
            {
                "vehicle_points": [
                    np.random.rand(5, 4).astype(np.float32),
                    np.random.rand(6, 4).astype(np.float32)
                ],
                "infrastructure_points": [
                    np.random.rand(7, 4).astype(np.float32),
                    np.random.rand(8, 4).astype(np.float32)
                ],
                "vehicle_labels": [
                    [{"id": "obj1", "category": "Car"}],
                    [{"id": "obj2", "category": "Van"}]
                ],
                "infrastructure_labels": [
                    [{"id": "inf1", "category": "Car"}],
                    [{"id": "inf2", "category": "Truck"}]
                ],
                "cooperative_labels": [
                    [{"id": "coop1", "category": "Car"}],
                    [{"id": "coop2", "category": "Bus"}]
                ],
                "transformation_matrices": [
                    {"matrix1": np.eye(4)},
                    {"matrix2": np.eye(4)}
                ],
                "metadata": {
                    "sequence_id": "seq_001", 
                    "frames": ["frame_001", "frame_002"]
                }
            },
            {
                "vehicle_points": [
                    np.random.rand(4, 4).astype(np.float32),
                    np.random.rand(5, 4).astype(np.float32)
                ],
                "infrastructure_points": [
                    np.random.rand(6, 4).astype(np.float32),
                    np.random.rand(7, 4).astype(np.float32)
                ],
                "vehicle_labels": [
                    [{"id": "obj3", "category": "Pedestrian"}],
                    [{"id": "obj4", "category": "Cyclist"}]
                ],
                "infrastructure_labels": [
                    [{"id": "inf3", "category": "Van"}],
                    [{"id": "inf4", "category": "Bus"}]
                ],
                "cooperative_labels": [
                    [{"id": "coop3", "category": "Truck"}],
                    [{"id": "coop4", "category": "Car"}]
                ],
                "transformation_matrices": [
                    {"matrix3": np.eye(4)},
                    {"matrix4": np.eye(4)}
                ],
                "metadata": {
                    "sequence_id": "seq_002", 
                    "frames": ["frame_003", "frame_004"]
                }
            }
        ]
        
        # Apply collate_fn
        collated = collate_fn(batch)
        
        # Verify structure and dimensions
        self.assertIn("vehicle_points", collated)
        self.assertIn("infrastructure_points", collated)
        self.assertIn("vehicle_labels", collated)
        self.assertIn("infrastructure_labels", collated)
        self.assertIn("cooperative_labels", collated)
        self.assertIn("transformation_matrices", collated)
        self.assertIn("point_masks", collated)
        
        # Check shapes
        self.assertEqual(len(collated["vehicle_points"]), 2)  # 2 frames
        self.assertEqual(collated["vehicle_points"][0].shape[0], 2)  # Batch size = 2
        self.assertEqual(collated["vehicle_points"][0].shape[1], 5)  # Max points in first frame = 5
        self.assertEqual(collated["vehicle_points"][1].shape[1], 6)  # Max points in second frame = 6
        
        # Check masks
        self.assertEqual(torch.sum(collated["point_masks"]["vehicle"][0][0]), 5)  # 5 real points
        self.assertEqual(torch.sum(collated["point_masks"]["vehicle"][0][1]), 4)  # 4 real points
    
    def test_create_dataloader(self):
        """Test create_dataloader function."""
        # Create dataloader
        dataset, dataloader = create_dataloader(
            dataset_path=self.test_dir,
            split="train",
            batch_size=2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            segment_length=2
        )
        
        # Check that dataset and dataloader were created
        self.assertIsInstance(dataset, V2XSeqDataset)
        self.assertIsInstance(dataloader, DataLoader)
        
        # Check dataloader properties
        self.assertEqual(dataloader.batch_size, 2)
        
        # Create a separate dataloader without using the create_dataloader function
        # This is to test that we can create a dataloader with different parameters
        test_dataset = V2XSeqDataset(
            dataset_path=self.test_dir,
            split="train",
            segment_length=2
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=4,  # Different batch size
            shuffle=False,  # Explicitly set shuffle to False
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Check the manually created dataloader
        self.assertEqual(test_dataloader.batch_size, 4)


if __name__ == "__main__":
    # Check if running in Jupyter notebook
    try:
        import IPython
        if IPython.get_ipython() is not None:
            # Running in Jupyter - avoid sys.exit() call
            unittest.main(argv=['first-arg-is-ignored'], exit=False)
        else:
            unittest.main()
    except (ImportError, AttributeError):
        # Not in Jupyter
        unittest.main()