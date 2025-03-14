"""
tracker module for V2X-Seq project.

This module provides the abstract base class for 3D object trackers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BaseTracker(ABC):
    """Abstract base class for all tracking algorithms.
    
    This defines the interface that all tracker implementations must follow.
    """
    
    def __init__(self, config: Dict):
        """Initialize the tracker with configuration parameters.
        
        Args:
            config: Dictionary containing tracker configuration parameters
        """
        self.config = config
        self.tracks = []  # List to store active tracks
        self.next_id = 1  # Counter for generating unique track IDs
    
    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update the tracker with new detections.
        
        Args:
            detections: List of detection dictionaries, each containing at least:
                - 'box': 3D bounding box parameters [x, y, z, w, l, h, yaw]
                - 'score': Detection confidence score
                - 'label': Object class label
                
        Returns:
            List of track dictionaries, each containing at least:
                - 'box': 3D bounding box parameters [x, y, z, w, l, h, yaw]
                - 'score': Detection confidence score
                - 'label': Object class label
                - 'id': Unique tracking ID
                - 'age': Track age in frames
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the tracker state."""
        pass


class Track:
    """Base class to represent a single tracked object."""
    
    def __init__(self, detection: Dict, track_id: int):
        """Initialize a new track from a detection.
        
        Args:
            detection: Detection dictionary containing object information
            track_id: Unique ID for this track
        """
        self.id = track_id
        self.box = detection['box']  # 3D bounding box [x, y, z, w, l, h, yaw]
        self.score = detection['score']
        self.label = detection['label']
        self.age = 1  # How many frames this track has existed
        self.hits = 1  # Number of times this track has been matched to a detection
        self.time_since_update = 0  # Frames since last update
        
    def update(self, detection: Dict):
        """Update this track with a new matched detection.
        
        Args:
            detection: Detection dictionary containing new object information
        """
        self.box = detection['box']
        self.score = detection['score']
        self.hits += 1
        self.time_since_update = 0
        
    def increment_age(self):
        """Increment the age of this track by one frame."""
        self.age += 1
        self.time_since_update += 1
        
    def to_dict(self) -> Dict:
        """Convert this track to a dictionary representation.
        
        Returns:
            Dictionary containing track information
        """
        return {
            'id': self.id,
            'box': self.box,
            'score': self.score,
            'label': self.label,
            'age': self.age,
            'time_since_update': self.time_since_update
        }


class SimpleTracker(BaseTracker):
    """A simple IoU-based 3D object tracker.
    
    This is a basic implementation that associates detections to tracks
    based on 3D IoU overlap.
    """
    
    def __init__(self, config: Dict):
        """Initialize the simple tracker.
        
        Args:
            config: Dictionary containing tracker configuration with:
                - 'iou_threshold': Minimum IoU for matching detection to track
                - 'max_age': Maximum number of frames a track can go unmatched
                - 'min_hits': Minimum number of hits needed to confirm a track
        """
        super().__init__(config)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.max_age = config.get('max_age', 3)
        self.min_hits = config.get('min_hits', 3)
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections using IoU matching.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of track dictionaries
        """
        # If no tracks exist yet, initialize tracks from detections
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return [track.to_dict() for track in self.tracks]
            
        # Increment age of existing tracks
        for track in self.tracks:
            track.increment_age()
            
        # If no detections, just return existing tracks
        if len(detections) == 0:
            valid_tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            self.tracks = valid_tracks
            return [track.to_dict() for track in self.tracks 
                    if track.hits >= self.min_hits]
        
        # Compute IoU matrix between tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou_3d(track.box, det['box'])
        
        # Hungarian algorithm for optimal assignment
        # This is a simplified greedy approach - in practice you would use the
        # Hungarian algorithm for optimal matching
        matched_indices = self._greedy_match(iou_matrix)
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Get unmatched detections and tracks
        matched_det_indices = [det_idx for _, det_idx in matched_indices]
        unmatched_detections = [detections[i] for i in range(len(detections)) 
                               if i not in matched_det_indices]
        
        matched_track_indices = [track_idx for track_idx, _ in matched_indices]
        unmatched_tracks = [self.tracks[i] for i in range(len(self.tracks)) 
                           if i not in matched_track_indices]
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            self.tracks.append(Track(det, self.next_id))
            self.next_id += 1
        
        # Remove old unmatched tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return tracks that meet the minimum hits requirement
        return [track.to_dict() for track in self.tracks 
                if track.hits >= self.min_hits]
    
    def reset(self):
        """Reset the tracker state."""
        self.tracks = []
        self.next_id = 1
    
    def _calculate_iou_3d(self, box1, box2) -> float:
        """Calculate 3D IoU between two bounding boxes.
        
        This is a simplified 2D IoU calculation based on BEV (bird's eye view).
        For a complete 3D IoU, you would need to consider overlap in all dimensions.
        
        Args:
            box1: First box parameters [x, y, z, w, l, h, yaw]
            box2: Second box parameters [x, y, z, w, l, h, yaw]
            
        Returns:
            IoU score between 0 and 1
        """
        # For simplicity, this calculates 2D IoU in BEV
        # In a real implementation, you would calculate proper 3D IoU
        # or use a library like nuscenes-devkit with rotation handling
        
        # Extract box centers and dimensions
        x1, y1 = box1[0], box1[1]
        w1, l1 = box1[3], box1[4]
        
        x2, y2 = box2[0], box2[1]
        w2, l2 = box2[3], box2[4]
        
        # Simplification: ignore rotation for this example
        # Calculate box corners in 2D
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - l1/2, y1 + l1/2
        
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - l2/2, y2 + l2/2
        
        # Calculate intersection area
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union area
        area1 = w1 * l1
        area2 = w2 * l2
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    def _greedy_match(self, iou_matrix):
        """Perform greedy matching based on IoU.
        
        This is a simplified matching strategy. For better performance,
        use Hungarian algorithm (e.g., from scipy.optimize.linear_sum_assignment).
        
        Args:
            iou_matrix: Matrix of IoU values between tracks and detections
            
        Returns:
            List of (track_idx, detection_idx) pairs for matched tracks/detections
        """
        # Find matches above the IoU threshold
        matches = []
        
        # Sort all possible matches by IoU (highest first)
        track_indices, det_indices = np.where(iou_matrix >= self.iou_threshold)
        iou_values = iou_matrix[track_indices, det_indices]
        
        # Sort by IoU in descending order
        sort_indices = np.argsort(-iou_values)
        track_indices = track_indices[sort_indices]
        det_indices = det_indices[sort_indices]
        
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if track_idx not in matched_tracks and det_idx not in matched_dets:
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        return matches