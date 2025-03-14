"""
ab3dmot module for V2X-Seq project.

This module provides an implementation of the AB3DMOT tracking algorithm.
Based on: "A Baseline for 3D Multi-Object Tracking" (Weng et al.)
https://arxiv.org/abs/1907.03961
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional, Any
import copy

from .tracker import BaseTracker


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox: np.ndarray, info: Dict):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box in the format [x, y, z, w, l, h, yaw]
            info: Additional information about the detection
        """
        # Define constant velocity model
        # State: [x, y, z, w, l, h, yaw, vx, vy, vz]
        # We include width, length, height, and yaw angle but assume they are constant
        self.kf = KalmanFilter(dim_x=10, dim_z=7)  
        
        # Initialize state transition matrix (motion model)
        self.kf.F = np.eye(10)
        # Add velocity component to position
        self.kf.F[0, 7] = 1.0  # x += vx
        self.kf.F[1, 8] = 1.0  # y += vy
        self.kf.F[2, 9] = 1.0  # z += vz
        
        # Initialize measurement matrix (measurement model)
        self.kf.H = np.zeros((7, 10))
        self.kf.H[:7, :7] = np.eye(7)  # We only measure position, size, and orientation
        
        # Initialize the covariance matrices
        self.kf.P[7:, 7:] *= 1000.  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        
        # Set process noise
        self.kf.Q[7:, 7:] *= 0.01  # Process noise
        
        # Set measurement noise
        self.kf.R[3:6, 3:6] *= 0.01  # Low noise for size measurements
        self.kf.R[0:3, 0:3] *= 1.0   # Higher noise for position measurements
        self.kf.R[6, 6] *= 0.01      # Low noise for orientation measurements
        
        # Initialize state with measurement
        self.kf.x[:7] = bbox.reshape(7, 1)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1           # Number of total hits including the first detection
        self.hit_streak = 1     # Number of continuing hits
        self.age = 1
        self.info = info        # Additional info from the detection
        
    def update(self, bbox: np.ndarray, info: Dict):
        """
        Update the state vector with observed bbox.
        
        Args:
            bbox: Bounding box in the format [x, y, z, w, l, h, yaw]
            info: Additional information about the detection
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.info = info  # Update with the latest detection info
        self.kf.update(bbox.reshape(7, 1))  # Update the Kalman filter
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        
        Returns:
            Predicted bounding box
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:7].reshape(-1))
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.
        
        Returns:
            Current bounding box [x, y, z, w, l, h, yaw]
        """
        return self.kf.x[:7].reshape(-1)


class KalmanFilter:
    """
    A simple Kalman filter implementation for 3D object tracking.
    
    This is a simplified version - in a real implementation you would use
    a library like filterpy for a complete Kalman filter.
    """
    
    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize Kalman Filter.
        
        Args:
            dim_x: Dimension of the state vector
            dim_z: Dimension of the measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State vector
        self.x = np.zeros((dim_x, 1)) 
        
        # State transition matrix
        self.F = np.eye(dim_x)
        
        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
        
        # Measurement noise covariance
        self.R = np.eye(dim_z)
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        
        # State covariance matrix
        self.P = np.eye(dim_x)
        
    def predict(self):
        """
        Predict next state based on motion model.
        """
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray):
        """
        Update state based on measurement.
        
        Args:
            z: Measurement vector
        """
        # Calculate innovation (residual)
        y = z - self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update state covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


class AB3DMOT(BaseTracker):
    """
    3D Multi-Object Tracker using 3D IoU and Kalman Filter.
    
    Implementation of the AB3DMOT algorithm from Weng et al.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the AB3DMOT tracker.
        
        Args:
            config: Dictionary containing tracker configuration with:
                - 'max_age': Maximum number of frames a track can go unmatched
                - 'min_hits': Minimum number of hits needed to confirm a track
                - 'iou_threshold': Minimum IoU for matching detection to track
        """
        super().__init__(config)
        self.max_age = config.get('max_age', 3)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.trackers = []  # List of KalmanBoxTracker objects
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections using 3D IoU and Kalman filtering.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of track dictionaries
        """
        # Convert detections to the format expected by KalmanBoxTracker
        detection_boxes = []
        detection_info = []
        
        for det in detections:
            box = np.array(det['box'])  # [x, y, z, w, l, h, yaw]
            info = {
                'score': det['score'],
                'label': det['label']
            }
            detection_boxes.append(box)
            detection_info.append(info)
            
        # Get predicted locations from existing trackers
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)
                
        # Remove invalid trackers
        trks = np.array(trks)
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers based on 3D IoU
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detection_boxes, trks)
            
        # Update matched trackers with assigned detections
        for t, trk_idx in enumerate(matched[:, 1]):
            self.trackers[trk_idx].update(detection_boxes[matched[t, 0]], detection_info[matched[t, 0]])
            
        # Create and initialize new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(detection_boxes[det_idx], detection_info[det_idx])
            self.trackers.append(trk)
            
        # Remove dead tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            elif trk.hits >= self.min_hits and trk.time_since_update <= self.max_age:
                # Return valid tracks
                d = trk.get_state()
                ret.append({
                    'id': trk.id,
                    'box': d.tolist(),
                    'score': trk.info['score'],
                    'label': trk.info['label'],
                    'age': trk.age,
                    'time_since_update': trk.time_since_update
                })
                
        return ret
        
    def reset(self):
        """Reset the tracker state."""
        self.trackers = []
        KalmanBoxTracker.count = 0
        
    def _associate_detections_to_trackers(self, detections: np.ndarray, 
                                          trackers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assigns detections to tracked objects using 3D IoU.
        
        Args:
            detections: Numpy array of shape (n, 7) with detections [x, y, z, w, l, h, yaw]
            trackers: Numpy array of shape (m, 7) with tracker states [x, y, z, w, l, h, yaw]
            
        Returns:
            Tuple of:
                - matched indices: Array of shape (k, 2) with (det_idx, trk_idx) pairs
                - unmatched_detections: Array of unmatched detection indices
                - unmatched_trackers: Array of unmatched tracker indices
        """
        if len(trackers) == 0:
            return (np.empty((0, 2), dtype=int), 
                    np.arange(len(detections)), 
                    np.empty(0, dtype=int))
            
        if len(detections) == 0:
            return (np.empty((0, 2), dtype=int), 
                    np.empty(0, dtype=int), 
                    np.arange(len(trackers)))
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._calculate_3d_iou(det, trk)
                
        # Use Hungarian algorithm for optimal assignment
        # Negate IoU because the algorithm minimizes cost
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        
        # Filter out assignments with low IoU
        iou_mask = iou_matrix[row_ind, col_ind] >= self.iou_threshold
        matched_indices = matched_indices[iou_mask]
        
        # Find unmatched detections and trackers
        unmatched_detections = np.array([d for d in range(len(detections)) 
                                       if d not in matched_indices[:, 0]])
        unmatched_trackers = np.array([t for t in range(len(trackers)) 
                                      if t not in matched_indices[:, 1]])
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    def _calculate_3d_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate 3D IoU between two boxes.
        
        This is a simplified implementation. For a complete 3D IoU with rotation,
        you would need more complex geometry calculations.
        
        Args:
            box1: First box parameters [x, y, z, w, l, h, yaw]
            box2: Second box parameters [x, y, z, w, l, h, yaw]
            
        Returns:
            IoU score between 0 and 1
        """
        # Extract box centers and dimensions
        x1, y1, z1 = box1[0], box1[1], box1[2]
        w1, l1, h1 = box1[3], box1[4], box1[5]
        
        x2, y2, z2 = box2[0], box2[1], box2[2]
        w2, l2, h2 = box2[3], box2[4], box2[5]
        
        # For simplicity, we ignore the rotation angle (yaw)
        # and calculate axis-aligned 3D IoU
        
        # Calculate min/max for each box
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - l1/2, y1 + l1/2
        z1_min, z1_max = z1 - h1/2, z1 + h1/2
        
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - l2/2, y2 + l2/2
        z2_min, z2_max = z2 - h2/2, z2 + h2/2
        
        # Calculate intersection volume
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        intersection = x_overlap * y_overlap * z_overlap
        
        # Calculate individual volumes
        vol1 = w1 * l1 * h1
        vol2 = w2 * l2 * h2
        
        # Calculate union volume
        union = vol1 + vol2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou