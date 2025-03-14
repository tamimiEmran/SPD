"""
late_fusion module for V2X-Seq project.

This module provides functionality for late fusion of detection results from vehicle and infrastructure sensors.
Late fusion combines already-processed detection results from both sources at the object level.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

from fusion.fusion_base import FusionBase
from data.calibration.coordinate_transform import transform_boxes_to_vehicle_frame

logger = logging.getLogger(__name__)

class LateFusion(FusionBase):
    """
    Late Fusion implementation for Vehicle-Infrastructure Cooperative 3D Tracking.
    
    This class implements a late fusion strategy where object detection results from
    both vehicle and infrastructure are combined at the object level.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the late fusion module.
        
        Args:
            config: Dictionary containing configuration parameters
                - match_threshold: Maximum distance for matching detections
                - confidence_weight: Weight for confidence score during fusion
                - nms_threshold: NMS threshold for removing duplicate detections
                - latency_compensation: Whether to compensate for infrastructure latency
        """
        super().__init__(config)
        self.match_threshold = config.get('match_threshold', 2.0)  # meters
        self.confidence_weight = config.get('confidence_weight', 0.7)  # vehicle confidence weight
        self.nms_threshold = config.get('nms_threshold', 0.1)
        self.latency_compensation = config.get('latency_compensation', True)
        
        logger.info(f"Initialized Late Fusion with match_threshold={self.match_threshold}m")
    
    def fuse(self, 
             vehicle_detections: Dict, 
             infrastructure_detections: Dict,
             vehicle_to_world_transform: np.ndarray,
             infrastructure_to_world_transform: np.ndarray,
             timestamp_diff: float = 0.0) -> Dict:
        """
        Fuse vehicle and infrastructure detections.
        
        Args:
            vehicle_detections: Dictionary containing vehicle detection results
                - boxes: (N, 7) array with x, y, z, w, l, h, yaw
                - scores: (N,) array with confidence scores
                - classes: (N,) array with class ids
            infrastructure_detections: Dictionary containing infrastructure detection results
                - boxes: (M, 7) array with x, y, z, w, l, h, yaw
                - scores: (M,) array with confidence scores
                - classes: (M,) array with class ids
            vehicle_to_world_transform: (4, 4) transformation matrix from vehicle to world
            infrastructure_to_world_transform: (4, 4) transformation matrix from infrastructure to world
            timestamp_diff: Time difference between vehicle and infrastructure frames (seconds)
            
        Returns:
            Dictionary containing fused detection results
                - boxes: (K, 7) array with x, y, z, w, l, h, yaw
                - scores: (K,) array with confidence scores
                - classes: (K,) array with class ids
                - sources: (K,) array indicating source (0: vehicle, 1: infrastructure, 2: both)
        """
        # Verify inputs
        if not vehicle_detections or not infrastructure_detections:
            logger.warning("One of the detection inputs is empty")
            if not vehicle_detections:
                return infrastructure_detections
            return vehicle_detections
        
        # Transform infrastructure detections to vehicle frame
        inf_boxes_vehicle_frame = transform_boxes_to_vehicle_frame(
            infrastructure_detections['boxes'],
            infrastructure_to_world_transform,
            vehicle_to_world_transform
        )
        
        # Compensate for latency if enabled
        if self.latency_compensation and timestamp_diff > 0:
            inf_boxes_vehicle_frame = self._compensate_latency(
                inf_boxes_vehicle_frame,
                infrastructure_detections['velocities'] if 'velocities' in infrastructure_detections else None,
                timestamp_diff
            )
        
        # Match detections between vehicle and infrastructure
        matches, unmatched_vehicle, unmatched_infrastructure = self._match_detections(
            vehicle_detections['boxes'],
            inf_boxes_vehicle_frame,
            vehicle_detections['classes'],
            infrastructure_detections['classes']
        )
        
        # Fuse matched detections
        fused_detections = self._fuse_matched_detections(
            vehicle_detections,
            infrastructure_detections,
            inf_boxes_vehicle_frame,
            matches
        )
        
        # Add unmatched detections
        fused_detections = self._add_unmatched_detections(
            fused_detections,
            vehicle_detections,
            infrastructure_detections,
            inf_boxes_vehicle_frame,
            unmatched_vehicle,
            unmatched_infrastructure
        )
        
        # Apply NMS to remove duplicates
        fused_detections = self._apply_nms(fused_detections)
        
        return fused_detections
    
    def _match_detections(self, 
                          vehicle_boxes: np.ndarray, 
                          inf_boxes_vehicle_frame: np.ndarray,
                          vehicle_classes: np.ndarray,
                          inf_classes: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections between vehicle and infrastructure using Hungarian algorithm.
        
        Args:
            vehicle_boxes: (N, 7) array of vehicle detection boxes
            inf_boxes_vehicle_frame: (M, 7) array of infrastructure detection boxes in vehicle frame
            vehicle_classes: (N,) array of vehicle detection classes
            inf_classes: (M,) array of infrastructure detection classes
            
        Returns:
            Tuple containing:
                - List of (vehicle_idx, inf_idx) tuples for matched detections
                - List of unmatched vehicle detection indices
                - List of unmatched infrastructure detection indices
        """
        from scipy.optimize import linear_sum_assignment
        
        n_vehicle = len(vehicle_boxes)
        n_inf = len(inf_boxes_vehicle_frame)
        
        if n_vehicle == 0 or n_inf == 0:
            return [], list(range(n_vehicle)), list(range(n_inf))
        
        # Compute pairwise distances between all detections
        distance_matrix = np.zeros((n_vehicle, n_inf))
        for i in range(n_vehicle):
            for j in range(n_inf):
                # Only match if class IDs are the same
                if vehicle_classes[i] != inf_classes[j]:
                    distance_matrix[i, j] = float('inf')
                else:
                    # Compute 3D IoU or distance-based metric
                    distance_matrix[i, j] = self._compute_box_distance(
                        vehicle_boxes[i], inf_boxes_vehicle_frame[j]
                    )
        
        # Apply Hungarian algorithm to find optimal matching
        vehicle_indices, inf_indices = linear_sum_assignment(distance_matrix)
        
        # Filter matches based on distance threshold
        matches = []
        unmatched_vehicle = list(range(n_vehicle))
        unmatched_inf = list(range(n_inf))
        
        for vehicle_idx, inf_idx in zip(vehicle_indices, inf_indices):
            if distance_matrix[vehicle_idx, inf_idx] <= self.match_threshold:
                matches.append((vehicle_idx, inf_idx))
                if vehicle_idx in unmatched_vehicle:
                    unmatched_vehicle.remove(vehicle_idx)
                if inf_idx in unmatched_inf:
                    unmatched_inf.remove(inf_idx)
        
        return matches, unmatched_vehicle, unmatched_inf
    
    def _compute_box_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute distance between two 3D bounding boxes.
        
        Args:
            box1: First box parameters [x, y, z, w, l, h, yaw]
            box2: Second box parameters [x, y, z, w, l, h, yaw]
            
        Returns:
            Distance between box centers
        """
        # Simple Euclidean distance between box centers
        return np.linalg.norm(box1[:3] - box2[:3])
    
    def _fuse_matched_detections(self,
                                vehicle_detections: Dict,
                                infrastructure_detections: Dict,
                                inf_boxes_vehicle_frame: np.ndarray,
                                matches: List[Tuple[int, int]]) -> Dict:
        """
        Fuse matched detections from vehicle and infrastructure.
        
        Args:
            vehicle_detections: Vehicle detection results
            infrastructure_detections: Infrastructure detection results
            inf_boxes_vehicle_frame: Infrastructure boxes in vehicle frame
            matches: List of matched detection indices
            
        Returns:
            Dictionary with fused detections
        """
        n_matches = len(matches)
        if n_matches == 0:
            return {
                'boxes': np.empty((0, 7), dtype=np.float32),
                'scores': np.empty(0, dtype=np.float32),
                'classes': np.empty(0, dtype=np.int32),
                'sources': np.empty(0, dtype=np.int32)  # 0: vehicle, 1: infrastructure, 2: both
            }
        
        fused_boxes = np.zeros((n_matches, 7), dtype=np.float32)
        fused_scores = np.zeros(n_matches, dtype=np.float32)
        fused_classes = np.zeros(n_matches, dtype=np.int32)
        sources = np.full(n_matches, 2, dtype=np.int32)  # 2 indicates fusion of both sources
        
        for i, (v_idx, i_idx) in enumerate(matches):
            v_box = vehicle_detections['boxes'][v_idx]
            i_box = inf_boxes_vehicle_frame[i_idx]
            v_score = vehicle_detections['scores'][v_idx]
            i_score = infrastructure_detections['scores'][i_idx]
            
            # Weighted average of box parameters based on confidence
            weight_v = self.confidence_weight * v_score
            weight_i = (1 - self.confidence_weight) * i_score
            total_weight = weight_v + weight_i
            
            # Normalize weights
            if total_weight > 0:
                weight_v /= total_weight
                weight_i /= total_weight
            else:
                weight_v = weight_i = 0.5
            
            # Fuse box parameters
            fused_boxes[i] = weight_v * v_box + weight_i * i_box
            
            # Take maximum of scores
            fused_scores[i] = max(v_score, i_score)
            
            # Class should be the same for matched detections
            fused_classes[i] = vehicle_detections['classes'][v_idx]
        
        return {
            'boxes': fused_boxes,
            'scores': fused_scores,
            'classes': fused_classes,
            'sources': sources
        }
    
    def _add_unmatched_detections(self,
                                  fused_detections: Dict,
                                  vehicle_detections: Dict,
                                  infrastructure_detections: Dict,
                                  inf_boxes_vehicle_frame: np.ndarray,
                                  unmatched_vehicle: List[int],
                                  unmatched_infrastructure: List[int]) -> Dict:
        """
        Add unmatched detections to the fused results.
        
        Args:
            fused_detections: Currently fused detections
            vehicle_detections: Vehicle detection results
            infrastructure_detections: Infrastructure detection results
            inf_boxes_vehicle_frame: Infrastructure boxes in vehicle frame
            unmatched_vehicle: Indices of unmatched vehicle detections
            unmatched_infrastructure: Indices of unmatched infrastructure detections
            
        Returns:
            Updated dictionary with all detections
        """
        n_fused = len(fused_detections['boxes'])
        n_unmatched_v = len(unmatched_vehicle)
        n_unmatched_i = len(unmatched_infrastructure)
        n_total = n_fused + n_unmatched_v + n_unmatched_i
        
        # Create arrays for all detections
        all_boxes = np.zeros((n_total, 7), dtype=np.float32)
        all_scores = np.zeros(n_total, dtype=np.float32)
        all_classes = np.zeros(n_total, dtype=np.int32)
        all_sources = np.zeros(n_total, dtype=np.int32)
        
        # Copy existing fused detections
        if n_fused > 0:
            all_boxes[:n_fused] = fused_detections['boxes']
            all_scores[:n_fused] = fused_detections['scores']
            all_classes[:n_fused] = fused_detections['classes']
            all_sources[:n_fused] = fused_detections['sources']
        
        # Add unmatched vehicle detections
        for i, v_idx in enumerate(unmatched_vehicle):
            idx = n_fused + i
            all_boxes[idx] = vehicle_detections['boxes'][v_idx]
            all_scores[idx] = vehicle_detections['scores'][v_idx]
            all_classes[idx] = vehicle_detections['classes'][v_idx]
            all_sources[idx] = 0  # 0 indicates vehicle source
        
        # Add unmatched infrastructure detections
        for i, i_idx in enumerate(unmatched_infrastructure):
            idx = n_fused + n_unmatched_v + i
            all_boxes[idx] = inf_boxes_vehicle_frame[i_idx]
            all_scores[idx] = infrastructure_detections['scores'][i_idx]
            all_classes[idx] = infrastructure_detections['classes'][i_idx]
            all_sources[idx] = 1  # 1 indicates infrastructure source
        
        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'classes': all_classes,
            'sources': all_sources
        }
    
    def _compensate_latency(self,
                           boxes: np.ndarray,
                           velocities: Optional[np.ndarray],
                           timestamp_diff: float) -> np.ndarray:
        """
        Compensate for latency by projecting boxes forward in time.
        
        Args:
            boxes: (N, 7) array of boxes
            velocities: (N, 3) array of velocities or None
            timestamp_diff: Time difference in seconds
            
        Returns:
            Updated boxes with compensated positions
        """
        if velocities is None or timestamp_diff <= 0:
            return boxes
        
        updated_boxes = boxes.copy()
        
        # Apply simple linear motion model
        updated_boxes[:, :3] += velocities * timestamp_diff
        
        return updated_boxes
    
    def _apply_nms(self, detections: Dict) -> Dict:
        """
        Apply non-maximum suppression to remove duplicate detections.
        
        Args:
            detections: Dictionary with detection results
            
        Returns:
            Dictionary with filtered detections
        """
        from fusion.utils.matching import nms_3d
        
        if len(detections['boxes']) == 0:
            return detections
        
        # Apply NMS for each class separately
        class_ids = np.unique(detections['classes'])
        keep_indices = []
        
        for class_id in class_ids:
            class_mask = detections['classes'] == class_id
            if np.sum(class_mask) <= 1:
                keep_indices.extend(np.where(class_mask)[0])
                continue
                
            boxes_class = detections['boxes'][class_mask]
            scores_class = detections['scores'][class_mask]
            
            # Get indices after NMS
            nms_indices = nms_3d(boxes_class, scores_class, self.nms_threshold)
            
            # Map back to original indices
            original_indices = np.where(class_mask)[0]
            keep_indices.extend(original_indices[nms_indices])
        
        # Sort indices to maintain original order
        keep_indices = sorted(keep_indices)
        
        return {
            'boxes': detections['boxes'][keep_indices],
            'scores': detections['scores'][keep_indices],
            'classes': detections['classes'][keep_indices],
            'sources': detections['sources'][keep_indices]
        }
    
    def get_bandwidth_usage(self, num_objects: int) -> int:
        """
        Calculate the bandwidth usage for transmitting detection results.
        
        Args:
            num_objects: Number of detected objects in infrastructure frame
            
        Returns:
            Approximate bandwidth usage in bytes
        """
        # Calculate approximate size of transmitted data
        # Each object typically includes:
        # - Box parameters (7 floats): 7 * 4 bytes
        # - Score (1 float): 4 bytes
        # - Class ID (1 int): 4 bytes
        # - Optional velocity (3 floats): 3 * 4 bytes
        bytes_per_object = (7 + 1 + 1 + 3) * 4
        
        # Add overhead for transmission protocol and timestamps
        overhead = 100  # bytes
        
        return num_objects * bytes_per_object + overhead