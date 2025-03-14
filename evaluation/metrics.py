"""
metrics module for V2X-Seq project.

This module provides standard evaluation metrics for 3D tracking performance,
including MOTA (Multi-Object Tracking Accuracy), MOTP (Multi-Object Tracking Precision),
and IDS (ID Switches).
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class TrackingMetrics:
    """
    Class to compute standard tracking metrics for 3D object tracking evaluation.
    """
    
    def __init__(self, distance_threshold=2.0):
        """
        Initialize tracking metrics calculator.
        
        Args:
            distance_threshold (float): Distance threshold (in meters) to consider
                                       a detection as a match to ground truth.
        """
        self.distance_threshold = distance_threshold
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        # Accumulated over all frames
        self.total_gt = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_matches = 0
        self.total_distance = 0.0
        
        # For ID mapping
        self.id_mapping = {}
        self.prev_matches = {}
    
    def compute_distance_matrix(self, detections, ground_truths):
        """
        Compute distance matrix between detections and ground truths.
        
        Args:
            detections: List of detection boxes with format [x, y, z, ...]
            ground_truths: List of ground truth boxes with format [x, y, z, ...]
            
        Returns:
            numpy.ndarray: Distance matrix with shape (num_dets, num_gts)
        """
        distance_matrix = np.zeros((len(detections), len(ground_truths)))
        
        for i, det in enumerate(detections):
            for j, gt in enumerate(ground_truths):
                # Compute Euclidean distance between 3D centers
                distance = np.linalg.norm(np.array(det[:3]) - np.array(gt[:3]))
                distance_matrix[i, j] = distance
                
        return distance_matrix
    
    def update(self, detections, detection_ids, ground_truths, gt_ids, frame_id):
        """
        Update metrics with detections and ground truths from a new frame.
        
        Args:
            detections: List of detection boxes with format [x, y, z, ...]
            detection_ids: List of detection IDs
            ground_truths: List of ground truth boxes with format [x, y, z, ...]
            gt_ids: List of ground truth IDs
            frame_id: Current frame ID
        """
        # Number of ground truths and detections
        num_gt = len(ground_truths)
        num_det = len(detections)
        
        self.total_gt += num_gt
        
        # If no detections or no ground truths, update FP/FN and return
        if num_det == 0:
            self.total_fn += num_gt
            return
        
        if num_gt == 0:
            self.total_fp += num_det
            return
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(detections, ground_truths)
        
        # Apply Hungarian algorithm to find optimal assignment
        det_indices, gt_indices = linear_sum_assignment(distance_matrix)
        
        matches = []
        
        for det_idx, gt_idx in zip(det_indices, gt_indices):
            # Check if the match is within the distance threshold
            if distance_matrix[det_idx, gt_idx] <= self.distance_threshold:
                det_id = detection_ids[det_idx]
                gt_id = gt_ids[gt_idx]
                matches.append((det_id, gt_id, distance_matrix[det_idx, gt_idx]))
                
                # Update matched distance for MOTP
                self.total_distance += distance_matrix[det_idx, gt_idx]
                self.total_matches += 1
            else:
                # This is a false positive because the distance is too large
                self.total_fp += 1
                
        # Count ID switches
        matched_det_ids = [match[0] for match in matches]
        matched_gt_ids = [match[1] for match in matches]
        
        # Count ID switches
        for det_id, gt_id, _ in matches:
            # Check if this ground truth was matched to a different detection in the previous frame
            if gt_id in self.prev_matches and self.prev_matches[gt_id] != det_id:
                self.total_id_switches += 1
        
        # Update previous matches for next frame
        self.prev_matches = {gt_id: det_id for det_id, gt_id, _ in matches}
        
        # Count false negatives (unmatched ground truths)
        self.total_fn += (num_gt - len(matched_gt_ids))
        
        # Count false positives (unmatched detections)
        self.total_fp += (num_det - len(matched_det_ids))
    
    def compute_metrics(self):
        """
        Compute final tracking metrics.
        
        Returns:
            dict: Dictionary containing MOTA, MOTP, and IDS metrics
        """
        # MOTA (Multi-Object Tracking Accuracy)
        if self.total_gt == 0:
            mota = 0
        else:
            mota = 1.0 - (self.total_fp + self.total_fn + self.total_id_switches) / self.total_gt
            mota = max(0, mota)  # MOTA can be negative, but we clip it at 0 for easier interpretation
        
        # MOTP (Multi-Object Tracking Precision)
        if self.total_matches == 0:
            motp = 0
        else:
            motp = self.total_distance / self.total_matches
        
        return {
            'MOTA': mota * 100,  # Convert to percentage
            'MOTP': motp,
            'IDS': self.total_id_switches,
            'FP': self.total_fp,
            'FN': self.total_fn,
            'GT': self.total_gt,
            'Matches': self.total_matches
        }


def evaluate_tracking(predictions, ground_truths, distance_threshold=2.0):
    """
    Evaluate tracking performance between predictions and ground truths.
    
    Args:
        predictions: Dictionary mapping frame_id to (detections, detection_ids)
        ground_truths: Dictionary mapping frame_id to (ground_truths, gt_ids)
        distance_threshold: Distance threshold for matching
        
    Returns:
        dict: Dictionary containing MOTA, MOTP, and IDS metrics
    """
    metrics = TrackingMetrics(distance_threshold)
    
    # Process frames in order
    sorted_frames = sorted(ground_truths.keys())
    
    for frame_id in sorted_frames:
        if frame_id not in predictions:
            # No predictions for this frame, count all as false negatives
            gt_boxes, gt_ids = ground_truths[frame_id]
            metrics.total_fn += len(gt_ids)
            metrics.total_gt += len(gt_ids)
            continue
        
        # Get predictions and ground truths for this frame
        dets, det_ids = predictions[frame_id]
        gts, gt_ids = ground_truths[frame_id]
        
        # Update metrics
        metrics.update(dets, det_ids, gts, gt_ids, frame_id)
    
    # Compute final metrics
    return metrics.compute_metrics()


def evaluate_by_category(predictions, ground_truths, categories, distance_threshold=2.0):
    """
    Evaluate tracking performance separately for each object category.
    
    Args:
        predictions: Dictionary mapping frame_id to (detections, detection_ids, detection_categories)
        ground_truths: Dictionary mapping frame_id to (ground_truths, gt_ids, gt_categories)
        categories: List of category names to evaluate
        distance_threshold: Distance threshold for matching
        
    Returns:
        dict: Dictionary mapping category to metrics
    """
    results = {}
    
    for category in categories:
        # Filter predictions and ground truths for this category
        category_predictions = {}
        category_ground_truths = {}
        
        for frame_id, (dets, det_ids, det_cats) in predictions.items():
            category_dets = [dets[i] for i in range(len(dets)) if det_cats[i] == category]
            category_det_ids = [det_ids[i] for i in range(len(det_ids)) if det_cats[i] == category]
            
            if len(category_dets) > 0:
                category_predictions[frame_id] = (category_dets, category_det_ids)
        
        for frame_id, (gts, gt_ids, gt_cats) in ground_truths.items():
            category_gts = [gts[i] for i in range(len(gts)) if gt_cats[i] == category]
            category_gt_ids = [gt_ids[i] for i in range(len(gt_ids)) if gt_cats[i] == category]
            
            if len(category_gts) > 0:
                category_ground_truths[frame_id] = (category_gts, category_gt_ids)
        
        # Evaluate for this category
        if category_ground_truths:  # Only evaluate if there are ground truths for this category
            results[category] = evaluate_tracking(category_predictions, category_ground_truths, distance_threshold)
    
    # Calculate overall metrics across all categories
    all_predictions = {}
    all_ground_truths = {}
    
    for frame_id, (dets, det_ids, _) in predictions.items():
        all_predictions[frame_id] = (dets, det_ids)
    
    for frame_id, (gts, gt_ids, _) in ground_truths.items():
        all_ground_truths[frame_id] = (gts, gt_ids)
    
    results['OVERALL'] = evaluate_tracking(all_predictions, all_ground_truths, distance_threshold)
    
    return results


if __name__ == "__main__":
    # Example usage
    predictions = {
        0: (
            [[1, 1, 0], [4, 5, 0], [8, 1, 0]],  # detections
            [1, 2, 3],  # detection IDs
            ['Car', 'Car', 'Pedestrian']  # categories
        ),
        1: (
            [[1.1, 1.2, 0], [4.1, 5.1, 0], [8.2, 1.1, 0]], 
            [1, 2, 3],
            ['Car', 'Car', 'Pedestrian']
        )
    }
    
    ground_truths = {
        0: (
            [[1, 1, 0], [4, 5, 0], [8, 1, 0]],  # ground truths
            [1, 2, 3],  # ground truth IDs
            ['Car', 'Car', 'Pedestrian']  # categories
        ),
        1: (
            [[1.2, 1.3, 0], [4.2, 5.2, 0], [8.3, 1.2, 0]],
            [1, 2, 3],
            ['Car', 'Car', 'Pedestrian']
        )
    }
    
    categories = ['Car', 'Pedestrian']
    
    results = evaluate_by_category(predictions, ground_truths, categories)
    print("Results by category:")
    for category, metrics in results.items():
        print(f"{category}: {metrics}")