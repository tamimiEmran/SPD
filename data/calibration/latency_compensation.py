"""
latency_compensation module for V2X-Seq project.

This module provides functionality for simulating and compensating for latency
in vehicle-infrastructure cooperative perception systems. It includes tools for:
- Simulating communication delays
- Interpolating or extrapolating object positions to account for latency
- Time synchronization between vehicle and infrastructure data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import copy


class LatencySimulator:
    """Simulates communication latency in V2X systems by delaying infrastructure data."""
    
    def __init__(self, latency_ms: float = 200.0):
        """
        Initialize the latency simulator.
        
        Args:
            latency_ms: Simulated communication latency in milliseconds
        """
        self.latency_ms = latency_ms
        
    def apply_latency(self, infrastructure_frames: List[Dict], 
                     current_timestamp: float) -> Dict:
        """
        Simulates latency by selecting an infrastructure frame from the past.
        
        Args:
            infrastructure_frames: List of infrastructure data frames with timestamps
            current_timestamp: Current vehicle timestamp in seconds
            
        Returns:
            The infrastructure frame that would be available at the current time
            considering the simulated latency
        """
        # Convert latency from ms to seconds
        latency_sec = self.latency_ms / 1000.0
        
        # Calculate the timestamp that would be available with latency
        available_timestamp = current_timestamp - latency_sec
        
        # Find the closest infrastructure frame before the available timestamp
        selected_frame = None
        min_time_diff = float('inf')
        
        for frame in infrastructure_frames:
            time_diff = current_timestamp - frame['timestamp']
            
            # Only consider frames that would have arrived by now
            if time_diff >= latency_sec and time_diff < min_time_diff:
                min_time_diff = time_diff
                selected_frame = frame
        
        if selected_frame is None and infrastructure_frames:
            # If no suitable frame found but we have some frames, use the oldest one
            # (this is a fallback that would only happen at the beginning of a sequence)
            selected_frame = min(infrastructure_frames, 
                                key=lambda f: current_timestamp - f['timestamp'])
        
        return copy.deepcopy(selected_frame) if selected_frame else None


class PositionCompensator:
    """Compensates for latency by predicting object positions at the current time."""
    
    def __init__(self, use_motion_model: bool = True, max_prediction_time: float = 0.5):
        """
        Initialize the position compensator.
        
        Args:
            use_motion_model: Whether to use motion model for prediction or simple linear interpolation
            max_prediction_time: Maximum time (in seconds) to allow for prediction
        """
        self.use_motion_model = use_motion_model
        self.max_prediction_time = max_prediction_time
    
    def compensate_detections(self, 
                             detections: List[Dict], 
                             timestamp: float,
                             detection_history: Optional[Dict[int, List[Dict]]] = None) -> List[Dict]:
        """
        Adjust detection positions to compensate for latency.
        
        Args:
            detections: List of detection dictionaries with position and timestamp
            timestamp: Target timestamp to predict positions for
            detection_history: Optional history of previous detections for better prediction
            
        Returns:
            List of detections with compensated positions
        """
        if detections is None:
            return []
            
        compensated_detections = []
        
        for detection in detections:
            # Calculate time difference
            time_diff = timestamp - detection.get('timestamp', timestamp)
            
            # Skip if negative time difference or exceeds max prediction time
            if time_diff < 0 or time_diff > self.max_prediction_time:
                compensated_detections.append(copy.deepcopy(detection))
                continue
            
            # Make a copy of the detection to modify
            compensated_detection = copy.deepcopy(detection)
            
            if self.use_motion_model and detection_history is not None:
                # Use motion model with history if available
                track_id = detection.get('track_id')
                if track_id in detection_history and len(detection_history[track_id]) >= 2:
                    compensated_detection = self._apply_motion_model(
                        detection, detection_history[track_id], timestamp)
                else:
                    # Fallback to linear prediction if no history
                    compensated_detection = self._apply_linear_prediction(detection, time_diff)
            else:
                # Simple linear prediction based on velocity if available
                compensated_detection = self._apply_linear_prediction(detection, time_diff)
            
            compensated_detections.append(compensated_detection)
        
        return compensated_detections
    
    def _apply_linear_prediction(self, detection: Dict, time_diff: float) -> Dict:
        """
        Apply linear prediction to compensate for latency.
        
        Args:
            detection: Detection dictionary
            time_diff: Time difference to predict for in seconds
            
        Returns:
            Updated detection with predicted position
        """
        result = copy.deepcopy(detection)
        
        # If velocity information is available, use it
        if 'velocity' in detection:
            velocity = detection['velocity']
            
            # Update position based on velocity
            if '3d_location' in result:
                result['3d_location']['x'] += velocity.get('x', 0) * time_diff
                result['3d_location']['y'] += velocity.get('y', 0) * time_diff
                result['3d_location']['z'] += velocity.get('z', 0) * time_diff
        
        return result
    
    def _apply_motion_model(self, 
                           current_detection: Dict, 
                           detection_history: List[Dict],
                           target_timestamp: float) -> Dict:
        """
        Apply a motion model using detection history.
        
        Args:
            current_detection: Current detection
            detection_history: List of previous detections for the same object
            target_timestamp: Target timestamp to predict for
            
        Returns:
            Updated detection with predicted position
        """
        result = copy.deepcopy(current_detection)
        
        # Sort history by timestamp
        sorted_history = sorted(detection_history, 
                              key=lambda x: x.get('timestamp', 0))
        
        # Get the two most recent detections
        if len(sorted_history) >= 2:
            prev_detection = sorted_history[-2]
            current_detection = sorted_history[-1]
            
            # Calculate time differences
            dt_history = current_detection.get('timestamp', 0) - prev_detection.get('timestamp', 0)
            dt_predict = target_timestamp - current_detection.get('timestamp', 0)
            
            # Skip if time differences are invalid
            if dt_history <= 0 or dt_predict <= 0:
                return result
            
            # Calculate velocity based on history
            if '3d_location' in current_detection and '3d_location' in prev_detection:
                velocity = {
                    'x': (current_detection['3d_location']['x'] - prev_detection['3d_location']['x']) / dt_history,
                    'y': (current_detection['3d_location']['y'] - prev_detection['3d_location']['y']) / dt_history,
                    'z': (current_detection['3d_location']['z'] - prev_detection['3d_location']['z']) / dt_history
                }
                
                # Update position based on calculated velocity
                result['3d_location']['x'] = current_detection['3d_location']['x'] + velocity['x'] * dt_predict
                result['3d_location']['y'] = current_detection['3d_location']['y'] + velocity['y'] * dt_predict
                result['3d_location']['z'] = current_detection['3d_location']['z'] + velocity['z'] * dt_predict
                
                # Optionally update the velocity in the result
                result['velocity'] = velocity
            
            # If the object has a rotation rate, predict rotation
            if 'rotation' in current_detection and 'rotation' in prev_detection:
                rot_diff = current_detection['rotation'] - prev_detection['rotation']
                # Handle angle wrap-around
                if rot_diff > np.pi:
                    rot_diff -= 2 * np.pi
                elif rot_diff < -np.pi:
                    rot_diff += 2 * np.pi
                    
                rot_rate = rot_diff / dt_history
                predicted_rotation = current_detection['rotation'] + rot_rate * dt_predict
                # Normalize to [-pi, pi]
                result['rotation'] = ((predicted_rotation + np.pi) % (2 * np.pi)) - np.pi
        
        return result


class TimeSynchronizer:
    """Synchronizes timestamps between vehicle and infrastructure data."""
    
    def __init__(self, max_time_diff_ms: float = 50.0):
        """
        Initialize the time synchronizer.
        
        Args:
            max_time_diff_ms: Maximum allowed time difference for matching frames in milliseconds
        """
        self.max_time_diff_ms = max_time_diff_ms
    
    def find_closest_frame(self, 
                          target_timestamp: float, 
                          frames: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Find the frame with timestamp closest to the target timestamp.
        
        Args:
            target_timestamp: Target timestamp in seconds
            frames: List of frames with timestamps
            
        Returns:
            Tuple of (closest frame, time difference in seconds)
        """
        if not frames:
            return None, float('inf')
        
        closest_frame = None
        min_time_diff = float('inf')
        
        for frame in frames:
            time_diff = abs(frame.get('timestamp', 0) - target_timestamp)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_frame = frame
        
        return closest_frame, min_time_diff
    
    def match_frames(self, 
                    vehicle_frames: List[Dict], 
                    infrastructure_frames: List[Dict]) -> List[Tuple[Dict, Optional[Dict]]]:
        """
        Match vehicle frames with the closest infrastructure frames.
        
        Args:
            vehicle_frames: List of vehicle data frames with timestamps
            infrastructure_frames: List of infrastructure data frames with timestamps
            
        Returns:
            List of tuples (vehicle_frame, matched_infrastructure_frame) where
            matched_infrastructure_frame may be None if no match within threshold
        """
        matched_frames = []
        
        for vehicle_frame in vehicle_frames:
            v_timestamp = vehicle_frame.get('timestamp', 0)
            closest_frame, time_diff = self.find_closest_frame(v_timestamp, infrastructure_frames)
            
            # Check if the time difference is within the threshold
            if time_diff <= (self.max_time_diff_ms / 1000.0):
                matched_frames.append((vehicle_frame, closest_frame))
            else:
                matched_frames.append((vehicle_frame, None))
        
        return matched_frames


def simulate_sample_delays(timestamps: List[float], 
                          mean_latency_ms: float = 200.0,
                          std_latency_ms: float = 50.0,
                          min_latency_ms: float = 50.0,
                          seed: Optional[int] = None) -> Dict[float, float]:
    """
    Simulate realistic, variable network latency for a sequence of timestamps.
    
    Args:
        timestamps: List of original timestamps
        mean_latency_ms: Mean latency in milliseconds
        std_latency_ms: Standard deviation of latency in milliseconds
        min_latency_ms: Minimum latency in milliseconds
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping original timestamps to delayed availability timestamps
    """
    if seed is not None:
        np.random.seed(seed)
    
    latency_map = {}
    
    for ts in timestamps:
        # Generate random latency with normal distribution
        latency = np.random.normal(mean_latency_ms, std_latency_ms)
        # Apply minimum latency constraint
        latency = max(latency, min_latency_ms)
        # Convert to seconds and add to timestamp
        latency_map[ts] = ts + (latency / 1000.0)
    
    return latency_map