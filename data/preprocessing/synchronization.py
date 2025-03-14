"""
synchronization module for V2X-Seq project.

This module provides functionality for synchronizing time series data from
vehicle and infrastructure sensors. It includes tools for:
- Aligning timestamps between different sensor sources
- Interpolating sensor data to match specific timestamps
- Managing data streams with different sampling rates
- Handling communication latency between infrastructure and vehicle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import bisect
import heapq
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import logging
import copy

logger = logging.getLogger(__name__)


class TimeSynchronizer:
    """
    Class for synchronizing data from different sources based on timestamps.
    """
    
    def __init__(self, 
                 max_time_diff_ms: float = 50.0,
                 interpolation_mode: str = 'linear'):
        """
        Initialize the time synchronizer.
        
        Args:
            max_time_diff_ms: Maximum allowed time difference for matching frames (ms)
            interpolation_mode: Interpolation method ('nearest', 'linear', 'cubic')
        """
        self.max_time_diff_sec = max_time_diff_ms / 1000.0
        self.interpolation_mode = interpolation_mode
    
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
                    infrastructure_frames: List[Dict],
                    return_time_diffs: bool = False) -> List[Tuple[Dict, Optional[Dict]]]:
        """
        Match vehicle frames with the closest infrastructure frames.
        
        Args:
            vehicle_frames: List of vehicle data frames with timestamps
            infrastructure_frames: List of infrastructure data frames with timestamps
            return_time_diffs: Whether to include time differences in the result
            
        Returns:
            List of tuples (vehicle_frame, matched_infrastructure_frame) where
            matched_infrastructure_frame may be None if no match within threshold
        """
        matched_frames = []
        
        # Sort frames by timestamp for more efficient matching
        vehicle_frames = sorted(vehicle_frames, key=lambda x: x.get('timestamp', 0))
        infrastructure_frames = sorted(infrastructure_frames, key=lambda x: x.get('timestamp', 0))
        
        # Extract timestamps for binary search
        infra_timestamps = [frame.get('timestamp', 0) for frame in infrastructure_frames]
        
        for vehicle_frame in vehicle_frames:
            v_timestamp = vehicle_frame.get('timestamp', 0)
            
            # Binary search for closest timestamp
            idx = bisect.bisect_left(infra_timestamps, v_timestamp)
            
            # Check bounds
            if idx == 0:
                closest_idx = 0
            elif idx == len(infra_timestamps):
                closest_idx = len(infra_timestamps) - 1
            else:
                # Compare with neighbors
                if abs(infra_timestamps[idx] - v_timestamp) < abs(infra_timestamps[idx-1] - v_timestamp):
                    closest_idx = idx
                else:
                    closest_idx = idx - 1
            
            # Get closest frame and time difference
            closest_frame = infrastructure_frames[closest_idx]
            time_diff = abs(infra_timestamps[closest_idx] - v_timestamp)
            
            # Check if time difference is within the threshold
            if time_diff <= self.max_time_diff_sec:
                if return_time_diffs:
                    matched_frames.append((vehicle_frame, closest_frame, time_diff))
                else:
                    matched_frames.append((vehicle_frame, closest_frame))
            else:
                if return_time_diffs:
                    matched_frames.append((vehicle_frame, None, float('inf')))
                else:
                    matched_frames.append((vehicle_frame, None))
        
        return matched_frames
    
    def interpolate_boxes(self, 
                        boxes1: np.ndarray, 
                        boxes2: np.ndarray, 
                        timestamp1: float,
                        timestamp2: float,
                        target_timestamp: float) -> np.ndarray:
        """
        Interpolate 3D bounding boxes between two timestamps.
        
        Args:
            boxes1: First set of boxes [N, 7] (x, y, z, w, l, h, yaw)
            boxes2: Second set of boxes [N, 7] with same IDs as boxes1
            timestamp1: Timestamp for boxes1
            timestamp2: Timestamp for boxes2
            target_timestamp: Target timestamp for interpolation
            
        Returns:
            Interpolated boxes at target_timestamp
        """
        if len(boxes1) != len(boxes2):
            raise ValueError(f"Box arrays must have same length, got {len(boxes1)} and {len(boxes2)}")
        
        if timestamp1 == timestamp2:
            return boxes1.copy()
        
        # Compute interpolation factor (between 0 and 1)
        alpha = (target_timestamp - timestamp1) / (timestamp2 - timestamp1)
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        interpolated_boxes = np.zeros_like(boxes1)
        
        # Interpolate position and size linearly
        interpolated_boxes[:, :6] = (1 - alpha) * boxes1[:, :6] + alpha * boxes2[:, :6]
        
        # Interpolate orientation (yaw angle)
        for i in range(len(boxes1)):
            # Handle angle wrap-around
            yaw1 = boxes1[i, 6]
            yaw2 = boxes2[i, 6]
            
            # Ensure the angle difference is within [-pi, pi]
            diff = (yaw2 - yaw1 + np.pi) % (2 * np.pi) - np.pi
            
            # Interpolate
            interpolated_yaw = yaw1 + alpha * diff
            interpolated_boxes[i, 6] = interpolated_yaw
        
        return interpolated_boxes
    
    def interpolate_pose(self,
                       pose1: np.ndarray,
                       pose2: np.ndarray,
                       timestamp1: float,
                       timestamp2: float,
                       target_timestamp: float) -> np.ndarray:
        """
        Interpolate between two poses (transformation matrices).
        
        Args:
            pose1: First pose as 4x4 transformation matrix
            pose2: Second pose as 4x4 transformation matrix
            timestamp1: Timestamp for pose1
            timestamp2: Timestamp for pose2
            target_timestamp: Target timestamp for interpolation
            
        Returns:
            Interpolated pose at target_timestamp
        """
        if timestamp1 == timestamp2:
            return pose1.copy()
        
        # Compute interpolation factor
        alpha = (target_timestamp - timestamp1) / (timestamp2 - timestamp1)
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        # Extract rotation and translation components
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]
        
        # Interpolate translation linearly
        t_interp = (1 - alpha) * t1 + alpha * t2
        
        # Interpolate rotation using SLERP
        rot1 = Rotation.from_matrix(R1)
        rot2 = Rotation.from_matrix(R2)
        
        # Create a Slerp object
        key_rots = Rotation.from_matrix(np.stack([R1, R2]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate
        R_interp = slerp([alpha])[0].as_matrix()
        
        # Construct interpolated pose
        pose_interp = np.eye(4)
        pose_interp[:3, :3] = R_interp
        pose_interp[:3, 3] = t_interp
        
        return pose_interp


class FrameSynchronizer:
    """
    Class for synchronizing and interpolating frame data based on timestamps.
    Handles multiple sensor types and different data structures.
    """
    
    def __init__(self,
                target_frequency: float = 10.0,  # Hz
                max_time_diff_ms: float = 50.0):
        """
        Initialize the frame synchronizer.
        
        Args:
            target_frequency: Target frequency for synchronized data in Hz
            max_time_diff_ms: Maximum allowed time difference for matching frames
        """
        self.target_frequency = target_frequency
        self.target_period = 1.0 / target_frequency
        self.max_time_diff_sec = max_time_diff_ms / 1000.0
        self.time_synchronizer = TimeSynchronizer(max_time_diff_ms)
    
    def generate_sync_timestamps(self, 
                              start_time: float, 
                              end_time: float) -> List[float]:
        """
        Generate evenly spaced timestamps at the target frequency.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of evenly spaced timestamps
        """
        num_frames = int((end_time - start_time) * self.target_frequency) + 1
        return [start_time + i * self.target_period for i in range(num_frames)]
    
    def synchronize_sequence(self,
                           vehicle_frames: List[Dict],
                           infrastructure_frames: List[Dict],
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> List[Dict]:
        """
        Synchronize a sequence of vehicle and infrastructure frames to a common timeline.
        
        Args:
            vehicle_frames: List of vehicle data frames with timestamps
            infrastructure_frames: List of infrastructure data frames with timestamps
            start_time: Optional start time, defaults to earliest timestamp
            end_time: Optional end time, defaults to latest timestamp
            
        Returns:
            List of synchronized frames with both vehicle and infrastructure data
        """
        if not vehicle_frames or not infrastructure_frames:
            return []
        
        # Sort frames by timestamp
        vehicle_frames = sorted(vehicle_frames, key=lambda x: x.get('timestamp', 0))
        infrastructure_frames = sorted(infrastructure_frames, key=lambda x: x.get('timestamp', 0))
        
        # Determine sequence time range
        if start_time is None:
            start_time = max(
                vehicle_frames[0].get('timestamp', 0),
                infrastructure_frames[0].get('timestamp', 0)
            )
        
        if end_time is None:
            end_time = min(
                vehicle_frames[-1].get('timestamp', 0),
                infrastructure_frames[-1].get('timestamp', 0)
            )
        
        # Generate target timestamps
        target_timestamps = self.generate_sync_timestamps(start_time, end_time)
        
        # Create synchronized frames
        sync_frames = []
        
        for target_ts in target_timestamps:
            # Find closest vehicle and infrastructure frames
            v_frame, v_diff = self.time_synchronizer.find_closest_frame(target_ts, vehicle_frames)
            i_frame, i_diff = self.time_synchronizer.find_closest_frame(target_ts, infrastructure_frames)
            
            # Skip if either frame is too far from target timestamp
            if v_diff > self.max_time_diff_sec or i_diff > self.max_time_diff_sec:
                continue
            
            # Create synchronized frame
            sync_frame = {
                'timestamp': target_ts,
                'vehicle_frame': v_frame,
                'infrastructure_frame': i_frame,
                'vehicle_time_diff': v_diff,
                'infrastructure_time_diff': i_diff
            }
            
            sync_frames.append(sync_frame)
        
        return sync_frames
    
    def interpolate_frame_data(self,
                             target_timestamp: float,
                             prev_frame: Dict,
                             next_frame: Dict,
                             data_keys: List[str],
                             interp_funcs: Dict[str, Callable] = None) -> Dict:
        """
        Interpolate frame data for a specific timestamp.
        
        Args:
            target_timestamp: Target timestamp for interpolation
            prev_frame: Frame before the target timestamp
            next_frame: Frame after the target timestamp
            data_keys: List of keys in the frames to interpolate
            interp_funcs: Dictionary mapping data keys to interpolation functions
                         Default interpolation is linear
            
        Returns:
            Frame with interpolated data
        """
        prev_ts = prev_frame.get('timestamp', 0)
        next_ts = next_frame.get('timestamp', 0)
        
        # Check if timestamps are valid
        if prev_ts >= next_ts:
            return prev_frame.copy()
        
        # Create interpolated frame
        interp_frame = {'timestamp': target_timestamp}
        
        # Compute interpolation factor
        alpha = (target_timestamp - prev_ts) / (next_ts - prev_ts)
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        # Default interpolation function (linear)
        def default_interp(prev_val, next_val, alpha):
            if isinstance(prev_val, np.ndarray) and isinstance(next_val, np.ndarray):
                return (1 - alpha) * prev_val + alpha * next_val
            elif isinstance(prev_val, (int, float)) and isinstance(next_val, (int, float)):
                return (1 - alpha) * prev_val + alpha * next_val
            else:
                return prev_val if alpha < 0.5 else next_val
        
        # Interpolate each data key
        for key in data_keys:
            if key in prev_frame and key in next_frame:
                if interp_funcs and key in interp_funcs:
                    # Use provided interpolation function
                    interp_frame[key] = interp_funcs[key](
                        prev_frame[key], next_frame[key], 
                        prev_ts, next_ts, target_timestamp
                    )
                else:
                    # Use default interpolation
                    interp_frame[key] = default_interp(
                        prev_frame[key], next_frame[key], alpha
                    )
        
        return interp_frame


class TrajectoryInterpolator:
    """
    Specialized class for interpolating trajectories of tracked objects.
    """
    
    def __init__(self, position_smoothing: float = 0.5, angle_smoothing: float = 0.3):
        """
        Initialize the trajectory interpolator.
        
        Args:
            position_smoothing: Smoothing factor for position interpolation (0-1)
            angle_smoothing: Smoothing factor for angle interpolation (0-1)
        """
        self.position_smoothing = position_smoothing
        self.angle_smoothing = angle_smoothing
    
    def interpolate_trajectory(self,
                            trajectory: List[Dict],
                            target_timestamp: float) -> Optional[Dict]:
        """
        Interpolate trajectory data for a specific timestamp.
        
        Args:
            trajectory: List of trajectory points, each with 'timestamp' and 'position'
            target_timestamp: Target timestamp for interpolation
            
        Returns:
            Interpolated trajectory point at target timestamp, or None if impossible
        """
        if not trajectory:
            return None
        
        # Sort trajectory by timestamp
        trajectory = sorted(trajectory, key=lambda x: x.get('timestamp', 0))
        
        # Find surrounding trajectory points
        timestamps = [point.get('timestamp', 0) for point in trajectory]
        
        # Binary search for insertion point
        idx = bisect.bisect_left(timestamps, target_timestamp)
        
        # Handle edge cases
        if idx == 0:
            return trajectory[0]
        elif idx == len(trajectory):
            return trajectory[-1]
        
        # Get surrounding points
        prev_point = trajectory[idx - 1]
        next_point = trajectory[idx]
        
        prev_ts = prev_point.get('timestamp', 0)
        next_ts = next_point.get('timestamp', 0)
        
        # Compute interpolation factor
        if prev_ts == next_ts:
            alpha = 0.5  # Equal weight if timestamps are the same
        else:
            alpha = (target_timestamp - prev_ts) / (next_ts - prev_ts)
            alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        # Create interpolated point
        interp_point = {
            'timestamp': target_timestamp,
            'track_id': prev_point.get('track_id')
        }
        
        # Interpolate position
        if 'position' in prev_point and 'position' in next_point:
            prev_pos = np.array(prev_point['position'])
            next_pos = np.array(next_point['position'])
            interp_point['position'] = (1 - alpha) * prev_pos + alpha * next_pos
        
        # Interpolate orientation (if available)
        if 'orientation' in prev_point and 'orientation' in next_point:
            prev_angle = prev_point['orientation']
            next_angle = next_point['orientation']
            
            # Handle angle wrap-around
            diff = (next_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
            interp_angle = prev_angle + alpha * diff
            
            # Normalize to [-pi, pi]
            interp_point['orientation'] = ((interp_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Interpolate velocity (if available)
        if 'velocity' in prev_point and 'velocity' in next_point:
            prev_vel = np.array(prev_point['velocity'])
            next_vel = np.array(next_point['velocity'])
            interp_point['velocity'] = (1 - alpha) * prev_vel + alpha * next_vel
        
        # Interpolate additional data fields
        for key in prev_point.keys():
            if key not in interp_point and key != 'timestamp':
                if key in next_point:
                    try:
                        prev_val = prev_point[key]
                        next_val = next_point[key]
                        
                        if isinstance(prev_val, (int, float)) and isinstance(next_val, (int, float)):
                            interp_point[key] = (1 - alpha) * prev_val + alpha * next_val
                        elif isinstance(prev_val, (list, tuple)) and isinstance(next_val, (list, tuple)):
                            if len(prev_val) == len(next_val):
                                interp_point[key] = [
                                    (1 - alpha) * p + alpha * n 
                                    for p, n in zip(prev_val, next_val)
                                ]
                        else:
                            # For non-numeric types, use nearest neighbor
                            interp_point[key] = prev_val if alpha < 0.5 else next_val
                    except:
                        # If interpolation fails, use nearest neighbor
                        interp_point[key] = prev_val if alpha < 0.5 else next_val
        
        return interp_point
    
    def resample_trajectory(self,
                         trajectory: List[Dict],
                         target_frequency: float = 10.0,
                         smooth: bool = True) -> List[Dict]:
        """
        Resample a trajectory to a target frequency.
        
        Args:
            trajectory: List of trajectory points with timestamps
            target_frequency: Target frequency in Hz
            smooth: Whether to apply smoothing
            
        Returns:
            Resampled trajectory at regular intervals
        """
        if not trajectory:
            return []
        
        # Sort trajectory by timestamp
        trajectory = sorted(trajectory, key=lambda x: x.get('timestamp', 0))
        
        # Get time range
        start_time = trajectory[0].get('timestamp', 0)
        end_time = trajectory[-1].get('timestamp', 0)
        
        # Generate target timestamps
        target_period = 1.0 / target_frequency
        num_samples = int((end_time - start_time) / target_period) + 1
        target_timestamps = [start_time + i * target_period for i in range(num_samples)]
        
        # Interpolate for each target timestamp
        resampled = []
        
        for ts in target_timestamps:
            interp_point = self.interpolate_trajectory(trajectory, ts)
            if interp_point:
                resampled.append(interp_point)
        
        # Apply smoothing if requested
        if smooth and len(resampled) > 2:
            smoothed = self._smooth_trajectory(resampled)
            return smoothed
        
        return resampled
    
    def _smooth_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Apply smoothing to a trajectory.
        
        Args:
            trajectory: List of trajectory points
            
        Returns:
            Smoothed trajectory
        """
        smoothed = copy.deepcopy(trajectory)
        
        # Apply moving average to position and orientation
        for i in range(1, len(smoothed) - 1):
            if 'position' in smoothed[i]:
                prev_pos = np.array(smoothed[i-1]['position'])
                curr_pos = np.array(smoothed[i]['position'])
                next_pos = np.array(smoothed[i+1]['position'])
                
                # Weighted average
                smoothed_pos = (
                    prev_pos * self.position_smoothing / 2 +
                    curr_pos * (1 - self.position_smoothing) +
                    next_pos * self.position_smoothing / 2
                )
                
                smoothed[i]['position'] = smoothed_pos
            
            if 'orientation' in smoothed[i]:
                prev_angle = smoothed[i-1]['orientation']
                curr_angle = smoothed[i]['orientation']
                next_angle = smoothed[i+1]['orientation']
                
                # Ensure angle continuity
                while next_angle - curr_angle > np.pi:
                    next_angle -= 2 * np.pi
                while next_angle - curr_angle < -np.pi:
                    next_angle += 2 * np.pi
                    
                while prev_angle - curr_angle > np.pi:
                    prev_angle -= 2 * np.pi
                while prev_angle - curr_angle < -np.pi:
                    prev_angle += 2 * np.pi
                
                # Weighted average
                smoothed_angle = (
                    prev_angle * self.angle_smoothing / 2 +
                    curr_angle * (1 - self.angle_smoothing) +
                    next_angle * self.angle_smoothing / 2
                )
                
                # Normalize to [-pi, pi]
                smoothed[i]['orientation'] = ((smoothed_angle + np.pi) % (2 * np.pi)) - np.pi
        
        return smoothed


class CommunicationLatencySimulator:
    """
    Simulates communication latency between infrastructure and vehicle.
    """
    
    def __init__(self, 
                mean_latency_ms: float = 200.0,
                std_latency_ms: float = 30.0,
                min_latency_ms: float = 50.0,
                seed: Optional[int] = None):
        """
        Initialize the latency simulator.
        
        Args:
            mean_latency_ms: Mean latency in milliseconds
            std_latency_ms: Standard deviation of latency in milliseconds
            min_latency_ms: Minimum latency in milliseconds
            seed: Random seed for reproducibility
        """
        self.mean_latency = mean_latency_ms / 1000.0  # Convert to seconds
        self.std_latency = std_latency_ms / 1000.0
        self.min_latency = min_latency_ms / 1000.0
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_latency(self) -> float:
        """
        Generate a random latency value.
        
        Returns:
            Latency in seconds
        """
        # Generate random latency with normal distribution
        latency = np.random.normal(self.mean_latency, self.std_latency)
        
        # Apply minimum latency constraint
        return max(latency, self.min_latency)
    
    def simulate_latency(self, 
                       infrastructure_frames: List[Dict],
                       constant_latency: Optional[float] = None) -> Dict[str, float]:
        """
        Simulate latency for infrastructure frames.
        
        Args:
            infrastructure_frames: List of infrastructure frames with timestamps
            constant_latency: If provided, use this constant latency value in seconds
                             instead of random latency
            
        Returns:
            Dictionary mapping original timestamps to delayed availability timestamps
        """
        latency_map = {}
        
        for frame in infrastructure_frames:
            # Get original timestamp
            timestamp = frame.get('timestamp', 0)
            
            # Generate latency
            if constant_latency is not None:
                latency = constant_latency
            else:
                latency = self.generate_latency()
            
            # Compute delayed timestamp
            latency_map[timestamp] = timestamp + latency
        
        return latency_map
    
    def apply_latency(self, 
                    infrastructure_frames: List[Dict],
                    vehicle_frames: List[Dict],
                    constant_latency: Optional[float] = None) -> List[Tuple[Dict, Optional[Dict]]]:
        """
        Apply latency to match vehicle frames with delayed infrastructure frames.
        
        Args:
            infrastructure_frames: List of infrastructure frames with timestamps
            vehicle_frames: List of vehicle frames with timestamps
            constant_latency: Optional constant latency to apply instead of random latency
            
        Returns:
            List of tuples (vehicle_frame, delayed_infrastructure_frame) where
            delayed_infrastructure_frame may be None if no match is available yet
        """
        # Sort frames by timestamp
        vehicle_frames = sorted(vehicle_frames, key=lambda x: x.get('timestamp', 0))
        infrastructure_frames = sorted(infrastructure_frames, key=lambda x: x.get('timestamp', 0))
        
        # Simulate latency
        latency_map = self.simulate_latency(infrastructure_frames, constant_latency)
        
        # Create infrastructure availability timeline
        availability_timeline = []
        for frame in infrastructure_frames:
            timestamp = frame.get('timestamp', 0)
            available_at = latency_map[timestamp]
            availability_timeline.append((available_at, frame))
        
        # Sort by availability time
        availability_timeline.sort()
        
        # Match vehicle frames with available infrastructure frames
        matched_frames = []
        
        for vehicle_frame in vehicle_frames:
            v_timestamp = vehicle_frame.get('timestamp', 0)
            
            # Find infrastructure frames available at this time
            available_frames = [frame for avail_time, frame in availability_timeline 
                              if avail_time <= v_timestamp]
            
            if available_frames:
                # Find closest infrastructure frame by original timestamp
                closest_frame = min(available_frames, 
                                  key=lambda f: abs(f.get('timestamp', 0) - v_timestamp))
                matched_frames.append((vehicle_frame, closest_frame))
            else:
                # No infrastructure frame available yet
                matched_frames.append((vehicle_frame, None))
        
        return matched_frames


class MultiSourceTimeSynchronizer:
    """
    Synchronize data from multiple sources with different sampling rates and latencies.
    """
    
    def __init__(self, 
                sources: List[str],
                target_frequency: float = 10.0,
                max_time_diff_ms: float = 50.0):
        """
        Initialize the multi-source synchronizer.
        
        Args:
            sources: List of source names
            target_frequency: Target frequency for synchronized data in Hz
            max_time_diff_ms: Maximum allowed time difference for matching frames
        """
        self.sources = sources
        self.target_frequency = target_frequency
        self.max_time_diff_sec = max_time_diff_ms / 1000.0
        self.target_period = 1.0 / target_frequency
    
    def synchronize(self, 
                  data_by_source: Dict[str, List[Dict]],
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[Dict]:
        """
        Synchronize data from multiple sources to a common timeline.
        
        Args:
            data_by_source: Dictionary mapping source names to data frames
            start_time: Optional start time, defaults to earliest timestamp
            end_time: Optional end time, defaults to latest timestamp
            
        Returns:
            List of synchronized frames with data from all sources
        """
        # Check for missing sources
        for source in self.sources:
            if source not in data_by_source or not data_by_source[source]:
                logger.warning(f"Source {source} missing or empty in synchronization")
        
        # Sort each source's data by timestamp
        sorted_data = {}
        for source, frames in data_by_source.items():
            if frames:
                sorted_data[source] = sorted(frames, key=lambda x: x.get('timestamp', 0))
        
        # Determine time range
        if start_time is None:
            # Use latest start time from all sources
            source_start_times = [
                frames[0].get('timestamp', 0) 
                for frames in sorted_data.values() if frames
            ]
            start_time = max(source_start_times) if source_start_times else 0
        
        if end_time is None:
            # Use earliest end time from all sources
            source_end_times = [
                frames[-1].get('timestamp', 0) 
                for frames in sorted_data.values() if frames
            ]
            end_time = min(source_end_times) if source_end_times else 0
        
        # Generate target timestamps
        num_frames = int((end_time - start_time) * self.target_frequency) + 1
        target_timestamps = [start_time + i * self.target_period for i in range(num_frames)]
        
        # Create synchronized frames
        sync_frames = []
        
        for target_ts in target_timestamps:
            sync_frame = {
                'timestamp': target_ts,
                'sources': {}
            }
            
            # Find closest frame from each source
            all_found = True
            for source in self.sources:
                if source in sorted_data and sorted_data[source]:
                    frames = sorted_data[source]
                    
                    # Binary search for closest frame
                    timestamps = [frame.get('timestamp', 0) for frame in frames]
                    idx = bisect.bisect_left(timestamps, target_ts)
                    
                    # Handle edge cases
                    if idx == 0:
                        closest_idx = 0
                    elif idx == len(timestamps):
                        closest_idx = len(timestamps) - 1
                    else:
                        # Compare with neighbors
                        if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
                            closest_idx = idx
                        else:
                            closest_idx = idx - 1
                    
                    closest_frame = frames[closest_idx]
                    time_diff = abs(timestamps[closest_idx] - target_ts)
                    
                    # Check if time difference is within threshold
                    if time_diff <= self.max_time_diff_sec:
                        sync_frame['sources'][source] = {
                            'frame': closest_frame,
                            'time_diff': time_diff
                        }
                    else:
                        all_found = False
                else:
                    all_found = False
            
            # Only add sync frame if all sources are found
            if all_found:
                sync_frames.append(sync_frame)
        
        return sync_frames


class LatencyCompensator:
    """
    Compensates for communication latency in vehicle-infrastructure systems.
    """
    
    def __init__(self, use_motion_model: bool = True):
        """
        Initialize the latency compensator.
        
        Args:
            use_motion_model: Whether to use motion model for prediction
        """
        self.use_motion_model = use_motion_model
    
    def compensate_objects(self, 
                         objects: List[Dict], 
                         timestamp: float,
                         latency: float,
                         tracking_history: Optional[Dict[str, List[Dict]]] = None) -> List[Dict]:
        """
        Adjust object positions to compensate for latency.
        
        Args:
            objects: List of object dictionaries with position and timestamp
            timestamp: Target timestamp to predict positions for
            latency: Latency in seconds
            tracking_history: Optional history of tracked objects for better prediction
            
        Returns:
            List of objects with compensated positions
        """
        if objects is None:
            return []
            
        compensated_objects = []
        
        for obj in objects:
            # Calculate time difference
            obj_timestamp = obj.get('timestamp', timestamp - latency)
            time_diff = timestamp - obj_timestamp
            
            # Skip if negative time difference (object from the future)
            if time_diff < 0:
                compensated_objects.append(copy.deepcopy(obj))
                continue
            
            # Make a copy of the object to modify
            compensated_obj = copy.deepcopy(obj)
            
            if self.use_motion_model and tracking_history is not None:
                # Use motion model with history if available
                object_id = obj.get('id') or obj.get('track_id')
                if object_id in tracking_history and len(tracking_history[object_id]) >= 2:
                    compensated_obj = self._apply_motion_model(
                        obj, tracking_history[object_id], timestamp)
                else:
                    # Fallback to linear prediction if no history
                    compensated_obj = self._apply_linear_prediction(obj, time_diff)
            else:
                # Simple linear prediction based on velocity if available
                compensated_obj = self._apply_linear_prediction(obj, time_diff)
            
            compensated_objects.append(compensated_obj)
        
        return compensated_objects
    
    def _apply_linear_prediction(self, obj: Dict, time_diff: float) -> Dict:
        """
        Apply linear prediction to compensate for latency.
        
        Args:
            obj: Object dictionary
            time_diff: Time difference to predict for in seconds
            
        Returns:
            Updated object with predicted position
        """
        result = copy.deepcopy(obj)
        
        # If velocity information is available, use it
        if 'velocity' in obj or 'vel' in obj:
            velocity = obj.get('velocity', obj.get('vel', {'x': 0, 'y': 0, 'z': 0}))
            
            # Handle different velocity formats
            if isinstance(velocity, dict):
                vx = velocity.get('x', 0)
                vy = velocity.get('y', 0)
                vz = velocity.get('z', 0)
            elif isinstance(velocity, (list, tuple, np.ndarray)) and len(velocity) >= 3:
                vx, vy, vz = velocity[:3]
            else:
                vx = vy = vz = 0
            
            # Update position based on velocity
            if '3d_location' in result:
                # Dictionary format
                result['3d_location']['x'] += vx * time_diff
                result['3d_location']['y'] += vy * time_diff
                result['3d_location']['z'] += vz * time_diff
            elif 'position' in result:
                if isinstance(result['position'], dict):
                    result['position']['x'] += vx * time_diff
                    result['position']['y'] += vy * time_diff
                    result['position']['z'] += vz * time_diff
                elif isinstance(result['position'], (list, tuple, np.ndarray)) and len(result['position']) >= 3:
                    result['position'] = [
                        result['position'][0] + vx * time_diff,
                        result['position'][1] + vy * time_diff,
                        result['position'][2] + vz * time_diff
                    ]
            elif 'box' in result and isinstance(result['box'], (list, tuple, np.ndarray)) and len(result['box']) >= 3:
                # Update box center coordinates
                result['box'][0] += vx * time_diff
                result['box'][1] += vy * time_diff
                result['box'][2] += vz * time_diff
        
        return result
    
    def _apply_motion_model(self, 
                           obj: Dict, 
                           history: List[Dict],
                           target_timestamp: float) -> Dict:
        """
        Apply a motion model using object tracking history.
        
        Args:
            obj: Current object state
            history: List of previous states for the same object
            target_timestamp: Target timestamp to predict for
            
        Returns:
            Updated object with predicted state
        """
        result = copy.deepcopy(obj)
        
        # Sort history by timestamp
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', 0))
        
        # Get the two most recent states
        if len(sorted_history) >= 2:
            prev_state = sorted_history[-2]
            curr_state = sorted_history[-1]
            
            # Calculate time differences
            dt_history = curr_state.get('timestamp', 0) - prev_state.get('timestamp', 0)
            dt_predict = target_timestamp - curr_state.get('timestamp', 0)
            
            # Skip if time differences are invalid
            if dt_history <= 0 or dt_predict <= 0:
                return result
            
            # Calculate velocity based on history
            if self._extract_position(curr_state) is not None and self._extract_position(prev_state) is not None:
                curr_pos = self._extract_position(curr_state)
                prev_pos = self._extract_position(prev_state)
                
                # Calculate velocity
                if isinstance(curr_pos, dict) and isinstance(prev_pos, dict):
                    vx = (curr_pos.get('x', 0) - prev_pos.get('x', 0)) / dt_history
                    vy = (curr_pos.get('y', 0) - prev_pos.get('y', 0)) / dt_history
                    vz = (curr_pos.get('z', 0) - prev_pos.get('z', 0)) / dt_history
                elif isinstance(curr_pos, (list, tuple, np.ndarray)) and isinstance(prev_pos, (list, tuple, np.ndarray)):
                    vx = (curr_pos[0] - prev_pos[0]) / dt_history
                    vy = (curr_pos[1] - prev_pos[1]) / dt_history
                    vz = (curr_pos[2] - prev_pos[2]) / dt_history
                else:
                    return result  # Can't calculate velocity
                
                # Update position based on calculated velocity
                pos = self._extract_position(result)
                if pos is not None:
                    if isinstance(pos, dict):
                        pos['x'] = curr_pos.get('x', 0) + vx * dt_predict
                        pos['y'] = curr_pos.get('y', 0) + vy * dt_predict
                        pos['z'] = curr_pos.get('z', 0) + vz * dt_predict
                    elif isinstance(pos, (list, tuple, np.ndarray)):
                        pos = [
                            curr_pos[0] + vx * dt_predict,
                            curr_pos[1] + vy * dt_predict,
                            curr_pos[2] + vz * dt_predict
                        ]
                    self._update_position(result, pos)
                
                # Update velocity
                self._update_velocity(result, {'x': vx, 'y': vy, 'z': vz})
            
            # If the object has a rotation, predict rotation too
            yaw1 = self._extract_rotation(prev_state)
            yaw2 = self._extract_rotation(curr_state)
            
            if yaw1 is not None and yaw2 is not None:
                # Handle angle wrap-around
                rot_diff = yaw2 - yaw1
                if rot_diff > np.pi:
                    rot_diff -= 2 * np.pi
                elif rot_diff < -np.pi:
                    rot_diff += 2 * np.pi
                    
                rot_rate = rot_diff / dt_history
                predicted_rotation = yaw2 + rot_rate * dt_predict
                
                # Normalize to [-pi, pi]
                norm_rotation = ((predicted_rotation + np.pi) % (2 * np.pi)) - np.pi
                self._update_rotation(result, norm_rotation)
        
        return result
    
    def _extract_position(self, obj: Dict) -> Optional[Union[Dict, List]]:
        """Extract position from an object in various formats."""
        if '3d_location' in obj:
            return obj['3d_location']
        elif 'position' in obj:
            return obj['position']
        elif 'box' in obj and isinstance(obj['box'], (list, tuple, np.ndarray)) and len(obj['box']) >= 3:
            return obj['box'][:3]
        return None
    
    def _update_position(self, obj: Dict, position: Union[Dict, List]) -> None:
        """Update position in an object based on its format."""
        if '3d_location' in obj:
            if isinstance(position, dict):
                obj['3d_location'] = position
            else:
                obj['3d_location'] = {'x': position[0], 'y': position[1], 'z': position[2]}
        elif 'position' in obj:
            obj['position'] = position
        elif 'box' in obj and isinstance(obj['box'], (list, tuple, np.ndarray)) and len(obj['box']) >= 3:
            if isinstance(position, dict):
                obj['box'][0] = position.get('x', 0)
                obj['box'][1] = position.get('y', 0)
                obj['box'][2] = position.get('z', 0)
            else:
                obj['box'][0] = position[0]
                obj['box'][1] = position[1]
                obj['box'][2] = position[2]
    
    def _update_velocity(self, obj: Dict, velocity: Dict) -> None:
        """Update velocity in an object."""
        if 'velocity' in obj:
            if isinstance(obj['velocity'], dict):
                obj['velocity'] = velocity
            else:
                obj['velocity'] = [velocity.get('x', 0), velocity.get('y', 0), velocity.get('z', 0)]
        else:
            obj['velocity'] = velocity
    
    def _extract_rotation(self, obj: Dict) -> Optional[float]:
        """Extract rotation (yaw) from an object in various formats."""
        if 'rotation' in obj:
            return obj['rotation']
        elif 'orientation' in obj:
            return obj['orientation']
        elif 'yaw' in obj:
            return obj['yaw']
        elif 'box' in obj and isinstance(obj['box'], (list, tuple, np.ndarray)) and len(obj['box']) >= 7:
            return obj['box'][6]
        return None
    
    def _update_rotation(self, obj: Dict, rotation: float) -> None:
        """Update rotation in an object based on its format."""
        if 'rotation' in obj:
            obj['rotation'] = rotation
        elif 'orientation' in obj:
            obj['orientation'] = rotation
        elif 'yaw' in obj:
            obj['yaw'] = rotation
        elif 'box' in obj and isinstance(obj['box'], (list, tuple, np.ndarray)) and len(obj['box']) >= 7:
            obj['box'][6] = rotation


def find_closest_timestamp(target_timestamp: float, timestamps: List[float]) -> Tuple[int, float]:
    """
    Find the index and value of the closest timestamp in a list.
    
    Args:
        target_timestamp: Target timestamp
        timestamps: List of timestamps
        
    Returns:
        Tuple of (closest_index, time_difference)
    """
    if not timestamps:
        return -1, float('inf')
    
    # Binary search for closest timestamp
    idx = bisect.bisect_left(timestamps, target_timestamp)
    
    if idx == 0:
        return 0, abs(timestamps[0] - target_timestamp)
    elif idx == len(timestamps):
        return len(timestamps) - 1, abs(timestamps[-1] - target_timestamp)
    else:
        prev_diff = abs(timestamps[idx-1] - target_timestamp)
        curr_diff = abs(timestamps[idx] - target_timestamp)
        
        if prev_diff < curr_diff:
            return idx - 1, prev_diff
        else:
            return idx, curr_diff


def interpolate_value(value1: Any, value2: Any, alpha: float) -> Any:
    """
    Interpolate between two values based on interpolation factor alpha.
    
    Args:
        value1: First value
        value2: Second value
        alpha: Interpolation factor (0-1)
        
    Returns:
        Interpolated value
    """
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return (1 - alpha) * value1 + alpha * value2
    elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)) and len(value1) == len(value2):
        return [(1 - alpha) * v1 + alpha * v2 for v1, v2 in zip(value1, value2)]
    elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and value1.shape == value2.shape:
        return (1 - alpha) * value1 + alpha * value2
    elif isinstance(value1, dict) and isinstance(value2, dict):
        result = {}
        for key in set(value1.keys()) & set(value2.keys()):
            result[key] = interpolate_value(value1[key], value2[key], alpha)
        return result
    else:
        # For non-numeric types, use nearest neighbor
        return value1 if alpha < 0.5 else value2


def apply_latency(data: Dict[str, Any], latency_ms: float = 200.0) -> Dict[str, Any]:
    """
    Simulates communication latency by selecting an earlier infrastructure frame.
    
    This is a convenience function for the V2X-Seq dataset that can be used directly
    in the data loader to simulate latency effects.
    
    Args:
        data: Infrastructure data dictionary with timestamps
        latency_ms: Latency to simulate in milliseconds
        
    Returns:
        The infrastructure data that would be available at the current time
        considering the simulated latency
    """
    # Convert latency from ms to seconds
    latency_sec = latency_ms / 1000.0
    
    # If data has a timestamp field, adjust it to reflect latency
    if 'timestamp' in data:
        data['original_timestamp'] = data['timestamp']
        data['timestamp'] += latency_sec
    
    # Add latency information to the data
    data['latency'] = latency_sec
    
    return data