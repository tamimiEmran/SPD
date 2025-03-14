"""
bandwidth module for V2X-Seq project.

This module provides functionality for measuring and analyzing bandwidth requirements
for different fusion strategies in vehicle-infrastructure cooperative perception.
"""

import numpy as np
import json
import time
import pickle
from collections import defaultdict


class BandwidthMeter:
    """
    Class to measure the bandwidth used by different fusion strategies.
    """
    
    def __init__(self):
        """Initialize the bandwidth meter."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated measurements."""
        self.total_bytes = 0
        self.num_frames = 0
        self.transmission_times = []
        self.data_types = defaultdict(int)
        self.start_time = None
        self.measurement_duration = 0
        
    def start_measurement(self):
        """Start a new measurement session."""
        self.reset()
        self.start_time = time.time()
    
    def end_measurement(self):
        """End the current measurement session."""
        if self.start_time is not None:
            self.measurement_duration = time.time() - self.start_time
    
    def add_transmission(self, data, data_type=None, timestamp=None):
        """
        Record a data transmission.
        
        Args:
            data: The data being transmitted (can be raw bytes, dict, list, etc.)
            data_type (str, optional): Type of data being transmitted for detailed analysis
            timestamp (float, optional): Timestamp of transmission, if None current time is used
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate size in bytes
        if isinstance(data, (bytes, bytearray)):
            size = len(data)
        elif isinstance(data, (dict, list, tuple)):
            # For non-byte data, estimate size using pickle
            size = len(pickle.dumps(data))
        elif isinstance(data, str):
            size = len(data.encode('utf-8'))
        elif isinstance(data, np.ndarray):
            size = data.nbytes
        else:
            # For other types, convert to string and then bytes
            size = len(str(data).encode('utf-8'))
        
        self.total_bytes += size
        self.num_frames += 1
        self.transmission_times.append(timestamp)
        
        if data_type is not None:
            self.data_types[data_type] += size
    
    def get_bytes_per_second(self):
        """
        Calculate the average bytes per second (BPS).
        
        Returns:
            float: Average bytes per second
        """
        if self.measurement_duration <= 0:
            if len(self.transmission_times) < 2:
                return 0
            # Calculate duration from first to last transmission
            duration = self.transmission_times[-1] - self.transmission_times[0]
            if duration <= 0:
                return 0
            return self.total_bytes / duration
        
        return self.total_bytes / self.measurement_duration
    
    def get_statistics(self):
        """
        Get comprehensive bandwidth statistics.
        
        Returns:
            dict: Dictionary containing various bandwidth statistics
        """
        bps = self.get_bytes_per_second()
        
        stats = {
            'total_bytes': self.total_bytes,
            'num_frames': self.num_frames,
            'bytes_per_second': bps,
            'kilobytes_per_second': bps / 1024,
            'megabytes_per_second': bps / (1024 * 1024),
            'bytes_per_frame': self.total_bytes / max(1, self.num_frames),
            'data_type_breakdown': dict(self.data_types),
            'measurement_duration': self.measurement_duration,
        }
        
        if len(self.transmission_times) >= 2:
            intervals = np.diff(self.transmission_times)
            stats['avg_transmission_interval'] = np.mean(intervals)
            stats['transmission_frequency'] = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        return stats


class LateFusionBandwidthEstimator:
    """
    Estimate bandwidth requirements for late fusion strategies.
    """
    
    @staticmethod
    def estimate_bandwidth(num_objects, object_properties, frequency=10.0):
        """
        Estimate bandwidth requirements for late fusion.
        
        Args:
            num_objects (int): Average number of objects transmitted per frame
            object_properties (dict): Dictionary mapping property names to their byte sizes
            frequency (float): Transmission frequency in Hz
            
        Returns:
            float: Estimated bytes per second
        """
        bytes_per_object = sum(object_properties.values())
        bytes_per_frame = num_objects * bytes_per_object
        bytes_per_second = bytes_per_frame * frequency
        
        return bytes_per_second
    
    @staticmethod
    def get_default_object_properties():
        """
        Get default byte sizes for common object properties.
        
        Returns:
            dict: Dictionary mapping property names to their byte sizes
        """
        return {
            'position': 12,  # 3 floats (x, y, z) * 4 bytes
            'dimensions': 12,  # 3 floats (width, length, height) * 4 bytes
            'orientation': 4,  # 1 float (yaw) * 4 bytes
            'object_id': 4,  # 1 integer * 4 bytes
            'class_id': 1,  # 1 byte for class
            'confidence': 4,  # 1 float * 4 bytes
        }


class MiddleFusionBandwidthEstimator:
    """
    Estimate bandwidth requirements for middle fusion strategies.
    """
    
    @staticmethod
    def estimate_bandwidth(feature_dimensions, compression_ratio=1.0, frequency=10.0):
        """
        Estimate bandwidth requirements for middle fusion.
        
        Args:
            feature_dimensions (tuple): Dimensions of feature tensor (C, H, W) or similar
            compression_ratio (float): Ratio of compressed size to original size (0-1)
            frequency (float): Transmission frequency in Hz
            
        Returns:
            float: Estimated bytes per second
        """
        # Calculate size of feature tensor
        feature_size = np.prod(feature_dimensions) * 4  # Assuming float32 (4 bytes)
        
        # Apply compression ratio
        compressed_size = feature_size * compression_ratio
        
        # Calculate bytes per second
        bytes_per_second = compressed_size * frequency
        
        return bytes_per_second


class FFTrackingBandwidthEstimator:
    """
    Estimate bandwidth requirements for Feature Flow Tracking.
    """
    
    @staticmethod
    def estimate_bandwidth(feature_dimensions, flow_dimensions=None, 
                           feature_compression=1.0, flow_compression=1.0, frequency=10.0):
        """
        Estimate bandwidth requirements for FF-Tracking.
        
        Args:
            feature_dimensions (tuple): Dimensions of feature tensor (C, H, W) or similar
            flow_dimensions (tuple, optional): Dimensions of flow tensor, defaults to feature_dimensions
            feature_compression (float): Compression ratio for features (0-1)
            flow_compression (float): Compression ratio for flow (0-1)
            frequency (float): Transmission frequency in Hz
            
        Returns:
            float: Estimated bytes per second
        """
        if flow_dimensions is None:
            flow_dimensions = feature_dimensions
        
        # Calculate size of feature tensor
        feature_size = np.prod(feature_dimensions) * 4  # Assuming float32 (4 bytes)
        
        # Calculate size of flow tensor
        flow_size = np.prod(flow_dimensions) * 4  # Assuming float32 (4 bytes)
        
        # Apply compression ratios
        compressed_feature_size = feature_size * feature_compression
        compressed_flow_size = flow_size * flow_compression
        
        # Calculate total size per transmission
        total_size = compressed_feature_size + compressed_flow_size
        
        # Calculate bytes per second
        bytes_per_second = total_size * frequency
        
        return bytes_per_second


def compare_fusion_strategies(num_objects=10, feature_dimensions=(64, 100, 100), frequency=10.0):
    """
    Compare bandwidth requirements for different fusion strategies.
    
    Args:
        num_objects (int): Average number of objects for late fusion
        feature_dimensions (tuple): Feature dimensions for middle fusion and FF-Tracking
        frequency (float): Transmission frequency in Hz
        
    Returns:
        dict: Dictionary with bandwidth estimates for each strategy
    """
    # Late fusion
    object_properties = LateFusionBandwidthEstimator.get_default_object_properties()
    late_fusion_bps = LateFusionBandwidthEstimator.estimate_bandwidth(
        num_objects, object_properties, frequency)
    
    # Middle fusion with 80% compression
    middle_fusion_bps = MiddleFusionBandwidthEstimator.estimate_bandwidth(
        feature_dimensions, compression_ratio=0.2, frequency=frequency)
    
    # FF-Tracking with 80% compression
    ff_tracking_bps = FFTrackingBandwidthEstimator.estimate_bandwidth(
        feature_dimensions, feature_compression=0.2, flow_compression=0.2, frequency=frequency)
    
    return {
        'late_fusion': {
            'bytes_per_second': late_fusion_bps,
            'kilobytes_per_second': late_fusion_bps / 1024,
            'megabytes_per_second': late_fusion_bps / (1024 * 1024),
        },
        'middle_fusion': {
            'bytes_per_second': middle_fusion_bps,
            'kilobytes_per_second': middle_fusion_bps / 1024,
            'megabytes_per_second': middle_fusion_bps / (1024 * 1024),
        },
        'ff_tracking': {
            'bytes_per_second': ff_tracking_bps,
            'kilobytes_per_second': ff_tracking_bps / 1024,
            'megabytes_per_second': ff_tracking_bps / (1024 * 1024),
        }
    }


if __name__ == "__main__":
    # Example usage of BandwidthMeter
    meter = BandwidthMeter()
    meter.start_measurement()
    
    # Simulate transmissions
    for i in range(10):
        # Late fusion example: transmit detection results
        detections = [
            {'position': [1.0, 2.0, 3.0], 'dimensions': [1.5, 4.5, 1.8], 'id': 1, 'class': 'Car'},
            {'position': [5.0, 6.0, 0.0], 'dimensions': [1.8, 4.8, 1.9], 'id': 2, 'class': 'Car'},
        ]
        meter.add_transmission(detections, data_type='late_fusion')
        time.sleep(0.1)  # Simulate 10Hz frequency
    
    meter.end_measurement()
    stats = meter.get_statistics()
    print("Bandwidth statistics:")
    for key, value in stats.items():
        if key != 'data_type_breakdown':
            print(f"{key}: {value}")
    
    # Example usage of bandwidth estimators
    comparison = compare_fusion_strategies()
    print("\nFusion strategy bandwidth comparison:")
    for strategy, metrics in comparison.items():
        print(f"{strategy}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")