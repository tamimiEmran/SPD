"""
evaluator module for V2X-Seq project.

This module provides functionality for evaluating VIC3D Tracking models,
including support for various fusion strategies and latency settings.
"""

import os
import time
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.config_parser import load_config
from data.v2x_seq_dataset import V2XSeqDataset
from evaluation.metrics import compute_tracking_metrics, aggregate_metrics
from evaluation.bandwidth import measure_bandwidth


class Evaluator:
    """
    Evaluator class for VIC3D Tracking models.
    
    This class handles the evaluation of tracking models with different fusion
    strategies and latency settings.
    """
    
    def __init__(self, config, checkpoint_path=None):
        """
        Initialize the evaluator with the given configuration.
        
        Args:
            config: Configuration dictionary or path to config file
            checkpoint_path: Path to the model checkpoint (optional)
        """
        # Load configuration if a path is provided
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config
            
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("Evaluator")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize dataset
        self._init_dataset()
        
        # Initialize models
        self._init_models(checkpoint_path)
        
        # Initialize experiment tracking
        self.results = {}
    
    def _init_dataset(self):
        """Initialize the dataset and data loader."""
        # Create dataset
        self.dataset = V2XSeqDataset(
            config=self.config['data'],
            split='test'
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['evaluation'].get('batch_size', 1),
            shuffle=False,
            num_workers=self.config['evaluation'].get('num_workers', 4),
            pin_memory=True,
            collate_fn=self.dataset.collate_fn
        )
        
        self.logger.info(f"Initialized test dataset with {len(self.dataset)} samples")
    
    def _init_models(self, checkpoint_path):
        """Initialize models for evaluation."""
        # Import models dynamically based on configuration
        if self.config['model']['detection']['type'] == 'pointpillars':
            from models.detection.pointpillars import PointPillarsDetector
            self.detector = PointPillarsDetector(self.config['model']['detection'])
        else:
            from models.detection.center_point import CenterPointDetector
            self.detector = CenterPointDetector(self.config['model']['detection'])
        
        if self.config['model']['tracking']['type'] == 'ab3dmot':
            from models.tracking.ab3dmot import AB3DMOT
            self.tracker = AB3DMOT(self.config['model']['tracking'])
        else:
            from models.tracking.tracker import Tracker
            self.tracker = Tracker(self.config['model']['tracking'])
        
        # Initialize fusion strategy
        fusion_type = self.config['fusion']['type']
        if fusion_type == 'late_fusion':
            from fusion.late_fusion import LateFusion
            self.fusion = LateFusion(self.config['fusion'])
        elif fusion_type == 'early_fusion':
            from fusion.early_fusion import EarlyFusion
            self.fusion = EarlyFusion(self.config['fusion'])
        elif fusion_type == 'middle_fusion':
            from fusion.middle_fusion import MiddleFusion
            self.fusion = MiddleFusion(self.config['fusion'])
        elif fusion_type == 'ff_tracking':
            from fusion.ff_tracking import FFTracking
            self.fusion = FFTracking(self.config['fusion'])
        else:
            self.fusion = None
            self.logger.info("No fusion strategy selected, using vehicle-only mode")
        
        # Move models to device
        self.detector.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Set models to evaluation mode
        self.detector.eval()
        
        self.logger.info(f"Initialized {fusion_type} model for evaluation")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.detector.load_state_dict(checkpoint['detector_state_dict'])
        
        if self.fusion is not None and 'fusion_state_dict' in checkpoint:
            self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self, latency_values=None):
        """
        Run evaluation with specified latency values.
        
        Args:
            latency_values: List of latency values in milliseconds to evaluate.
                If None, uses the latencies from the config file.
        
        Returns:
            Dictionary containing evaluation results for all latency settings.
        """
        # Get latency values to evaluate
        if latency_values is None:
            latency_values = self.config['evaluation'].get(
                'latency_values', [0, 100, 200, 300]
            )
        
        self.logger.info(f"Starting evaluation with latency values: {latency_values}")
        
        results = {}
        
        # First evaluate the vehicle-only baseline for comparison
        self.logger.info("Evaluating vehicle-only baseline")
        vehicle_only_metrics = self._evaluate_vehicle_only()
        results['vehicle_only'] = vehicle_only_metrics
        
        # Only evaluate fusion if a fusion strategy is specified
        if self.fusion is not None:
            # Evaluate each latency setting
            for latency in latency_values:
                self.logger.info(f"Evaluating with latency: {latency}ms")
                fusion_metrics = self._evaluate_with_latency(latency)
                results[f'fusion_latency_{latency}ms'] = fusion_metrics
        
        self.results = results
        return results
    
    def _evaluate_vehicle_only(self):
        """Evaluate using only vehicle data."""
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Vehicle-Only Evaluation")):
                # Extract vehicle data
                vehicle_data = batch['vehicle']
                vehicle_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in vehicle_data.items()}
                
                # Get ground truth for evaluation
                ground_truth = batch['ground_truth']
                
                # Run detection on vehicle data
                vehicle_detections = self.detector(vehicle_data)
                
                # Run tracking
                vehicle_tracks = self.tracker.track(vehicle_detections)
                
                # Compute metrics for this batch
                batch_metrics = compute_tracking_metrics(
                    tracks=vehicle_tracks,
                    ground_truth=ground_truth,
                    config=self.config['evaluation']
                )
                
                all_metrics.append(batch_metrics)
        
        # Aggregate metrics across all batches
        aggregated_metrics = aggregate_metrics(all_metrics)
        
        # Log summary
        self.logger.info(f"Vehicle-Only Results: MOTA: {aggregated_metrics['MOTA']:.2f}, "
                         f"MOTP: {aggregated_metrics['MOTP']:.2f}, "
                         f"IDS: {aggregated_metrics['IDS']}")
        
        return aggregated_metrics
    
    def _evaluate_with_latency(self, latency_ms):
        """
        Evaluate using fusion with the specified latency.
        
        Args:
            latency_ms: Latency in milliseconds
        
        Returns:
            Dictionary of evaluation metrics
        """
        all_metrics = []
        bandwidth_measurements = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Fusion Evaluation ({latency_ms}ms)")):
                # Extract data
                vehicle_data = batch['vehicle']
                vehicle_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in vehicle_data.items()}
                
                infrastructure_data = batch['infrastructure']
                infrastructure_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in infrastructure_data.items()}
                
                # Apply latency to infrastructure data
                infrastructure_data = self._apply_latency(infrastructure_data, latency_ms)
                
                # Get ground truth for evaluation
                ground_truth = batch['ground_truth']
                
                # Start bandwidth measurement
                start_time = time.time()
                
                # Apply fusion strategy
                fused_data = self.fusion.fuse(vehicle_data, infrastructure_data)
                
                # Measure bandwidth
                bandwidth = measure_bandwidth(
                    self.fusion,
                    latency_ms,
                    self.config['evaluation'].get('transmission_frequency', 10)  # 10Hz default
                )
                bandwidth_measurements.append(bandwidth)
                
                # Run detection on fused data
                fused_detections = self.detector(fused_data)
                
                # Run tracking
                fused_tracks = self.tracker.track(fused_detections)
                
                # Compute metrics for this batch
                batch_metrics = compute_tracking_metrics(
                    tracks=fused_tracks,
                    ground_truth=ground_truth,
                    config=self.config['evaluation']
                )
                
                all_metrics.append(batch_metrics)
        
        # Aggregate metrics across all batches
        aggregated_metrics = aggregate_metrics(all_metrics)
        
        # Add bandwidth measurement
        aggregated_metrics['BPS'] = np.mean(bandwidth_measurements)
        
        # Log summary
        self.logger.info(f"Fusion Results ({latency_ms}ms): MOTA: {aggregated_metrics['MOTA']:.2f}, "
                         f"MOTP: {aggregated_metrics['MOTP']:.2f}, "
                         f"IDS: {aggregated_metrics['IDS']}, "
                         f"BPS: {aggregated_metrics['BPS']:.2e}")
        
        return aggregated_metrics
    
    def _apply_latency(self, infrastructure_data, latency_ms):
        """
        Apply simulated latency to infrastructure data.
        
        Args:
            infrastructure_data: Dictionary containing infrastructure sensor data
            latency_ms: Latency to simulate in milliseconds
            
        Returns:
            Modified infrastructure data with simulated latency
        """
        # This would be implemented in the data.calibration.latency_compensation module
        # For now, we'll just return the data with a timestamp adjustment
        from data.calibration.latency_compensation import apply_latency
        
        return apply_latency(infrastructure_data, latency_ms)
    
    def save_results(self, output_dir):
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        import json
        import datetime
        
        # Create results dictionary with metadata
        full_results = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            'config': self.config,
            'results': self.results
        }
        
        # Save JSON results
        results_path = os.path.join(output_dir, 'eval_results.json')
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {results_path}")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("V2X-Seq Evaluation Results\n")
            f.write("==========================\n\n")
            
            # Vehicle-only results
            vehicle_only = self.results.get('vehicle_only', {})
            f.write("Vehicle-Only Baseline:\n")
            f.write(f"  MOTA: {vehicle_only.get('MOTA', 'N/A'):.2f}\n")
            f.write(f"  MOTP: {vehicle_only.get('MOTP', 'N/A'):.2f}\n")
            f.write(f"  IDS:  {vehicle_only.get('IDS', 'N/A')}\n\n")
            
            # Fusion results for each latency
            f.write("Fusion Results:\n")
            for result_key, metrics in self.results.items():
                if result_key != 'vehicle_only':
                    latency = result_key.split('_')[-1]
                    f.write(f"  Latency {latency}:\n")
                    f.write(f"    MOTA: {metrics.get('MOTA', 'N/A'):.2f}\n")
                    f.write(f"    MOTP: {metrics.get('MOTP', 'N/A'):.2f}\n")
                    f.write(f"    IDS:  {metrics.get('IDS', 'N/A')}\n")
                    f.write(f"    BPS:  {metrics.get('BPS', 'N/A'):.2e}\n\n")
        
        self.logger.info(f"Saved summary to {summary_path}")
        
        # Visualize results if visualization module is available
        try:
            from evaluation.visualization import visualize_results
            vis_path = os.path.join(output_dir, 'visualization')
            visualize_results(self.results, vis_path)
            self.logger.info(f"Saved visualizations to {vis_path}")
        except ImportError:
            self.logger.warning("Visualization module not available, skipping result visualization")


def main():
    """Main function to run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate V2X-Seq tracking models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--latencies", type=int, nargs="+", help="Latency values to evaluate (ms)")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.config, args.checkpoint)
    
    # Run evaluation
    results = evaluator.evaluate(args.latencies)
    
    # Save results
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()