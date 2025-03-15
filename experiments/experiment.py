"""
experiment module for V2X-Seq project.

This module provides functionality for running experiments with different configurations,
models, and fusion strategies for Vehicle-Infrastructure Cooperative 3D Tracking (VIC3D).
It handles experiment setup, execution, and result collection.

The module includes two main classes:
1. Experiment - Handles a single experiment with a specific configuration
2. ExperimentSuite - Manages multiple experiments for comparative analysis
"""

import os
import time
import json
import logging
import argparse
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.config_parser import load_config
from utils.logger import setup_logger
from experiments.trainer import Trainer
from experiments.evaluator import Evaluator
from evaluation.metrics import aggregate_metrics
from evaluation.visualization import TrackingVisualizer


class Experiment:
    """
    Class for managing experiments in the V2X-Seq project.
    
    This class handles the complete lifecycle of an experiment:
    - Configuration loading and validation
    - Training setup and execution
    - Evaluation and result analysis
    - Result visualization and comparison
    """
    
    def __init__(self, 
                 config_path: str, 
                 experiment_name: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the experiment with the given configuration.
        
        Args:
            config_path: Path to the experiment configuration file
            experiment_name: Optional name for this experiment
            output_dir: Optional output directory for results
            seed: Optional random seed for reproducibility
        """
        # Set random seed for reproducibility if provided
        if seed is not None:
            self._set_random_seed(seed)
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Set experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = self.config.get('experiment_name', f"experiment_{timestamp}")
        self.experiment_name = experiment_name
        
        # Set output directory
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'outputs')
        self.output_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger('experiment', 
                                 os.path.join(log_dir, 'experiment.log'),
                                 level=logging.INFO)
        
        # Save the configuration
        self.config_path = os.path.join(self.output_dir, 'config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Initialize trainer and evaluator
        self.trainer = None
        self.evaluator = None
        
        # Initialize experiment state
        self.results = {}
        self.best_checkpoint = None
        
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(f"Configuration saved to: {self.config_path}")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment (training and evaluation).
        
        Returns:
            Dictionary of experiment results
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Run training if enabled
        if self.config.get('run_training', True):
            self._run_training()
        
        # Run evaluation if enabled
        if self.config.get('run_evaluation', True):
            self._run_evaluation()
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        self._save_results()
        
        # Log total experiment time
        total_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _run_training(self):
        """Run the training phase of the experiment."""
        self.logger.info("Starting training phase")
        
        # Create trainer
        self.trainer = Trainer(self.config)
        
        # Run training
        best_val_metric = self.trainer.train()
        
        # Save best checkpoint path
        checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.best_checkpoint = os.path.join(checkpoints_dir, 'best_model.pth')
        
        # Save training results
        self.results['training'] = {
            'best_val_metric': best_val_metric,
            'best_checkpoint': self.best_checkpoint,
            'epochs': self.trainer.start_epoch
        }
        
        self.logger.info(f"Training completed with best validation metric: {best_val_metric:.4f}")
    
    def _run_evaluation(self):
        """Run the evaluation phase of the experiment."""
        self.logger.info("Starting evaluation phase")
        
        # Determine checkpoint to use
        checkpoint_path = self.best_checkpoint
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            # Try to use provided checkpoint in config
            checkpoint_path = self.config.get('evaluation', {}).get('checkpoint_path')
            
        # Create evaluator
        self.evaluator = Evaluator(self.config, checkpoint_path)
        
        # Get latency values for evaluation
        latency_values = self.config.get('evaluation', {}).get(
            'latency_values', [0, 100, 200, 300]
        )
        
        # Run evaluation
        eval_results = self.evaluator.evaluate(latency_values)
        
        # Save results
        self.results['evaluation'] = eval_results
        
        # Save evaluation results to separate files
        eval_dir = os.path.join(self.output_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        self.evaluator.save_results(eval_dir)
        
        # Log summary
        vehicle_only = eval_results.get('vehicle_only', {})
        fusion_0ms = eval_results.get('fusion_latency_0ms', {})
        
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"  Vehicle-Only MOTA: {vehicle_only.get('MOTA', 0):.2f}")
        if 'fusion_latency_0ms' in eval_results:
            self.logger.info(f"  Fusion (0ms) MOTA: {fusion_0ms.get('MOTA', 0):.2f}")
        
    def _analyze_results(self):
        """Analyze experiment results and generate visualizations."""
        self.logger.info("Analyzing experiment results")
        
        # Only analyze if we have evaluation results
        if 'evaluation' not in self.results:
            self.logger.warning("No evaluation results to analyze")
            return
        
        # Extract evaluation results
        eval_results = self.results['evaluation']
        
        # Analyze performance across latency values
        latency_analysis = self._analyze_latency_impact(eval_results)
        self.results['analysis'] = {'latency_impact': latency_analysis}
        
        # Generate visualizations
        self._generate_visualizations(eval_results)
        
        self.logger.info("Results analysis completed")
    
    def _analyze_latency_impact(self, eval_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the impact of latency on tracking performance.
        
        Args:
            eval_results: Dictionary of evaluation results for different latencies
            
        Returns:
            Dictionary with latency analysis
        """
        # Extract latency values and corresponding metrics
        latencies = []
        mota_values = []
        motp_values = []
        ids_values = []
        bps_values = []
        
        # Add vehicle-only baseline for reference
        vehicle_only = eval_results.get('vehicle_only', {})
        if vehicle_only:
            latencies.append(-100)  # Use negative value to distinguish from fusion
            mota_values.append(vehicle_only.get('MOTA', 0))
            motp_values.append(vehicle_only.get('MOTP', 0))
            ids_values.append(vehicle_only.get('IDS', 0))
            bps_values.append(0)  # No bandwidth usage for vehicle-only
        
        # Extract metrics for each latency value
        for key, metrics in eval_results.items():
            if key.startswith('fusion_latency_'):
                # Extract latency value from key
                latency = int(key.split('_')[-1].rstrip('ms'))
                latencies.append(latency)
                mota_values.append(metrics.get('MOTA', 0))
                motp_values.append(metrics.get('MOTP', 0))
                ids_values.append(metrics.get('IDS', 0))
                bps_values.append(metrics.get('BPS', 0))
        
        # Compute summary statistics
        avg_mota = np.mean(mota_values[1:]) if len(mota_values) > 1 else 0  # Skip vehicle-only
        max_mota = np.max(mota_values[1:]) if len(mota_values) > 1 else 0
        max_mota_latency = latencies[np.argmax(mota_values)] if mota_values else 0
        
        # Create analysis results
        analysis = {
            'latencies': latencies,
            'mota_values': mota_values,
            'motp_values': motp_values,
            'ids_values': ids_values,
            'bps_values': bps_values,
            'avg_mota': avg_mota,
            'max_mota': max_mota,
            'max_mota_latency': max_mota_latency,
            'vehicle_only_mota': vehicle_only.get('MOTA', 0)
        }
        
        return analysis
    
    def _generate_visualizations(self, eval_results: Dict[str, Dict[str, Any]]):
        """
        Generate visualizations of experiment results.
        
        Args:
            eval_results: Dictionary of evaluation results
        """
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Only generate visualizations if we have latency analysis
        if 'analysis' not in self.results or 'latency_impact' not in self.results['analysis']:
            return
        
        # Extract latency analysis
        analysis = self.results['analysis']['latency_impact']
        
        # Plot MOTA vs. Latency
        self._plot_metric_vs_latency(
            analysis['latencies'],
            analysis['mota_values'],
            'MOTA (%)',
            os.path.join(vis_dir, 'mota_vs_latency.png')
        )
        
        # Plot MOTP vs. Latency
        self._plot_metric_vs_latency(
            analysis['latencies'],
            analysis['motp_values'],
            'MOTP',
            os.path.join(vis_dir, 'motp_vs_latency.png')
        )
        
        # Plot IDS vs. Latency
        self._plot_metric_vs_latency(
            analysis['latencies'],
            analysis['ids_values'],
            'ID Switches',
            os.path.join(vis_dir, 'ids_vs_latency.png')
        )
        
        # Plot Bandwidth vs. Latency
        valid_latencies = [l for l, b in zip(analysis['latencies'], analysis['bps_values']) if l >= 0]
        valid_bps = [b for l, b in zip(analysis['latencies'], analysis['bps_values']) if l >= 0]
        if valid_latencies and valid_bps:
            self._plot_metric_vs_latency(
                valid_latencies,
                valid_bps,
                'Bandwidth (Bytes/s)',
                os.path.join(vis_dir, 'bandwidth_vs_latency.png'),
                log_scale=True
            )
        
        # Plot MOTA vs. Bandwidth
        if valid_latencies and valid_bps:
            valid_mota = [m for l, m in zip(analysis['latencies'], analysis['mota_values']) if l >= 0]
            self._plot_mota_vs_bandwidth(
                valid_bps,
                valid_mota,
                os.path.join(vis_dir, 'mota_vs_bandwidth.png')
            )
        
        self.logger.info(f"Generated visualizations in {vis_dir}")
    
    def _plot_metric_vs_latency(self, 
                              latencies: List[int],
                              metric_values: List[float],
                              metric_name: str,
                              output_path: str,
                              log_scale: bool = False):
        """
        Plot a metric against latency.
        
        Args:
            latencies: List of latency values
            metric_values: List of metric values
            metric_name: Name of the metric
            output_path: Output path for the plot
            log_scale: Whether to use log scale for y-axis
        """
        plt.figure(figsize=(10, 6))
        
        # Split vehicle-only and fusion points
        vehicle_indices = [i for i, l in enumerate(latencies) if l < 0]
        fusion_indices = [i for i, l in enumerate(latencies) if l >= 0]
        
        fusion_x = [latencies[i] for i in fusion_indices]
        fusion_y = [metric_values[i] for i in fusion_indices]
        
        # Plot fusion points with line
        plt.plot(fusion_x, fusion_y, 'o-', color='blue', linewidth=2, markersize=8, label='Fusion')
        
        # Plot vehicle-only as horizontal line
        if vehicle_indices:
            vehicle_y = metric_values[vehicle_indices[0]]
            plt.axhline(y=vehicle_y, color='red', linestyle='--', linewidth=2, 
                       label='Vehicle-Only')
            
            # Add text label for vehicle-only
            plt.text(5, vehicle_y * 1.05, f"Vehicle-Only: {vehicle_y:.2f}", 
                    color='red', fontsize=10)
        
        plt.xlabel('Latency (ms)')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs. Latency')
        plt.grid(True, alpha=0.3)
        if log_scale and min(fusion_y) > 0:
            plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_mota_vs_bandwidth(self, 
                              bandwidth_values: List[float],
                              mota_values: List[float],
                              output_path: str):
        """
        Plot MOTA against bandwidth.
        
        Args:
            bandwidth_values: List of bandwidth values
            mota_values: List of MOTA values
            output_path: Output path for the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Sort by bandwidth
        sorted_indices = np.argsort(bandwidth_values)
        x = [bandwidth_values[i] for i in sorted_indices]
        y = [mota_values[i] for i in sorted_indices]
        
        plt.plot(x, y, 'o-', color='green', linewidth=2, markersize=8)
        
        plt.xlabel('Bandwidth (Bytes/s)')
        plt.ylabel('MOTA (%)')
        plt.title('MOTA vs. Bandwidth')
        plt.grid(True, alpha=0.3)
        if min(x) > 0:
            plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_results(self):
        """Save experiment results to a JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        # Add metadata
        serializable_results['metadata'] = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_path': self.config_path
        }
        
        # Save to JSON file
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved experiment results to {results_path}")
    
    def _set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Set random seed to {seed} for reproducibility")


class ExperimentSuite:
    """
    Class for running multiple experiments with different configurations.
    
    This class allows for systematic comparison of different model architectures,
    fusion strategies, or hyperparameter settings.
    """
    
    def __init__(self, 
                 suite_name: str,
                 base_config_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the experiment suite.
        
        Args:
            suite_name: Name for this suite of experiments
            base_config_path: Optional path to base configuration file
            output_dir: Optional output directory for results
        """
        self.suite_name = suite_name
        self.base_config_path = base_config_path
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'outputs/suites/{suite_name}_{timestamp}'
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger('experiment_suite', 
                                  os.path.join(self.output_dir, 'suite.log'),
                                  level=logging.INFO)
        
        # Initialize suite state
        self.experiments = []
        self.results = {}
        
        # Load base configuration if provided
        self.base_config = None
        if base_config_path is not None:
            self.base_config = load_config(base_config_path)
        
        self.logger.info(f"Initialized experiment suite: {suite_name}")
    
    def add_experiment(self, 
                     config: Union[Dict, str],
                     experiment_name: Optional[str] = None) -> None:
        """
        Add an experiment to the suite.
        
        Args:
            config: Configuration dictionary or path to configuration file
            experiment_name: Optional name for this experiment
        """
        # Load configuration if path is provided
        if isinstance(config, str):
            config = load_config(config)
        
        # Merge with base configuration if available
        if self.base_config is not None:
            config = self._merge_configs(self.base_config, config)
        
        # Generate temporary config file
        config_path = os.path.join(self.output_dir, f'temp_config_{len(self.experiments)}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Generate default name if not provided
        if experiment_name is None:
            experiment_name = f"experiment_{len(self.experiments)}"
        
        # Add to experiments list
        self.experiments.append({
            'name': experiment_name,
            'config_path': config_path
        })
        
        self.logger.info(f"Added experiment '{experiment_name}' to suite")
    
    def run(self) -> Dict[str, Any]:
        """
        Run all experiments in the suite.
        
        Returns:
            Dictionary with results from all experiments
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment suite: {self.suite_name} "
                       f"with {len(self.experiments)} experiments")
        
        # Run each experiment
        for i, exp_info in enumerate(self.experiments):
            exp_name = exp_info['name']
            config_path = exp_info['config_path']
            
            self.logger.info(f"Running experiment {i+1}/{len(self.experiments)}: {exp_name}")
            
            try:
                # Create and run experiment
                experiment = Experiment(
                    config_path=config_path,
                    experiment_name=exp_name,
                    output_dir=os.path.join(self.output_dir, exp_name)
                )
                
                # Run experiment
                results = experiment.run()
                
                # Store results
                self.results[exp_name] = results
                
                self.logger.info(f"Experiment {exp_name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error running experiment {exp_name}: {e}", exc_info=True)
        
        # Compare experiment results
        self._compare_experiments()
        
        # Save suite results
        self._save_suite_results()
        
        # Log total time
        total_time = time.time() - start_time
        self.logger.info(f"Experiment suite completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _compare_experiments(self):
        """Compare results across experiments in the suite."""
        self.logger.info("Comparing experiment results")
        
        # Extract metrics across experiments
        experiment_names = []
        mota_values = []
        motp_values = []
        ids_values = []
        bps_values = []
        
        for exp_name, results in self.results.items():
            # Skip experiments without evaluation results
            if 'evaluation' not in results:
                continue
                
            eval_results = results['evaluation']
            
            # Get metrics from fusion with 0ms latency if available
            if 'fusion_latency_0ms' in eval_results:
                metrics = eval_results['fusion_latency_0ms']
                experiment_names.append(exp_name)
                mota_values.append(metrics.get('MOTA', 0))
                motp_values.append(metrics.get('MOTP', 0))
                ids_values.append(metrics.get('IDS', 0))
                bps_values.append(metrics.get('BPS', 0))
        
        # Generate comparison visualizations
        self._generate_comparison_plots(
            experiment_names, mota_values, motp_values, ids_values, bps_values
        )
    
    def _generate_comparison_plots(self,
                                 experiment_names: List[str],
                                 mota_values: List[float],
                                 motp_values: List[float],
                                 ids_values: List[float],
                                 bps_values: List[float]):
        """
        Generate plots comparing metrics across experiments.
        
        Args:
            experiment_names: List of experiment names
            mota_values: List of MOTA values
            motp_values: List of MOTP values
            ids_values: List of IDS values
            bps_values: List of BPS values
        """
        if not experiment_names:
            return
            
        # Create output directory
        vis_dir = os.path.join(self.output_dir, 'comparisons')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot MOTA comparison
        self._plot_metric_comparison(
            experiment_names, mota_values, 'MOTA (%)',
            os.path.join(vis_dir, 'mota_comparison.png')
        )
        
        # Plot MOTP comparison
        self._plot_metric_comparison(
            experiment_names, motp_values, 'MOTP',
            os.path.join(vis_dir, 'motp_comparison.png')
        )
        
        # Plot IDS comparison
        self._plot_metric_comparison(
            experiment_names, ids_values, 'ID Switches',
            os.path.join(vis_dir, 'ids_comparison.png')
        )
        
        # Plot bandwidth comparison
        self._plot_metric_comparison(
            experiment_names, bps_values, 'Bandwidth (Bytes/s)',
            os.path.join(vis_dir, 'bandwidth_comparison.png'),
            log_scale=True
        )
        
        # Plot MOTA vs. bandwidth
        self._plot_mota_vs_bandwidth_comparison(
            experiment_names, mota_values, bps_values,
            os.path.join(vis_dir, 'mota_vs_bandwidth.png')
        )
    
    def _plot_metric_comparison(self,
                              experiment_names: List[str],
                              metric_values: List[float],
                              metric_name: str,
                              output_path: str,
                              log_scale: bool = False):
        """
        Plot comparison of a metric across experiments.
        
        Args:
            experiment_names: List of experiment names
            metric_values: List of metric values
            metric_name: Name of the metric
            output_path: Output path for the plot
            log_scale: Whether to use log scale for y-axis
        """
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(experiment_names, metric_values, alpha=0.7)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, value * 1.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Experiment')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name} Across Experiments')
        plt.grid(True, alpha=0.3, axis='y')
        if log_scale and min(metric_values) > 0:
            plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_mota_vs_bandwidth_comparison(self,
                                         experiment_names: List[str],
                                         mota_values: List[float],
                                         bps_values: List[float],
                                         output_path: str):
        """
        Plot MOTA against bandwidth for all experiments.
        
        Args:
            experiment_names: List of experiment names
            mota_values: List of MOTA values
            bps_values: List of bandwidth values
            output_path: Output path for the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(bps_values, mota_values, s=100, alpha=0.7)
        
        # Add experiment name labels
        for name, x, y in zip(experiment_names, bps_values, mota_values):
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Bandwidth (Bytes/s)')
        plt.ylabel('MOTA (%)')
        plt.title('MOTA vs. Bandwidth Across Experiments')
        plt.grid(True, alpha=0.3)
        if min(bps_values) > 0:
            plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _save_suite_results(self):
        """Save suite results to a JSON file."""
        # Convert results to serializable format
        serializable_results = {}
        
        for exp_name, results in self.results.items():
            # Extract key metrics only
            exp_summary = {
                'name': exp_name,
                'metrics': {}
            }
            
            if 'evaluation' in results:
                eval_results = results['evaluation']
                
                # Get metrics from fusion with 0ms latency if available
                if 'fusion_latency_0ms' in eval_results:
                    metrics = eval_results['fusion_latency_0ms']
                    exp_summary['metrics'] = {
                        'MOTA': metrics.get('MOTA', 0),
                        'MOTP': metrics.get('MOTP', 0),
                        'IDS': metrics.get('IDS', 0),
                        'BPS': metrics.get('BPS', 0)
                    }
            
            serializable_results[exp_name] = exp_summary
        
        # Add metadata
        serializable_results['metadata'] = {
            'suite_name': self.suite_name,
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(self.experiments)
        }
        
        # Save to JSON file
        results_path = os.path.join(self.output_dir, 'suite_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved suite results to {results_path}")
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration dictionary with overrides
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if (key in merged and isinstance(merged[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override or add value
                merged[key] = value
                
        return merged


def main():
    """
    Main function to run experiments from command line.
    
    This function parses command line arguments and runs either a single
    experiment or an experiment suite based on the provided arguments.
    """
    parser = argparse.ArgumentParser(description="Run V2X-Seq experiments")
    
    # Common arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Experiment-specific arguments
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    
    # Suite-specific arguments
    parser.add_argument("--suite", action="store_true", help="Run an experiment suite")
    parser.add_argument("--suite_name", type=str, default=None, help="Suite name")
    parser.add_argument("--configs", type=str, nargs="+", help="List of config files for suite")
    
    args = parser.parse_args()
    
    if args.suite:
        # Run experiment suite
        suite_name = args.suite_name or f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        suite = ExperimentSuite(
            suite_name=suite_name,
            base_config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Add experiments from config files
        if args.configs:
            for i, config_path in enumerate(args.configs):
                exp_name = f"experiment_{i}"
                suite.add_experiment(config_path, exp_name)
        
        # Run suite
        suite.run()
    else:
        # Run single experiment
        if not args.config:
            parser.error("--config is required for single experiment")
        
        experiment = Experiment(
            config_path=args.config,
            experiment_name=args.name,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        experiment.run()


if __name__ == "__main__":
    main()