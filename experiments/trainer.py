"""
trainer module for V2X-Seq project.

This module provides functionality for training VIC3D Tracking models,
including support for various fusion strategies and model architectures.
"""

import os
import time
import datetime
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.config_parser import load_config
from utils.logger import setup_logger
from data.v2x_seq_dataset import V2XSeqDataset, create_dataloader
from evaluation.metrics import compute_tracking_metrics, aggregate_metrics
from evaluation.bandwidth import BandwidthMeter


class Trainer:
    """
    Trainer class for VIC3D Tracking models.
    
    This class handles the training of tracking models with different fusion
    strategies and model architectures.
    """
    
    def __init__(self, config, resume_from=None):
        """
        Initialize the trainer with the given configuration.
        
        Args:
            config: Configuration dictionary or path to config file
            resume_from: Path to checkpoint to resume training from (optional)
        """
        # Load configuration if a path is provided
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config
            
        # Set up logging
        self.experiment_name = self.config.get('experiment_name', 
                                              f"experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        self.output_dir = os.path.join(self.config.get('output_dir', 'outputs'), 
                                      self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            'trainer', 
            os.path.join(self.log_dir, 'train.log'),
            level=logging.INFO
        )
        
        # Set up TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize datasets and loaders
        self._init_datasets()
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Set up training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)
            
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    def _init_datasets(self):
        """Initialize training and validation datasets."""
        # Create training dataset and dataloader
        self.train_dataset, self.train_loader = create_dataloader(
            dataset_path=self.config['data']['dataset_path'],
            split='train',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            shuffle=True,
            drop_last=True,
            segment_length=self.config['data'].get('segment_length', 10),
            use_infrastructure=True,
            use_image=self.config['data'].get('use_image', False),
            simulate_latency=self.config['training'].get('simulate_latency', False),
            latency_ms=self.config['training'].get('latency_ms', 0),
            augment=self.config['training'].get('augment', True)
        )
        
        # Create validation dataset and dataloader
        self.val_dataset, self.val_loader = create_dataloader(
            dataset_path=self.config['data']['dataset_path'],
            split='val',
            batch_size=self.config['evaluation']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            shuffle=False,
            drop_last=False,
            segment_length=self.config['data'].get('segment_length', 10),
            use_infrastructure=True,
            use_image=self.config['data'].get('use_image', False),
            simulate_latency=self.config['evaluation'].get('simulate_latency', True),
            latency_ms=self.config['evaluation'].get('latency_ms', 200)
        )
        
        self.logger.info(f"Initialized datasets - Train: {len(self.train_dataset)} samples, "
                         f"Val: {len(self.val_dataset)} samples")
    
    def _init_models(self):
        """Initialize models based on configuration."""
        # Import and initialize detection model
        if self.config['model']['detection']['type'] == 'pointpillars':
            from models.detection.pointpillars import PointPillarsDetector
            self.detector = PointPillarsDetector(self.config['model']['detection'])
        else:
            from models.detection.center_point import CenterPointDetector
            self.detector = CenterPointDetector(self.config['model']['detection'])
        
        # Import and initialize fusion model
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
            self.logger.warning(f"Unknown fusion type: {fusion_type}, using vehicle-only mode")
        
        # Create combined model for easier training
        self.model = VIC3DTrackingModel(
            detector=self.detector,
            fusion=self.fusion,
            config=self.config
        )
        
        # Move model to device
        self.model.to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Get training parameters
        lr = self.config['training'].get('learning_rate', 0.001)
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training'].get('num_epochs', 40)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 5)
            )
        else:
            self.scheduler = None
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}, not using a scheduler")
            
        self.logger.info(f"Initialized optimizer with LR={lr} and scheduler={scheduler_type}")
    
    def _init_loss_functions(self):
        """Initialize loss functions for detection and tracking."""
        # Detection losses
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
        
        # Loss weights
        self.cls_weight = self.config['training'].get('cls_weight', 1.0)
        self.reg_weight = self.config['training'].get('reg_weight', 1.0)
        self.dir_weight = self.config['training'].get('dir_weight', 0.2)
        
        self.logger.info(f"Initialized losses with weights: cls={self.cls_weight}, "
                        f"reg={self.reg_weight}, dir={self.dir_weight}")
    
    def train(self, num_epochs=None):
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for, defaults to config value
        """
        if num_epochs is None:
            num_epochs = self.config['training'].get('num_epochs', 40)
        
        total_epochs = self.start_epoch + num_epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs "
                       f"({self.start_epoch+1} to {total_epochs})")
        
        # Training loop
        for epoch in range(self.start_epoch, total_epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self._validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation loss for plateau scheduler
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['loss']
                
            self._save_checkpoint(epoch, is_best)
            
            # Early stopping
            early_stop = self.config['training'].get('early_stop', False)
            patience = self.config['training'].get('patience', 10)
            
            if early_stop and hasattr(self, 'early_stop_counter'):
                if is_best:
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    
                if self.early_stop_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                self.early_stop_counter = 0
        
        self.logger.info("Training completed")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.best_val_metric
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        epoch_dir_loss = 0
        
        bandwidth_meter = BandwidthMeter()
        bandwidth_meter.start_measurement()
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = {k: self._to_device(v) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, bandwidth_usage = self.model(batch)
            
            # Calculate loss
            cls_loss = self.cls_loss_fn(predictions['cls_preds'], batch['cls_targets'])
            reg_loss = self.reg_loss_fn(predictions['reg_preds'], batch['reg_targets'])
            dir_loss = self.cls_loss_fn(predictions['dir_preds'], batch['dir_targets']) if 'dir_preds' in predictions else 0
            
            total_loss = (
                self.cls_weight * cls_loss + 
                self.reg_weight * reg_loss + 
                self.dir_weight * dir_loss
            )
            
            # Backward pass and optimize
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', False):
                clip_value = self.config['training'].get('clip_value', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                
            self.optimizer.step()
            
            # Update bandwidth meter
            if bandwidth_usage is not None:
                bandwidth_meter.add_transmission(bandwidth_usage)
            
            # Update running losses
            batch_size = batch['vehicle_points'][0].shape[0]
            epoch_loss += total_loss.item() * batch_size
            epoch_cls_loss += cls_loss.item() * batch_size
            epoch_reg_loss += reg_loss.item() * batch_size
            if isinstance(dir_loss, torch.Tensor):
                epoch_dir_loss += dir_loss.item() * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'cls_loss': cls_loss.item(),
                'reg_loss': reg_loss.item()
            })
            
            # Log to TensorBoard every N steps
            log_interval = self.config['training'].get('log_interval', 10)
            if batch_idx % log_interval == 0:
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/cls_loss', cls_loss.item(), self.global_step)
                self.writer.add_scalar('train/reg_loss', reg_loss.item(), self.global_step)
                if isinstance(dir_loss, torch.Tensor):
                    self.writer.add_scalar('train/dir_loss', dir_loss.item(), self.global_step)
                
                # Log learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', lr, self.global_step)
            
            self.global_step += 1
        
        # End bandwidth measurement
        bandwidth_meter.end_measurement()
        bps = bandwidth_meter.get_bytes_per_second()
        
        # Calculate average losses
        num_samples = len(self.train_loader.dataset)
        epoch_loss /= num_samples
        epoch_cls_loss /= num_samples
        epoch_reg_loss /= num_samples
        epoch_dir_loss /= num_samples
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Return metrics
        metrics = {
            'loss': epoch_loss,
            'cls_loss': epoch_cls_loss,
            'reg_loss': epoch_reg_loss,
            'dir_loss': epoch_dir_loss,
            'time': epoch_time,
            'bandwidth': bps
        }
        
        return metrics
    
    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        epoch_dir_loss = 0
        
        bandwidth_meter = BandwidthMeter()
        bandwidth_meter.start_measurement()
        
        detection_predictions = []
        detection_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            # Create progress bar
            pbar = tqdm(self.val_loader, desc=f"Validate Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = {k: self._to_device(v) for k, v in batch.items()}
                
                # Forward pass
                predictions, bandwidth_usage = self.model(batch)
                
                # Calculate loss
                cls_loss = self.cls_loss_fn(predictions['cls_preds'], batch['cls_targets'])
                reg_loss = self.reg_loss_fn(predictions['reg_preds'], batch['reg_targets'])
                dir_loss = self.cls_loss_fn(predictions['dir_preds'], batch['dir_targets']) if 'dir_preds' in predictions else 0
                
                total_loss = (
                    self.cls_weight * cls_loss + 
                    self.reg_weight * reg_loss + 
                    self.dir_weight * dir_loss
                )
                
                # Update bandwidth meter
                if bandwidth_usage is not None:
                    bandwidth_meter.add_transmission(bandwidth_usage)
                
                # Update running losses
                batch_size = batch['vehicle_points'][0].shape[0]
                epoch_loss += total_loss.item() * batch_size
                epoch_cls_loss += cls_loss.item() * batch_size
                epoch_reg_loss += reg_loss.item() * batch_size
                if isinstance(dir_loss, torch.Tensor):
                    epoch_dir_loss += dir_loss.item() * batch_size
                
                # Collect predictions and targets for tracking metrics
                if 'detections' in predictions:
                    detection_predictions.extend(predictions['detections'])
                    detection_targets.extend(batch['ground_truth'])
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': total_loss.item()
                })
        
        # End bandwidth measurement
        bandwidth_meter.end_measurement()
        bps = bandwidth_meter.get_bytes_per_second()
        
        # Calculate average losses
        num_samples = len(self.val_loader.dataset)
        epoch_loss /= num_samples
        epoch_cls_loss /= num_samples
        epoch_reg_loss /= num_samples
        epoch_dir_loss /= num_samples
        
        # Calculate tracking metrics if predictions are available
        tracking_metrics = {}
        if detection_predictions:
            tracking_metrics = compute_tracking_metrics(
                detection_predictions, 
                detection_targets,
                self.config['evaluation']
            )
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Return metrics
        metrics = {
            'loss': epoch_loss,
            'cls_loss': epoch_cls_loss,
            'reg_loss': epoch_reg_loss,
            'dir_loss': epoch_dir_loss,
            'time': epoch_time,
            'bandwidth': bps
        }
        
        # Add tracking metrics
        metrics.update(tracking_metrics)
        
        return metrics
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """
        Log metrics to console and TensorBoard.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        # Log to console
        self.logger.info(f"Epoch {epoch+1} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Time: {train_metrics['time']:.2f}s")
        
        if 'MOTA' in val_metrics:
            self.logger.info(f"Tracking Metrics - "
                           f"MOTA: {val_metrics.get('MOTA', 0):.2f}, "
                           f"MOTP: {val_metrics.get('MOTP', 0):.2f}, "
                           f"IDS: {val_metrics.get('IDS', 0)}")
        
        # Log to TensorBoard
        self.writer.add_scalars('loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('cls_loss', {
            'train': train_metrics['cls_loss'],
            'val': val_metrics['cls_loss']
        }, epoch)
        
        self.writer.add_scalars('reg_loss', {
            'train': train_metrics['reg_loss'],
            'val': val_metrics['reg_loss']
        }, epoch)
        
        self.writer.add_scalars('dir_loss', {
            'train': train_metrics['dir_loss'],
            'val': val_metrics['dir_loss']
        }, epoch)
        
        self.writer.add_scalars('bandwidth', {
            'train': train_metrics['bandwidth'],
            'val': val_metrics['bandwidth']
        }, epoch)
        
        # Log tracking metrics
        for metric in ['MOTA', 'MOTP', 'IDS']:
            if metric in val_metrics:
                self.writer.add_scalar(f'tracking/{metric}', val_metrics[metric], epoch)
    
    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if hasattr(self, 'early_stop_counter'):
            checkpoint['early_stop_counter'] = self.early_stop_counter
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint to {best_path}")
        
        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if exists
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_val_metric = checkpoint['best_val_metric']
            
            if 'early_stop_counter' in checkpoint:
                self.early_stop_counter = checkpoint['early_stop_counter']
                
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
    
    def _to_device(self, data):
        """
        Move data to device (recursive for nested structures).
        
        Args:
            data: Input data (can be tensor, list, dict, etc.)
            
        Returns:
            Data moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._to_device(item) for item in data)
        else:
            return data


class VIC3DTrackingModel(nn.Module):
    """
    Combined model for Vehicle-Infrastructure Cooperative 3D Tracking.
    
    This class combines the detector and fusion models into a single module
    for easier training and inference.
    """
    
    def __init__(self, detector, fusion, config):
        """
        Initialize the combined tracking model.
        
        Args:
            detector: Detection model
            fusion: Fusion model (can be None for vehicle-only mode)
            config: Configuration dictionary
        """
        super().__init__()
        self.detector = detector
        self.fusion = fusion
        self.config = config
        
        # Whether to use tracking during training
        self.use_tracking = config['training'].get('use_tracking', True)
        
        # Initialize tracker if needed
        if self.use_tracking:
            if config['model']['tracking']['type'] == 'ab3dmot':
                from models.tracking.ab3dmot import AB3DMOT
                self.tracker = AB3DMOT(config['model']['tracking'])
            else:
                from models.tracking.tracker import SimpleTracker
                self.tracker = SimpleTracker(config['model']['tracking'])
    
    def forward(self, batch):
        """
        Forward pass of the combined model.
        
        Args:
            batch: Input batch containing vehicle and infrastructure data
            
        Returns:
            Tuple of (predictions, bandwidth_usage)
        """
        # Extract data
        vehicle_data = batch['vehicle']
        infrastructure_data = batch.get('infrastructure')
        
        # Initialize bandwidth usage
        bandwidth_usage = None
        
        # Apply fusion if available
        if self.fusion is not None and infrastructure_data is not None:
            # Extract transformation matrices
            vehicle_to_world = batch.get('vehicle_to_world', None)
            infrastructure_to_world = batch.get('infrastructure_to_world', None)
            
            # Get timestamp difference
            vehicle_timestamp = batch.get('vehicle_timestamp', 0)
            infrastructure_timestamp = batch.get('infrastructure_timestamp', 0)
            timestamp_diff = vehicle_timestamp - infrastructure_timestamp
            
            # Apply fusion
            fused_data, bandwidth_usage = self.fusion.fuse(
                vehicle_data,
                infrastructure_data,
                vehicle_to_world,
                infrastructure_to_world,
                timestamp_diff
            )
        else:
            # Vehicle-only mode
            fused_data = vehicle_data
        
        # Apply detection model
        detector_output = self.detector(fused_data)
        
        # Apply tracking if in evaluation mode and tracking is enabled
        if not self.training and self.use_tracking and hasattr(self, 'tracker'):
            # Get detections from detector output
            detections = detector_output.get('detections', [])
            
            # Apply tracking
            tracks = self.tracker.track(detections)
            
            # Add tracks to output
            detector_output['tracks'] = tracks
        
        return detector_output, bandwidth_usage