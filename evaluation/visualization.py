"""
visualization module for V2X-Seq project.

This module provides functionality for visualizing tracking results, both for single-view 
and cooperative perception approaches. It includes 3D visualization of tracked objects,
comparison tools, and temporal visualization sequences.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import open3d as o3d

class TrackingVisualizer:
    """Class for visualizing 3D tracking results."""
    
    def __init__(self, save_dir='./visualizations'):
        """
        Initialize the tracking visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Define standard colors for object categories
        self.category_colors = {
            'Car': '#1f77b4',      # blue
            'Truck': '#ff7f0e',    # orange
            'Van': '#2ca02c',      # green
            'Bus': '#d62728',      # red
            'Pedestrian': '#9467bd',  # purple
            'Cyclist': '#8c564b',   # brown
            'Tricyclist': '#e377c2',  # pink
            'Motorcyclist': '#7f7f7f',  # gray
            'Barrowlist': '#bcbd22',  # olive
            'TrafficCone': '#17becf'   # cyan
        }
        
        # Define colors for vehicle vs infrastructure
        self.source_colors = {
            'vehicle': '#1f77b4',      # blue
            'infrastructure': '#ff7f0e',  # orange
            'fused': '#2ca02c'         # green
        }
    
    def visualize_frame(self, point_cloud, tracking_results, frame_idx, 
                        view='bev', show_labels=True, show_ids=True, 
                        source=None, filename=None):
        """
        Visualize tracking results for a single frame.
        
        Args:
            point_cloud: Nx3 numpy array of point cloud data
            tracking_results: List of dict containing tracking results
                Each dict should have keys: 'bbox', 'category', 'track_id'
                where bbox is [x, y, z, l, w, h, yaw]
            frame_idx: Frame index for title
            view: 'bev' for bird's eye view or '3d' for 3D view
            show_labels: Whether to show category labels
            show_ids: Whether to show tracking IDs
            source: Source of detections ('vehicle', 'infrastructure', or 'fused')
            filename: Filename to save the visualization (without extension)
        """
        if view == 'bev':
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot point cloud from bird's eye view
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                      s=0.5, color='gray', alpha=0.5)
            
            # Plot bounding boxes
            for result in tracking_results:
                bbox = result['bbox']  # [x, y, z, l, w, h, yaw]
                category = result['category']
                track_id = result['track_id']
                
                color = self.category_colors.get(category, '#333333')
                if source:
                    color = self.source_colors.get(source, '#333333')
                
                # Draw rotated rectangle
                self._draw_rotated_box_bev(ax, bbox, color)
                
                # Add text label if requested
                if show_labels and show_ids:
                    label = f"{category}-{track_id}"
                elif show_labels:
                    label = f"{category}"
                elif show_ids:
                    label = f"{track_id}"
                else:
                    label = None
                    
                if label:
                    ax.text(bbox[0], bbox[1], label, color='black',
                           backgroundcolor='white', fontsize=8)
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Frame {frame_idx}')
            
            # Set limits - adjust as needed based on your point cloud range
            ax.set_xlim([0, 100])
            ax.set_ylim([-39.68, 39.68])
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=300)
                plt.close()
            else:
                plt.show()
                
        elif view == '3d':
            # Create Open3D visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            vis.add_geometry(pcd)
            
            # Create bounding boxes
            for result in tracking_results:
                bbox = result['bbox']  # [x, y, z, l, w, h, yaw]
                category = result['category']
                
                color = self.category_colors.get(category, [0.2, 0.2, 0.2])
                if isinstance(color, str) and color.startswith('#'):
                    # Convert hex color to RGB
                    color = mcolors.hex2color(color)
                
                # Create oriented bounding box
                box = self._create_o3d_bbox(bbox, color)
                vis.add_geometry(box)
            
            # Set view
            view_control = vis.get_view_control()
            view_control.set_zoom(0.1)
            
            # Render
            vis.run()
            vis.destroy_window()
    
    def compare_results(self, point_cloud, vehicle_results, fusion_results, 
                        frame_idx, show_ids=True, filename=None):
        """
        Compare tracking results between vehicle-only and fusion approaches.
        
        Args:
            point_cloud: Nx3 numpy array of point cloud data
            vehicle_results: List of dict containing vehicle-only tracking results
            fusion_results: List of dict containing fusion tracking results
            frame_idx: Frame index for title
            show_ids: Whether to show tracking IDs
            filename: Filename to save the visualization (without extension)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot point cloud and tracking results for vehicle-only
        ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                   s=0.5, color='gray', alpha=0.5)
        
        for result in vehicle_results:
            bbox = result['bbox']
            category = result['category']
            track_id = result['track_id']
            
            color = self.category_colors.get(category, '#333333')
            self._draw_rotated_box_bev(ax1, bbox, color)
            
            if show_ids:
                ax1.text(bbox[0], bbox[1], str(track_id), color='black',
                        backgroundcolor='white', fontsize=8)
        
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Frame {frame_idx} - Vehicle Only')
        ax1.set_xlim([0, 100])
        ax1.set_ylim([-39.68, 39.68])
        
        # Plot point cloud and tracking results for fusion
        ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                   s=0.5, color='gray', alpha=0.5)
        
        for result in fusion_results:
            bbox = result['bbox']
            category = result['category']
            track_id = result['track_id']
            
            color = self.category_colors.get(category, '#333333')
            self._draw_rotated_box_bev(ax2, bbox, color)
            
            if show_ids:
                ax2.text(bbox[0], bbox[1], str(track_id), color='black',
                        backgroundcolor='white', fontsize=8)
        
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Frame {frame_idx} - Fusion')
        ax2.set_xlim([0, 100])
        ax2.set_ylim([-39.68, 39.68])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_sequence(self, sequence_data, output_filename=None, fps=10):
        """
        Create animation of tracking results over a sequence.
        
        Args:
            sequence_data: List of tuples (point_cloud, tracking_results, frame_idx)
            output_filename: Filename to save the animation (without extension)
            fps: Frames per second in the animation
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame_idx):
            ax.clear()
            point_cloud, tracking_results, _ = sequence_data[frame_idx]
            
            # Plot point cloud from bird's eye view
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], 
                      s=0.5, color='gray', alpha=0.5)
            
            # Plot bounding boxes
            for result in tracking_results:
                bbox = result['bbox']
                category = result['category']
                track_id = result['track_id']
                
                color = self.category_colors.get(category, '#333333')
                self._draw_rotated_box_bev(ax, bbox, color)
                
                ax.text(bbox[0], bbox[1], str(track_id), color='black',
                       backgroundcolor='white', fontsize=8)
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Frame {frame_idx}')
            
            # Set limits
            ax.set_xlim([0, 100])
            ax.set_ylim([-39.68, 39.68])
        
        anim = animation.FuncAnimation(fig, update, frames=len(sequence_data), interval=1000/fps)
        
        if output_filename:
            anim.save(os.path.join(self.save_dir, f"{output_filename}.mp4"), writer='ffmpeg', fps=fps)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_trajectory(self, trajectory_data, show_categories=True, filename=None):
        """
        Plot trajectories of tracked objects over time.
        
        Args:
            trajectory_data: Dict mapping track_id to list of positions and categories
                Format: {track_id: [(x1, y1, category), (x2, y2, category), ...]}
            show_categories: Whether to color trajectories by category
            filename: Filename to save the visualization (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        for track_id, positions in trajectory_data.items():
            x_values = [pos[0] for pos in positions]
            y_values = [pos[1] for pos in positions]
            
            if show_categories and positions:
                # Use the category of the first point (should be consistent for a track)
                category = positions[0][2]
                color = self.category_colors.get(category, '#333333')
            else:
                color = None
                
            ax.plot(x_values, y_values, '-', linewidth=2, 
                   alpha=0.7, label=f"ID: {track_id}", color=color)
            
            # Mark start and end points
            ax.plot(x_values[0], y_values[0], 'o', color='green', markersize=6)
            ax.plot(x_values[-1], y_values[-1], 's', color='red', markersize=6)
            
            # Add track ID at the end of the trajectory
            ax.text(x_values[-1], y_values[-1], str(track_id), fontsize=10)
        
        # Add legend for categories if showing them
        if show_categories:
            legend_elements = [Line2D([0], [0], color=color, lw=4, label=cat)
                              for cat, color in self.category_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Object Trajectories')
        
        # Set limits
        ax.set_xlim([0, 100])
        ax.set_ylim([-39.68, 39.68])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_metrics_over_latency(self, latencies, metrics_dict, filename=None):
        """
        Plot tracking metrics over different latency values.
        
        Args:
            latencies: List of latency values (in ms)
            metrics_dict: Dict mapping metric names to lists of values
                Format: {'MOTA': [mota1, mota2, ...], 'MOTP': [motp1, motp2, ...], ...}
            filename: Filename to save the visualization (without extension)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in metrics_dict.items():
            ax.plot(latencies, values, '-o', linewidth=2, 
                   label=metric_name, alpha=0.8)
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Metric Value')
        ax.set_title('Tracking Performance vs. Latency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_fusion_comparison(self, metrics, methods, filename=None):
        """
        Plot comparison of different fusion methods.
        
        Args:
            metrics: Dict mapping metric names to lists of values for each method
                Format: {'MOTA': [mota1, mota2, ...], 'MOTP': [motp1, motp2, ...], ...}
            methods: List of method names
            filename: Filename to save the visualization (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(methods))
        width = 0.2
        offset = 0
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.bar(x + offset, values, width, label=metric_name)
            offset += width
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Metric Value')
        ax.set_title('Comparison of Fusion Methods')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(methods)
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _draw_rotated_box_bev(self, ax, bbox, color):
        """
        Draw a rotated 2D box in Bird's Eye View.
        
        Args:
            ax: Matplotlib axis
            bbox: Bounding box [x, y, z, l, w, h, yaw]
            color: Box color
        """
        x, y, _, l, w, _, yaw = bbox
        
        # Create box corners
        corners = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # Rotation matrix
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s],
            [s, c]
        ])
        
        # Rotate corners
        corners = np.dot(corners, R.T)
        
        # Translate corners
        corners[:, 0] += x
        corners[:, 1] += y
        
        # Draw the box
        corners = np.vstack([corners, corners[0]])  # Close the box
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
    
    def _create_o3d_bbox(self, bbox, color):
        """
        Create an Open3D oriented bounding box.
        
        Args:
            bbox: Bounding box [x, y, z, l, w, h, yaw]
            color: Box color as RGB tuple
            
        Returns:
            Open3D oriented bounding box
        """
        x, y, z, l, w, h, yaw = bbox
        
        box = o3d.geometry.OrientedBoundingBox()
        box.center = [x, y, z + h/2]  # Adjust center to account for height
        box.extent = [l, w, h]
        
        # Create rotation matrix from yaw angle (rotate around z-axis)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        box.R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Set color
        box.color = color
        
        return box


def visualize_detections_by_source(point_cloud, detections_dict, frame_idx, 
                                  output_filename=None):
    """
    Visualize detections from different sources (vehicle, infrastructure, fused).
    
    Args:
        point_cloud: Nx3 numpy array of point cloud data
        detections_dict: Dict mapping source name to detection results
            Format: {'vehicle': [...], 'infrastructure': [...], 'fused': [...]}
        frame_idx: Frame index for title
        output_filename: Filename to save the visualization (without extension)
    """
    fig, axes = plt.subplots(1, len(detections_dict), figsize=(6*len(detections_dict), 6))
    
    if len(detections_dict) == 1:
        axes = [axes]
    
    for ax, (source, detections) in zip(axes, detections_dict.items()):
        # Plot point cloud
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.5, color='gray', alpha=0.5)
        
        # Source colors
        source_color = {
            'vehicle': '#1f77b4',      # blue
            'infrastructure': '#ff7f0e',  # orange
            'fused': '#2ca02c'         # green
        }.get(source, '#333333')
        
        # Plot detections
        for det in detections:
            bbox = det['bbox']
            
            # Create box corners
            x, y, _, l, w, _, yaw = bbox
            corners = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2]
            ])
            
            # Rotation matrix
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([
                [c, -s],
                [s, c]
            ])
            
            # Rotate corners
            corners = np.dot(corners, R.T)
            
            # Translate corners
            corners[:, 0] += x
            corners[:, 1] += y
            
            # Draw the box
            corners = np.vstack([corners, corners[0]])  # Close the box
            ax.plot(corners[:, 0], corners[:, 1], color=source_color, linewidth=2)
            
        ax.set_aspect('equal')
        ax.set_xlim([0, 100])
        ax.set_ylim([-39.68, 39.68])
        ax.set_title(f'{source.capitalize()} Detections')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    plt.suptitle(f'Frame {frame_idx} - Detection Comparison')
    plt.tight_layout()
    
    if output_filename:
        visualizer = TrackingVisualizer()
        plt.savefig(os.path.join(visualizer.save_dir, f"{output_filename}.png"), dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_tracking_with_history(point_cloud, current_tracking, history_tracking, 
                                   frame_idx, history_frames=5, output_filename=None):
    """
    Visualize current tracking results with historical trajectories.
    
    Args:
        point_cloud: Nx3 numpy array of point cloud data
        current_tracking: List of dict containing current tracking results
        history_tracking: Dict mapping track_id to list of historical positions
            Format: {track_id: [(x1, y1), (x2, y2), ...]}
        frame_idx: Frame index for title
        history_frames: Number of historical frames to show
        output_filename: Filename to save the visualization (without extension)
    """
    visualizer = TrackingVisualizer()
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.5, color='gray', alpha=0.5)
    
    # Plot current tracking boxes
    for result in current_tracking:
        bbox = result['bbox']
        category = result['category']
        track_id = result['track_id']
        
        color = visualizer.category_colors.get(category, '#333333')
        visualizer._draw_rotated_box_bev(ax, bbox, color)
        
        ax.text(bbox[0], bbox[1], str(track_id), color='black',
               backgroundcolor='white', fontsize=8)
    
    # Plot historical trajectories
    for track_id, history in history_tracking.items():
        if len(history) <= 1:
            continue
            
        # Limit history length
        limited_history = history[-history_frames:]
        
        # Plot trajectory line
        x_values = [pos[0] for pos in limited_history]
        y_values = [pos[1] for pos in limited_history]
        
        ax.plot(x_values, y_values, '-', linewidth=2, alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Frame {frame_idx} - Tracking with History')
    
    # Set limits
    ax.set_xlim([0, 100])
    ax.set_ylim([-39.68, 39.68])
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(os.path.join(visualizer.save_dir, f"{output_filename}.png"), dpi=300)
        plt.close()
    else:
        plt.show()


import os
# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    #make sure ./example_visualizations exists

    dir_path = r'M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\v2x_tracking\evaluation\visualizationExamples'

    os.makedirs(dir_path, exist_ok=True)
    
    # Sample point cloud
    point_cloud = np.random.rand(1000, 3)
    point_cloud[:, 0] *= 100  # X range: 0-100
    point_cloud[:, 1] = (point_cloud[:, 1] * 2 - 1) * 39.68  # Y range: -39.68 to 39.68
    point_cloud[:, 2] *= 3  # Z range: 0-3
    
    # Sample tracking results
    tracking_results = [
        {'bbox': [20, 10, 1, 4, 2, 1.5, 0.2], 'category': 'Car', 'track_id': 1},
        {'bbox': [30, -15, 1, 4, 2, 1.5, 0.5], 'category': 'Car', 'track_id': 2},
        {'bbox': [40, 5, 1, 8, 2.5, 3, 0.1], 'category': 'Bus', 'track_id': 3},
        {'bbox': [15, 20, 1, 4, 2, 1.5, -0.3], 'category': 'Car', 'track_id': 4},
        {'bbox': [60, -5, 1, 4, 2, 1.5, 0], 'category': 'Van', 'track_id': 5},
    ]
    
    # Sample fusion results (with additional detections)
    fusion_results = tracking_results.copy()
    fusion_results.extend([
        {'bbox': [70, 15, 1, 4, 2, 1.5, 0], 'category': 'Car', 'track_id': 6},
        {'bbox': [50, -25, 1, 4, 2, 1.5, 0.7], 'category': 'Car', 'track_id': 7},
    ])
    
    # Create visualizer
    visualizer = TrackingVisualizer(save_dir=dir_path)
    
    # Visualize single frame - BEV
    visualizer.visualize_frame(
        point_cloud, tracking_results, frame_idx=1, 
        view='bev', filename='single_frame_bev'
    )
    
    # Compare vehicle-only vs fusion results
    visualizer.compare_results(
        point_cloud, tracking_results, fusion_results, 
        frame_idx=1, filename='comparison'
    )
    
    # Visualize detections by source
    visualize_detections_by_source(
        point_cloud, 
        {'vehicle': tracking_results, 'infrastructure': fusion_results, 'fused': fusion_results}, 
        frame_idx=1, output_filename='detections_by_source'
    )
    
    print("Example visualizations saved to ./example_visualizations/")