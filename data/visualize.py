"""
visualize module for V2X-Seq project.

This module provides functionality for visualizing different aspects of the V2X-Seq dataset,
including:
- 3D visualizations of point clouds and bounding boxes
- Bird's-eye view (BEV) visualizations
- Camera image visualizations with projected 3D boxes
- Visualization of fusion results from different strategies
- Video creation utilities for recording sequences
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import cv2
import math
from PIL import Image
from io import BytesIO
import json
# Try to import open3d for 3D visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available. Some 3D visualizations will be disabled.")

# Try to import trimesh as an alternative to open3d
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    if not OPEN3D_AVAILABLE:
        logging.warning("Neither Open3D nor Trimesh available. 3D visualizations will be limited.")
import copy

logger = logging.getLogger(__name__)


# Define colors for different object categories
CATEGORY_COLORS = {
    'Car': '#1f77b4',       # blue
    'Truck': '#ff7f0e',     # orange
    'Van': '#2ca02c',       # green
    'Bus': '#d62728',       # red
    'Pedestrian': '#9467bd',# purple
    'Cyclist': '#8c564b',   # brown
    'Tricyclist': '#e377c2',# pink
    'Motorcyclist': '#7f7f7f',# gray
    'Barrowlist': '#bcbd22',# olive
    'TrafficCone': '#17becf'# cyan
}

# Define colors for different data sources
SOURCE_COLORS = {
    'vehicle': '#1f77b4',       # blue
    'infrastructure': '#ff7f0e', # orange
    'fused': '#2ca02c'          # green
}


def get_color_for_category(category: str) -> Tuple[float, float, float]:
    """
    Get RGB color tuple for a specific object category.
    
    Args:
        category: Object category name
        
    Returns:
        Tuple of RGB values (0-1 range)
    """
    hex_color = CATEGORY_COLORS.get(category, '#333333')  # Default to dark gray
    return mcolors.hex2color(hex_color)


def get_color_for_source(source: str) -> Tuple[float, float, float]:
    """
    Get RGB color tuple for a specific data source.
    
    Args:
        source: Data source name ('vehicle', 'infrastructure', or 'fused')
        
    Returns:
        Tuple of RGB values (0-1 range)
    """
    hex_color = SOURCE_COLORS.get(source, '#333333')  # Default to dark gray
    return mcolors.hex2color(hex_color)


def draw_box_3d(ax, vertices, faces, color=(0, 1, 0), alpha=0.5, linewidth=1):
    """
    Draw a 3D bounding box on a matplotlib axis.
    
    Args:
        ax: Matplotlib 3D axis
        vertices: 8 vertices of the box
        faces: List of vertex indices for each face
        color: RGB color tuple
        alpha: Transparency (0-1)
        linewidth: Width of box edges
    """
    # Draw faces as transparent surfaces
    for face in faces:
        face_vertices = vertices[face]
        # Create polygon for this face
        poly = Axes3D.art3d.Poly3DCollection([face_vertices], alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor('k')
        poly.set_linewidth(linewidth)
        ax.add_collection3d(poly)

    # Draw edges
    for i, j in [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]:
        ax.plot([vertices[i][0], vertices[j][0]],
                [vertices[i][1], vertices[j][1]],
                [vertices[i][2], vertices[j][2]], 'k-', alpha=0.5, linewidth=linewidth)


def get_box_corners(box: List[float]) -> np.ndarray:
    """
    Convert box parameters to 8 corner points.
    
    Args:
        box: Box parameters [x, y, z, w, l, h, yaw]
            - (x, y, z): Center position
            - (w, l, h): Width, length, height
            - yaw: Rotation around Z-axis
            
    Returns:
        Numpy array of shape (8, 3) with corner coordinates
    """
    # Extract box parameters
    x, y, z, w, l, h, yaw = box
    
    # Define corners in canonical frame (centered at origin, aligned with axes)
    corners = np.array([
        [l/2, w/2, h/2],   # Front-right-top
        [l/2, -w/2, h/2],  # Front-left-top
        [-l/2, -w/2, h/2], # Rear-left-top
        [-l/2, w/2, h/2],  # Rear-right-top
        [l/2, w/2, -h/2],  # Front-right-bottom
        [l/2, -w/2, -h/2], # Front-left-bottom
        [-l/2, -w/2, -h/2],# Rear-left-bottom
        [-l/2, w/2, -h/2]  # Rear-right-bottom
    ])
    
    # Create rotation matrix for yaw
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotate corners
    corners = corners @ rotation.T
    
    # Translate corners to box center
    corners += np.array([x, y, z])
    
    return corners


def get_box_faces():
    """Get faces for 3D box, where each face is defined by vertex indices."""
    return [
        [0, 1, 2, 3],  # Top face
        [4, 5, 6, 7],  # Bottom face
        [0, 1, 5, 4],  # Front face
        [1, 2, 6, 5],  # Left face
        [2, 3, 7, 6],  # Rear face
        [3, 0, 4, 7]   # Right face
    ]


def get_box_params_from_label(label: Dict) -> List[float]:
    """
    Extract box parameters from a label dictionary.
    
    Args:
        label: Label dictionary with 3D box information
        
    Returns:
        List of box parameters [x, y, z, w, l, h, yaw]
    """
    # Check if the label has a 'box' field directly
    if 'box' in label:
        return label['box']
    
    # Otherwise, extract from separate fields
    try:
        x = label.get('3d_location', {}).get('x', 0)
        y = label.get('3d_location', {}).get('y', 0)
        z = label.get('3d_location', {}).get('z', 0)
        
        w = label.get('3d_dimensions', {}).get('w', 1)
        l = label.get('3d_dimensions', {}).get('l', 1)
        h = label.get('3d_dimensions', {}).get('h', 1)
        
        yaw = label.get('rotation', 0)
        
        return [x, y, z, w, l, h, yaw]
    except (KeyError, AttributeError):
        logger.warning(f"Failed to extract box parameters from label: {label}")
        return [0, 0, 0, 1, 1, 1, 0]  # Default box


def plot_point_cloud_3d(points: np.ndarray, title: str = None, 
                      color=None, size=1, ax=None, equal_aspect=True):
    """
    Plot 3D point cloud.
    
    Args:
        points: Nx3 array of points (x, y, z)
        title: Plot title
        color: Point colors or colormap name
        size: Point size
        ax: Matplotlib axis to use (creates new axis if None)
        equal_aspect: Whether to use equal aspect ratio for axes
        
    Returns:
        Matplotlib figure and axis
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot points
    if color is None:
        # Color by height
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap='viridis',
                          s=size, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Height')
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c=color, s=size, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio if requested
    if equal_aspect:
        set_axes_equal(ax)
    
    return fig, ax


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    
    Args:
        ax: Matplotlib 3D axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call it a box. Set the radius of the box to be 0.5
    # times the max of the ranges of x, y, and z.
    max_range = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])


def plot_point_cloud_bev(points: np.ndarray, title: str = None, 
                       x_range=(0, 100), y_range=(-40, 40),
                       color=None, size=1, ax=None, cmap='viridis'):
    """
    Plot bird's-eye view (BEV) of point cloud.
    
    Args:
        points: Nx3+ array of points (x, y, z, ...)
        title: Plot title
        x_range: Range of x-axis
        y_range: Range of y-axis
        color: Point colors or colormap name
        size: Point size
        ax: Matplotlib axis to use (creates new axis if None)
        cmap: Colormap name if coloring by height
        
    Returns:
        Matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    # Plot points
    if color is None:
        # Color by height
        scatter = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2],
                          cmap=cmap, s=size, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Height')
    else:
        ax.scatter(points[:, 0], points[:, 1], c=color, s=size, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    return fig, ax


def draw_box_bev(ax, box: List[float], color='b', linewidth=1.5, alpha=0.7, 
               fill=False, label=None, zorder=2):
    """
    Draw a 2D bounding box in bird's-eye view.
    
    Args:
        ax: Matplotlib axis
        box: Box parameters [x, y, z, w, l, h, yaw]
        color: Box color
        linewidth: Width of box edges
        alpha: Transparency
        fill: Whether to fill the box
        label: Optional label to display
        zorder: Drawing order (higher is on top)
    """
    # Extract box parameters
    x, y, _, w, l, _, yaw = box
    
    # Create rotation matrix
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    
    # Generate box corners in BEV (centered at origin)
    corners = np.array([
        [l/2, w/2],
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])
    
    # Rotate and translate corners
    corners = corners @ R.T + np.array([x, y])
    
    # Create polygon
    poly = Polygon(corners, fill=fill, color=color, alpha=alpha, 
                  linewidth=linewidth, zorder=zorder)
    ax.add_patch(poly)
    
    # Add label
    if label:
        ax.text(x, y, label, color='black', fontsize=8,
               ha='center', va='center', weight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
               zorder=zorder+1)
    
    # Draw heading arrow
    front_mid = np.array([l/2, 0]) @ R.T + np.array([x, y])
    ax.arrow(x, y, front_mid[0]-x, front_mid[1]-y, 
            color=color, alpha=alpha, width=0.1, 
            head_width=0.5, head_length=1.0, zorder=zorder)


def visualize_frame_bev(points: np.ndarray, labels: List[Dict], 
                       title: str = None, x_range=(0, 100), y_range=(-40, 40),
                       color_by_category=True, show_track_ids=True, 
                       ax=None, save_path=None):
    """
    Visualize a frame in bird's-eye view with point cloud and 3D boxes.
    
    Args:
        points: Nx3+ array of points (x, y, z, ...)
        labels: List of label dictionaries
        title: Plot title
        x_range: Range of x-axis
        y_range: Range of y-axis
        color_by_category: Whether to color boxes by category
        show_track_ids: Whether to show tracking IDs
        ax: Matplotlib axis to use (creates new axis if None)
        save_path: Path to save the visualization (if None, just displays)
    
    Returns:
        Matplotlib figure and axis
    """
    # Plot point cloud
    fig, ax = plot_point_cloud_bev(points, title, x_range, y_range, size=0.5, ax=ax)
    
    # Plot 3D boxes in BEV
    for label in labels:
        box = get_box_params_from_label(label)
        
        # Get box color based on category
        if color_by_category:
            category = label.get('type', 'Unknown')
            color = CATEGORY_COLORS.get(category, '#333333')
        else:
            color = 'b'
        
        # Add label
        if show_track_ids and 'track_id' in label:
            label_text = f"{label.get('track_id')}"
        else:
            label_text = None
        
        # Draw box
        draw_box_bev(ax, box, color=color, label=label_text)
    
    # Add legend if coloring by category
    if color_by_category:
        # Get unique categories
        categories = set(label.get('type', 'Unknown') for label in labels)
        
        # Create legend
        legend_handles = []
        for category in categories:
            color = CATEGORY_COLORS.get(category, '#333333')
            handle = Line2D([0], [0], color=color, linewidth=2, label=category)
            legend_handles.append(handle)
        
        if legend_handles:
            ax.legend(handles=legend_handles, loc='best')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_sequence_bev(frames: List[Dict], 
                         output_path: str = None,
                         color_by_category: bool = True,
                         show_track_ids: bool = True,
                         fps: int = 10,
                         dpi: int = 100):
    """
    Visualize a sequence in bird's-eye view as a video.
    
    Args:
        frames: List of frame dictionaries, each containing:
            - points: Nx3+ array of points
            - labels: List of label dictionaries
            - timestamp: Optional timestamp for title
        output_path: Path to save the video (if None, displays animation)
        color_by_category: Whether to color boxes by category
        show_track_ids: Whether to show tracking IDs
        fps: Frames per second for the video
        dpi: DPI for the output video
    """
    # First frame for setting up the plot
    first_frame = frames[0]
    fig, ax = visualize_frame_bev(
        points=first_frame['points'],
        labels=first_frame['labels'],
        title=f"Frame 0" + (f" - {first_frame.get('timestamp', 0):.2f}s" 
                         if 'timestamp' in first_frame else ""),
        color_by_category=color_by_category,
        show_track_ids=show_track_ids
    )
    
    # Function to update plot for each frame
    def update(frame_idx):
        ax.clear()
        frame = frames[frame_idx]
        
        # Title with frame index and timestamp
        title = f"Frame {frame_idx}"
        if 'timestamp' in frame:
            title += f" - {frame['timestamp']:.2f}s"
        
        # Plot the frame
        visualize_frame_bev(
            points=frame['points'],
            labels=frame['labels'],
            title=title,
            color_by_category=color_by_category,
            show_track_ids=show_track_ids,
            ax=ax
        )
        
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000/fps, blit=False
    )
    
    # Save animation if path provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as MP4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='V2X-Seq'), bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def visualize_3d_boxes(points: np.ndarray, labels: List[Dict],
                     title: str = None, color_by_category=True,
                     show_track_ids=True, open3d_viewer=False,
                     ax=None, save_path=None):
    """
    Visualize 3D point cloud with 3D bounding boxes.
    
    Args:
        points: Nx3+ array of points (x, y, z, ...)
        labels: List of label dictionaries
        title: Plot title
        color_by_category: Whether to color boxes by category
        show_track_ids: Whether to show tracking IDs
        open3d_viewer: Whether to use Open3D viewer instead of matplotlib
        ax: Matplotlib axis to use (creates new axis if None)
        save_path: Path to save the visualization (if None, just displays)
    
    Returns:
        Matplotlib figure and axis, or None if using Open3D
    """
    if open3d_viewer and OPEN3D_AVAILABLE:
        return visualize_3d_open3d(points, labels, color_by_category, show_track_ids)
    
    # Fallback to matplotlib
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=points[:, 2], cmap='viridis', s=0.5, alpha=0.5)
    
    # Get box faces
    faces = get_box_faces()
    
    # Plot 3D boxes
    for label in labels:
        box = get_box_params_from_label(label)
        corners = get_box_corners(box)
        
        # Get box color based on category
        if color_by_category:
            category = label.get('type', 'Unknown')
            color = get_color_for_category(category)
        else:
            color = (0, 0, 1)  # Blue
        
        # Draw 3D box
        draw_box_3d(ax, corners, faces, color=color)
        
        # Add label for tracking ID
        if show_track_ids and 'track_id' in label:
            # Position text above the box
            text_pos = np.mean(corners[0:4], axis=0)  # Average of top corners
            text_pos[2] += 0.5  # Raise slightly above box
            
            ax.text(text_pos[0], text_pos[1], text_pos[2], 
                  str(label['track_id']), color='black', fontsize=10,
                  backgroundcolor='white', ha='center', va='center')
    
    # Set axis limits based on points
    x_min, y_min, z_min = np.min(points[:, :3], axis=0)
    x_max, y_max, z_max = np.max(points[:, :3], axis=0)
    
    # Add some padding
    padding = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
    ax.set_xlim([x_min - padding, x_max + padding])
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.set_zlim([z_min - padding, z_max + padding])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    
    # Make axes equal
    set_axes_equal(ax)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_3d_open3d(points: np.ndarray, labels: List[Dict],
                      color_by_category=True, show_track_ids=True):
    """
    Visualize 3D point cloud with boxes using Open3D.
    
    Args:
        points: Nx3+ array of points (x, y, z, ...)
        labels: List of label dictionaries
        color_by_category: Whether to color boxes by category
        show_track_ids: Whether to show tracking IDs
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D not available. Falling back to matplotlib.")
        return visualize_3d_boxes(points, labels, color_by_category=color_by_category, 
                               show_track_ids=show_track_ids)
    
    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Set point cloud colors (color by height)
    colors = np.zeros_like(points[:, :3])
    normalized_heights = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))
    # Apply colormap (similar to viridis)
    colors[:, 0] = 1 - normalized_heights  # Red
    colors[:, 1] = normalized_heights      # Green
    colors[:, 2] = normalized_heights      # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add point cloud to visualization
    vis.add_geometry(pcd)
    
    # Add 3D boxes
    for label in labels:
        box = get_box_params_from_label(label)
        x, y, z, w, l, h, yaw = box
        
        # Get box color based on category
        if color_by_category:
            category = label.get('type', 'Unknown')
            color = get_color_for_category(category)
        else:
            color = (0, 0, 1)  # Blue
        
        # Create Open3D box
        center = [x, y, z + h/2]  # Box center is at bottom in our format, but Open3D uses center
        size = [l, w, h]
        box_3d = o3d.geometry.OrientedBoundingBox(center, np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ]), size)
        box_3d.color = color
        vis.add_geometry(box_3d)
        
        # Add tracking ID if requested - Open3D doesn't support text directly
        # We could create a line set with the ID, but it's complex
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 2
    
    # Run the visualization
    vis.run()
    vis.destroy_window()


def visualize_fusion_comparison(veh_points: np.ndarray, inf_points: np.ndarray, 
                              veh_labels: List[Dict], inf_labels: List[Dict],
                              fused_labels: List[Dict] = None,
                              title: str = None, save_path=None):
    """
    Visualize comparison between vehicle-only, infrastructure-only, and fusion results.
    
    Args:
        veh_points: Nx3+ array of vehicle points
        inf_points: Nx3+ array of infrastructure points
        veh_labels: List of vehicle label dictionaries
        inf_labels: List of infrastructure label dictionaries
        fused_labels: Optional list of fused label dictionaries
        title: Plot title
        save_path: Path to save the visualization
    
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots
    if fused_labels is not None:
        n_plots = 3
    else:
        n_plots = 2
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6), squeeze=False)
    axes = axes[0]  # Remove extra dimension
    
    # Vehicle-only view
    visualize_frame_bev(veh_points, veh_labels, "Vehicle-only", 
                      show_track_ids=True, ax=axes[0])
    
    # Infrastructure-only view
    # Transform infrastructure points to vehicle frame
    # This assumes infrastructure points are already in vehicle frame
    visualize_frame_bev(inf_points, inf_labels, "Infrastructure-only", 
                      show_track_ids=True, ax=axes[1])
    
    # Fusion view if provided
    if fused_labels is not None:
        # Combine points for visualization
        combined_points = np.vstack([veh_points, inf_points])
        visualize_frame_bev(combined_points, fused_labels, "Fusion result", 
                          show_track_ids=True, ax=axes[2])
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_fusion_sequence_comparison(sequence: List[Dict], output_path: str = None, fps: int = 10):
    """
    Visualize a sequence with vehicle-only, infrastructure-only, and fusion results.
    
    Args:
        sequence: List of frame dictionaries, each containing:
            - veh_points: Vehicle point cloud
            - inf_points: Infrastructure point cloud
            - veh_labels: Vehicle labels
            - inf_labels: Infrastructure labels
            - fused_labels: Fusion result labels
            - timestamp: Optional timestamp
        output_path: Path to save the video (if None, displays animation)
        fps: Frames per second for the video
    """
    # First frame for setting up the plot
    first_frame = sequence[0]
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes = axes[0]  # Remove extra dimension
    
    # Initialize the three views
    visualize_frame_bev(
        points=first_frame['veh_points'],
        labels=first_frame['veh_labels'],
        title="Vehicle-only",
        ax=axes[0]
    )
    
    visualize_frame_bev(
        points=first_frame['inf_points'],
        labels=first_frame['inf_labels'],
        title="Infrastructure-only",
        ax=axes[1]
    )
    
    # Combine points for fusion view
    combined_points = np.vstack([first_frame['veh_points'], first_frame['inf_points']])
    visualize_frame_bev(
        points=combined_points,
        labels=first_frame['fused_labels'],
        title="Fusion result",
        ax=axes[2]
    )
    
    # Add timestamp to suptitle if available
    if 'timestamp' in first_frame:
        fig.suptitle(f"Timestamp: {first_frame['timestamp']:.2f}s", fontsize=16)
    
    # Function to update plot for each frame
    def update(frame_idx):
        frame = sequence[frame_idx]
        
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Update vehicle view
        visualize_frame_bev(
            points=frame['veh_points'],
            labels=frame['veh_labels'],
            title="Vehicle-only",
            ax=axes[0]
        )
        
        # Update infrastructure view
        visualize_frame_bev(
            points=frame['inf_points'],
            labels=frame['inf_labels'],
            title="Infrastructure-only",
            ax=axes[1]
        )
        
        # Update fusion view
        combined_points = np.vstack([frame['veh_points'], frame['inf_points']])
        visualize_frame_bev(
            points=combined_points,
            labels=frame['fused_labels'],
            title="Fusion result",
            ax=axes[2]
        )
        
        # Update timestamp in suptitle if available
        if 'timestamp' in frame:
            fig.suptitle(f"Timestamp: {frame['timestamp']:.2f}s", fontsize=16)
        
        return axes
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(sequence), interval=1000/fps, blit=False
    )
    
    # Save animation if path provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as MP4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='V2X-Seq'), bitrate=5000)
        ani.save(output_path, writer=writer, dpi=100)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def visualize_latency_effect(
    veh_points: np.ndarray, 
    inf_points_no_latency: np.ndarray,
    inf_points_with_latency: np.ndarray,
    veh_labels: List[Dict], 
    inf_labels_no_latency: List[Dict],
    inf_labels_with_latency: List[Dict],
    latency_ms: int,
    title: str = None,
    save_path: str = None
):
    """
    Visualize the effect of latency on infrastructure data.
    
    Args:
        veh_points: Vehicle point cloud
        inf_points_no_latency: Infrastructure points without latency
        inf_points_with_latency: Infrastructure points with simulated latency
        veh_labels: Vehicle labels
        inf_labels_no_latency: Infrastructure labels without latency
        inf_labels_with_latency: Infrastructure labels with simulated latency
        latency_ms: Latency in milliseconds
        title: Plot title
        save_path: Path to save the visualization
    
    Returns:
        Matplotlib figure
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes = axes[0]  # Remove extra dimension
    
    # Vehicle-only view
    visualize_frame_bev(
        points=veh_points,
        labels=veh_labels,
        title="Vehicle-only",
        ax=axes[0]
    )
    
    # Infrastructure without latency
    visualize_frame_bev(
        points=inf_points_no_latency,
        labels=inf_labels_no_latency,
        title="Infrastructure (No latency)",
        ax=axes[1]
    )
    
    # Infrastructure with latency
    visualize_frame_bev(
        points=inf_points_with_latency,
        labels=inf_labels_with_latency,
        title=f"Infrastructure (Latency: {latency_ms}ms)",
        ax=axes[2]
    )
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f"Effect of {latency_ms}ms Latency on Infrastructure Data", fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def overlay_detection_on_image(image: np.ndarray, boxes_3d: List[List[float]], 
                             intrinsic: np.ndarray, extrinsic: np.ndarray,
                             labels: List[Dict] = None, color_by_category: bool = True,
                             show_track_ids: bool = True, linewidth: int = 2):
    """
    Overlay 3D bounding boxes on a camera image.
    
    Args:
        image: Camera image as numpy array (H, W, 3)
        boxes_3d: List of 3D box parameters [x, y, z, w, l, h, yaw]
        intrinsic: Camera intrinsic matrix (3x3)
        extrinsic: Camera extrinsic matrix (4x4, transform from lidar to camera)
        labels: Optional list of label dictionaries (must match boxes_3d in length)
        color_by_category: Whether to color boxes by category
        show_track_ids: Whether to show tracking IDs
        linewidth: Width of box lines
    
    Returns:
        Image with overlaid 3D boxes
    """
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Process each box
    for i, box in enumerate(boxes_3d):
        # Get box corners in 3D
        corners_3d = get_box_corners(box)
        
        # Project corners to image
        corners_2d = []
        for corner in corners_3d:
            # Convert to homogeneous coordinates
            corner_hom = np.array([corner[0], corner[1], corner[2], 1])
            
            # Transform to camera frame
            corner_camera = extrinsic @ corner_hom
            
            # Check if behind camera
            if corner_camera[2] <= 0:
                continue
                
            # Project to image
            corner_img = intrinsic @ corner_camera[:3]
            corner_img = corner_img / corner_img[2]
            
            corners_2d.append(corner_img[:2].astype(int))
        
        # Skip if not enough corners are visible
        if len(corners_2d) < 2:
            continue
            
        # Determine box color
        if color_by_category and labels is not None:
            category = labels[i].get('type', 'Unknown') if i < len(labels) else 'Unknown'
            color = CATEGORY_COLORS.get(category, '#333333')
            # Convert hex color to BGR (for OpenCV)
            color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        else:
            # Default green
            color = (0, 255, 0)
        
        # Draw 3D box edges
        # Bottom face
        cv2.line(image_with_boxes, tuple(corners_2d[0]), tuple(corners_2d[1]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[1]), tuple(corners_2d[2]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[2]), tuple(corners_2d[3]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[3]), tuple(corners_2d[0]), color, linewidth)
        
        # Top face
        cv2.line(image_with_boxes, tuple(corners_2d[4]), tuple(corners_2d[5]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[5]), tuple(corners_2d[6]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[6]), tuple(corners_2d[7]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[7]), tuple(corners_2d[4]), color, linewidth)
        
        # Connecting edges
        cv2.line(image_with_boxes, tuple(corners_2d[0]), tuple(corners_2d[4]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[1]), tuple(corners_2d[5]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[2]), tuple(corners_2d[6]), color, linewidth)
        cv2.line(image_with_boxes, tuple(corners_2d[3]), tuple(corners_2d[7]), color, linewidth)
        
        # Add track ID
        if show_track_ids and labels is not None and i < len(labels) and 'track_id' in labels[i]:
            track_id = labels[i]['track_id']
            # Calculate center of box in image
            center = np.mean(corners_2d, axis=0).astype(int)
            # Draw text with background
            text = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw text background
            cv2.rectangle(
                image_with_boxes,
                (center[0] - text_size[0]//2 - 5, center[1] - text_size[1]//2 - 5),
                (center[0] + text_size[0]//2 + 5, center[1] + text_size[1]//2 + 5),
                (255, 255, 255),
                -1
            )
            
            # Draw text
            cv2.putText(
                image_with_boxes,
                text,
                (center[0] - text_size[0]//2, center[1] + text_size[1]//2),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
    
    return image_with_boxes


def project_point_cloud_to_image(image: np.ndarray, points: np.ndarray,
                               intrinsic: np.ndarray, extrinsic: np.ndarray,
                               min_dist: float = 0.5, max_dist: float = 70.0,
                               point_size: int = 2):
    """
    Project point cloud onto camera image.
    
    Args:
        image: Camera image as numpy array (H, W, 3)
        points: Point cloud as numpy array (N, 3+)
        intrinsic: Camera intrinsic matrix (3x3)
        extrinsic: Camera extrinsic matrix (4x4, transform from lidar to camera)
        min_dist: Minimum distance for points to be projected
        max_dist: Maximum distance for points to be projected
        point_size: Size of projected points
    
    Returns:
        Image with projected point cloud
    """
    # Create a copy of the image to draw on
    image_with_points = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Transform points to camera frame
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    points_camera = points_hom @ extrinsic.T
    
    # Filter points in front of camera
    mask_in_front = points_camera[:, 2] > 0
    
    # Filter by distance
    dist = np.linalg.norm(points_camera[:, :3], axis=1)
    mask_dist = (dist > min_dist) & (dist < max_dist)
    
    # Combine masks
    mask = mask_in_front & mask_dist
    
    # Project points to image
    points_img = points_camera[mask, :3] @ intrinsic.T
    points_img = points_img[:, :2] / points_img[:, 2:3]
    points_img = points_img.astype(int)
    
    # Filter points within image
    mask_in_img = (
        (points_img[:, 0] >= 0) & (points_img[:, 0] < width) &
        (points_img[:, 1] >= 0) & (points_img[:, 1] < height)
    )
    points_img = points_img[mask_in_img]
    depths = dist[mask][mask_in_img]
    
    # Normalize depths for coloring
    if len(depths) > 0:
        normalized_depths = (depths - min_dist) / (max_dist - min_dist)
        normalized_depths = np.clip(normalized_depths, 0, 1)
        
        # Color points by depth
        for i, (x, y) in enumerate(points_img):
            # Use jet colormap
            r = int(255 * min(1, 2 - 2 * normalized_depths[i]))
            g = int(255 * min(1, 2 * normalized_depths[i]))
            b = int(255 * min(1, 2 * normalized_depths[i] - 1))
            
            # Draw colored point
            cv2.circle(image_with_points, (x, y), point_size, (b, g, r), -1)
    
    return image_with_points


def visualize_frame_multiview(
    veh_points: np.ndarray,
    veh_image: np.ndarray,
    veh_labels: List[Dict],
    inf_points: Optional[np.ndarray] = None,
    inf_image: Optional[np.ndarray] = None,
    inf_labels: Optional[List[Dict]] = None,
    transformation_matrices: Optional[Dict] = None,
    title: str = None,
    save_path: str = None
):
    """
    Create a multi-view visualization with BEV, 3D, and camera views.
    
    Args:
        veh_points: Vehicle point cloud
        veh_image: Vehicle camera image
        veh_labels: Vehicle labels
        inf_points: Optional infrastructure point cloud
        inf_image: Optional infrastructure camera image
        inf_labels: Optional infrastructure labels
        transformation_matrices: Optional transformation matrices
        title: Plot title
        save_path: Path to save the visualization
    
    Returns:
        Matplotlib figure
    """
    # Determine number of rows and columns based on available data
    has_inf = inf_points is not None and inf_labels is not None
    
    if has_inf:
        # 2x3 layout: BEV + 3D + camera for vehicle and infrastructure
        fig = plt.figure(figsize=(18, 12))
        grid = plt.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.3)
        
        # Vehicle views
        ax_veh_bev = fig.add_subplot(grid[0, 0])
        ax_veh_3d = fig.add_subplot(grid[0, 1], projection='3d')
        ax_veh_img = fig.add_subplot(grid[0, 2])
        
        # Infrastructure views
        ax_inf_bev = fig.add_subplot(grid[1, 0])
        ax_inf_3d = fig.add_subplot(grid[1, 1], projection='3d')
        ax_inf_img = fig.add_subplot(grid[1, 2])
    else:
        # 1x3 layout: BEV + 3D + camera for vehicle only
        fig = plt.figure(figsize=(18, 6))
        grid = plt.GridSpec(1, 3, figure=fig, wspace=0.4, hspace=0.3)
        
        # Vehicle views
        ax_veh_bev = fig.add_subplot(grid[0, 0])
        ax_veh_3d = fig.add_subplot(grid[0, 1], projection='3d')
        ax_veh_img = fig.add_subplot(grid[0, 2])
        
        # Placeholder variables
        ax_inf_bev = None
        ax_inf_3d = None
        ax_inf_img = None
    
    # Vehicle BEV view
    visualize_frame_bev(
        points=veh_points,
        labels=veh_labels,
        title="Vehicle BEV",
        ax=ax_veh_bev
    )
    
    # Vehicle 3D view
    visualize_3d_boxes(
        points=veh_points,
        labels=veh_labels,
        title="Vehicle 3D",
        ax=ax_veh_3d,
        open3d_viewer=False
    )
    
    # Vehicle camera view
    # Display image
    ax_veh_img.imshow(veh_image)
    ax_veh_img.set_title("Vehicle Camera")
    ax_veh_img.axis('off')
    
    # If we have transformation matrices, overlay 3D boxes on image
    if transformation_matrices is not None:
        # Extract box parameters
        boxes_3d = [get_box_params_from_label(label) for label in veh_labels]
        
        # Get intrinsic and extrinsic matrices
        if 'veh_cam_intrinsic' in transformation_matrices and 'veh_lidar_to_cam' in transformation_matrices:
            intrinsic = transformation_matrices['veh_cam_intrinsic']
            extrinsic = transformation_matrices['veh_lidar_to_cam']
            
            # Overlay boxes on image
            image_with_boxes = overlay_detection_on_image(
                image=veh_image,
                boxes_3d=boxes_3d,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                labels=veh_labels,
                color_by_category=True,
                show_track_ids=True
            )
            
            # Update image display
            ax_veh_img.imshow(image_with_boxes)
    
    # Infrastructure views if available
    if has_inf:
        # Infrastructure BEV view
        visualize_frame_bev(
            points=inf_points,
            labels=inf_labels,
            title="Infrastructure BEV",
            ax=ax_inf_bev
        )
        
        # Infrastructure 3D view
        visualize_3d_boxes(
            points=inf_points,
            labels=inf_labels,
            title="Infrastructure 3D",
            ax=ax_inf_3d,
            open3d_viewer=False
        )
        
        # Infrastructure camera view
        if inf_image is not None:
            # Display image
            ax_inf_img.imshow(inf_image)
            ax_inf_img.set_title("Infrastructure Camera")
            ax_inf_img.axis('off')
            
            # If we have transformation matrices, overlay 3D boxes on image
            if transformation_matrices is not None:
                # Extract box parameters
                boxes_3d = [get_box_params_from_label(label) for label in inf_labels]
                
                # Get intrinsic and extrinsic matrices
                if 'inf_cam_intrinsic' in transformation_matrices and 'inf_lidar_to_cam' in transformation_matrices:
                    intrinsic = transformation_matrices['inf_cam_intrinsic']
                    extrinsic = transformation_matrices['inf_lidar_to_cam']
                    
                    # Overlay boxes on image
                    image_with_boxes = overlay_detection_on_image(
                        image=inf_image,
                        boxes_3d=boxes_3d,
                        intrinsic=intrinsic,
                        extrinsic=extrinsic,
                        labels=inf_labels,
                        color_by_category=True,
                        show_track_ids=True
                    )
                    
                    # Update image display
                    ax_inf_img.imshow(image_with_boxes)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_tracking_results(
    sequence: List[Dict],
    track_ids: List[str] = None,
    output_path: str = None,
    fps: int = 10,
    show_trajectories: bool = True,
    trajectory_length: int = 10
):
    """
    Visualize tracking results for a sequence.
    
    Args:
        sequence: List of frame dictionaries, each containing:
            - points: Point cloud
            - labels: Tracking labels
            - timestamp: Optional timestamp
        track_ids: Optional list of track IDs to highlight
        output_path: Path to save the video (if None, displays animation)
        fps: Frames per second for the video
        show_trajectories: Whether to show trajectories
        trajectory_length: Number of previous frames to show in trajectory
    """
    # First frame for setting up the plot
    first_frame = sequence[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Initialize plot
    visualize_frame_bev(
        points=first_frame['points'],
        labels=first_frame['labels'],
        title=f"Frame 0" + (f" - {first_frame.get('timestamp', 0):.2f}s" 
                          if 'timestamp' in first_frame else ""),
        ax=ax
    )
    
    # Initialize trajectory storage
    trajectories = {}  # track_id -> list of (x,y) positions
    
    # Function to update plot for each frame
    def update(frame_idx):
        ax.clear()
        frame = sequence[frame_idx]
        
        # Title with frame index and timestamp
        title = f"Frame {frame_idx}"
        if 'timestamp' in frame:
            title += f" - {frame['timestamp']:.2f}s"
        
        # Plot the frame
        visualize_frame_bev(
            points=frame['points'],
            labels=frame['labels'],
            title=title,
            ax=ax
        )
        
        # Update trajectories
        for label in frame['labels']:
            if 'track_id' not in label:
                continue
                
            track_id = label['track_id']
            
            # Skip if we're only showing specific tracks and this isn't one
            if track_ids is not None and track_id not in track_ids:
                continue
                
            # Get box center
            box = get_box_params_from_label(label)
            x, y = box[0], box[1]
            
            # Initialize trajectory if needed
            if track_id not in trajectories:
                trajectories[track_id] = []
            
            # Add current position
            trajectories[track_id].append((x, y))
            
            # Limit trajectory length
            if len(trajectories[track_id]) > trajectory_length:
                trajectories[track_id] = trajectories[track_id][-trajectory_length:]
        
        # Draw trajectories
        if show_trajectories:
            for track_id, positions in trajectories.items():
                # Skip if we're only showing specific tracks and this isn't one
                if track_ids is not None and track_id not in track_ids:
                    continue
                    
                # Skip if no positions
                if len(positions) < 2:
                    continue
                
                # Get positions as arrays
                xs, ys = zip(*positions)
                
                # Get category for coloring if possible
                color = None
                for label in frame['labels']:
                    if label.get('track_id') == track_id:
                        category = label.get('type', 'Unknown')
                        color = CATEGORY_COLORS.get(category, '#333333')
                        break
                
                # Draw trajectory
                ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.7)
        
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(sequence), interval=1000/fps, blit=False
    )
    
    # Save animation if path provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as MP4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='V2X-Seq'), bitrate=5000)
        ani.save(output_path, writer=writer, dpi=100)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def create_tracking_metrics_plots(
    metrics_over_time: Dict[str, List[float]],
    timestamps: List[float] = None,
    title: str = None,
    save_path: str = None
):
    """
    Create plots for tracking metrics over time.
    
    Args:
        metrics_over_time: Dictionary mapping metric names to lists of values
        timestamps: Optional list of timestamps
        title: Plot title
        save_path: Path to save the visualization
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(len(metrics_over_time), 1, figsize=(10, 3*len(metrics_over_time)), 
                           sharex=True, squeeze=False)
    axes = axes.flatten()
    
    # X-axis values
    if timestamps is None:
        x = np.arange(len(next(iter(metrics_over_time.values()))))
        x_label = 'Frame'
    else:
        x = timestamps
        x_label = 'Time (s)'
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics_over_time.items()):
        ax = axes[i]
        ax.plot(x, values, '-o', linewidth=1.5, markersize=4)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        
        # Set y-limits based on metric type
        if metric_name in ['MOTA', 'MOTP', 'Recall', 'Precision']:
            ax.set_ylim([0, 1])
    
    # Set x-label for bottom plot
    axes[-1].set_xlabel(x_label)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_visualization_to_html(fig, save_path: str):
    """
    Save a matplotlib figure as an HTML file with interactive plot.
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save the HTML file
    """
    try:
        from mpld3 import fig_to_html, save_html
        
        # Convert figure to HTML
        html = fig_to_html(fig)
        
        # Save HTML
        with open(save_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Interactive visualization saved to {save_path}")
        
    except ImportError:
        logger.warning("mpld3 not available. Saving as static image instead.")
        # Save as PNG instead
        png_path = save_path.replace('.html', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        logger.info(f"Static visualization saved to {png_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V2X-Seq visualization utilities")
    parser.add_argument("--sample_path", type=str, help="Path to sample data", default= r'M:\Documents\Mwasalat\dataset\Full Dataset (train & val)-20250313T155844Z\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\infrastructure-side')
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--mode", type=str, default="bev",
                       choices=["bev", "3d", "fusion", "tracking"],
                       help="Visualization mode")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.sample_path:
        # Load sample data
        if os.path.isdir(args.sample_path):
            # Try to load a sample frame
            logger.info(f"Loading sample data from directory: {args.sample_path}")
            
            # Find point cloud files
            pcd_files = list(Path(args.sample_path).glob("**/*.pcd"))
            
            if not pcd_files:
                logger.error("No PCD files found in the directory")
                exit(1)
                
            # Load first point cloud
            pc_path = pcd_files[0]
            
            try:
                if OPEN3D_AVAILABLE:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(str(pc_path))
                    points = np.asarray(pcd.points)
                    if pcd.has_colors():
                        colors = np.asarray(pcd.colors)
                        points = np.hstack([points, np.mean(colors, axis=1, keepdims=True)])
                    else:
                        points = np.hstack([points, np.ones((points.shape[0], 1))])
                else:
                    # Fallback to numpy
                    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
                
                # Find label files
                label_files = list(Path(args.sample_path).glob("**/*.json"))
                labels = []
                
                if label_files:
                    # Load first label file
                    label_path = label_files[0]
                    with open(label_path, 'r') as f:
                        try:
                            label_data = json.load(f)
                            if isinstance(label_data, list):
                                labels = label_data
                            elif isinstance(label_data, dict) and 'objects' in label_data:
                                labels = label_data['objects']
                            
                            logger.info(f"Loaded {len(labels)} labels from {label_path}")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON from {label_path}")
                
                # Run visualization based on the selected mode
                if args.mode == "bev":
                    # Bird's-eye view
                    fig, _ = visualize_frame_bev(
                        points=points,
                        labels=labels,
                        title=f"Sample Data - {pc_path.name}"
                    )
                    plt.savefig(os.path.join(args.output_dir, "sample_bev.png"), dpi=300)
                    plt.close(fig)
                    logger.info(f"BEV visualization saved to {args.output_dir}/sample_bev.png")
                    
                elif args.mode == "3d":
                    # 3D visualization
                    if OPEN3D_AVAILABLE:
                        # Use Open3D for 3D visualization (no saving)
                        visualize_3d_open3d(
                            points=points,
                            labels=labels
                        )
                    else:
                        # Fallback to matplotlib
                        fig, _ = visualize_3d_boxes(
                            points=points,
                            labels=labels,
                            title=f"Sample Data - {pc_path.name}"
                        )
                        plt.savefig(os.path.join(args.output_dir, "sample_3d.png"), dpi=300)
                        plt.close(fig)
                        logger.info(f"3D visualization saved to {args.output_dir}/sample_3d.png")
                
                elif args.mode == "fusion":
                    # For fusion, we need both vehicle and infrastructure data
                    # Just create a dummy visualization with the same data
                    fig = visualize_fusion_comparison(
                        veh_points=points,
                        inf_points=points,
                        veh_labels=labels,
                        inf_labels=labels,
                        title="Fusion Comparison Example (with same data)"
                    )
                    plt.savefig(os.path.join(args.output_dir, "sample_fusion.png"), dpi=300)
                    plt.close(fig)
                    logger.info(f"Fusion visualization saved to {args.output_dir}/sample_fusion.png")
                
                elif args.mode == "tracking":
                    # For tracking, we need a sequence of frames
                    # Just create a dummy sequence with the same data repeated


                    dummy_sequence = [
                        {'points': points, 'labels': labels, 'timestamp': i * 0.1}
                        for i in range(10)
                    ]
                    visualize_tracking_results(
                        sequence=dummy_sequence,
                        output_path=os.path.join(args.output_dir, "sample_tracking.mp4"),
                        fps=5
                    )
                    logger.info(f"Tracking visualization saved to {args.output_dir}/sample_tracking.mp4")
                
            except Exception as e:
                logger.error(f"Error processing sample data: {e}")
                
        else:
            # Single file
            logger.error("Single file input not yet supported, please provide a directory")
    else:
        # No sample data, generate a sample visualization
        logger.info("No sample data provided, generating demo visualization")
        
        # Create sample data
        np.random.seed(42)
        
        # Sample point cloud
        num_points = 1000
        x = np.random.uniform(0, 50, num_points)
        y = np.random.uniform(-30, 30, num_points)
        z = np.random.uniform(-1, 2, num_points)
        intensity = np.random.uniform(0, 1, num_points)
        points = np.column_stack([x, y, z, intensity])
        
        # Sample labels
        labels = []
        for i in range(5):
            x = np.random.uniform(10, 40)
            y = np.random.uniform(-20, 20)
            z = 0
            w = np.random.uniform(1.5, 2.0)
            l = np.random.uniform(3.5, 5.0)
            h = np.random.uniform(1.4, 1.8)
            yaw = np.random.uniform(-np.pi, np.pi)
            
            category = np.random.choice(['Car', 'Truck', 'Van', 'Bus'])
            
            # Create label
            label = {
                'track_id': f"{i+1}",
                'type': category,
                '3d_location': {'x': x, 'y': y, 'z': z},
                '3d_dimensions': {'w': w, 'l': l, 'h': h},
                'rotation': yaw
            }
            labels.append(label)
        
        # Run visualization based on the selected mode
        if args.mode == "bev":
            # Bird's-eye view
            fig, _ = visualize_frame_bev(
                points=points,
                labels=labels,
                title="Demo BEV Visualization"
            )
            plt.savefig(os.path.join(args.output_dir, "demo_bev.png"), dpi=300)
            plt.close(fig)
            logger.info(f"BEV visualization saved to {args.output_dir}/demo_bev.png")
            
        elif args.mode == "3d":
            # 3D visualization
            fig, _ = visualize_3d_boxes(
                points=points,
                labels=labels,
                title="Demo 3D Visualization"
            )
            plt.savefig(os.path.join(args.output_dir, "demo_3d.png"), dpi=300)
            plt.close(fig)
            logger.info(f"3D visualization saved to {args.output_dir}/demo_3d.png")
        
        elif args.mode == "fusion":
            # Create some infrastructure data with different viewpoint
            inf_points = points.copy()
            inf_points[:, 0] += 10  # Shift x position
            
            inf_labels = copy.deepcopy(labels)
            # Shift infrastructure labels
            for label in inf_labels:
                label['3d_location']['x'] += 10
            
            # Add a few additional objects only visible from infrastructure
            for i in range(2):
                x = np.random.uniform(50, 60)
                y = np.random.uniform(-20, 20)
                z = 0
                w = np.random.uniform(1.5, 2.0)
                l = np.random.uniform(3.5, 5.0)
                h = np.random.uniform(1.4, 1.8)
                yaw = np.random.uniform(-np.pi, np.pi)
                
                category = np.random.choice(['Car', 'Truck', 'Van', 'Bus'])
                
                # Create label
                label = {
                    'track_id': f"{i+10}",
                    'type': category,
                    '3d_location': {'x': x, 'y': y, 'z': z},
                    '3d_dimensions': {'w': w, 'l': l, 'h': h},
                    'rotation': yaw
                }
                inf_labels.append(label)
            
            # Create fusion labels (combine both sets)
            fused_labels = copy.deepcopy(labels) + [label for label in inf_labels if label['track_id'] not in [l['track_id'] for l in labels]]
            
            # Fusion visualization
            fig = visualize_fusion_comparison(
                veh_points=points,
                inf_points=inf_points,
                veh_labels=labels,
                inf_labels=inf_labels,
                fused_labels=fused_labels,
                title="Demo Fusion Comparison"
            )
            plt.savefig(os.path.join(args.output_dir, "demo_fusion.png"), dpi=300)
            plt.close(fig)
            logger.info(f"Fusion visualization saved to {args.output_dir}/demo_fusion.png")
        
        elif args.mode == "tracking":
            # Create a sequence by moving objects
            sequence = []
            num_frames = 20
            
            for i in range(num_frames):
                frame_labels = copy.deepcopy(labels)
                
                # Move objects
                for label in frame_labels:
                    # Move forward based on orientation
                    yaw = label['rotation']
                    speed = np.random.uniform(0.5, 2.0)
                    label['3d_location']['x'] += speed * np.cos(yaw)
                    label['3d_location']['y'] += speed * np.sin(yaw)
                
                # Create frame
                frame = {
                    'points': points,
                    'labels': frame_labels,
                    'timestamp': i * 0.1
                }
                sequence.append(frame)
            
            # Tracking visualization
            visualize_tracking_results(
                sequence=sequence,
                output_path=os.path.join(args.output_dir, "demo_tracking.mp4"),
                fps=5
            )
            logger.info(f"Tracking visualization saved to {args.output_dir}/demo_tracking.mp4")