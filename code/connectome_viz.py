# ==============================================================================
# Title: code/connectome_viz.py
# ==============================================================================
# Description: This module provides functions for visualizing brain connectivity
# matrices using bezier curves and node images. It includes utilities for
# converting connectivity indices, ordering nodes based on spatial coordinates,
# and creating detailed connectome plots with customizable aesthetics.
# ==============================================================================

import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
import matplotlib.patches as patches
import os
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional

def linear_to_matrix_indices(linear_indices, n_nodes):
    """
    Convert linear indices from flattened upper triangle back to matrix indices.
    
    Parameters:
    -----------
    linear_indices : array-like
        Linear indices from the flattened upper triangle
    n_nodes : int
        Number of nodes in the connectivity matrix
        
    Returns:
    --------
    row_indices : array
        Row indices in the matrix
    col_indices : array
        Column indices in the matrix
    """
    # Get all upper triangle indices
    upper_indices = np.triu_indices(n_nodes, k=1)
    
    # Convert to linear indices for mapping
    all_linear_indices = upper_indices[0] * n_nodes + upper_indices[1]
    
    # Find positions of our specific indices in the upper triangle
    positions = np.searchsorted(all_linear_indices, linear_indices)
    
    # Get corresponding row and column indices
    row_indices = upper_indices[0][positions]
    col_indices = upper_indices[1][positions]
    
    return row_indices, col_indices


def connectivity_indices_to_matrix_indices(conn_indices, n_nodes):
    """
    Convert connectivity feature indices (0-based) to matrix indices.
    
    Parameters:
    -----------
    conn_indices : array-like
        Connectivity feature indices (e.g., from significant results)
    n_nodes : int
        Number of nodes in the connectivity matrix
        
    Returns:
    --------
    row_indices : array
        Row indices in the matrix
    col_indices : array
        Column indices in the matrix
    """
    # Get upper triangle indices in order
    upper_indices = np.triu_indices(n_nodes, k=1)
    
    # Connectivity indices directly correspond to positions in upper triangle
    row_indices = upper_indices[0][conn_indices]
    col_indices = upper_indices[1][conn_indices]
    
    return row_indices, col_indices

def order_nodes_by_coordinates(labels: List[str], coordinates_df: pd.DataFrame, 
                              label_col: str = 'label') -> List[int]:
    """
    Orders nodes based on their spatial coordinates from a DataFrame.
    
    Parameters:
    -----------
    labels : list
        Node labels to be ordered
    coordinates_df : pandas.DataFrame
        DataFrame containing node coordinates with columns 'x', 'y', 'z', and label_col
    label_col : str
        Name of the column containing node labels
    
    Returns:
    --------
    list : Indices for ordering the nodes
    """
    # Check if required columns exist
    required_cols = ['x', 'y', label_col]
    
    for col in required_cols:
        if col not in coordinates_df.columns:
            print(f"Warning: Column '{col}' not found in coordinates DataFrame. Columns available: {coordinates_df.columns.tolist()}")
            return list(range(len(labels)))
    
    # Check for 'z' column - it's optional
    has_z = 'z' in coordinates_df.columns
    
    # Create a mapping of labels to their coordinates
    label_to_coords = {}
    for idx, row in coordinates_df.iterrows():
        label = str(row[label_col])
        if has_z:
            coords = (row['x'], row['y'], row['z'])
        else:
            coords = (row['x'], row['y'])
        label_to_coords[label] = coords
    
    # Check which labels have coordinates
    missing_labels = []
    for label in labels:
        if label not in label_to_coords:
            missing_labels.append(label)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} labels don't have coordinates. First few: {missing_labels[:5]}")
    
    # Calculate the angle for each node based on coordinates
    label_to_angle = {}
    for label, coords in label_to_coords.items():
        # For 2D, use x and y directly
        if len(coords) == 2:
            x, y = coords
            # Calculate angle in the x-y plane (theta)
            theta = np.arctan2(y, x)
            label_to_angle[label] = theta
        # For 3D, project onto the x-y plane first
        else:
            x, y, z = coords
            # Calculate angle in the x-y plane (theta)
            theta = np.arctan2(y, x)
            label_to_angle[label] = theta
    
    # Sort labels by their angles
    # First, get indices for labels that have coordinates
    indices_with_coords = [i for i, label in enumerate(labels) if label in label_to_angle]
    
    # Sort these indices by the angle
    sorted_indices = sorted(indices_with_coords, key=lambda i: label_to_angle[labels[i]])
    
    # Add indices for labels without coordinates at the end
    indices_without_coords = [i for i, label in enumerate(labels) if label not in label_to_angle]
    
    # Return the full ordering
    return sorted_indices + indices_without_coords


def create_bezier_connectome(
    matrix: np.ndarray, 
    labels: List[str], 
    image_paths: Optional[Dict[str, str]] = None, 
    output_path: Optional[str] = None, 
    figsize: Tuple[int, int] = (12, 12),  # Increased default figure size
    dpi: int = 300, 
    image_scale: float = 0.45,  # Increased image scale even more
    curve_height: float = 0.15,  # Further decreased for smaller curves
    min_line_width: float = 0.5,  # Reduced minimum line width for less visual clutter
    coordinates_df: Optional[pd.DataFrame] = None,
    label_col: str = 'label',
    edge_color: Tuple[float, float, float] = (0.4, 0.7, 0.9),  # Light blue by default (matches your image)
    show_circle_boundary: bool = True,
    show_matrix_thumbnail: bool = False,  # New parameter to show connectivity matrix
    matrix_thumbnail_size: float = 0.3,  # New parameter for matrix thumbnail size
    connection_threshold: float = 75,  # Percentile threshold for displaying connections (0-100)
    node_size: float = 15,  # Increased node size for better visibility
    label_fontsize: int = 9  # Increased label font size
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a connectome visualization with bezier curves and optional images.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Connectivity matrix
    labels : list
        Node labels
    image_paths : dict, optional
        Dictionary mapping node labels to image file paths
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple
        Figure size (inches)
    dpi : int
        DPI for the output figure
    image_scale : float
        Scale factor for the images
    curve_height : float
        Height factor for the bezier curves (0-1)
    min_line_width : float
        Minimum line width for connections
    coordinates_df : pandas.DataFrame, optional
        DataFrame with columns 'x', 'y', 'z' (optional), and label_col
    label_col : str
        Column name for node labels in the DataFrame
    edge_color : tuple
        RGB color for edges (default is light blue)
    show_circle_boundary : bool
        Whether to show a dashed circle boundary
    show_matrix_thumbnail : bool
        Whether to show a thumbnail of the connectivity matrix
    matrix_thumbnail_size : float
        Size of the matrix thumbnail relative to the main figure
    connection_threshold : float
        Percentile threshold (0-100) for displaying connections, higher values show fewer connections
    node_size : float
        Size of the node circles
    label_fontsize : int
        Font size for node labels
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import os
    import pandas as pd
    from typing import List, Dict, Tuple, Optional
    
    # Function to order nodes based on their spatial coordinates
    def order_nodes_by_coordinates(labels, df, label_col):
        # Find the angle of each node in the horizontal plane
        df = df.copy()
        
        # Ensure label column exists
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
        
        # Compute angles in the horizontal plane (x-y)
        df['angle'] = np.arctan2(df['y'], df['x'])
        
        # Sort by angle
        sorted_df = df.sort_values('angle')
        
        # Create mapping from original labels to indices
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Get indices in the sorted order
        sorted_indices = [label_to_idx[label] for label in sorted_df[label_col] if label in label_to_idx]
        
        return sorted_indices
    
    n_nodes = len(labels)
    
    # Ensure image_paths is a dictionary, even if empty
    if image_paths is None:
        image_paths = {}
    
    # Use coordinates for ordering if provided
    if coordinates_df is not None:
        order_indices = order_nodes_by_coordinates(
            labels, 
            coordinates_df, 
            label_col
        )
        
        # Reorder matrix, labels based on the spatial ordering
        matrix = matrix[order_indices, :][:, order_indices]
        labels = [labels[i] for i in order_indices]
        print(f"Reordered nodes based on spatial coordinates")
    
    # Create figure with square aspect ratio and grid for layout
    if show_matrix_thumbnail:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Calculate grid proportions based on thumbnail size
        main_size = 1.0 - matrix_thumbnail_size
        gs = GridSpec(1, 2, width_ratios=[main_size, matrix_thumbnail_size])
        
        # Create main connectome axis and thumbnail axis
        ax = fig.add_subplot(gs[0, 0])
        ax_thumbnail = fig.add_subplot(gs[0, 1])
        
        # Add the connectivity matrix thumbnail
        im = ax_thumbnail.imshow(matrix, cmap='viridis', aspect='auto')
        ax_thumbnail.set_title('Connectivity Matrix')
        ax_thumbnail.set_xticks([])
        ax_thumbnail.set_yticks([])
        plt.colorbar(im, ax=ax_thumbnail)
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    
    # Calculate node positions on a circle
    radius = 10
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    # Rotate to start from the top
    angles = angles + np.pi/2
    
    # Calculate x,y coordinates
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Draw bezier edges (connections) first so they're in the background
    max_weight = np.max(matrix)
    
    # Optional parameters for connection filtering
    threshold_percentile = connection_threshold  # Only show connections above this percentile
    threshold_value = np.percentile(matrix[matrix > 0], threshold_percentile) if np.sum(matrix > 0) > 0 else 0
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Only process each pair once
            # Only draw connections that exceed the threshold
            if matrix[i, j] > threshold_value:
                # Normalize weight for color intensity and width
                weight = matrix[i, j] / max_weight
                # Make connections more transparent overall
                alpha = min(0.7, max(0.05, weight * 0.8))  # Reduced from 0.9 to 0.7 maximum
                line_width = min_line_width + weight * 1.5  # Reduced multiplier from 2 to 1.5
                
                # Calculate control points for the bezier curve
                # Start and end points
                start = (x_pos[i], y_pos[i])
                end = (x_pos[j], y_pos[j])
                
                # Calculate midpoint
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                # Calculate distance between nodes
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx*dx + dy*dy)
                
                # Calculate angle between nodes
                angle_between = np.arctan2(end[1] - start[1], end[0] - start[0])
                
                # Calculate whether connection is crossing through large portion of circle
                # by checking angular distance
                angular_dist = min(
                    abs(angles[i] - angles[j]),
                    2 * np.pi - abs(angles[i] - angles[j])
                )
                
                # For long arcs, we need more aggressive curving toward the center
                # For short arcs, less curvature is needed
                if angular_dist > np.pi:
                    # Long arc (going across the circle)
                    # Calculate how far from center the midpoint is
                    dist_to_center = np.sqrt(mid_x**2 + mid_y**2)
                    
                    # Calculate vector from midpoint to center
                    center_dir_x = -mid_x / (dist_to_center + 1e-10)
                    center_dir_y = -mid_y / (dist_to_center + 1e-10)
                    
                    # Pull control point toward center based on angular distance
                    # The longer the arc, the more we pull toward center
                    # Reduced curve factor for smaller curves
                    pull_factor = curve_height * 1.2  # Reduced from 1.5
                    control_x = mid_x + center_dir_x * radius * pull_factor
                    control_y = mid_y + center_dir_y * radius * pull_factor
                    
                else:
                    # Short arc (around the perimeter)
                    if length > 0:
                        # Calculate perpendicular direction
                        perpx = -dy / length
                        perpy = dx / length
                        
                        # Always curve inward
                        direction = -1
                        
                        # Adjust curvature based on arc length
                        # Longer arcs need more curve but overall reduced
                        curvature = curve_height * (0.4 + angular_dist / np.pi)  # Reduced factor
                        
                        # Calculate control point
                        control_x = mid_x + direction * perpx * radius * curvature
                        control_y = mid_y + direction * perpy * radius * curvature
                    else:
                        control_x = mid_x
                        control_y = mid_y
                
                # Safety check: ensure control point is inside the circle
                control_dist = np.sqrt(control_x**2 + control_y**2)
                if control_dist > radius * 0.95:
                    # If control point is too close to the edge, pull it inward
                    scale_factor = (radius * 0.8) / control_dist
                    control_x *= scale_factor
                    control_y *= scale_factor
                
                # Create the bezier path
                verts = [
                    start,
                    (control_x, control_y),
                    end
                ]
                codes = [
                    Path.MOVETO,
                    Path.CURVE3,
                    Path.CURVE3
                ]
                
                bezier_path = Path(verts, codes)
                
                # Create the patch
                edge_color_with_alpha = (*edge_color, alpha)
                patch = patches.PathPatch(
                    bezier_path, 
                    facecolor='none', 
                    edgecolor=edge_color_with_alpha,
                    linewidth=line_width,
                    zorder=1
                )
                ax.add_patch(patch)
    
    # Draw nodes and add labels
    for i, label in enumerate(labels):
        # Draw node (white circle with black edge)
        ax.scatter(x_pos[i], y_pos[i], s=node_size, color='white', 
                  edgecolor='black', linewidth=1, zorder=5)
        
        # Add text label
        # Position label slightly outside circle
        label_radius = radius * 1.05
        label_x = label_radius * np.cos(angles[i])
        label_y = label_radius * np.sin(angles[i])
        
        # Adjust text alignment based on position in circle
        ha = 'left' if np.cos(angles[i]) >= 0 else 'right'
        va = 'center'
        
        # Add the label with a white background for better visibility
        text = ax.text(label_x, label_y, label, fontsize=label_fontsize,
                      ha=ha, va=va, fontweight='bold', zorder=6)
        
        # Try to add image if available
        if label in image_paths:
            try:
                # Convert path to absolute string path
                img_path = os.path.abspath(str(image_paths[label]))
                
                # Verify the file exists
                if not os.path.exists(img_path):
                    print(f"Image file does not exist: {img_path}")
                    continue
                
                # Load image
                img = mpimg.imread(img_path)
                
                # Position image outside the label
                img_radius = radius * 1.35  # Positioned further out for better visibility
                img_x = img_radius * np.cos(angles[i])
                img_y = img_radius * np.sin(angles[i])
                
                # Create OffsetImage with explicitly controlled size
                imagebox = OffsetImage(img, zoom=image_scale)
                # Add white background and frame to image
                ab = AnnotationBbox(
                    imagebox, (img_x, img_y), 
                    frameon=False, 
                    bboxprops=dict(facecolor='white', edgecolor='gray', linewidth=1, boxstyle="round,pad=0.3"),
                    pad=0.5,
                    zorder=7
                )
                ax.add_artist(ab)
                print(f"Successfully added image for node {label} at ({img_x:.2f}, {img_y:.2f})")
            except Exception as e:
                print(f"Failed to add image for node {label}: {e}")
    
    # Draw the circle boundary
    if show_circle_boundary:
        circle = plt.Circle((0, 0), radius, fill=False, edgecolor='gray', 
                          linestyle='--', linewidth=0.5, zorder=0)
        ax.add_artist(circle)
    
    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    # Increase margin to accommodate larger images
    ax.set_xlim(-radius*1.6, radius*1.6)
    ax.set_ylim(-radius*1.6, radius*1.6)
    ax.axis('off')
    
    # Save the figure if output path is provided
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show_matrix_thumbnail:
        return fig, (ax, ax_thumbnail)
    else:
        return fig, ax

def mask_dlabel_data(dscalar_filename, dlabel_filename):
    """
    Extracts the parcellation data from a CIFTI file and fills in the data for unmasked vertices.

    Parameters
    ----------
    dscalar_filename : str
        Path to the dscalar file containing the data.
    dlabel_filename : str
        Path to the dlabel file containing the parcellation.

    Returns
    -------
    left_full : array
        Array containing the parcellation data for the left hemisphere.
    right_full : array
        Array containing the parcellation data for the right hemisphere
    """
    cifti_file = nib.load(dscalar_filename)
    brain_models = cifti_file.header.get_index_map(1).brain_models

    # Extract vertex mappings
    for model in brain_models:
        if model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            left_vertices = model.vertex_indices
        elif model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            right_vertices = model.vertex_indices

    # Load the parcellation data
    dlabel_file = nib.load(dlabel_filename)

    # Get the mapping axis (usually axis 1 in CIFTI files)
    brain_models = dlabel_file.header.get_index_map(1).brain_models

    # Initialize variables to store start and stop indices
    left_start = None
    left_stop = None
    right_start = None
    right_stop = None

    # Track the current position in the data array
    current_pos = 0

    # Iterate through all brain models in the file
    for model in brain_models:
        # Number of vertices in this model
        index_count = model.index_count

        # Check if this is the left cortex
        if model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            left_start = current_pos
            left_stop = current_pos + index_count

        # Check if this is the right cortex
        elif model.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            right_start = current_pos
            right_stop = current_pos + index_count

        # Update position counter
        current_pos += index_count

    # Now extract the data for each hemisphere
    data = dlabel_file.get_fdata()

    # Handle the different possible data shapes
    if len(data.shape) > 1:
        # If data has multiple dimensions (e.g., time series)
        data = data[0]  # For label files, usually just take the first dimension

    if left_start is not None and left_stop is not None:
        left_parcellation = data[left_start:left_stop]
        print(f"Left hemisphere: {len(left_parcellation)} vertices")

    if right_start is not None and right_stop is not None:
        right_parcellation = data[right_start:right_stop]
        print(f"Right hemisphere: {len(right_parcellation)} vertices")

    # Fill in the data for the unmasked vertices
    left_full = np.full(32492, np.nan)
    right_full = np.full(32492, np.nan)
    left_full[left_vertices] = left_parcellation
    right_full[right_vertices] = right_parcellation

    return left_full, right_full

def extract_lower_triangle(mats):
    # Extract the upper triangle from each matrix
    triu_mats = []

    # Each row is a flattened connectivity matrix
    for i, row in mats.iterrows():
        mat_2d = row.values.reshape(
            int(np.sqrt(row.values.size)),
            int(np.sqrt(row.values.size))
        )
        mat_triu = sym_matrix_to_vec(mat_2d, discard_diagonal=True)
        triu_mats.append(mat_triu)

    mats_df = pd.DataFrame(triu_mats, index=mats.index)
    mats_df.rename(
        columns={i: f'conn_{i+1}' for i in range(mats_df.shape[1])},
        inplace=True
    )

    return mats_df