# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:14:09 2025

@author: Admin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tifffile import imread
import pandas as pd
import time

tif_path = '167208_AFstrain_labels.tif'  # update this
csv_path = '167208_tissuestrain-sw75.Lstr.csv'

results_filename = '167208_radcirc_strains.csv'

# ----------------------------------
# Step 1: read files
# ----------------------------------

def read_AF_mask(tif_path):
    donut_mask = imread(tif_path)
    
    # If needed: fix axis order
    if donut_mask.shape[0] == 2160:  # likely Z-first
        donut_mask = np.transpose(donut_mask, (1, 2, 0))

    # Ensure it's binary (if needed)
    donut_mask = (donut_mask > 0).astype(np.uint8)
    return donut_mask
    
def read_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # now uses header=0 by default
    points = df[['x', 'y', 'z']].values
    strain = df[['exx', 'eyy', 'ezz', 'exy', 'eyz', 'exz']].values
    exx, eyy, ezz, exy, eyz, exz = strain.T  # shape (N,)

    N = strain.shape[0]
    strain_tensors = np.zeros((N, 3, 3))

    # Fill tensors symmetrically
    strain_tensors[:, 0, 0] = exx
    strain_tensors[:, 1, 1] = eyy
    strain_tensors[:, 2, 2] = ezz

    strain_tensors[:, 0, 1] = strain_tensors[:, 1, 0] = exy
    strain_tensors[:, 1, 2] = strain_tensors[:, 2, 1] = eyz
    strain_tensors[:, 0, 2] = strain_tensors[:, 2, 0] = exz
    
    return points, strain_tensors

# ----------------------------------
# Step 2: Extract contours and compute centerline
# ----------------------------------
def extract_inner_outer_contours(slice_img):
    """Extract inner and outer contours using hierarchy (cv2.RETR_TREE)"""
    slice_img = (slice_img * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(slice_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return None, None

    contours = [cnt[:, 0, :] for cnt in contours]
    hierarchy = hierarchy[0]  # shape (n_contours, 4)

    outer_contour = None
    inner_contour = None

    for i, h in enumerate(hierarchy):
        next_idx, prev_idx, child_idx, parent_idx = h

        if parent_idx == -1 and child_idx != -1:
            # This contour has a child: likely the outer boundary
            outer_contour = contours[i]
        elif parent_idx != -1:
            # This contour is inside another: likely the inner boundary
            inner_contour = contours[i]

    if outer_contour is not None and inner_contour is not None:
        return inner_contour, outer_contour
    else:
        return None, None

def align_contours(inner_contour, outer_contour, n_points=1000):
    """
    Interpolates contours and ensures matching start point directions.
    Returns (aligned_inner_points, aligned_outer_points)
    """
    inner_pts = resample_contour(inner_contour, n_points)
    outer_pts = resample_contour(outer_contour, n_points)

    # Measure distances between start points
    dist_normal = np.linalg.norm(inner_pts[0] - outer_pts[0])
    dist_flipped = np.linalg.norm(inner_pts[-1] - outer_pts[0])

    # If flipping epi makes the match closer, flip it
    if dist_flipped < dist_normal:
        inner_pts = inner_pts[::-1]

    return inner_pts, outer_pts

def resample_contour(contour, n_points=1000):
    contour = np.vstack([contour, contour[0]])  # close loop
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(contour, axis=0), axis=1)])
    total_length = distances[-1]
    interp_points = np.linspace(0, total_length, n_points)
    x = np.interp(interp_points, distances, contour[:, 0])
    y = np.interp(interp_points, distances, contour[:, 1])
    return np.vstack((x, y)).T

def compute_centerline(inner_pts, outer_pts):
    """
    For each point on the outer contour, find closest point on inner.
    Then compute the midpoint to define the centerline.
    """
    tree = cKDTree(inner_pts)
    distances, indices = tree.query(outer_pts)
    matched_inner = inner_pts[indices]
    centerline = (outer_pts + matched_inner) / 2
    return centerline

# ----------------------------------
# Step 4: Compute tangent, normal vectors and strain components
# ----------------------------------
def compute_tangent_normal_vectors(contour_points, smooth_window=5):
    """
    Compute tangent and normal vectors for each point on a contour.
    
    Parameters:
    - contour_points: Array of shape (n_points, 2) representing the contour
    - smooth_window: Window size for smoothing the tangent calculation
    
    Returns:
    - tangent_vectors: Array of shape (n_points, 2) - unit tangent vectors
    - normal_vectors: Array of shape (n_points, 2) - unit normal vectors (pointing inward)
    """
    n_points = len(contour_points)
    tangent_vectors = np.zeros_like(contour_points)
    
    # Compute tangent vectors using central differences with smoothing
    for i in range(n_points):
        # Use a window around current point for smoothing
        indices = []
        for j in range(-smooth_window//2, smooth_window//2 + 1):
            idx = (i + j) % n_points
            indices.append(idx)
        
        # Compute average tangent direction
        tangent_sum = np.zeros(2)
        for j in range(len(indices)-1):
            curr_idx = indices[j]
            next_idx = indices[j+1]
            tangent_sum += contour_points[next_idx] - contour_points[curr_idx]
        
        tangent_vectors[i] = tangent_sum
    
    # Normalize tangent vectors
    tangent_magnitudes = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    tangent_magnitudes[tangent_magnitudes == 0] = 1  # Avoid division by zero
    tangent_vectors = tangent_vectors / tangent_magnitudes
    
    # Compute normal vectors (perpendicular to tangent, pointing inward for outer contour)
    normal_vectors = np.zeros_like(tangent_vectors)
    normal_vectors[:, 0] = -tangent_vectors[:, 1]  # Rotate 90 degrees clockwise
    normal_vectors[:, 1] = tangent_vectors[:, 0]
    
    return tangent_vectors, normal_vectors

def compute_strain_components(strain_tensor, tangent_vector, normal_vector):
    """
    Compute strain components along tangent and normal directions.
    
    Parameters:
    - strain_tensor: 3x3 strain tensor
    - tangent_vector: 2D tangent vector (will be extended to 3D)
    - normal_vector: 2D normal vector (will be extended to 3D)
    
    Returns:
    - strain_tangent: strain component along tangent direction
    - strain_normal: strain component along normal direction
    - strain_through_plane: strain component through the plane (z-direction)
    - strain_shear_tn: shear strain in tangent-normal plane
    - strain_shear_tz: shear strain in tangent-z plane
    - strain_shear_nz: shear strain in normal-z plane
    """
    # Extend 2D vectors to 3D (assuming z-component is 0 for in-plane vectors)
    tangent_3d = np.array([tangent_vector[0], tangent_vector[1], 0.0])
    normal_3d = np.array([normal_vector[0], normal_vector[1], 0.0])
    through_plane_3d = np.array([0.0, 0.0, 1.0])  # z-direction
    
    # Normalize vectors (should already be normalized, but ensure)
    tangent_3d = tangent_3d / np.linalg.norm(tangent_3d)
    normal_3d = normal_3d / np.linalg.norm(normal_3d)
    
    # Compute strain components using the formula: strain_component = n^T * ε * n
    # where n is the unit vector in the direction of interest
    
    # Normal strain components
    strain_tangent = np.dot(tangent_3d, np.dot(strain_tensor, tangent_3d))
    strain_normal = np.dot(normal_3d, np.dot(strain_tensor, normal_3d))
    strain_through_plane = np.dot(through_plane_3d, np.dot(strain_tensor, through_plane_3d))
    
    # Shear strain components using the formula: shear_strain = n1^T * ε * n2
    strain_shear_tn = np.dot(tangent_3d, np.dot(strain_tensor, normal_3d))
    strain_shear_tz = np.dot(tangent_3d, np.dot(strain_tensor, through_plane_3d))
    strain_shear_nz = np.dot(normal_3d, np.dot(strain_tensor, through_plane_3d))
    
    return {
        'tangent': strain_tangent,
        'normal': strain_normal,
        'through_plane': strain_through_plane,
        'shear_tn': strain_shear_tn,
        'shear_tz': strain_shear_tz,
        'shear_nz': strain_shear_nz
    }

def find_closest_contour_points(point_cloud, strain_tensors, centerline_data):
    """
    For each point in the point cloud, find the closest point on contours/centerlines
    and compute corresponding tangent and normal vectors and strain components.
    
    Parameters:
    - point_cloud: Array of 3D points
    - strain_tensors: Array of 3x3 strain tensors for each point
    - centerline_data: Contour data for each slice
    
    Returns:
    - results: List of dictionaries containing closest points, vectors, and strain analysis
    """
    results = []
    
    for i, point in enumerate(point_cloud):
        x, y, z = point
        z_slice = int(np.round(z))
        
        # Find the slice data
        slice_data = None
        for data in centerline_data:
            if data['z'] == z_slice:
                slice_data = data
                break
        
        if slice_data is None:
            continue
        
        point_2d = np.array([x, y])
        strain_tensor = strain_tensors[i]
        
        # Find closest points on each contour/centerline
        contours = {
            'inner': slice_data['inner'],
            'outer': slice_data['outer'], 
            'centerline': slice_data['centerline'],
            'inner_to_center': slice_data['inner_to_center'],
            'center_to_outer': slice_data['center_to_outer']
        }
        
        closest_info = {}
        
        for contour_name, contour_points in contours.items():
            # Build KDTree for this contour
            tree = cKDTree(contour_points)
            distance, closest_idx = tree.query(point_2d)
            
            closest_point = contour_points[closest_idx]
            
            # Compute tangent and normal vectors for this contour
            tangent_vectors, normal_vectors = compute_tangent_normal_vectors(contour_points)
            
            # Get vectors at the closest point
            tangent_at_closest = tangent_vectors[closest_idx]
            normal_at_closest = normal_vectors[closest_idx]
            
            # Compute strain components
            strain_components = compute_strain_components(
                strain_tensor, tangent_at_closest, normal_at_closest
            )
            
            closest_info[contour_name] = {
                'closest_point': closest_point,
                'distance': distance,
                'index': closest_idx,
                'tangent_vector': tangent_at_closest,
                'normal_vector': normal_at_closest,
                'strain_components': strain_components
            }
        
        # Find the overall closest contour
        min_distance = float('inf')
        closest_contour = None
        for contour_name, info in closest_info.items():
            if info['distance'] < min_distance:
                min_distance = info['distance']
                closest_contour = contour_name
        
        result = {
            'cloud_point': point,
            'strain_tensor': strain_tensor,
            'slice': z_slice,
            'closest_contour': closest_contour,
            'all_contours': closest_info
        }
        
        results.append(result)
    
    return results

# ----------------------------------
# Step 5: Enhanced Visualization with Strain Analysis
# ----------------------------------
def plot_slice_with_strain_analysis(centerline_data, point_cloud_results, z_index, strain_component='tangent'):
    """
    Plot slice contours with point cloud colored by strain components
    
    Parameters:
    - strain_component: 'tangent', 'normal', 'through_plane', 'shear_tn', 'shear_tz', 'shear_nz'
    """
    # Get slice data
    slice_data = next((d for d in centerline_data if d['z'] == z_index), None)
    if slice_data is None:
        print(f"No data for z={z_index}")
        return
    
    # Filter point cloud results for this slice
    slice_points = [r for r in point_cloud_results if r['slice'] == z_index]
    
    if not slice_points:
        print(f"No point cloud data for z={z_index}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Point cloud colored by strain component
    ax1.plot(slice_data['outer'][:, 0], slice_data['outer'][:, 1], label='Outer', color='blue', linewidth=2)
    ax1.plot(slice_data['inner'][:, 0], slice_data['inner'][:, 1], label='Inner', color='red', linewidth=2)
    ax1.plot(slice_data['centerline'][:, 0], slice_data['centerline'][:, 1], '--', label='Centerline', color='green', linewidth=2)
    
    # Extract strain values for coloring
    cloud_x = [r['cloud_point'][0] for r in slice_points]
    cloud_y = [r['cloud_point'][1] for r in slice_points]
    strain_values = [r['all_contours'][r['closest_contour']]['strain_components'][strain_component] 
                    for r in slice_points]
    
    scatter = ax1.scatter(cloud_x, cloud_y, c=strain_values, s=30, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(scatter, ax=ax1, label=f'Strain {strain_component}')
    
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title(f"Point Cloud - {strain_component.capitalize()} Strain at z = {z_index}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Strain distribution histogram
    ax2.hist(strain_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel(f'{strain_component.capitalize()} Strain')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of {strain_component.capitalize()} Strain')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: All strain components comparison
    all_strain_data = {}
    strain_types = ['tangent', 'normal', 'through_plane', 'shear_tn', 'shear_tz', 'shear_nz']
    
    for strain_type in strain_types:
        values = [r['all_contours'][r['closest_contour']]['strain_components'][strain_type] 
                 for r in slice_points]
        all_strain_data[strain_type] = values
    
    # Box plot of all strain components
    strain_labels = []
    strain_data_list = []
    for strain_type, values in all_strain_data.items():
        strain_labels.append(strain_type.replace('_', '\n'))
        strain_data_list.append(values)
    
    box_plot = ax3.boxplot(strain_data_list, labels=strain_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Strain Value')
    ax3.set_title('Strain Components Distribution')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Strain tensor visualization (principal strains)
    principal_strains = []
    for result in slice_points:
        strain_tensor = result['strain_tensor']
        eigenvals, _ = np.linalg.eig(strain_tensor)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        principal_strains.append(eigenvals)
    
    principal_strains = np.array(principal_strains)
    
    ax4.scatter(cloud_x, cloud_y, c=principal_strains[:, 0], s=30, cmap='viridis', alpha=0.8)
    ax4.plot(slice_data['centerline'][:, 0], slice_data['centerline'][:, 1], '--', color='white', linewidth=2, alpha=0.8)
    
    scatter4 = ax4.scatter(cloud_x, cloud_y, c=principal_strains[:, 0], s=30, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter4, ax=ax4, label='Maximum Principal Strain')
    
    ax4.invert_yaxis()
    ax4.set_aspect('equal')
    ax4.set_title(f"Maximum Principal Strain at z = {z_index}")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_strain_summary_report(point_cloud_results):
    """
    Create a comprehensive summary report of strain analysis
    """
    print("\n" + "="*60)
    print("STRAIN ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    if not point_cloud_results:
        print("No point cloud results to analyze.")
        return
    
    # Collect all strain data
    strain_data = {
        'tangent': [],
        'normal': [],
        'through_plane': [],
        'shear_tn': [],
        'shear_tz': [],
        'shear_nz': []
    }
    
    contour_counts = {}
    principal_strains = []
    
    for result in point_cloud_results:
        closest_contour = result['closest_contour']
        contour_counts[closest_contour] = contour_counts.get(closest_contour, 0) + 1
        
        strain_components = result['all_contours'][closest_contour]['strain_components']
        for strain_type in strain_data.keys():
            strain_data[strain_type].append(strain_components[strain_type])
        
        # Calculate principal strains
        strain_tensor = result['strain_tensor']
        eigenvals, _ = np.linalg.eig(strain_tensor)
        eigenvals = np.sort(eigenvals)[::-1]
        principal_strains.append(eigenvals)
    
    principal_strains = np.array(principal_strains)
    
    # Print summary statistics
    print(f"Total points analyzed: {len(point_cloud_results)}")
    print(f"Slices with data: {len(set(r['slice'] for r in point_cloud_results))}")
    
    print("\nPoints closest to each contour type:")
    for contour, count in contour_counts.items():
        percentage = (count / len(point_cloud_results)) * 100
        print(f"  {contour:15s}: {count:4d} points ({percentage:5.1f}%)")
    
    print("\nStrain Component Statistics:")
    print("-" * 60)
    print(f"{'Component':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for strain_type, values in strain_data.items():
        values = np.array(values)
        print(f"{strain_type:<15} {np.mean(values):< 9.4f} {np.std(values):< 9.4f} "
              f"{np.min(values):< 9.4f} {np.max(values):< 9.4f}")
    
    print("\nPrincipal Strain Statistics:")
    print("-" * 60)
    print(f"{'Principal':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for i in range(3):
        values = principal_strains[:, i]
        print(f"λ{i+1} (max->min)  {np.mean(values):< 9.4f} {np.std(values):< 9.4f} "
              f"{np.min(values):< 9.4f} {np.max(values):< 9.4f}")
    
    # Calculate strain invariants
    I1 = np.trace(np.mean([r['strain_tensor'] for r in point_cloud_results], axis=0))  # First invariant
    print(f"\nMean First Strain Invariant (I1 = εxx + εyy + εzz): {I1:.6f}")
    
    print("\n" + "="*60)
def plot_slice_with_point_cloud(centerline_data, point_cloud_results, z_index, show_vectors=True):
    """
    Plot slice contours with point cloud and tangent/normal vectors
    """
    # Get slice data
    slice_data = next((d for d in centerline_data if d['z'] == z_index), None)
    if slice_data is None:
        print(f"No data for z={z_index}")
        return
    
    # Filter point cloud results for this slice
    slice_points = [r for r in point_cloud_results if r['slice'] == z_index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Contours and point cloud
    ax1.plot(slice_data['outer'][:, 0], slice_data['outer'][:, 1], label='Outer', color='blue', linewidth=2)
    ax1.plot(slice_data['inner'][:, 0], slice_data['inner'][:, 1], label='Inner', color='red', linewidth=2)
    ax1.plot(slice_data['centerline'][:, 0], slice_data['centerline'][:, 1], '--', label='Centerline', color='green', linewidth=2)
    ax1.plot(slice_data['inner_to_center'][:, 0], slice_data['inner_to_center'][:, 1], '--', label='Inner–Center', color='orange')
    ax1.plot(slice_data['center_to_outer'][:, 0], slice_data['center_to_outer'][:, 1], '--', label='Center–Outer', color='purple')
    
    # Plot point cloud
    if slice_points:
        cloud_x = [r['cloud_point'][0] for r in slice_points]
        cloud_y = [r['cloud_point'][1] for r in slice_points]
        ax1.scatter(cloud_x, cloud_y, c='black', s=10, alpha=0.6, label='Point Cloud')
        
        # Draw lines to closest points
        for result in slice_points[:20]:  # Show only first 20 to avoid clutter
            cloud_pt = result['cloud_point'][:2]
            closest_contour = result['closest_contour']
            closest_pt = result['all_contours'][closest_contour]['closest_point']
            ax1.plot([cloud_pt[0], closest_pt[0]], [cloud_pt[1], closest_pt[1]], 
                    'k--', alpha=0.3, linewidth=0.5)
    
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title(f"Contours and Point Cloud at z = {z_index}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tangent and normal vectors
    if show_vectors and slice_points:
        # Plot contours
        ax2.plot(slice_data['centerline'][:, 0], slice_data['centerline'][:, 1], 
                '--', label='Centerline', color='green', linewidth=2)
        
        # Compute and plot tangent/normal vectors for centerline
        tangent_vecs, normal_vecs = compute_tangent_normal_vectors(slice_data['centerline'])
        
        # Sample every 5th point to avoid clutter
        sample_indices = range(0, len(slice_data['centerline']), 5)
        for i in sample_indices:
            pt = slice_data['centerline'][i]
            tangent = tangent_vecs[i] * 5  # Scale for visibility
            normal = normal_vecs[i] * 5
            
            # Plot tangent vector (blue)
            ax2.arrow(pt[0], pt[1], tangent[0], tangent[1], 
                     head_width=1, head_length=1, fc='blue', ec='blue', alpha=0.7)
            # Plot normal vector (red)
            ax2.arrow(pt[0], pt[1], normal[0], normal[1], 
                     head_width=1, head_length=1, fc='red', ec='red', alpha=0.7)
        
        # Add legend for vectors
        ax2.arrow([], [], [], [], fc='blue', ec='blue', label='Tangent')
        ax2.arrow([], [], [], [], fc='red', ec='red', label='Normal')
    
    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.set_title(f"Tangent and Normal Vectors at z = {z_index}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_slice_contours(centerline_data, z_index):
    data = next((d for d in centerline_data if d['z'] == z_index), None)
    if data is None:
        print(f"No data for z={z_index}")
        return

    plt.figure(figsize=(6, 6))
    plt.plot(data['outer'][:, 0], data['outer'][:, 1], label='Outer', color='blue')
    plt.plot(data['inner'][:, 0], data['inner'][:, 1], label='Inner', color='red')
    plt.plot(data['centerline'][:, 0], data['centerline'][:, 1], '--', label='Centerline', color='green')
    plt.plot(data['inner_to_center'][:, 0], data['inner_to_center'][:, 1], '--', label='Inner–Center', color='orange')
    plt.plot(data['center_to_outer'][:, 0], data['center_to_outer'][:, 1], '--', label='Center–Outer', color='purple')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title(f"Contours at z = {z_index}")
    plt.legend()
    plt.show()

# ----------------------------------
# Run the pipeline
# ----------------------------------
if __name__ == "__main__":
    start_time = time.time()  # Start timer
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Load real mask from 3D .tif file
    donut_mask = read_AF_mask(tif_path)
   
    centerline_data = []

    # Step 2: Loop through slices and extract contours
    for z in range(donut_mask.shape[2]):
        slice_img = donut_mask[:, :, z]
        inner_contour, outer_contour = extract_inner_outer_contours(slice_img)

        if inner_contour is None or outer_contour is None:
            continue  # Skip slices without valid contours

        # Align contours
        inner_pts, outer_pts = align_contours(inner_contour, outer_contour, n_points=100)

        # Compute centerline
        centerline = compute_centerline(inner_pts, outer_pts)
        
        inner_to_center = compute_centerline(inner_pts, centerline)
        center_to_outer = compute_centerline(outer_pts, centerline)

        # Store data
        centerline_data.append({
            'z': z,
            'inner': inner_pts,
            'outer': outer_pts,
            'centerline': centerline,
            'inner_to_center': inner_to_center,
            'center_to_outer': center_to_outer
        })

    print(f"Processed {len(centerline_data)} slices with valid contours")

    # Step 3: Read point cloud and create strain tensors
    point_cloud, strain_tensors = read_csv(csv_path)
    print(f"Generated point cloud with {len(point_cloud)} points and strain tensors")

    # Step 4: Find closest contour points and compute strain components
    print("Computing closest contour points and strain components...")
    point_cloud_results = find_closest_contour_points(point_cloud, strain_tensors, centerline_data)
    print(f"Processed {len(point_cloud_results)} point cloud results")

    # Step 5: Enhanced visualization with strain analysis
    print("Creating strain visualizations...")
    for z_idx in [ 1200, 1400]:  # Modify as needed
        if any(d['z'] == z_idx for d in centerline_data):
            # Show different strain components
            for strain_comp in ['tangent', 'normal', 'shear_tn']:
                plot_slice_with_strain_analysis(centerline_data, point_cloud_results, z_idx, strain_comp)

    # Step 6: Comprehensive strain analysis report
    create_strain_summary_report(point_cloud_results)

    # Summary statistics
    print("\n=== Summary Statistics ===")
    contour_counts = {}
    distances_by_contour = {}
    
    for result in point_cloud_results:
        closest = result['closest_contour']
        contour_counts[closest] = contour_counts.get(closest, 0) + 1
        
        if closest not in distances_by_contour:
            distances_by_contour[closest] = []
        distances_by_contour[closest].append(result['all_contours'][closest]['distance'])
    
    print("Points closest to each contour type:")
    for contour, count in contour_counts.items():
        avg_dist = np.mean(distances_by_contour[contour])
        print(f"  {contour}: {count} points (avg distance: {avg_dist:.2f})")
        
# Step 7: Save x, y, z and strain components to CSV
output_data = []

for result in point_cloud_results:
    point = result['cloud_point']
    x, y, z = point

    contour_name = result['closest_contour']
    strain_comp = result['all_contours'][contour_name]['strain_components']

    output_data.append({
        'x': x,
        'y': y,
        'z': z,
        'contour': contour_name,
        'strain_tangent': strain_comp['tangent'],
        'strain_normal': strain_comp['normal'],
        'strain_through_plane': strain_comp['through_plane'],
        'shear_tangent_normal': strain_comp['shear_tn'],
        'shear_tangent_z': strain_comp['shear_tz'],
        'shear_normal_z': strain_comp['shear_nz']
    })

# Create DataFrame and save
df_out = pd.DataFrame(output_data)
df_out.to_csv(results_filename, index=False)

print(f"\nSaved strain data with direction components to: {results_filename}")

# === End timer and print elapsed time ===
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal runtime: {elapsed:.2f} seconds")
        