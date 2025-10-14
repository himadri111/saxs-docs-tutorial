# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:58:16 2025

@author: Admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TomoSAXS metrics calculation - Python version

Finds fibre tangent values, curvature, displacements and strain along fibres

INPUTS: 
    - Reference point_spacecurve .npz file from fibre orientation extraction
    - .disp file from iDVC
          
OUTPUTS: 
    - metrics.csv containing x, y, z, k, m, L, Lsmooth, Lobjmin
    - updated_pointcloud.csv
    - new_orientation.csv
    - pointcloud_mask.csv
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial import KDTree
from scipy.interpolate import splrep, splev
from concurrent.futures import ProcessPoolExecutor
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsConfig:
    """Configuration for metrics calculation"""
    def __init__(self):
        # Input files
        self.pointcloud_spacecurve = "167208_fibres_new_inc4_spacecurve.npz"
        self.disp_file = "167208_new_whole.disp"
        
        # Output files
        self.output_metrics = "167208_Python_metrics.csv"
        self.output_pointcloud = "167208_Python_updated_pointcloud.csv"
        self.output_orientation = "167208_Python_new_orientation.csv"
        self.output_mask = "167208_Python_new_pointcloud_mask.csv"
        
        # Parameters
        self.inc = 5.0  # spacing increment from original processing
        self.inc_mask = 1.0  # spacing for mask generation
        self.coord_tolerance = 1e-2  # tolerance for coordinate matching
        self.polyfit_degree = 3  # polynomial degree for space curve fitting
        self.min_points_for_fit = 4


def load_disp_file(filepath: str) -> np.ndarray:
    """Load displacement file from iDVC"""
    logger.info(f"Loading displacement file: {filepath}")
    disp_data = np.loadtxt(filepath, skiprows=1)
    logger.info(f"Loaded {len(disp_data)} displacement points")
    return disp_data


def reorder_disp_to_pointcloud(point_cloud: np.ndarray, disp_data: np.ndarray, 
                               tolerance: float = 1e-2) -> np.ndarray:
    """
    Reorder displacement file to match point cloud coordinate order
    
    Args:
        point_cloud: Reference point cloud array
        disp_data: Displacement data from iDVC
        tolerance: Tolerance for coordinate matching
        
    Returns:
        Reordered displacement data matching point cloud order
    """
    logger.info("Reordering displacement data to match point cloud")
    
    # Extract coordinates
    coords_pc = point_cloud[:, 4:7]  # x, y, z from point cloud
    coords_disp = disp_data[:, 1:4]  # x, y, z from disp file (columns 2-4 in 1-indexed)
    
    # Use KDTree for efficient nearest neighbor search
    tree = KDTree(coords_disp)
    distances, indices = tree.query(coords_pc)
    
    # Check for matches within tolerance
    within_tolerance = distances <= tolerance
    num_outside = np.sum(~within_tolerance)
    
    if num_outside > 0:
        logger.warning(f"{num_outside} points outside tolerance (max distance: {distances.max():.6f})")
    
    # Reorder displacement data
    reordered = disp_data[indices]
    
    logger.info("Reordering complete")
    return reordered


def update_pointcloud_with_displacements(point_cloud: np.ndarray, 
                                        disp_data: np.ndarray) -> np.ndarray:
    """
    Update point cloud coordinates with displacement vectors
    
    Returns:
        Updated point cloud with displaced coordinates
    """
    logger.info("Updating point cloud with displacements")
    
    point_cloud_new = point_cloud.copy()
    
    # Extract displacements (u, v, w are columns 7-9 in 1-indexed, or 6-8 in 0-indexed)
    u = disp_data[:, 6]
    v = disp_data[:, 7]
    w = disp_data[:, 8]
    
    # Update coordinates (columns 4-6 are x, y, z)
    point_cloud_new[:, 4] += u
    point_cloud_new[:, 5] += v
    point_cloud_new[:, 6] += w
    
    return point_cloud_new


def interpolate_fibre_mask(fibre_coords: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate points along fibre for mask generation
    
    Args:
        fibre_coords: Nx3 array of fibre coordinates
        spacing: Target spacing between points
        
    Returns:
        positions: Interpolated positions
        directions: Direction vectors
    """
    if len(fibre_coords) < 2:
        return fibre_coords, np.zeros((len(fibre_coords), 3))
    
    positions = [fibre_coords[0]]
    directions = []
    
    current_pos = fibre_coords[0]
    next_idx = 1
    
    while next_idx < len(fibre_coords):
        next_point = fibre_coords[next_idx]
        vec = next_point - current_pos
        dist = norm(vec)
        
        if dist < 1e-10:
            next_idx += 1
            continue
        
        if dist >= spacing:
            direction = vec / dist
            new_pos = current_pos + spacing * direction
            positions.append(new_pos)
            directions.append(direction)
            current_pos = new_pos
        else:
            next_idx += 1
            if next_idx < len(fibre_coords):
                current_pos = next_point
    
    # Add initial zero direction
    directions = [np.zeros(3)] + directions
    
    return np.array(positions), np.array(directions)


def generate_mask_pointcloud(point_cloud_new: np.ndarray, inc_mask: float) -> np.ndarray:
    """
    Generate mask point cloud with finer spacing
    
    Args:
        point_cloud_new: Updated point cloud
        inc_mask: Spacing for mask generation
        
    Returns:
        Mask point cloud array
    """
    logger.info(f"Generating mask point cloud with spacing {inc_mask}")
    
    fibre_ids = np.unique(point_cloud_new[:, 1]).astype(int)
    mask_cloud_list = []
    global_count = 0
    
    for fibre_id in fibre_ids:
        fibre_data = point_cloud_new[point_cloud_new[:, 1] == fibre_id]
        fibre_coords = fibre_data[:, 4:7]
        
        positions, directions = interpolate_fibre_mask(fibre_coords, inc_mask)
        num_points = len(positions)
        
        if num_points == 0:
            continue
        
        # Build mask point cloud entry
        mask_entry = np.column_stack([
            np.ones(num_points),  # lamella
            np.full(num_points, fibre_id),  # fibre ID
            np.arange(1, num_points + 1),  # local ID
            np.arange(global_count + 1, global_count + num_points + 1),  # global ID
            positions,
            directions
        ])
        
        mask_cloud_list.append(mask_entry)
        global_count += num_points
    
    mask_cloud = np.vstack(mask_cloud_list)
    logger.info(f"Generated mask with {len(mask_cloud)} points")
    
    return mask_cloud


def fit_space_curve_updated(fibre_data: np.ndarray, degree: int = 3) -> Optional[Dict]:
    """
    Fit polynomial space curve to updated fibre data
    
    Returns:
        Dictionary with polynomial coefficients and arc length
    """
    npts = len(fibre_data)
    
    if npts < 4:
        return None
    
    # Calculate arc length parameterization
    positions = fibre_data[:, 4:7]
    segments = positions[1:] - positions[:-1]
    distances = norm(segments, axis=1)
    arc_length = np.insert(np.cumsum(distances), 0, 0.0)
    
    try:
        fx = np.polyfit(arc_length, positions[:, 0], degree)
        fy = np.polyfit(arc_length, positions[:, 1], degree)
        fz = np.polyfit(arc_length, positions[:, 2], degree)
        
        return {
            'fx': fx,
            'fy': fy,
            'fz': fz,
            'arc_length': arc_length
        }
    except np.linalg.LinAlgError:
        return None


def fit_all_updated_fibres(point_cloud_new: np.ndarray, config: MetricsConfig) -> Dict:
    """Fit space curves to all updated fibres"""
    logger.info("Fitting space curves to updated fibres")
    
    fibre_ids = np.unique(point_cloud_new[:, 1]).astype(int)
    curve_fits = {}
    
    for fibre_id in fibre_ids:
        fibre_data = point_cloud_new[point_cloud_new[:, 1] == fibre_id]
        result = fit_space_curve_updated(fibre_data, config.polyfit_degree)
        if result is not None:
            curve_fits[fibre_id] = result
    
    logger.info(f"Fitted {len(curve_fits)}/{len(fibre_ids)} fibres")
    return curve_fits


def compute_tangent_from_fit(fit: Dict) -> np.ndarray:
    """Compute tangent vectors from polynomial fit"""
    dfx = np.polyder(fit['fx'])
    dfy = np.polyder(fit['fy'])
    dfz = np.polyder(fit['fz'])
    
    s = fit['arc_length']
    
    tx = np.polyval(dfx, s)
    ty = np.polyval(dfy, s)
    tz = np.polyval(dfz, s)
    
    return np.column_stack([tx, ty, tz])


def compute_curvature(fit: Dict) -> np.ndarray:
    """
    Compute curvature from polynomial fit
    
    Curvature formula: k = ||r' × r''|| / ||r'||^3
    """
    s = fit['arc_length']
    
    # First derivatives
    fxd1 = np.polyder(fit['fx'])
    fyd1 = np.polyder(fit['fy'])
    fzd1 = np.polyder(fit['fz'])
    
    fxd1_vals = np.polyval(fxd1, s)
    fyd1_vals = np.polyval(fyd1, s)
    fzd1_vals = np.polyval(fzd1, s)
    
    # Second derivatives
    fxd2 = np.polyder(fxd1)
    fyd2 = np.polyder(fyd1)
    fzd2 = np.polyder(fzd1)
    
    fxd2_vals = np.polyval(fxd2, s)
    fyd2_vals = np.polyval(fyd2, s)
    fzd2_vals = np.polyval(fzd2, s)
    
    # Cross product components
    a = fzd2_vals * fyd1_vals - fyd2_vals * fzd1_vals
    b = fxd2_vals * fzd1_vals - fzd2_vals * fxd1_vals
    c = fyd2_vals * fxd1_vals - fxd2_vals * fyd1_vals
    
    # Magnitude of first derivative
    d = fxd1_vals**2 + fyd1_vals**2 + fzd1_vals**2
    
    # Curvature
    curvature = np.sqrt(a**2 + b**2 + c**2) * d**(-1.5)
    
    return curvature


def compute_orientation_angles(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orientation angles theta and phi from tangent vectors"""
    # Reference direction 1 (in-plane)
    ref_dir1 = tangents.copy()
    ref_dir1[:, 2] = 0
    
    # Determine sign convention
    sign_agreement = np.mean(np.sign(tangents[:, 0]) == np.sign(tangents[:, 1]))
    
    if sign_agreement > 0.5:
        ref_dir1[:, 0] = np.abs(ref_dir1[:, 0])
        ref_dir1[:, 1] = np.abs(ref_dir1[:, 1])
    else:
        ref_dir1[:, 0] = np.abs(ref_dir1[:, 0])
        ref_dir1[:, 1] = -np.abs(ref_dir1[:, 1])
    
    # Compute theta
    cross1 = np.cross(tangents, ref_dir1)
    cross1_norm = norm(cross1, axis=1)
    dot1 = np.sum(tangents * ref_dir1, axis=1)
    theta = np.degrees(np.arctan2(cross1_norm, dot1))
    
    # Reference direction 2 (y-axis)
    ref_dir2 = np.zeros_like(tangents)
    ref_dir2[:, 1] = 1
    
    # Compute phi
    cross2 = np.cross(tangents, ref_dir2)
    cross2_norm = norm(cross2, axis=1)
    dot2 = np.sum(tangents * ref_dir2, axis=1)
    phi = np.degrees(np.arctan2(cross2_norm, dot2))
    
    return theta, phi


def compute_strain_simple(tangent: np.ndarray, displacement: np.ndarray, 
                         inc: float) -> np.ndarray:
    """
    Compute simple strain L using gradient
    
    L = dm/ds + (dm/ds)^2 / 2
    where m is displacement in tangent direction
    """
    # Displacement in tangent direction
    m = np.sum(tangent * displacement, axis=1)
    
    # Gradient
    dm_ds = np.gradient(m, inc)
    
    # Strain
    L = dm_ds + (dm_ds**2) / 2
    
    return L


def compute_strain_smooth(tangent: np.ndarray, displacement: np.ndarray, 
                         inc: float, npts: int) -> np.ndarray:
    """
    Compute smoothed strain using polynomial fit
    
    Degree adapts based on fibre length
    """
    # Displacement in tangent direction
    m = np.sum(tangent * displacement, axis=1)
    
    # Determine polynomial degree
    degree = 2 + round(npts * inc / 320)
    degree = min(degree, 9)  # Cap at 9
    degree = min(degree, len(m) - 1)  # Can't exceed npts - 1
    
    # Spacing array
    spacing = np.arange(len(m)) * inc
    
    # Fit polynomial
    try:
        p = np.polyfit(spacing, m, degree)
        dp = np.polyder(p)
        
        # Evaluate derivative
        dm_ds = np.polyval(dp, spacing)
        
        # Strain
        L_smooth = dm_ds + (dm_ds**2) / 2
    except:
        L_smooth = np.full(len(m), np.nan)
    
    return L_smooth


def compute_strain_weighted(tangent: np.ndarray, displacement: np.ndarray, 
                           objmin: np.ndarray, inc: float, npts: int) -> np.ndarray:
    """
    Compute strain with objmin weighting
    
    Uses weighted polynomial fit based on objmin values
    """
    # Displacement in tangent direction
    m = np.sum(tangent * displacement, axis=1)
    
    # Determine polynomial degree
    degree = 2 + round(npts * inc / 320)
    degree = min(degree, 9)
    degree = min(degree, len(m) - 1)
    
    # Spacing array
    spacing = np.arange(len(m)) * inc
    
    # Weights based on objmin
    weights = 1.0 / ((objmin + 0.0001)**3)
    
    # Weighted polynomial fit
    try:
        p = np.polyfit(spacing, m, degree, w=weights)
        dp = np.polyder(p)
        
        # Evaluate derivative
        dm_ds = np.polyval(dp, spacing)
        
        # Strain
        L_objmin = dm_ds + (dm_ds**2) / 2
    except:
        L_objmin = np.full(len(m), np.nan)
    
    return L_objmin


def compute_all_metrics(point_cloud: np.ndarray, point_cloud_new: np.ndarray,
                       original_fits: Dict, updated_fits: Dict, 
                       disp_data: np.ndarray, config: MetricsConfig) -> Dict:
    """
    Compute all metrics for each fibre
    
    Returns dictionary with arrays for all metrics
    """
    logger.info("Computing metrics for all fibres")
    
    fibre_ids = np.unique(point_cloud[:, 1]).astype(int)
    
    # Initialize output arrays
    n_total = len(point_cloud)
    
    metrics = {
        'x': point_cloud[:, 4],
        'y': point_cloud[:, 5],
        'z': point_cloud[:, 6],
        'theta': np.full(n_total, np.nan),
        'phi': np.full(n_total, np.nan),
        'delta_theta': np.full(n_total, np.nan),
        'delta_phi': np.full(n_total, np.nan),
        'k': np.full(n_total, np.nan),
        'delta_k': np.full(n_total, np.nan),
        'm': np.full(n_total, np.nan),
        'L': np.full(n_total, np.nan),
        'Lsmooth': np.full(n_total, np.nan),
        'Lobjmin': np.full(n_total, np.nan)
    }
    
    for fibre_id in fibre_ids:
        # Get indices for this fibre
        idx_orig = point_cloud[:, 1] == fibre_id
        idx_new = point_cloud_new[:, 1] == fibre_id
        
        npts = np.sum(idx_orig)
        
        if npts < config.min_points_for_fit:
            continue
        
        if fibre_id not in original_fits or fibre_id not in updated_fits:
            continue
        
        # Original fibre metrics
        fit_orig = original_fits[fibre_id]
        tangent_orig = compute_tangent_from_fit(fit_orig)
        theta_orig, phi_orig = compute_orientation_angles(tangent_orig)
        k_orig = compute_curvature(fit_orig)
        
        # Updated fibre metrics
        fit_new = updated_fits[fibre_id]
        tangent_new = compute_tangent_from_fit(fit_new)
        theta_new, phi_new = compute_orientation_angles(tangent_new)
        k_new = compute_curvature(fit_new)
        
        # Store original orientation and curvature
        metrics['theta'][idx_orig] = theta_orig
        metrics['phi'][idx_orig] = phi_orig
        metrics['k'][idx_orig] = k_orig
        
        # Store changes
        metrics['delta_theta'][idx_orig] = theta_new - theta_orig
        metrics['delta_phi'][idx_orig] = phi_new - phi_orig
        metrics['delta_k'][idx_orig] = k_new - k_orig
        
        # Compute strain metrics if displacement data available
        if len(disp_data) > 0:
            displacement = disp_data[idx_orig, 6:9]  # u, v, w
            objmin = disp_data[idx_orig, 5]  # objmin column
            
            # Displacement in tangent direction
            m = np.sum(tangent_orig * displacement, axis=1)
            metrics['m'][idx_orig] = m
            
            # Simple strain
            L = compute_strain_simple(tangent_orig, displacement, config.inc)
            metrics['L'][idx_orig] = L
            
            # Smoothed strain
            L_smooth = compute_strain_smooth(tangent_orig, displacement, config.inc, npts)
            metrics['Lsmooth'][idx_orig] = L_smooth
            
            # Weighted strain
            L_objmin = compute_strain_weighted(tangent_orig, displacement, objmin, config.inc, npts)
            metrics['Lobjmin'][idx_orig] = L_objmin
    
    logger.info("Metrics computation complete")
    return metrics


def save_all_outputs(point_cloud_new: np.ndarray, mask_cloud: np.ndarray,
                    tangents_new: np.ndarray, theta_new: np.ndarray, 
                    phi_new: np.ndarray, metrics: Dict, config: MetricsConfig):
    """Save all output files"""
    logger.info("Saving output files")
    
    # Save updated point cloud
    pc_df = pd.DataFrame(point_cloud_new, columns=[
        'lamella', 'fibre_id', 'local_id', 'global_id', 
        'x', 'y', 'z', 'dx', 'dy', 'dz'
    ])
    pc_df.to_csv(config.output_pointcloud, index=False)
    
    # Save mask point cloud (just coordinates)
    np.savetxt(config.output_mask, mask_cloud[:, 4:7], delimiter=',', 
               header='x,y,z', comments='')
    
    # Save new orientation
    orient_df = pd.DataFrame({
        'x': point_cloud_new[:, 4],
        'y': point_cloud_new[:, 5],
        'z': point_cloud_new[:, 6],
        'fibre_id': point_cloud_new[:, 1].astype(int),
        'theta': theta_new,
        'phi': phi_new
    })
    orient_df.to_csv(config.output_orientation, index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(config.output_metrics, index=False)
    
    logger.info(f"Outputs saved:")
    logger.info(f"  - {config.output_pointcloud}")
    logger.info(f"  - {config.output_mask}")
    logger.info(f"  - {config.output_orientation}")
    logger.info(f"  - {config.output_metrics}")


def main():
    """Main execution function"""
    start_time = time.time()
    
    config = MetricsConfig()
    
    try:
        # Load point cloud from NPZ file
        logger.info(f"Loading point cloud from {config.pointcloud_spacecurve}")
        data = np.load(config.pointcloud_spacecurve)
        point_cloud = data['point_cloud']
        logger.info(f"Loaded point cloud with {len(point_cloud)} points")
        
        # Load displacement file
        disp_data = load_disp_file(config.disp_file)
        
        # Reorder displacement data to match point cloud
        disp_reordered = reorder_disp_to_pointcloud(point_cloud, disp_data, 
                                                     config.coord_tolerance)
        
        # Update point cloud with displacements
        point_cloud_new = update_pointcloud_with_displacements(point_cloud, disp_reordered)
        
        # Generate mask point cloud
        mask_cloud = generate_mask_pointcloud(point_cloud_new, config.inc_mask)
        
        # Fit space curves to original and updated fibres
        logger.info("Fitting space curves to original fibres")
        original_fits = {}
        fibre_ids = np.unique(point_cloud[:, 1]).astype(int)
        for fibre_id in fibre_ids:
            fibre_data = point_cloud[point_cloud[:, 1] == fibre_id]
            result = fit_space_curve_updated(fibre_data, config.polyfit_degree)
            if result is not None:
                original_fits[fibre_id] = result
        
        updated_fits = fit_all_updated_fibres(point_cloud_new, config)
        
        # Compute tangents and orientation for updated fibres
        logger.info("Computing tangents and orientation for updated fibres")
        tangents_new = []
        theta_new_list = []
        phi_new_list = []
        
        for fibre_id in fibre_ids:
            fibre_data = point_cloud_new[point_cloud_new[:, 1] == fibre_id]
            npts = len(fibre_data)
            
            if npts < config.min_points_for_fit or fibre_id not in updated_fits:
                tangents_new.append(np.full((npts, 3), np.nan))
                theta_new_list.append(np.full(npts, np.nan))
                phi_new_list.append(np.full(npts, np.nan))
            else:
                fit = updated_fits[fibre_id]
                t = compute_tangent_from_fit(fit)
                theta, phi = compute_orientation_angles(t)
                
                tangents_new.append(t)
                theta_new_list.append(theta)
                phi_new_list.append(phi)
        
        tangents_new = np.vstack(tangents_new)
        theta_new = np.concatenate(theta_new_list)
        phi_new = np.concatenate(phi_new_list)
        
        # Compute all metrics
        metrics = compute_all_metrics(point_cloud, point_cloud_new, 
                                      original_fits, updated_fits,
                                      disp_reordered, config)
        
        # Save outputs
        save_all_outputs(point_cloud_new, mask_cloud, tangents_new, 
                        theta_new, phi_new, metrics, config)
        
        elapsed = time.time() - start_time
        logger.info(f"Processing complete in {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()