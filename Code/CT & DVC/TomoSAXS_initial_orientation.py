#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fibre orientation extraction from Excel-exported Avizo fibre tracing

Improvements:
- Added configuration class for easy parameter management
- Better error handling and validation
- Optimized numpy operations
- Improved code organization with functions
- Added progress tracking
- Better memory efficiency
- Enhanced documentation
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration parameters for fibre orientation extraction"""
    def __init__(self):
        self.input_filename = "167208_fibres_new.xlsx"
        self.pointcloud_output = "167208_fibres_new_inc4_pointcloud.txt"
        self.orientation_output = "167208_fibres_new_orientation.csv"
        self.spacecurve_output = "167208_fibres_new_inc4_spacecurve.npz"
        
        # Column ranges
        self.coord_range = "C:E"
        self.point_id_col = "Point IDs"
        self.lamella_col = "Lamella"
        
        # Processing parameters
        self.spacing_increment = 4.0
        self.min_points_for_fit = 4
        self.polyfit_degree = 3
        self.max_workers = None  # None uses default (cpu_count)


def validate_input_file(filepath: str) -> Path:
    """Validate that input file exists"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    return path


def load_data(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load coordinates, point IDs, and lamella data from Excel file
    
    Returns:
        coord: Nx3 array of coordinates
        point_ids: Array of point ID strings
        lamella: Array of lamella indices
    """
    logger.info(f"Loading data from {config.input_filename}")
    
    # Load coordinates
    coord = pd.read_excel(
        config.input_filename,
        sheet_name="Points",
        usecols=config.coord_range,
        header=0
    ).to_numpy(dtype=float)
    
    # Load segments
    segments_df = pd.read_excel(
        config.input_filename, 
        sheet_name="Segments", 
        header=0
    )
    
    # Handle lamella column
    if config.lamella_col in segments_df.columns and segments_df[config.lamella_col].notna().any():
        lamella = segments_df[config.lamella_col].fillna(1).to_numpy(dtype=int)
    else:
        lamella = np.ones(len(segments_df), dtype=int)
    
    # Load point IDs
    point_ids = segments_df[config.point_id_col].to_numpy()
    
    logger.info(f"Loaded {len(coord)} points and {len(point_ids)} fibres")
    return coord, point_ids, lamella


def parse_point_ids(point_id_str: str) -> np.ndarray:
    """Parse comma-separated point ID string to integer array"""
    return np.array([int(x.strip()) for x in str(point_id_str).split(',')])


def interpolate_fibre(fibre_coords: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate points along a fibre at regular spacing intervals
    
    Args:
        fibre_coords: Nx3 array of fibre coordinates
        spacing: Target spacing between interpolated points
        
    Returns:
        positions: Interpolated positions
        directions: Unit direction vectors at each position (padded to match positions length)
    """
    if len(fibre_coords) < 2:
        return fibre_coords, np.zeros((len(fibre_coords), 3))
    
    positions = [fibre_coords[0]]
    directions = [np.array([0.0, 0.0, 0.0])]  # First point has zero direction
    
    current_pos = fibre_coords[0]
    target_idx = 1
    
    while target_idx < len(fibre_coords):
        next_point = fibre_coords[target_idx]
        vec = next_point - current_pos
        dist = norm(vec)
        
        if dist < 1e-10:  # Skip duplicate points
            target_idx += 1
            continue
        
        if dist >= spacing:
            # Interpolate new point
            direction = vec / dist
            new_pos = current_pos + spacing * direction
            positions.append(new_pos)
            directions.append(direction)
            current_pos = new_pos
        else:
            # Move to next segment
            target_idx += 1
            if target_idx < len(fibre_coords):
                current_pos = next_point
    
    positions = np.array(positions)
    directions = np.array(directions)
    
    # Ensure directions array matches positions length
    if len(directions) < len(positions):
        # Pad with last direction or zeros
        padding = np.zeros((len(positions) - len(directions), 3))
        if len(directions) > 0:
            padding[:] = directions[-1]
        directions = np.vstack([directions, padding])
    
    return positions, directions


def build_point_cloud(coord: np.ndarray, point_ids: np.ndarray, 
                     lamella: np.ndarray, config: Config) -> np.ndarray:
    """
    Build point cloud with interpolated points along fibres
    
    Returns:
        point_cloud: Array with columns [lamella, fibre_id, local_id, global_id, x, y, z, dx, dy, dz]
    """
    logger.info("Building point cloud with interpolated points")
    
    point_cloud_list = []
    global_point_count = 0
    
    for fibre_idx, point_id_str in enumerate(point_ids):
        # Parse point IDs and extract coordinates
        point_indices = parse_point_ids(point_id_str)
        fibre_coords = coord[point_indices, :]
        
        # Interpolate points
        positions, directions = interpolate_fibre(fibre_coords, config.spacing_increment)
        num_points = len(positions)
        
        if num_points == 0:
            continue
        
        # Build fibre point cloud entry
        fibre_pc = np.column_stack([
            np.full(num_points, lamella[fibre_idx]),
            np.full(num_points, fibre_idx + 1),
            np.arange(1, num_points + 1),
            np.arange(global_point_count + 1, global_point_count + num_points + 1),
            positions,
            directions
        ])
        
        point_cloud_list.append(fibre_pc)
        global_point_count += num_points
    
    point_cloud = np.vstack(point_cloud_list)
    logger.info(f"Generated {len(point_cloud)} interpolated points")
    
    return point_cloud


def fit_space_curve(fibre_data: np.ndarray, degree: int = 3) -> Optional[Dict]:
    """
    Fit polynomial space curve to fibre data
    
    Returns:
        Dictionary with polynomial coefficients and arc length, or None if insufficient points
    """
    npts = len(fibre_data)
    
    if npts < 4:
        return None
    
    # Calculate arc length parameterization
    positions = fibre_data[:, 4:7]
    segments = positions[1:] - positions[:-1]
    distances = norm(segments, axis=1)
    arc_length = np.insert(np.cumsum(distances), 0, 0.0)
    
    # Fit polynomials for each coordinate
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
        logger.warning(f"Polynomial fitting failed for fibre")
        return None


def fit_fibre_wrapper(args: Tuple[int, np.ndarray, int]) -> Tuple[int, Optional[Dict]]:
    """Wrapper for parallel processing of space curve fitting"""
    fibre_id, point_cloud, degree = args
    fibre_data = point_cloud[point_cloud[:, 1] == fibre_id]
    result = fit_space_curve(fibre_data, degree)
    return fibre_id, result


def fit_all_fibres(point_cloud: np.ndarray, config: Config) -> Dict[int, Dict]:
    """Fit space curves to all fibres in parallel"""
    logger.info("Fitting space curves to fibres")
    
    fibre_ids = np.unique(point_cloud[:, 1]).astype(int)
    args_list = [(fid, point_cloud, config.polyfit_degree) for fid in fibre_ids]
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(executor.map(fit_fibre_wrapper, args_list))
    
    # Store results in dictionary
    curve_fits = {fid: result for fid, result in results if result is not None}
    
    logger.info(f"Successfully fit {len(curve_fits)}/{len(fibre_ids)} fibres")
    return curve_fits


def compute_tangent_vector(arc_length: np.ndarray, fx: np.ndarray, 
                          fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Compute tangent vectors from polynomial derivatives"""
    # Derivative coefficients
    dfx = np.polyder(fx)
    dfy = np.polyder(fy)
    dfz = np.polyder(fz)
    
    # Evaluate derivatives at arc length positions
    tx = np.polyval(dfx, arc_length)
    ty = np.polyval(dfy, arc_length)
    tz = np.polyval(dfz, arc_length)
    
    return np.column_stack([tx, ty, tz])


def compute_tangents_wrapper(args: Tuple[int, np.ndarray, Dict]) -> np.ndarray:
    """Wrapper for parallel tangent computation"""
    fibre_id, point_cloud, curve_fits = args
    fibre_data = point_cloud[point_cloud[:, 1] == fibre_id]
    
    if fibre_id not in curve_fits:
        return np.full((len(fibre_data), 3), np.nan)
    
    fit = curve_fits[fibre_id]
    return compute_tangent_vector(fit['arc_length'], fit['fx'], fit['fy'], fit['fz'])


def compute_all_tangents(point_cloud: np.ndarray, curve_fits: Dict, 
                        config: Config) -> np.ndarray:
    """Compute tangent vectors for all fibres in parallel"""
    logger.info("Computing tangent vectors")
    
    fibre_ids = np.unique(point_cloud[:, 1]).astype(int)
    args_list = [(fid, point_cloud, curve_fits) for fid in fibre_ids]
    
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        tangent_results = list(executor.map(compute_tangents_wrapper, args_list))
    
    return np.vstack(tangent_results)


def compute_orientation_angles(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation angles theta and phi from tangent vectors
    
    Returns:
        theta: Angles in degrees
        phi: Angles in degrees
    """
    logger.info("Computing orientation angles")
    
    # Reference direction 1 (in-plane)
    ref_dir1 = tangents.copy()
    ref_dir1[:, 2] = 0
    
    # Determine sign convention based on majority
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


def save_results(point_cloud: np.ndarray, theta: np.ndarray, 
                phi: np.ndarray, config: Config):
    """Save all output files"""
    logger.info("Saving results")
    
    # Save point cloud (global_id, x, y, z)
    # Format: integer for global_id, floats for coordinates
    np.savetxt(
        config.pointcloud_output,
        point_cloud[:, 3:7],
        fmt="%d\t%.6f\t%.6f\t%.6f",
        delimiter="\t"
    )
    
    # Save space curve data as NumPy compressed archive (.npz)
    # This is an open-source format that can be loaded with np.load()
    np.savez_compressed(
        config.spacecurve_output,
        point_cloud=point_cloud,
        lamella=point_cloud[:, 0],
        fibre_id=point_cloud[:, 1],
        local_id=point_cloud[:, 2],
        global_id=point_cloud[:, 3],
        coordinates=point_cloud[:, 4:7],
        directions=point_cloud[:, 7:10]
    )
    
    # Save orientation data
    orientation_df = pd.DataFrame({
        "x": point_cloud[:, 4],
        "y": point_cloud[:, 5],
        "z": point_cloud[:, 6],
        "fibre_id": point_cloud[:, 1].astype(int),
        "theta": theta,
        "phi": phi
    })
    orientation_df.to_csv(config.orientation_output, index=False)
    
    logger.info(f"Results saved to:")
    logger.info(f"  - {config.pointcloud_output}")
    logger.info(f"  - {config.orientation_output}")
    logger.info(f"  - {config.spacecurve_output} (NumPy compressed format)")
    logger.info(f"To load: data = np.load('{config.spacecurve_output}')")


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Initialize configuration
    config = Config()
    
    try:
        # Validate input
        validate_input_file(config.input_filename)
        
        # Load data
        coord, point_ids, lamella = load_data(config)
        
        # Build point cloud
        point_cloud = build_point_cloud(coord, point_ids, lamella, config)
        
        # Fit space curves
        curve_fits = fit_all_fibres(point_cloud, config)
        
        # Compute tangents
        tangents = compute_all_tangents(point_cloud, curve_fits, config)
        
        # Compute orientation angles
        theta, phi = compute_orientation_angles(tangents)
        
        # Save results
        save_results(point_cloud, theta, phi, config)
        
        elapsed = time.time() - start_time
        logger.info(f"Processing complete in {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()