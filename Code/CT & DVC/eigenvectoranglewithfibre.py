# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:50:54 2025

@author: Admin
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.linalg import eigh
import warnings

# Read data files
straindata = pd.read_csv("167208_new_whole-sw75.Lstr.csv", header=0).values
fibreorientationdata = pd.read_csv("167208_orientation.csv", header=0).values
outputfilename = '167208_Python_strain_direction.csv'

# Reorder strain file to same order as orientation
# Extract the x, y, z coordinates from both files
coords1 = fibreorientationdata[:, 0:3]
coords2 = straindata[:, 1:4]

# Define tolerance for floating-point comparison
tolerance = 1e-2

# Use KDTree to find the nearest neighbors in coords2 for each row in coords1
tree = KDTree(coords2)
distances, match_idx = tree.query(coords1)

# Initialize an array to hold the reordered data
reordered_straindata = np.zeros_like(straindata)

# Loop over each row in coords1 and find the approximately matching row in coords2
for i in range(len(coords1)):
    # Distance is already computed by KDTree
    dist = distances[i]
    
    # Check if the closest match is within the tolerance
    if dist <= tolerance:
        reordered_straindata[i, :] = straindata[match_idx[i], :]
    else:
        warnings.warn(f'No matching coordinates found for row {i} in file1 within tolerance.')
        print(dist)
        reordered_straindata[i, :] = straindata[match_idx[i], :]

print('Reordering complete with tolerance.')

# Extract strain components (using 0-based indexing)
exx = reordered_straindata[:, 9]
eyy = reordered_straindata[:, 10]
ezz = reordered_straindata[:, 11]
exy = reordered_straindata[:, 12]
eyz = reordered_straindata[:, 13]
exz = reordered_straindata[:, 14]

# Extract fibre orientation angles (using 0-based indexing)
theta_v = fibreorientationdata[:, 4]  # fibre orientation with vertical
phi_h = fibreorientationdata[:, 5]    # fibre orientation with horizontal

# Convert angles to radians
theta_v = np.deg2rad(theta_v)
phi_h = np.deg2rad(phi_h)

# Initialize arrays for fibre line vector L, eigenvalues, eigenvectors, and angles
N = len(exx)
L = np.zeros((N, 3))
eigenvalues = np.zeros((N, 3))
eigenvectors = np.zeros((N, 3, 3))
angles = np.zeros((N, 3))

# Loop through each point and compute eigenvalues/vectors and angles
for i in range(N):
    # Calculate line direction vector (Lx, Ly, Lz)
    L[i, 0] = np.cos(theta_v[i]) * np.sin(phi_h[i])
    L[i, 1] = np.sin(theta_v[i]) * np.sin(phi_h[i])
    L[i, 2] = np.cos(phi_h[i])
    
    # Normalize L
    L[i, :] = L[i, :] / np.linalg.norm(L[i, :])
    
    # Form the strain tensor for the current point
    strain_tensor = np.array([[exx[i], exy[i], exz[i]],
                              [exy[i], eyy[i], eyz[i]],
                              [exz[i], eyz[i], ezz[i]]])
    
    # Compute eigenvalues and eigenvectors
    D, V = eigh(strain_tensor)
    
    # Store eigenvalues and eigenvectors
    eigenvalues[i, :] = D
    eigenvectors[i, :, :] = V
    
    # Compute angles with fibre
    for j in range(3):  # Loop over the three eigenvectors
        cos_alpha = np.dot(V[:, j], L[i, :])  # Dot product with line direction
        angles[i, j] = np.arccos(np.clip(cos_alpha, -1, 1))  # Angle in radians
        
        if angles[i, j] > (np.pi / 2):
            angles[i, j] = np.pi - angles[i, j]

# Convert angles from radians to degrees
angles = np.rad2deg(angles)

# Prepare output data
header = ['x', 'y', 'z', 'ep1_alpha', 'ep3_alpha']
straindirdiff_data = np.column_stack([coords1, angles[:, 2], angles[:, 0]])

# Write to CSV
output_df = pd.DataFrame(straindirdiff_data, columns=header)
output_df.to_csv(outputfilename, index=False)