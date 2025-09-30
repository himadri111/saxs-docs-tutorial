#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:18:39 2024

@author: via83767
"""

"""
tomoSAXS / 3DXRD I(chi) Experimental Library Builder (Annotated)

Purpose
-------
- Load registered SAXS/CT index matrices (alpha, beta, fibre index, weights, counts).
- Build fibril parameter library and scanning geometry (beam paths).
- For each beam path / rotation: load or interpolate SAXS frames, perform azimuthal (chi) and radial (q) integrations.
- Construct an experimental I(chi) library for later fibre analysis.

Key Dependencies
----------------
- numpy, scipy, opencv (cv2)
- lmfit (for 1D peak modelling)
- pyFAI (AzimuthalIntegrator for chi/q integration)
- h5py (SAXS frames in .h5)
- a custom threeDXRD module for vector rotations

Reading Guide
-------------
- "UTILITY / GEOMETRY / INTEGRATION" sections describe reusable functions.
- "INPUTS & CONSTANTS" contains processing parameters.
- "DATA LOADING" pulls indices/masks/calibration from disk.
- "FIBRIL LIBRARY" and "BEAM PATHS" prepare the model of the experiment.
- The bottom section orchestrates per-beam-path I(chi)/I(q) extraction.

This file is intended as a drop-in replacement of the original script, but with commentary.
"""

# === IMPORTS & PATH SETUP ===
from multiprocessing import Pool, cpu_count 
import pickle
import re
#import PySimpleGUI as sg
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import time

from scipy import interpolate
import cv2

from lmfit import Parameters
from lmfit.models import GaussianModel,ExponentialModel,LinearModel,VoigtModel,LorentzianModel
from lmfit.models import SkewedGaussianModel,SkewedVoigtModel,Model,ConstantModel

from scipy.signal import savgol_filter

from pathlib import Path
from h5py import File
from numpy import array, ones
import hdf5plugin

import os,glob
#.chdir(r"C:/Users/Himadri/Desktop/Academic work/TomoSAXS/papers/outline scripts/")
import sys
sys.path.append(r'/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline')
#import threeDXRD as t3d
import threeDXRD_080923 as t3d

from lmfit import Model, CompositeModel

#from mayavi import mlab

#from mpi4py import MPI
import copy

from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator

# === DEFAULT INPUT/OUTPUT LOCATIONS ===
input_folder = '/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded'

saxs_slice = 1

alpha_file = input_folder+'/full_alpha_index_matrices.npy'
beta_file = input_folder+'/full_beta_index_matrices.npy'
index_file = input_folder+'/full_fibre_index_matrices.npy'
count_file = input_folder+'/full_count_index_matrices.npy'
weight_file = input_folder+'/full_fibre_weight_matrices.npy'
slice_files = input_folder
output_folder = input_folder

mask_file = '/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline/processing/SAXS_mask.nxs'
calib_file = '/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline/processing/SAXS_calibration.nxs'

# === PROCESS-WIDE OPTIONS (e.g., binning) ===
binning = 1

# === UTILITY: SMALL RANDOM PERTURBATIONS ===
def getVal(x0, dx, noiseterm=0.1):
    x = x0+dx*noiseterm*np.random.normal(0,1,1)[0]
    return x

# fibrilParams,r,rot,scan_data,fib_sample_data = fibrilParams,rotation,r_index,slice_scan_data,fib_sample_data

# === GEOMETRY: ROTATE/EXTRACT FIBRIL ATTRIBUTES PER ROTATION ===
def getRotatedFibrilPoints(fibrilParams,r,rot,scan_data,fib_sample_data):
    
    slice_scan_Data = np.copy(scan_data)
    
    slice_alpha_index,slice_beta_index,slice_index_index,slice_counts_index,slice_weights_index = slice_scan_Data[0:5]
    
    dxs,samplex1,dzs,samplez1 = fib_sample_data[0:4]
    
    fibre_index = [[slice_index_index[rot][i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != int 
                    and len(slice_index_index[rot][i,k])>0 and np.max(slice_index_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]

    fibreindex = [item for sublist in fibre_index for item in sublist]
    #fibreindex = [item for sublist in fibreindex for item in sublist]
    if len([k for k in np.arange(0,len(fibreindex),1) if type(fibreindex[k]) == int and fibreindex[k] == 500]) == 1:
                   
        kapton_edge_ID = np.max([np.max(k) for k in fibreindex])+100

        """
        Registration process labels kapton edge with a 500 - 
        check if this is still present or data has already been corrected 
        """
        
        kapton_edge_vox = [[[j,l] for l in np.arange(0,slice_alpha_index[rot].shape[1],1) if slice_alpha_index[rot][j,l] == 500] 
                           for j in np.arange(0,slice_alpha_index[rot].shape[0],1)]
        
        if len([k for k in kapton_edge_vox if len(k)>0])>0:
        
            kapton_edge_vox = [k for k in kapton_edge_vox if len(k)>0][0][0]
            
            kapton_edge_weight_vox = [[[j,l] for l in np.arange(0,slice_weights_index[rot].shape[1],1) if slice_weights_index[rot][j,l] == 500] 
                               for j in np.arange(0,slice_weights_index[rot].shape[0],1)]
            kapton_edge_weight_vox = [k for k in kapton_edge_weight_vox if len(k)>0][0][0]
            
            if type(slice_alpha_index[rot][kapton_edge_vox[0],kapton_edge_vox[1]]) != list:
                slice_alpha_index[rot][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
                slice_beta_index[rot][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
                slice_index_index[rot][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
                
                slice_weights_index[rot][kapton_edge_weight_vox[0],kapton_edge_weight_vox[1]] = [[100,kapton_edge_ID]]
                slice_counts_index[rot][kapton_edge_weight_vox[0],kapton_edge_weight_vox[1]] = [[100,kapton_edge_ID]]
    
    """
    index alpha, beta, and index values of registered voxels for this rotation
    """
    
    r_alpha_index = [[slice_alpha_index[rot][i,k][0:len(slice_alpha_index[rot][i,k])] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != int and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]
    r_alpha_voxs = [[[i,k] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]

    r_beta_index = [[slice_beta_index[rot][i,k][0:len(slice_beta_index[rot][i,k])] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != int and len(slice_beta_index[rot][i,k]) !=0 and np.max(slice_beta_index[rot][i,k]) !=0 and 
                     type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]
    r_beta_voxs = [[[i,k] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != int and len(slice_beta_index[rot][i,k]) !=0 and np.max(slice_beta_index[rot][i,k]) !=0 and 
                    type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]

    r_fibre_index = [[slice_index_index[rot][i,k][0:len(slice_index_index[rot][i,k])] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != int and len(slice_index_index[rot][i,k]) !=0 and np.max(slice_index_index[rot][i,k]) !=0 and 
                      type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]
    r_fibre_voxs = [[[i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != int and len(slice_index_index[rot][i,k]) !=0 and np.max(slice_index_index[rot][i,k]) !=0 and 
                     type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]

    r_count_index = [[slice_counts_index[rot][i,k][0:len(slice_counts_index[rot][i,k])] for k in np.arange(0,slice_counts_index[rot].shape[1],1) if type(slice_counts_index[rot][i,k]) != int and len(slice_counts_index[rot][i,k]) !=0 and np.max(slice_counts_index[rot][i,k]) !=0] for i in np.arange(0,slice_counts_index[rot].shape[0],1)]
    r_weight_index = [[slice_weights_index[rot][i,k][0:len(slice_weights_index[rot][i,k])] for k in np.arange(0,slice_weights_index[rot].shape[1],1) if type(slice_weights_index[rot][i,k]) != int and len(slice_weights_index[rot][i,k]) !=0 and np.max(slice_weights_index[rot][i,k]) !=0] for i in np.arange(0,slice_weights_index[rot].shape[0],1)]    
    r_vox_index = [[[i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != int and len(slice_index_index[rot][i,k]) !=0 and np.max(slice_index_index[rot][i,k]) !=0 and 
                    type(slice_alpha_index[rot][i,k]) != int and len(slice_alpha_index[rot][i,k]) !=0 and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]
    
    r_alphaindex = [item for sublist in r_alpha_index for item in sublist]
    r_alphavoxs = np.asarray([item for sublist in r_alpha_voxs for item in sublist])
    r_betaindex = [item for sublist in r_beta_index for item in sublist]
    r_fibreindex = [item for sublist in r_fibre_index for item in sublist]
    r_fibrevoxs = np.asarray([item for sublist in r_fibre_voxs for item in sublist])
    r_voxindex = [item for sublist in r_vox_index for item in sublist]
        
    r_fibreIndex = [item for sublist in r_fibreindex for item in sublist]
    r_fibreVoxs = np.asarray([item for sublist in r_fibrevoxs for item in sublist])
    r_alphaIndex = [item for sublist in r_alphaindex for item in sublist]
    r_alphaVoxs = np.asarray([item for sublist in r_alphavoxs for item in sublist])
    r_betaIndex = [item for sublist in r_betaindex for item in sublist]
    r_voxIndex = [item for sublist in r_voxindex for item in sublist]
    
    """
    check if there are any indexed fibres that don't have a corresponding alpha or beta value
    if there are - fill in the corresponding alpha and beta voxels with the values for the fibre from
    the original rotation
    """
    
    if len(r_alphaIndex)<len(r_fibreIndex):
        missing_alphas = [k for k in np.arange(0,len(r_alpha_index),1) if len(r_alpha_index[k]) != len(r_fibre_index[k])]
        missing_alpha_voxs = [[l for l in r_fibre_voxs[k] if l not in r_alpha_voxs[k]] for k in missing_alphas]
        missing_alpha_idxs = [[slice_index_index[rot][k[l][0],k[l][1]] for l in np.arange(0,len(k),1)]  for k in missing_alpha_voxs]
        
        og_indxs = np.asarray([k["indx"] for k in fibrilParams])
        og_alphas = np.asarray([k["alpha"] for k in fibrilParams])
        
        alphas_to_fill = [[[[og_alphas[np.where(og_indxs == j)][0] for j in i] for i in l] for l in missing_alpha_idxs]]
        
        for i in range(0,len(missing_alpha_voxs)):
            for k in range(0,len(missing_alpha_voxs[i])):
                
                v2f = missing_alpha_voxs[i][k]
                
                slice_alpha_index[rot][v2f[0],v2f[1]] = alphas_to_fill[0][i][k]
                
        r_alpha_index = [[slice_alpha_index[rot][i,k][0:len(slice_alpha_index[rot][i,k])] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != int and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]
        r_alphaindex = [item for sublist in r_alpha_index for item in sublist]
        r_alphaIndex = [item for sublist in r_alphaindex for item in sublist]
                
    if len(r_betaIndex)<len(r_fibreIndex):
        missing_betas = [k for k in np.arange(0,len(r_beta_index),1) if len(r_beta_index[k]) != len(r_fibre_index[k])]
        missing_beta_voxs = [[l for l in r_fibre_voxs[k] if l not in r_beta_voxs[k]] for k in missing_betas]
        missing_beta_idxs = [[slice_index_index[rot][k[l][0],k[l][1]] for l in np.arange(0,len(k),1)]  for k in missing_beta_voxs]
        
        og_indxs = np.asarray([k["indx"] for k in fibrilParams])
        og_betas = np.asarray([k["beta"] for k in fibrilParams])
        
        betas_to_fill = [[[[og_betas[np.where(og_indxs == j)][0] for j in i] for i in l] for l in missing_beta_idxs]]
        
        for i in range(0,len(missing_beta_voxs)):
            for k in range(0,len(missing_beta_voxs[i])):
                
                v2f = missing_beta_voxs[i][k]
                
                slice_beta_index[rot][v2f[0],v2f[1]] = betas_to_fill[0][i][k]
                
        r_beta_index = [[slice_beta_index[rot][i,k][0:len(slice_beta_index[rot][i,k])] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != int and np.max(slice_beta_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]

        r_betaindex = [item for sublist in r_beta_index for item in sublist]  
        r_betaIndex = [item for sublist in r_betaindex for item in sublist]
                    
    
    r_countindex = [item for sublist in r_count_index for item in sublist]
    r_weightindex = [item for sublist in r_weight_index for item in sublist]
    
    r_countIndex = [item for sublist in r_countindex for item in sublist]
    r_weightIndex = [item for sublist in r_weightindex for item in sublist]
         
    r_fibreindex = [item for sublist in r_fibre_index for item in sublist]
    r_fibreIndex = [item for sublist in r_fibreindex for item in sublist]
    
    r_fibre_voxels = [[[r_vox_index[i][k] for k in np.arange(0,len(r_vox_index[i]),1) if vox in r_fibre_index[i][k]] for i in np.arange(0,len(r_fibre_index),1)]for vox in r_fibreIndex]
    r_fibre_voxels = [[k[0] for k in vox if len(k)>0] for vox in r_fibre_voxels]    
    r_fibre_voxels = np.asarray([k[0] for k in r_fibre_voxels if len(k)>0]) 
    
    r_fibre_alphas = [[[[r_alpha_index[i][k][l] for l in np.arange(0,len(r_alpha_index[i][k]),1) if r_fibre_index[i][k][l] == vox] 
                     for k in np.arange(0,len(r_alpha_index[i]),1)] for i in np.arange(0,len(r_fibre_index),1)] 
                   for vox in r_fibreIndex]
    r_fibre_alphas = [[[r_fibre_alphas[i][k][l][0] for l in np.arange(0,len(r_fibre_alphas[i][k]),1) if len(r_fibre_alphas[i][k][l])>0] 
                    for k in np.arange(0,len(r_fibre_alphas[i]),1)] for i in np.arange(0,len(r_fibre_alphas),1)]    
    r_fibre_alphas = [[r_fibre_alphas[i][k][0] for k in np.arange(0,len(r_fibre_alphas[i]),1) if len(r_fibre_alphas[i][k])>0] 
                   for i in np.arange(0,len(r_fibre_alphas),1)]
    r_fibre_alphas = np.asarray([k[0] for k in r_fibre_alphas])
    
    r_fibre_betas = [[[[r_beta_index[i][k][l] for l in np.arange(0,len(r_beta_index[i][k]),1) if r_fibre_index[i][k][l] == vox] 
                     for k in np.arange(0,len(r_beta_index[i]),1)] for i in np.arange(0,len(r_fibre_index),1)] 
                   for vox in r_fibreIndex]
    r_fibre_betas = [[[r_fibre_betas[i][k][l][0] for l in np.arange(0,len(r_fibre_betas[i][k]),1) if len(r_fibre_betas[i][k][l])>0] 
                    for k in np.arange(0,len(r_fibre_betas[i]),1)] for i in np.arange(0,len(r_fibre_betas),1)]    
    r_fibre_betas = [[r_fibre_betas[i][k][0] for k in np.arange(0,len(r_fibre_betas[i]),1) if len(r_fibre_betas[i][k])>0] 
                   for i in np.arange(0,len(r_fibre_betas),1)]
    r_fibre_betas = np.asarray([k[0] if type(k) == list else k for k in r_fibre_betas])
    
    r_fibre_indexes = [[[[r_fibre_index[i][k][l] for l in np.arange(0,len(r_fibre_index[i][k]),1) if np.asarray(r_fibre_index[i][k])[l] == vox] 
                     for k in np.arange(0,len(r_fibre_index[i]),1)] for i in np.arange(0,len(r_fibre_index),1)] 
                   for vox in r_fibreIndex]
    r_fibre_indexes = [[[r_fibre_indexes[i][k][l][0] for l in np.arange(0,len(r_fibre_indexes[i][k]),1) if len(r_fibre_indexes[i][k][l])>0] 
                    for k in np.arange(0,len(r_fibre_indexes[i]),1)] for i in np.arange(0,len(r_fibre_indexes),1)]    
    r_fibre_indexes = [[r_fibre_indexes[i][k][0] for k in np.arange(0,len(r_fibre_indexes[i]),1) if len(r_fibre_indexes[i][k])>0] 
                   for i in np.arange(0,len(r_fibre_indexes),1)]
    r_fibre_indexes = np.asarray([k[0] if type(k) == list else k for k in r_fibre_indexes])
    
   
    """
    new adjustment:
    """
    r_fibre_alphas = np.copy(r_fibre_alphas)
    r_fibre_betas = np.copy(r_fibre_betas)
    r_fibre_alphas = np.where(r_fibre_alphas<90,r_fibre_alphas+90,90+(180-r_fibre_alphas))
    #r_fibre_alphas = r_fibre_alphas+90
    #r_fibre_betas = r_fibre_betas-20
    
    og_fibre_idxs = np.asarray([k["indx"] for k in fibrilParams])
    
    """
    fill rotated fibril dictionary
    """

    rotatedFibrilParams = [None]*len(r_fibre_indexes)
    test = []
    r_xTest,r_zTest = [],[]
    r_weight_test = []
    for i in range(0,len(r_fibre_voxels)):
        idx = r_fibre_indexes[i]
        if len(np.where(og_fibre_idxs == idx)[0])>0 and len(np.where(r_fibre_indexes == idx)[0])>0:
            test.append(idx)
            og_fibre = fibrilParams[np.where(og_fibre_idxs == idx)[0][0]]
            xr = r_fibre_voxels[i][1]*dxs+samplex1
            r_xTest.append(xr)
            yr = og_fibre["y"]
            zr = r_fibre_voxels[i][0]*dzs+samplez1
            r_zTest.append(zr)
            alpha = r_fibre_alphas[i]
            beta = r_fibre_betas[i]+r
            
            weight = np.mean([k[0] for k in r_weightIndex if k[1] == idx])
            r_weight_test.append(weight)
            count = np.mean([k[0] for k in r_countIndex if k[1] == idx])
            solved = og_fibre["solved"]
            intersected = og_fibre["intersected"]
            simu = og_fibre["simu"]
            fit = og_fibre["fit"]
            amplitude = og_fibre["amplitude"]
            number = og_fibre["number"]
            amp_est = og_fibre["initial_amp_est"]
            
            rotatedFibrilParams[i]={"indx": idx, "number": number,"x":xr,"y":yr,"z":zr, 
                                    "alpha": alpha, "beta": beta,
                                    "solved": solved,"intersected": intersected,
                                    "simu": simu, "fit": fit, "amplitude": amplitude,
                                    "weight": weight,"count": count,"initial_amp_est":amp_est}
                    
    return rotatedFibrilParams,[r_xTest,r_zTest,r_weight_test]


#beam_Paths,fibril_Params,weightIndex,beamradius,sig = beamPaths,rotatedFibrilParams,slice_weights_index[r],beam_size/2.0,1

# === GEOMETRY ↔︎ EXPOSURE: MAP VOXELS TO BEAM PATHS WITH GAUSSIAN WEIGHTS ===
def assignVoxelsPerPath(beam_Paths,fibril_Params,weightIndex,beamradius=0.05,sig = 3):
    """
    Input: Take a list of beam paths as input. The beam paths are given as dictionaries
    Output: returns the list of {"beampath": beam path dict, "fibril list": list of 
    intersected fibrils, each fibril given by a dictionary with its attributes as 
    well as overlap index and weight fraction}
    
    The voxels are assigned to a beampath if the center of the voxel is 
    at a distance which is < 3*sigma = beam radius from the beam. The 
    weight term is exp(-(d^2/2 sigma^2)) which is 1 if the beam is hitting 
    the centre and drops off sharply as one goes away from it.
    """
    beamdiameter = 2*beamradius
    #
    # beam diameter is FWHM of a Gaussian profile
    # if exp(-x^2/2sigma^2) = intensity fall off
    # 1/2 = exp(-(FWHM/2/)**2/2*sigma**2)
    # sqrt(2*log(2)) = FWHM/2 sigma
    # sigma = FWHM / (2*srt(2*log(2)))
    #
    sigma=beamdiameter/(2*np.sqrt(2*np.log(2))) # 
    #print("beamdiameter: ", beamdiameter, ", sigma: ", sigma)
    lenBeamPaths = len(beam_Paths)
    intersectedVoxels = [None]*lenBeamPaths
    #print("lenBeamPaths: ", lenBeamPaths)
    #intersectionThreshold = 5e-3
    t1 = time()
    test = []
    
    weights_test = []
    
    for i, beamPath in enumerate(beam_Paths[0:weightIndex.shape[1]]):
        
        bp_weights = weightIndex[:,i]
        bp_weights = [k for k in bp_weights if k !=0 and type(k) != int]
        bp_weights = [item for sublist in bp_weights for item in sublist]
        
        weight_idxs = np.asarray([k[1] for k in bp_weights])
        weight_vals = np.asarray([k[0] for k in bp_weights])
        
        intersectedVoxels[i]           = {}
        intersectedVoxels[i]["path"]   = beamPath
        intersectedVoxels[i]["ichi"]   = None
        intersectedVoxels[i]["voxels"] = None
        
        x1, y1, z1 = beamPath["start"][0],beamPath["start"][1],beamPath["start"][2]
        x2, y2, z2 = beamPath["end"][0],  beamPath["end"][1],  beamPath["end"][2]
        #print(x1,x2,y1,y2,z1,z2)
        """
        https://www.nagwa.com/en/explainers/939127418581/#:~:text=The%20perpendicular%20distance%20between%20a%20point%20and%20a%20line%20is,any%20point%20on%20the%20line.
        calculate direction vector d
        """
        xA, yA, zA = x1, y1, z1
        d_len = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        d1, d2, d3 = (x2-x1)/d_len, (y2-y1)/d_len, (z2-z1)/d_len
        """
        scan over all points in the fibril_Params to calculate distance
        """
        test1 = []
        for idx,fibril_param in enumerate(fibril_Params):
            if fibril_param != None:
                x, y, z = fibril_param['x'],fibril_param['y'],fibril_param['z']
                """
                cross product AP x d/mod(d)
                """
                ap_x, ap_y, ap_z = (x-xA), (y-yA), (z-zA)
                apd_x, apd_y, apd_z = (ap_y*d3-ap_z*d2),(ap_z*d1-ap_x*d3),(ap_x*d2-ap_y*d1)
                apd_len = np.sqrt(apd_x**2 + apd_y**2 + apd_z**2)
                perp_dist = apd_len/d_len
                #if perp_dist<sig*sigma or fibril_param["indx"] in weight_idxs:
                if fibril_param["indx"] in weight_idxs:
                    test1.append([x,z])
                    
                    vox_weight = weight_vals[np.where(weight_idxs == fibril_param["indx"])]
                    
                    if len(vox_weight)>1:
                        vox_weight = vox_weight[0]
                    
                    #print("close point: ", x, y, z)
                    if intersectedVoxels[i]["voxels"] == None:
                        intersectedVoxels[i]["voxels"] = []
                    voxelDict = {}
                    voxelDict["fibril_param"] = fibril_param
                    voxelDict["fibril_param"]["weight"] = vox_weight
                    voxelDict["perp_dist"]    = perp_dist
                    voxelDict["coloured"]    = False
                    #voxelDict["weight"] = 1.0 #Change this to proportional to overlap
                    #voxelDict["weight"] = (1.0-(perp_dist/intersectionThreshold)) #Change this to proportional to overlap
                    #voxelDict["weight"] = np.exp(-0.5*((perp_dist/sigma)**2)) #Change this to proportional to overlap
                    voxelDict["weight"] = vox_weight
                    #voxelDict["solved"] = False
                    intersectedVoxels[i]["voxels"].append(voxelDict)
                    
                    weights_test.append([i,idx,vox_weight])
                    
        if len(test1)>0:
            test.append(test1)
        
            
    t2 = time()
    #print("second time: ", t2-t1)
    return intersectedVoxels




# === DATA MODEL: DEFAULT FIBRIL PARAMETER RECORD ===
def InitialiseFibrilParameters(i):
    """
    returns a dictionary of fibril parameters for each voxel i
    set positions to NULL initially
    """
    fibrildict  = {"indx": None, "x": None, "y": None, "alpha": 0, "beta": 0,"weight": 0,"count": 0,\
                   "solved": False, "intersected": False, "simu": {}, "fit": {},\
                       "amplitude": 1}
    """
    code for initialising alpha, beta, q0, ...
    """
    return fibrildict

# === GEOMETRY: BEAM PATH GENERATION THROUGH SAMPLE ===
def findBeamPaths(scanParams,length=1,rot=0,eps=1e-6,pointsize=0.1,linewidth=1.0,\
                  linecolor=(0,0,1),display=False,beamsize=0.05,bp_count = False):
    """
    default to a x-y scan
    
    0) calculate coordinates of x-y scan point in z=0 plane
    1) displace all points by +/- 1/2 length along z (two sets of (xyz) planes)
    2) rotate all points in the planes
    3) return tuples of the points on each side of the plane (line coords) as
       list of dictionaries: "start": (x1-K, y1-K, z1-K), "end": (xK-2, yK-2, zK-2)
    """
    """
    scanParams = {"xstart": scanx1, "xend": scanx2, "xstep": dx,
                  "ystart": scany1, "yend": scany2, "ystep": dy}
    """
    scanx1=scanParams["xstart"]
    scanx2=scanParams["xend"]
    dx=scanParams["xstep"]
    scany1=scanParams["ystart"]
    scany2=scanParams["yend"]
    dy=scanParams["ystep"]
    
    if bp_count == False:
        if scany1!=scany2:
            scanxP, scanyP = np.mgrid[scanx1:scanx2+eps:dx,
                                      scany1:scany2+eps:dy]
        else:
            scanxP = np.arange(scanx1,scanx2+eps,dx)
            scanyP = scany1*np.ones_like(scanxP)
    else:
        if scany1!=scany2:
            scanxP, scanyP = np.mgrid[scanx1:scanx2+eps:dx,
                                      scany1:scany2+eps:dy]
        else:
            scanxP = np.linspace(scanx1,scanx2+eps,bp_count)
            scanyP = scany1*np.ones_like(scanxP)
        
    #print(scanx1,scanx2,dx,scany1,scany2,dy)
    #print("scanxP: ", scanxP)
    #print("scanyP: ", scanyP)
    #print(scanxP)
    scanzP = (length/2.0)*np.ones_like(scanxP)
    scanzM = (-length/2.0)*np.ones_like(scanxP)
    
    beam_Paths = []
    rotn = np.radians(rot)
    if scany1!=scany2:
        for x1,y1,z1,x2,y2,z2 in zip(scanxP, scanyP, scanzP, scanxP, scanyP, scanzM):
            for xx1, yy1, zz1, xx2, yy2, zz2 in zip(x1,y1,z1,x2,y2,z2):
                xx1r,yy1r,zz1r = t3d.rotated_vectors(xx1,yy1,zz1,alpha=0,beta=rotn)
                xx2r,yy2r,zz2r = t3d.rotated_vectors(xx2,yy2,zz2,alpha=0,beta=rotn)
                xvec = [xx1r,xx2r]
                yvec = [yy1r,yy2r]
                zvec = [zz1r,zz2r]
                #print(xvec,yvec,zvec)
                beamLine = {}
                beamLine["start"] = [xx1,yy1,zz1]
                beamLine["end"]   = [xx2,yy2,zz2]
                beamLine["diameter"] = beamsize
                beam_Paths.append(beamLine)
                
                #if display==True:
                    #mlab.plot3d(xvec,yvec,zvec,line_width=linewidth,color=linecolor,tube_radius=None)
    else:
        for x1,y1,z1,x2,y2,z2 in zip(scanxP, scanyP, scanzP, scanxP, scanyP, scanzM):
            xx1r,yy1r,zz1r = t3d.rotated_vectors(x1,y1,z1,alpha=0,beta=rotn)
            xx2r,yy2r,zz2r = t3d.rotated_vectors(x2,y2,z2,alpha=0,beta=rotn)
            xvec = [xx1r,xx2r]
            yvec = [yy1r,yy2r]
            zvec = [zz1r,zz2r]
            #print(xvec,yvec,zvec)
            beamLine = {}
            beamLine["start"] = [x1,y1,z1]
            beamLine["end"]   = [x2,y2,z2]
            beamLine["diameter"] = beamsize
            beam_Paths.append(beamLine)
            
            #if display==True:
                #mlab.plot3d(xvec,yvec,zvec,line_width=linewidth,color=linecolor,tube_radius=None)
            
            
    #if display==True:
        #mlab.points3d(scanxP, scanyP, scanzP,scale_factor=pointsize) 
        #mlab.points3d(scanxP, scanyP, scanzM,scale_factor=pointsize) 
        #mlab.show(stop=True)
            
    return beam_Paths


 
# === MODELLING: SIMPLE GAUSSIAN FITTER (LMFIT) ===
def fit_gauss(xData,yData):
    mod = GaussianModel()
    pars = mod.guess(yData,xData)
    result = mod.fit(yData, pars, x=xData)
    model = mod.eval(result.params,x=xData)
    params = result.params
    
    return model,params


# === UTILITY: IMAGE RESIZING FOR INTERPOLATION ===
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# === DATA: FRAME LOADING & INTERPOLATION (MASK-AWARE) ===
def interp_frame(rotation,frameIndex,mask,subsample,Method,slice_saxs_file):
    
    if type(frameIndex) == int:
        SAMPLE_PATH = Path(slice_saxs_file)
        
        with File(SAMPLE_PATH) as sample_file:
            entry = list(sample_file.keys())[rotation]
            frame = array(sample_file[entry][frameIndex])
            sample_file.close()
            
        frame = (frame+1)*-1
    else:
        frame = frameIndex
    
    array1 = np.copy(frame)
    array1[array1 == -1] = 0
    if len(np.unique(array1))>1:
        array1[array1==np.unique(array1)[1]] = 250
    mask_array = np.copy(mask)

    array1 = image_resize(array1.astype(np.uint8), width = int(np.round(array1.shape[1]/subsample)), height = int(np.round(array1.shape[0]/subsample)))
    new_col = np.zeros_like(array1.sum(0)[...,None])
    array1 = np.append(array1, new_col.T, 0)

    mask_array = image_resize(mask_array.astype(np.uint8), width = int(np.round(frame.shape[1]/subsample)), height = int(np.round(frame.shape[0]/subsample)))
    mask_array = np.append(mask_array, new_col.T, 0)

    z = np.where(mask_array == False,array1,np.nan)

    M, N = z.shape[0]-1, z.shape[1]-1
    zi = np.vstack((z[::-1,:],z))
    zi = np.hstack((zi[:,::-1], zi))
    y, x = np.mgrid[0:2*(M+1), 0:2*(N+1)]
    #y *= 5 # anisotropic interpolation if needed.
    
    z_d = interpolate.griddata((y[~np.isnan(zi)], x[~np.isnan(zi)]),
                    zi[~np.isnan(zi)], (y, x), method=Method)
    z_d = z_d[:(M+1),:(N+1)][::-1,::-1]
    
    z2 = image_resize(z_d, width = frame.shape[1], height = frame.shape[0])
    z2 = z2-np.min(z2)
    z2 = z2*(np.max(frame)/np.max(z2))
            
    return z2,frame


# === INTEGRATION: BUILD I(chi) / I(q) PROFILES (pyFAI) ===
def ichi_sample(rotation,frameIndex,ichiSim,chiSim,cake_params_chi,
                a1,Mask,slice_saxs_file,chiRange,fibre_chi = False,fit_chi=False,iq_plot = False):
    
    """
    function for ichi sampling (normalised against background intensity )
    
    add in dictionary for sampling values to input
    """
        
    nchi = cake_params_chi["nchi"]
    nq0 = cake_params_chi["nq"]
    
    q0i_low,q0i_high = cake_params_chi["q1i"],cake_params_chi["q2i"]
    q0o_low,q0o_high = cake_params_chi["q1o"],cake_params_chi["q2o"]
    
    q0c_low,q0c_high = cake_params_chi["q1"],cake_params_chi["q2"]
     
    if type(frameIndex) == int:
        SAMPLE_PATH = Path(slice_saxs_file)
        
        with File(SAMPLE_PATH,'r') as sample_file:
            entry = list(sample_file.keys())[rotation]
            frame = array(sample_file[entry][frameIndex])
            sample_file.close()
            
        frame = (frame+1)*-1
    else:
        frame = frameIndex
    
    if fibre_chi == True and ichiSim !=0:
        mod = GaussianModel()
        pars = mod.guess(ichiSim,chiSim)
        result = mod.fit(ichiSim, pars, x=chiSim)
        model = mod.eval(result.params,x=chiSim)
        params = result.params
        
        chiRange = [180-params["center"].value - (params["sigma"].value*5),180-params["center"].value + (params["sigma"].value*5)]
    
    
    if type(Mask) != int:
        print("incl mask")
        inner_bg = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0i_low,q0i_high),azimuth_range = (chiRange[0],chiRange[1]),mask=Mask)
        outer_bg = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0o_low,q0o_high),azimuth_range = (chiRange[0],chiRange[1]),mask=Mask)
        peak_chi = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0c_low,q0c_high),azimuth_range = (chiRange[0],chiRange[1]),mask=Mask)
    else:
        inner_bg = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0i_low,q0i_high),azimuth_range = (chiRange[0],chiRange[1]))
        outer_bg = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0o_low,q0o_high),azimuth_range = (chiRange[0],chiRange[1]))
        peak_chi = a1.integrate_radial(frame,npt = nchi,npt_rad = nq0,radial_range = 
                                       (q0c_low,q0c_high),azimuth_range = (chiRange[0],chiRange[1]))
    

    mean_bg = (inner_bg[1]+outer_bg[1])/2
    
    norm_chi = peak_chi[1]-mean_bg
                
    chi_model = None
    peak_iq = None
    peak_iq = None
    
    if fit_chi==True:
        chi_model = fit_gauss(inner_bg[0],norm_chi)
        
        
    if iq_plot==True:
        if type(Mask) != int:
            print("q incl. mask")
            peak_iq = a1.integrate1d(frame,nq0,radial_range=(q0i_low,q0o_high),
                                     azimuth_range=(chiRange[0],chiRange[1]),mask=Mask)
        else:
            peak_iq = a1.integrate1d(frame,nq0,radial_range=(q0i_low,q0o_high),
                                     azimuth_range=(chiRange[0],chiRange[1]))
        
    peak_iq = [np.copy(peak_iq[0]),np.copy(peak_iq[1])]
        
    return [peak_chi[0],norm_chi],chi_model,peak_iq,frame,[inner_bg[1],outer_bg[1],peak_chi[1]]


# === UTILITY: NEAREST VALUE IN ARRAY ===
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

# === MODELLING: BACKGROUND FIT/SUBTRACTION FOR I(q) ===
def remove_bg(profile,modelQrange,fullQrange,exp = False):
    fullQrange = np.asarray(fullQrange)
    IQProfile = profile
    if len(modelQrange)<3:
        IQXaxis = np.linspace(modelQrange[0],modelQrange[1],profile.shape[0])
    else:
        IQXaxis = np.asarray(modelQrange)
    if exp == False:
        model = LinearModel()
        pars = model.guess(IQProfile, IQXaxis) 
    else:
        expModel = ExponentialModel()
        pars = expModel.guess(IQProfile, IQXaxis) 
        pars["amplitude"].min = pars["amplitude"].value/10
        pars["amplitude"].max = pars["amplitude"].value*10
        pars["decay"].min = pars["decay"].value-0.01
        pars["decay"].max = pars["decay"].value+0.01
        #result = expModel.fit(IQProfile, pars, x=IQXaxis)
                
        cModel = ConstantModel()
        cPars = Parameters()
        cPars.add("c",value = np.min(profile),min = 0,max = 10)
        
        pars.add(cPars['c'])
               
        model = ExponentialModel()+ConstantModel()                
    result = model.fit(IQProfile, pars, x=IQXaxis)
    modelData = model.eval(result.params,x=fullQrange)
    
    return modelData

# === UTILITY: BINNING 1D DATA ===
def bin_data(data,width):
    result = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
    return result

# === PIPELINED TASK: PER-BEAMPATH PROCESS (I(chi)/I(q), KAPTON CHECK) ===
def testFunc(i,path_dict,slice_saxs_file,pyFai_params):
    
    max_int,sample_bps = None,None
    
    Mask = pyFai_params[0]
    cake_params_chi = pyFai_params[1]
    cake_params_test = pyFai_params[9]
    dict_chi_range = pyFai_params[10]
    
    kapton_edge_ID = pyFai_params[-1]
    
    a1 = AzimuthalIntegrator(wavelength = pyFai_params[2])
    a1.setFit2D(directDist = pyFai_params[3], centerX = pyFai_params[4],centerY = pyFai_params[5], 
                pixelX = pyFai_params[6],pixelY = pyFai_params[7])
                
    if kapton_edge_ID in [k["fibril_param"]["indx"] for k in path_dict["voxels"]]:
                            
        #print("######\nkapton edge found at BP: ",str(i),"\n######")
        
        """
        if the beampath does contain the CT kapton edge = plot the SAXS frame
        """
        SAMPLE_PATH = Path(slice_saxs_file)
        
        with File(SAMPLE_PATH) as sample_file:
            entry = list(sample_file.keys())[r]
            kapton_frame = array(sample_file[entry][i])
            sample_file.close()
            
        #kapton_frame = (kapton_frame
        
        lib_idx ={"bp": i,"ichi":0,"kapton":kapton_frame,"frame":kapton_frame}
        
        #sample_bps = i
        
    else:
        
        """
        else if the beampath doesn't contain the kapton edge - but does contain 
        indexed fibres:
            perform I(chi) and I(q) intergration over the frame and save to the SVD dictionary 
        """
        
        #print("######\nRotation: ",str(r),", BP: ",str(i),"\ninterpolating and integrating\n#####")
        
        rot_frame = interp_frame(r,i,Mask,4,"linear",slice_saxs_file)
        
        #if mask_interp == True:
        smpl_1d = ichi_sample(r,rot_frame[0][0:Mask.shape[0],:],0,0,cake_params_chi,a1,Mask,slice_saxs_file,[dict_chi_range[0],dict_chi_range[1]],
                                  fibre_chi = False,fit_chi=False,iq_plot = True)
        #else:
            #smpl_1d = ichi_sample(r,rot_frame[1],0,0,cake_params_chi,a1,Mask,slice_saxs_file,[dict_chi_range[0],dict_chi_range[1]],
                                      #fibre_chi = False,fit_chi=False,iq_plot = True)
                                                                                                                            
        smpl_ichi = np.copy(smpl_1d[0])
        smpl_chi = smpl_ichi[0]*-1
        smpl_I = np.flip(smpl_ichi[1])
        smpl_Q = np.copy(smpl_1d[2])
        
        """
        The I(q) is always performed from -180-to-0 degrees (to counteract the clockwise integration of pyFai vs the 
                                                             anticlockwise direction of threeDXRD)
        
        but the I(chi) is tailored to the "dict_chi_range" variable.
        If this is not [-180,0] - then we reporform the integration.
        """
         
        if dict_chi_range !=[-180,0]:
            
            nv_start = dict_chi_range[0]
            nv_end = 180
            
            pv_start = -180
            pv_end = dict_chi_range[1]-360
                                                                                                            
            smpl_nv90 = ichi_sample(pyFai_params[8],rot_frame[0][0:Mask.shape[0],:],0,0,cake_params_test,a1,Mask,slice_saxs_file,[nv_start,nv_end],
                                  fibre_chi = False,fit_chi=False,iq_plot = True)
            
            smpl_pv90 = ichi_sample(r,rot_frame[0][0:Mask.shape[0],:],0,0,cake_params_test,a1,Mask,slice_saxs_file,[pv_start,pv_end],
                                  fibre_chi = False,fit_chi=False,iq_plot = True)
            
            smpl_I = np.flip(np.concatenate((smpl_nv90[0][1],smpl_pv90[0][1])))
    
        
        #smpl_smooth = savgol_filter(smpl_I,window,pol)
        
        max_int = np.max(smpl_I)
                                     
        lib_idx = {"bp": i,"ichi":smpl_I,"kapton":False,"iq": smpl_Q}
        
        sample_bps = i
        
    return i,lib_idx,sample_bps,max_int    

"""
# === TOP-LEVEL EXECUTION ===
""""""
start script
""""""
"""
 
# === INPUTS & CONSTANTS (SAMPLING, ROTATIONS, FIT WINDOWS) ===
"""
Input variables
"""
mask_interp = True #option for interpolating over masks in 2D frames

q_order = 3 #order of interest

#global estimates for scattering parameters
q0_m, wMu_m, wa_m, delta_m, amp_m, alpha_m, beta_m = \
    ((2*np.pi)*q_order)/66.8, 0.2, 0.002, 0.5, 1.0, 30, 45

#Variation in scanning parameters for populating "fibrilParam" library
dq0_m, dwMu_m, dwa_m, ddelta_m, damp_m, dalpha_m, dbeta_m = \
    0.008, 0.008, 0.0008, 0.01, 0.8, 20, 45

#noise term for simulations
noiseterm = 0.1

#list of global scattering parameters
sim_vals = [q0_m,wa_m,wMu_m,amp_m,delta_m]

#wavelength of real data
wavelen = 0.08856

#angular rotations used in real scan
#rot_range = [-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]
rot_range = np.linspace(-85,95,9)

#parameters for plotting
pointcolor=(0,0,1)
pointSize=0.05 

#chi parameters for I(chi) sampling        
chi1, chi2, nchi = 90, 270, 180
chirange = np.linspace(chi1, chi2, nchi)

#sampling range for I(q) integratin
dq0_int, nq0_int = 0.02, 30
q0m_low, q0m_high, nq0 = q0_m-dq0_int,q0_m+dq0_int, nq0_int

#sampling parameters for I(q) fitting
q_fitIq1, q_fitIq2, nq_fitIq = q0_m-0.02, q0_m+0.03, 120
dq_fitIq = (q_fitIq2-q_fitIq1)/nq_fitIq
q_fitIqr = np.arange(q_fitIq1, q_fitIq2, dq_fitIq)

dq0_bgr = 0.005
q0i_low,q0i_high = q0m_low - dq0_bgr,q0m_low
q0o_low,q0o_high = q0m_high,q0m_high+dq0_bgr

#sampling parameters for background correction in I(chi) integration
dchi_bgr = 0.001
ichi_low,ichi_high = q0m_low - dchi_bgr,q0m_low

#smoothing parameters for I(chi) integration
window,pol = 9,1

# width of sampling in Chi for I(iq) sampling and fitting
dchiIq=3

#number of I(q) slices for SV I(q) fitting
nslices=3

#dict_chi_range = [-180,0]
dict_chi_range = [90,270]

#binning value for I(chi) data
bin_val = 4

"""
cake parameters for chi integration
"""
cake_params_chi = {"chi1": chi1, "chi2": chi2, "nchi": nchi, "q1": q0m_low, 
               "q2": q0m_high, "nq": nq0, "wavelen": wavelen,"q1i":q0i_low,"q2i":q0i_high,
               "q1o":q0o_low,"q2o": q0o_high}

cake_params_test = copy.deepcopy(cake_params_chi)                                
cake_params_test["nchi"] = 90
cake_params_test["q1"] = cake_params_chi['q1']+0.01
cake_params_test["q2"] = cake_params_chi['q2']-0.01
cake_params_test["q2i"] = cake_params_test['q1']
cake_params_test["q1i"] = cake_params_test['q1i']+0.01
cake_params_test["q1o"] = cake_params_test['q2']
cake_params_test["q2o"] = cake_params_test['q2o']-0.01

# === DATA LOADING: INDEX MATRICES, MASK & CALIBRATION ===
"""
Create data paths and load data
"""
   

alpha_data = np.load(alpha_file,allow_pickle = True)
beta_data = np.load(beta_file,allow_pickle = True)
index_data = np.load(index_file,allow_pickle = True)
counts_data = np.load(count_file,allow_pickle = True)      
weights_data = np.load(weight_file,allow_pickle = True)  

MASK_PATH = Path(mask_file)
CALIBRANT_PATH = Path(calib_file)

with File(MASK_PATH) as maskFile:
    Mask = array(maskFile["entry/mask/mask"])
    maskFile.close()

with File(CALIBRANT_PATH) as calibFile:
    beam_center_x = array(calibFile["entry1/instrument/detector/beam_center_x"])
    beam_center_y = array(calibFile["entry1/instrument/detector/beam_center_y"])
    x_pixel_size = array(calibFile["entry1/instrument/detector/x_pixel_size"])
    y_pixel_size = array(calibFile["entry1/instrument/detector/y_pixel_size"])
    sample_detector_separation = array(calibFile["entry1/instrument/detector/distance"])
    Wavelength = np.copy(array(calibFile["entry1/calibration_sample/beam/incident_wavelength"]).tolist())/1e+10
    #Wavelength = 9.442870632672332e-11
    calibFile.close()
    
a1 = AzimuthalIntegrator(wavelength = Wavelength)        
a1.setFit2D(directDist = sample_detector_separation, centerX = beam_center_x.item()/x_pixel_size.item(),centerY = beam_center_y.item()/y_pixel_size.item(), pixelX = x_pixel_size.item()*1000,pixelY = y_pixel_size.item()*1000)
 

# === SAXS SLICE DISCOVERY (FOLDER → .h5 FILE) ===
"""
For each background corrected (i.e. adcorr corrected) tomoSAXS slice,
create Ichi_exp library
"""

if len(glob.glob(os.path.join(slice_files, "*.h5")))>0:
    saxs_files = glob.glob(os.path.join(slice_files, "*.h5"))
else:
    saxs_files = glob.glob(slice_files+"/*/", recursive = True)
    
saxs_files = [k for k in saxs_files if k.split("/")[-2][-1].isnumeric()]

slice_saxs_folder = [k for k in saxs_files if k[-3:-1] ==  "_"+str(saxs_slice)][0]

og_output_folder = copy.deepcopy(output_folder)

#binning = int(coreg_files["-BINNING-"])

#for saxs_slice in range(0,len(glob.glob(os.path.join(slice_files, "*.h5")))):
#for saxs_slice in range(0,len(saxs_files)):    
                
slice_saxs_file = glob.glob(os.path.join(slice_saxs_folder, "*.h5"))[0]
#slice_saxs_file = saxs_files[saxs_slice]
#if slice_saxs_file[-2:len(slice_saxs_file)] is not ".h5":
    #slice_saxs_file = glob.glob(os.path.join(slice_saxs_file, "*.h5"))[0]
        
vert_slice = int(slice_saxs_file.split("/")[-2][-1])  
#vert_slice = int(re.findall(r'\d+', vert_slice)[0])

#if og_output_folder == slice_files:
    #if slice_saxs_file[-2:len(slice_saxs_file)] is not ".h5":
        #output_folder = saxs_files[saxs_slice]

output_folder = slice_saxs_file

#SAMPLE_PATH = Path(glob.glob(os.path.join(slice_files, "*.h5"))[saxs_slice])
SAMPLE_PATH = Path(slice_saxs_file)


with File(SAMPLE_PATH) as sample_file:
    entry = list(sample_file.keys())[0]
    sample_file.close()

# === REGISTERED VOXEL INDICES (alpha/beta/fibre/weights/counts) ===
"""
Load registered data
"""

slice_alpha_index = np.copy(np.asarray([k[vert_slice] for k in alpha_data]))
slice_beta_index = np.copy(np.asarray([k[vert_slice] for k in beta_data]))
slice_index_index = np.copy(np.asarray([k[vert_slice] for k in index_data]))
slice_counts_index = np.copy(np.asarray([k[vert_slice] for k in counts_data]))
slice_weights_index = np.copy(np.asarray([k[vert_slice] for k in weights_data]))

slice_scan_data = [slice_alpha_index,slice_beta_index,slice_index_index,slice_counts_index,slice_weights_index]

# === REGISTRATION CHECK: KAPTON EDGE TAGGING ===
"""
Create index for kapton tube edge (for checking registration quality)
index is maximum fibre index plus 100

need to check if this has already been applied to the data
"""

fibre_index = [[slice_index_index[0][i,k] for k in np.arange(0,slice_index_index[0].shape[1],1) if type(slice_index_index[0][i,k]) != int and np.max(slice_index_index[0][i,k]) !=0] for i in np.arange(0,slice_index_index[0].shape[0],1)]

fibreindex = [item for sublist in fibre_index for item in sublist]
#fibreindex = [item for sublist in fibreindex for item in sublist]

if type(fibreindex[0]) == list:
    fibreindex = [item for sublist in fibreindex for item in sublist]
    
if len(np.where(np.asarray(fibreindex) == 500)[0]) == 1:
               
    kapton_edge_ID = np.max([np.max(k) for k in fibreindex])+100

    """
    Registration process labels kapton edge with a 500 - 
    check if this is still present or data has already been corrected 
    """
    
    kapton_edge_vox = [[[j,l] for l in np.arange(0,slice_alpha_index[0].shape[1],1) if slice_alpha_index[0][j,l] == 500] 
                       for j in np.arange(0,slice_alpha_index[0].shape[0],1)]
    
    if len([k for k in kapton_edge_vox if len(k)>0])>0:
    
        kapton_edge_vox = [k for k in kapton_edge_vox if len(k)>0][0][0]
        
        kapton_edge_weight_vox = [[[j,l] for l in np.arange(0,slice_weights_index[0].shape[1],1) if slice_weights_index[0][j,l] == 500] 
                           for j in np.arange(0,slice_weights_index[0].shape[0],1)]
        kapton_edge_weight_vox = [k for k in kapton_edge_weight_vox if len(k)>0][0][0]
        
        if type(slice_alpha_index[0][kapton_edge_vox[0],kapton_edge_vox[1]]) != list:
            slice_alpha_index[0][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
            slice_beta_index[0][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
            slice_index_index[0][kapton_edge_vox[0],kapton_edge_vox[1]] = [kapton_edge_ID]
            
            slice_weights_index[0][kapton_edge_weight_vox[0],kapton_edge_weight_vox[1]] = [[100,kapton_edge_ID]]
            slice_counts_index[0][kapton_edge_weight_vox[0],kapton_edge_weight_vox[1]] = [[100,kapton_edge_ID]]

else:
    
    kapton_edge_ID = np.max([np.max(k) for k in fibreindex])+100
    

alpha_index = [[slice_alpha_index[0][i,k][0:len(slice_alpha_index[0][i,k])] for k in np.arange(0,slice_alpha_index[0].shape[1],1) if type(slice_alpha_index[0][i,k]) != int and len(slice_alpha_index[0][i,k])>0  and np.max(slice_alpha_index[0][i,k]) !=0] for i in np.arange(0,slice_alpha_index[0].shape[0],1)]
#alpha_index = [[slice_alpha_index[0][i,k] for k in np.arange(0,slice_alpha_index[0].shape[1],1) if type(slice_alpha_index[0][i,k]) != "int" and np.max(slice_alpha_index[0][i,k]) !=0] for i in np.arange(0,slice_alpha_index[0].shape[0],1)]
beta_index = [[slice_beta_index[0][i,k][0:len(slice_beta_index[0][i,k])] for k in np.arange(0,slice_beta_index[0].shape[1],1) if type(slice_beta_index[0][i,k]) != int and len(slice_beta_index[0][i,k])>0 and np.max(slice_beta_index[0][i,k]) !=0 and 
               type(slice_alpha_index[0][i,k]) != int and len(slice_alpha_index[0][i,k])>0  and np.max(slice_alpha_index[0][i,k]) !=0] for i in np.arange(0,slice_beta_index[0].shape[0],1)]
#beta_index = [[slice_beta_index[0][i,k] for k in np.arange(0,slice_beta_index[0].shape[1],1) if type(slice_beta_index[0][i,k]) != int and np.max(slice_beta_index[0][i,k]) !=0] for i in np.arange(0,slice_beta_index[0].shape[0],1)]
fibre_index = [[slice_index_index[0][i,k][0:len(slice_index_index[0][i,k])] for k in np.arange(0,slice_index_index[0].shape[1],1) if type(slice_index_index[0][i,k]) != int and len(slice_index_index[0][i,k])>0 and np.max(slice_index_index[0][i,k]) !=0 and 
                type(slice_alpha_index[0][i,k]) != int and len(slice_alpha_index[0][i,k])>0  and np.max(slice_alpha_index[0][i,k]) !=0] for i in np.arange(0,slice_index_index[0].shape[0],1)]
#fibre_index = [[slice_index_index[0][i,k] for k in np.arange(0,slice_index_index[0].shape[1],1) if type(slice_index_index[0][i,k]) != "int" and np.max(slice_index_index[0][i,k]) !=0] for i in np.arange(0,slice_index_index[0].shape[0],1)]
count_index = [[slice_counts_index[0][i,k][0:len(slice_counts_index[0][i,k])] for k in np.arange(0,slice_counts_index[0].shape[1],1) if type(slice_counts_index[0][i,k]) != int and len(slice_counts_index[0][i,k])>0 and np.max(slice_counts_index[0][i,k]) !=0] for i in np.arange(0,slice_counts_index[0].shape[0],1)]
#count_index = [[slice_counts_index[0][i,k] for k in np.arange(0,slice_counts_index[0].shape[1],1) if type(slice_counts_index[0][i,k]) != "int" and np.max(slice_counts_index[0][i,k]) !=0] for i in np.arange(0,slice_counts_index[0].shape[0],1)]
weight_index = [[slice_weights_index[0][i,k][0:len(slice_weights_index[0][i,k])] for k in np.arange(0,slice_weights_index[0].shape[1],1) if type(slice_weights_index[0][i,k]) != int and len(slice_weights_index[0][i,k])>0 and np.max(slice_weights_index[0][i,k]) !=0] for i in np.arange(0,slice_weights_index[0].shape[0],1)]    
#weight_index = [[slice_weights_index[0][i,k] for k in np.arange(0,slice_weights_index[0].shape[1],1) if type(slice_weights_index[0][i,k]) != "int" and np.max(slice_weights_index[0][i,k]) !=0] for i in np.arange(0,slice_weights_index[0].shape[0],1)]    
vox_index = [[[i,k] for k in np.arange(0,slice_alpha_index[0].shape[1],1) if type(slice_alpha_index[0][i,k]) != int and len(slice_alpha_index[0][i,k])>0 and np.max(slice_alpha_index[0][i,k]) !=0] for i in np.arange(0,slice_alpha_index[0].shape[0],1)]
            
# === FLATTEN LIST-OF-LISTS TO 1D ARRAYS (CONVENIENCE) ===
alphaindex = [item for sublist in alpha_index for item in sublist]
betaindex = [item for sublist in beta_index for item in sublist]
fibreindex = [item for sublist in fibre_index for item in sublist]
countindex = [item for sublist in count_index for item in sublist]
weightindex = [item for sublist in weight_index for item in sublist]
voxindex = [item for sublist in vox_index for item in sublist]
    
fibreIndex = [item for sublist in fibreindex for item in sublist]
alphaIndex = [item for sublist in alphaindex for item in sublist]
betaIndex = [item for sublist in betaindex for item in sublist]
countIndex = [item for sublist in countindex for item in sublist]
weightIndex = [item for sublist in weightindex for item in sublist]
voxIndex = [item for sublist in voxindex for item in sublist]

# === FIBRIL LIBRARY INITIALISATION ===
"""
Create empty library of scattering fibres 
"""

Nfibrils = len(fibreIndex)

fibrilParams = [None]*Nfibrils
print("InitialiseFibrilParameters")
for i in range(Nfibrils):
    fibrilParams[i]={}
    fibrilParams[i]=InitialiseFibrilParameters(i)
    
# === SCAN GEOMETRY & BEAM PATHS (SAVED TO PICKLE) ===
"""
Need the above data for creating scan parameters and beampath objects
"""
Nx, Ny = slice_alpha_index[0].shape[0], 1
Nz = slice_alpha_index[0].shape[1]

x0,y0,z0 = 0,0,0
eps=1e-3
samplex1, samplex2, Nxf = -.6,.6,Nx
sampley1, sampley2, Nyf = -.1,.1,Ny
sampley1, sampley2, Nyf = .1,.1,Ny
samplez1, samplez2, Nzf = -.6,.6,Nz

sample_xw, sample_zw = samplex2-samplex1, samplez2-samplez1
sample_xc, sample_zc = (samplex2+samplex1)/2.0, (samplez2+samplez1)/2.0

dxs, dys, dzs = (samplex2-samplex1)/Nx,(sampley2-sampley1)/Ny,(samplez2-samplez1)/Nz

fib_sample_data = [dxs,samplex1,dzs,samplez1]
    
scany1,scany2,dy = sampley1, sampley2, dys

scanx1,scanx2,dx = samplex1,samplex2, dxs

scanParameters = {"xstart": scanx1, "xend": scanx2, "xstep": dx,
                  "ystart": scany1, "yend": scany2, "ystep": dy}

r = 0
length = 2

beampath = np.linspace(scanx1,scanx2+eps,Nx)

beam_size = dx
beamPaths = findBeamPaths(scanParameters,length=length,rot=r,\
                          display=False,beamsize=dx)#,bp_count = Nx)
    
with open(output_folder+"slice_"+str(vert_slice)+"_beampaths.pkl", 'wb') as f:
    pickle.dump(beamPaths, f)
        

# === MAP FIBRE IDS ↔︎ VOXELS; FIX MISSING alpha/beta ===
"""
Allocate fibre coordinates
"""

idx_test = [[[i,k] for k in np.arange(0,len(beta_index[i]),1) if type(beta_index[i][k]) == list and len(beta_index[i][k])!=len(alpha_index[i][k])]
            for i in np.arange(0,len(fibre_index),1)]

idx_test = [k for k in idx_test if len(k)!=0]

if len(idx_test)>0:
    
    idx_test = idx_test[0]   
    
    for idx in range(0,len(idx_test)):
        
        idx_i,idx_k = idx_test[idx][0],idx_test[idx][1]
        
        if len(vox_index[idx_i][idx_k]) == len(beta_index[idx_i][idx_k]):
            alpha_val = alpha_index[idx_i][idx_k][0]
            
            alpha_index[idx_i][idx_k] = [alpha_val,alpha_val]
            
        else:
            
            beta_val = beta_index[idx_i][idx_k][0]
            
            beta_index[idx_i][idx_k] = [beta_val,beta_val]
        
        
        #if idx_test[idx][0] == idx_test[idx+1][0] and idx_test[idx][1] == (idx_test[idx+1][1] - 1):
            
            #idx_i,idx_k = idx_test[idx][0],idx_test[idx][1]
            
            #if len(beta_index[idx_i][idx_k])>len(alpha_index[idx_i][idx_k]):
                                                
                #beta_index[idx_i][idx_k+1] = [beta_index[idx_i][idx_k][1]] + beta_index[idx_i][idx_k+1]
                #beta_index[idx_i][idx_k] = [beta_index[idx_i][idx_k][0]]
                
            #else:
                
                #beta_index[idx_i][idx_k] = beta_index[idx_i][idx_k] + [beta_index[idx_i][idx_k+1][0]]
                #beta_index[idx_i][idx_k+1] = [beta_index[idx_i][idx_k+1][1]]
                
                
                
    
    
fibre_voxels = [[[vox_index[i][k] for k in np.arange(0,len(vox_index[i]),1) if vox in fibre_index[i][k]] for i in np.arange(0,len(fibre_index),1)]for vox in fibreIndex]
fibre_voxels = [[k[0] for k in vox if len(k)>0] for vox in fibre_voxels]    
fibre_voxels = np.asarray([k[0] for k in fibre_voxels if len(k)>0] )  

fibre_alphas = [[[[alpha_index[i][k][l] for l in np.arange(0,len(alpha_index[i][k]),1) if fibre_index[i][k][l] == vox] 
                 for k in np.arange(0,len(vox_index[i]),1)] for i in np.arange(0,len(fibre_index),1)] 
               for vox in fibreIndex]
fibre_alphas = [[[fibre_alphas[i][k][l][0] for l in np.arange(0,len(fibre_alphas[i][k]),1) if len(fibre_alphas[i][k][l])>0] 
                for k in np.arange(0,len(fibre_alphas[i]),1)] for i in np.arange(0,len(fibre_alphas),1)]    
fibre_alphas = [[fibre_alphas[i][k][0] for k in np.arange(0,len(fibre_alphas[i]),1) if len(fibre_alphas[i][k])>0] 
               for i in np.arange(0,len(fibre_alphas),1)]
fibre_alphas = np.asarray([k[0] for k in fibre_alphas])

fibre_betas = [[[[beta_index[i][k][l] for l in np.arange(0,len(beta_index[i][k]),1) if fibre_index[i][k][l] == vox] 
                 for k in np.arange(0,len(vox_index[i]),1)] for i in np.arange(0,len(fibre_index),1)] 
               for vox in fibreIndex]
fibre_betas = [[[fibre_betas[i][k][l][0] for l in np.arange(0,len(fibre_betas[i][k]),1) if len(fibre_betas[i][k][l])>0] 
                for k in np.arange(0,len(fibre_betas[i]),1)] for i in np.arange(0,len(fibre_betas),1)]    
fibre_betas = [[fibre_betas[i][k][0] for k in np.arange(0,len(fibre_betas[i]),1) if len(fibre_betas[i][k])>0] 
               for i in np.arange(0,len(fibre_betas),1)]
fibre_betas = np.asarray([k[0] for k in fibre_betas])

fibre_indexes = [[[[fibre_index[i][k][l] for l in np.arange(0,len(fibre_index[i][k]),1) if fibre_index[i][k][l] == vox] 
                 for k in np.arange(0,len(vox_index[i]),1)] for i in np.arange(0,len(fibre_index),1)] 
               for vox in fibreIndex]
fibre_indexes = [[[fibre_indexes[i][k][l][0] for l in np.arange(0,len(fibre_indexes[i][k]),1) if len(fibre_indexes[i][k][l])>0] 
                for k in np.arange(0,len(fibre_indexes[i]),1)] for i in np.arange(0,len(fibre_indexes),1)]    
fibre_indexes = [[fibre_indexes[i][k][0] for k in np.arange(0,len(fibre_indexes[i]),1) if len(fibre_indexes[i][k])>0] 
               for i in np.arange(0,len(fibre_indexes),1)]
fibre_indexes = np.asarray([k[0] for k in fibre_indexes])


"""
Adjust betas and alphas beyond tomoSAXS sampling (e.g. alpha>90)
"""
fibre_alphas = np.copy(fibre_alphas)
fibre_betas = np.copy(fibre_betas)
fibre_alphas = np.where(fibre_alphas<90,fibre_alphas+90,90+(180-fibre_alphas))
#fibre_alphas = fibre_alphas+90
#fibre_betas = fibre_betas-20


xTest = []
zTest = []
fib_idxs = []
fib_alphas = []
fib_betas = []
fib_q0s,fib_was,fib_wMus,fib_amps = [],[],[],[]
fib_weights = []
for i in range(0,len(fibre_voxels)):
    indx = int(fibreIndex[i])
    #if indx !=
    fib_idxs.append(indx)
    fibrilParams[i]["indx"] = fibre_indexes[i]
    fibrilParams[i]["number"] = i
    fibrilParams[i]["x"]=fibre_voxels[i][1]*dxs+samplex1
    xTest.append(fibre_voxels[i][1])#*dxs+samplex1)
    fibrilParams[i]["x_full"] = fibre_voxels[i][1]
    fibrilParams[i]["y"]=0.1
    fibrilParams[i]["z"]=fibre_voxels[i][0]*dzs+samplez1
    zTest.append(fibre_voxels[i][0])#*dzs+samplez1)
    fibrilParams[i]["z_full"] = fibre_voxels[i][0]
    fib_alpha = fibre_alphas[i]#+20
    fibrilParams[i]["alpha"] = fib_alpha
    fib_alphas.append(fib_alpha)
    fib_beta = fibre_betas[i]+rot_range[0]
    if fib_beta<0:
        fib_beta = 360 + fib_beta
    if fib_beta>360:
        fib_beta = fib_beta-360
    fibrilParams[i]["beta"] = fib_beta
    fib_betas.append(fib_beta)
    fibrilParams[i]["initial_amp_est"] = 1
    
    """
    stand-in values for count and weights for now (mean of all vals for first rotation)
    """
    fibrilParams[i]["weight"] = np.mean([k[0] for k in weightIndex if k[1] == indx])
    fib_weights.append(np.mean([k[0] for k in weightIndex if k[1] == indx]))
    fibrilParams[i]["count"] = np.mean([k[0] for k in weightIndex if k[1] == indx])
    
    fibrilParams[i]["simu"]["q0"] = getVal(q0_m,dq0_m,noiseterm=noiseterm)
    fib_q0s.append(fibrilParams[i]["simu"]["q0"])
    fibrilParams[i]["simu"]["wa"] = getVal(wa_m,dwa_m,noiseterm=noiseterm)
    if fibrilParams[i]["simu"]["wa"]<0:
        fibrilParams[i]["simu"]["wa"]=wa_m-dwa_m
    fib_was.append(fibrilParams[i]["simu"]["wa"])
    
    fibrilParams[i]["simu"]["wMu"] = getVal(wMu_m,dwMu_m,noiseterm=noiseterm)
    if fibrilParams[i]["simu"]["wMu"]<0:
        fibrilParams[i]["simu"]["wMu"]=wMu_m-dwMu_m
    fib_wMus.append(fibrilParams[i]["simu"]["wMu"])
    fibrilParams[i]["simu"]["amp"] = getVal(amp_m,damp_m,noiseterm=noiseterm)
    fib_amps.append(fibrilParams[i]["simu"]["amp"])
    fibrilParams[i]["simu"]["delta"] = getVal(delta_m,ddelta_m,noiseterm=noiseterm)           
    if fibrilParams[i]["simu"]["delta"]<0:
        fibrilParams[i]["simu"]["delta"]=0.0001
    if fibrilParams[i]["simu"]["delta"]>1:
        fibrilParams[i]["simu"]["delta"]=0.99999
        

with open(output_folder+"slice_"+str(vert_slice)+"_90-to-270_fibril_params_raw.pkl", 'wb') as f:
    pickle.dump(fibrilParams, f)
    
with open(output_folder+"slice_"+str(vert_slice)+"_ichi_dict_cake_params.pkl", 'wb') as f:
    pickle.dump(cake_params_chi, f)
        
    
params0 = {"q0": q0_m, "wa": wa_m, "wMu": wMu_m, "delta": delta_m}

#dict_chi_range = [-180,0]
dict_chi_range = [90,270]

ncpus = cpu_count()

pyFai_params = [Mask,cake_params_chi,Wavelength,sample_detector_separation,beam_center_x.item()/x_pixel_size.item(),
                beam_center_y.item()/y_pixel_size.item(),x_pixel_size.item()*1000,y_pixel_size.item()*1000,r,
                cake_params_test,dict_chi_range,kapton_edge_ID]

ichiExp = {}

max_int = []

rotated_beampaths = []

               
for r in range(0,len(rot_range)):
#for r in range(0,2):
    ichiExp[r]={"rotation":r}
    
    bps,path_dicts,saxs_file_id,params = [],[],[],[]
    
    if r>0:                
        rotation = rot_range[r]
        r_index = r                
        rotatedFibrilParams,rotCoords = getRotatedFibrilPoints(fibrilParams,rotation,r_index,slice_scan_data,fib_sample_data)
    else:
        rotatedFibrilParams = np.copy(fibrilParams)
        
    voxelsPerPath = assignVoxelsPerPath(beamPaths,rotatedFibrilParams,slice_weights_index[r],beamradius=beam_size/2.0,sig=1)
    rotated_beampaths.append(voxelsPerPath)
                
    sample_bps = []
    
    for i, path_dict in enumerate(voxelsPerPath):
        if path_dict != None:
                                            
            if path_dict["voxels"] != None:
                bps.append(i)
                path_dicts.append(path_dict)
                saxs_file_id.append(slice_saxs_file)
                params.append(pyFai_params)
                
    if __name__ == '__main__':
        #os.chdir(scanFiles['-SCRIPTFOLDER-']+"/")
        #import testFunc3_v2
        with Pool(ncpus) as p:
            test = p.starmap(testFunc, zip(bps,path_dicts,saxs_file_id,params))
            
    fibre_bps = np.asarray([k[0] for k in test])
            
    for idx,i, in enumerate(fibre_bps):
                        
        max_int.append(test[idx][3])
                                     
        if type(test[idx][1]["kapton"]) != bool:
            ichiExp[r][i]={"bp": test[idx][1]["bp"],"ichi":test[idx][1]["ichi"],"kapton":test[idx][1]["kapton"],
                           "frame": test[idx][1]["frame"]}
        else:
            ichiExp[r][i]={"bp": test[idx][1]["bp"],"ichi":test[idx][1]["ichi"],"kapton":test[idx][1]["kapton"],
                           "iq": test[idx][1]["iq"]}
        
        sample_bps.append(test[idx][2])
        
"""
Save this version of the dictionary in case we want to start the proceeding 
actions from scratch
"""
with open(output_folder+"slice_"+str(vert_slice)+"_90-to-270_ichi_dict_raw.pkl", 'wb') as f:
    pickle.dump(ichiExp, f) 

with open(output_folder+"slice_"+str(vert_slice)+"_mask_cake_params.pkl", 'wb') as f:
    pickle.dump(cake_params_test, f)         

"""
Integrate over mask to identify masked regions in I(chi) data
"""

if dict_chi_range !=[-180,0]:
    
    nv_start = dict_chi_range[0]
    nv_end = 180
    
    pv_start = -180
    pv_end = dict_chi_range[1]-360
    
    
if dict_chi_range !=[-180,0]:

    mask_nv90_1 = ichi_sample(r,Mask,0,0,cake_params_test,a1,0,slice_saxs_file,[nv_start,nv_end],
                          fibre_chi = False,fit_chi=False,iq_plot = True)
    
    mask_nv90_2 = ichi_sample(r,Mask,0,0,cake_params_test,a1,0,slice_saxs_file,[pv_start,pv_end],
                          fibre_chi = False,fit_chi=False,iq_plot = True)
    
    mask_I = np.flip(np.concatenate((mask_nv90_1[0][1],mask_nv90_2[0][1])))
    
else:
    mask_I = np.flip(ichi_sample(r,Mask,0,0,cake_params_test,a1,0,slice_saxs_file,[dict_chi_range[0],dict_chi_range[1]],
                          fibre_chi = False,fit_chi=False,iq_plot = True)[0][1])
            
chi = np.linspace(dict_chi_range[0],dict_chi_range[1],len(mask_I)+1)

mask_binned = bin_data(mask_I,bin_val)

no_masks = np.where(np.diff(mask_I)==0)[0]
mask_bounds = np.where(np.diff(no_masks)>1)[0]
mask_bounds = [[no_masks[k]+1,no_masks[k+1]] for k in mask_bounds]

no_masks_bin = np.where(np.diff(mask_binned)==0)[0]
mask_bounds_bin = np.where(np.diff(no_masks_bin)>1)[0]
mask_bounds_bin = [[no_masks_bin[k]+1,no_masks_bin[k+1]] for k in mask_bounds_bin]
    
data_bounds = [np.arange(mask_bounds[k][1],mask_bounds[k+1][0],1).tolist() for k in np.arange(0,len(mask_bounds)-1,1)]
data_bounds = [np.arange(0,mask_bounds[0][0],1).tolist()]+data_bounds+[np.arange(mask_bounds[-1][1],len(mask_I),1).tolist()]
binned_bounds = np.asarray([bin_data(np.asarray(k),bin_val) for k in data_bounds])

just_data = [item for sublist in data_bounds for item in sublist]
just_binned_data = [item for sublist in binned_bounds for item in sublist]
data_chi = [chi[k] for k in data_bounds]

chi_binned = np.asarray([bin_data(k,bin_val) for k in data_chi])


np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_data_bounds.npy",data_bounds)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_unmasked_I.npy",just_data)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_unmasked_chi.npy",data_chi)

np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_bin_bounds.npy",binned_bounds)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_unmasked_binned_I.npy",just_binned_data)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_unmasked_binned_chi.npy",chi_binned)


mask_lens = [k[1]-k[0] for k in mask_bounds]
mask_lens_bin = [k[1]-k[0] for k in mask_bounds_bin]
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_mask_ichi.npy",mask_I)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_mask_bounds.npy",mask_bounds)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_masks.npy",mask_lens)

np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_mask_ichi_bin.npy",mask_binned)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_mask_bounds_bin.npy",mask_bounds_bin)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_masks_bin.npy",mask_lens_bin)

masked_lens,bin_lens = [],[]
masked_len_sums,bin_len_sums = [],[]
masked_maxs,bin_maxs = [],[]
rot_masked_lens,rot_bin_lens = [],[]
rot_masked_maxs,rot_bin_maxs = [],[]

"""
Fill I(chi) dictionary
    a.I(chi) data
    b. Mask true/false array
    c. masked I(chi) array - linear integration across deleted mask regions
    d. binned I(chi) array
    e. smoothed I(chi)
    
Also create arrays outlining the length of each I(chi) dataset 
(allows flexible formatting of SVD arrays in SVD module)
"""

tths,Is = [],[]
    
for r in range(0,len(rot_range)):
    
    rot_mask_len = []
    rot_mask_max = []
    
    rot_bin_len = []
    rot_bin_max = []
    
    for i in range(0,len(voxelsPerPath)):
        if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:  
            ichi=ichiExp[r][i]["ichi"]
            raw_smooth = savgol_filter(ichi,window,pol)
            #ichi_mask_bounds = [[np.mean(ichi[k[0]-3:k[0]]),np.mean(ichi[k[1]:k[1]+3])] for k in mask_bounds]
            #mask_vals = [np.linspace(ichi_mask_bounds[k][0],ichi_mask_bounds[k][1],mask_lens[k]) for k in np.arange(0,len(mask_lens),1)]
            #ichi_masked = np.copy(ichi)
            #for m in range(0,len(mask_vals)):
                #ichi_masked[mask_bounds[m][0]:mask_bounds[m][1]] = mask_vals[m]
                
            data_masked = [ichi[k] for k in data_bounds]
            ichi_masked = np.asarray([item for sublist in data_masked for item in sublist])  
            data_smooth = [savgol_filter(k,window,pol) for k in data_masked]
            smpl_smooth = np.asarray([item for sublist in data_smooth for item in sublist])   
            #smpl_smooth = savgol_filter(ichi_masked,window,pol)    
            data_binned = np.asarray([bin_data(k,bin_val) for k in data_masked])
            ichi_binned = np.asarray([item for sublist in data_binned for item in sublist])
            
            smooth_bins = [savgol_filter(k,3,pol) for k in data_binned]
            smoothed_binned = np.asarray([item for sublist in smooth_bins for item in sublist])
            
            ichi_binned = bin_data(ichi_masked,bin_val)   
                            
            ichi_mask = np.where(mask_I==0,True,False)
            ichiExp[r][i].update({"mask":ichi_mask})
            ichiExp[r][i].update({"ichi_masked": ichi_masked})
            ichiExp[r][i].update({"ichi_binned": smoothed_binned})
            ichiExp[r][i].update({"ichi_smoothed": smpl_smooth})
            
            ichiExp[r][i].update({"full_indx": np.sum(masked_lens)})
            ichiExp[r][i].update({"bin_indx": np.sum(bin_lens)})
            masked_len_sums.append(np.sum(masked_lens))
            bin_len_sums.append(np.sum(bin_lens))
            #ichiExp[r][i]["ichi"] = ichi
            masked_lens.append(len(smpl_smooth))
            bin_lens.append(len(smoothed_binned))
            
            masked_maxs.append(np.max(smpl_smooth))
            bin_maxs.append(np.max(smoothed_binned))
            
            rot_mask_len.append(len(smpl_smooth))
            rot_bin_len.append(len(smoothed_binned))
            
            rot_mask_max.append(np.max(smpl_smooth))
            rot_bin_max.append(np.max(smoothed_binned))
            
            tths.append(ichiExp[r][i]["iq"][0])
            Is.append(ichiExp[r][i]["iq"][1])
            
    rot_masked_lens.append(rot_mask_len)
    rot_masked_maxs.append(rot_mask_max)
    
    rot_bin_lens.append(rot_bin_len)
    rot_bin_maxs.append(rot_bin_max)
    
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_masked_lens.npy",masked_lens)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_bin_lens.npy",bin_lens)
with open(output_folder+"slice_"+str(vert_slice)+"_90-to-270_ichi_dict_filled.pkl", 'wb') as f:
    pickle.dump(ichiExp, f)

q0i_low,q0i_high = cake_params_chi["q1i"],cake_params_chi["q2i"]
q0o_low,q0o_high = cake_params_chi["q1o"],cake_params_chi["q2o"]

q0c_low,q0c_high = cake_params_chi["q1"],cake_params_chi["q2"]

peak_bg_ints = [np.linspace(Is[k][find_nearest(tths[k],q0i_high)[1]],Is[k][find_nearest(tths[k],q0o_low)[1]],(find_nearest(tths[k],q0o_low)[1]-find_nearest(tths[k],q0i_high)[1])+1).tolist() for k in np.arange(0,len(Is),1)]
peak_bg_Qs = [np.linspace(tths[k][find_nearest(tths[k],q0i_high)[1]],tths[k][find_nearest(tths[k],q0o_low)[1]],(find_nearest(tths[k],q0o_low)[1]-find_nearest(tths[k],q0i_high)[1])+1).tolist() for k in np.arange(0,len(Is),1)]

bgIs = [Is[k][find_nearest(tths[k],q0i_low)[1]:find_nearest(tths[k],q0i_high)[1]].tolist()+peak_bg_ints[k]+Is[k][find_nearest(tths[k],q0o_low)[1]:find_nearest(tths[k],q0o_high)[1]].tolist() for k in np.arange(0,len(Is),1)]
bgQs = [tths[k][find_nearest(tths[k],q0i_low)[1]:find_nearest(tths[k],q0i_high)[1]].tolist()+peak_bg_Qs[k]+tths[k][find_nearest(tths[k],q0o_low)[1]:find_nearest(tths[k],q0o_high)[1]].tolist() for k in np.arange(0,len(Is),1)]

bgIs = [k-np.min(k) for k in bgIs]

flat_models = [remove_bg(bgIs[k],bgQs[k],tths[k],exp = False) for k in np.arange(0,len(Is),1) ]

flattened_iqs = [Is[k] - flat_models[k] for k in np.arange(0,len(Is),1)] 

#test = []

q0s = np.zeros((len(flattened_iqs)))
was = np.zeros((len(flattened_iqs)))

qAxis = tths[0]

flatModels = []

for i in range(0,len(flattened_iqs)):
    
    flatModel = flattened_iqs[i]
    
    gaussModel = SkewedGaussianModel()
    constantModel = ConstantModel() 
                                              
    pars = gaussModel.guess(flatModel, qAxis) 
    cPars = constantModel.guess(flatModel, qAxis)
    pars.add(cPars["c"])
    gaussModel = SkewedGaussianModel()+ConstantModel()  
    
    result = gaussModel.fit(flatModel, pars, x=qAxis)
    model = gaussModel.eval(result.params,x=qAxis)
    gaussParams = result.params
    
    if gaussParams["center"].value>cake_params_test["q1i"]:
        if gaussParams["center"].value<cake_params_test["q2o"] and gaussParams["sigma"].value<0.01:
            q0s[i] = gaussParams["center"].value
            was[i] = gaussParams["sigma"].value    
            
            flatModels.append([model,flatModel])
            
q0_mode = plt.hist(q0s[q0s>0],100) 
modal_q0 = q0_mode[1][np.argmax(q0_mode[0])]

wa_mode = plt.hist(was[was>0],bins=10)
wa_mode = wa_mode[1][np.argmax(wa_mode[0])]
wa_mean = np.mean(was[was>0])

initStruct = {"q0"    : {"m": modal_q0, "sd": dq0_m},
              "q0_min" : {"m": np.min(q0s[q0s>0]),"sd":dq0_m},
              "q0_max" : {"m": np.max(q0s[q0s>0]),"sd":dq0_m},
              "wMu"   : {"m": wMu_m, "sd": dwMu_m},
              "wa"    : {"m": wa_mean, "sd": dwa_m},
              "wa_min" : {"m": np.min(was[was>0]),"sd":dq0_m},
              "wa_max" : {"m": np.max(was[was>0]),"sd":dq0_m},
              "delta" : {"m": delta_m, "sd": ddelta_m},
              "amp"   : {"m": amp_m, "sd": damp_m},
              "alpha" : {"m": alpha_m, "sd": dalpha_m},
              "beta"  : {"m": beta_m, "sd": dbeta_m}
             }


with open(output_folder+"slice_"+str(vert_slice)+"_init_structure.pkl", 'wb') as f:
    pickle.dump(initStruct, f)
    
rotated_beampaths = []
for ridx, r in enumerate(rot_range):
    # ridx goes from 1 to M
    print("ridx: ", ridx," angle: ", r)
    
    if ridx>0:
        
        rotation = rot_range[ridx]
        r_index = ridx
        rfPars = getRotatedFibrilPoints(fibrilParams,rotation,r_index,slice_scan_data,fib_sample_data)[0]
    else:
        rfPars = np.copy(fibrilParams)
    
    voxelsPerPath = assignVoxelsPerPath(beamPaths,rfPars,slice_weights_index[ridx],beamradius=beam_size/2.0,sig=1)
    rotated_beampaths.append(voxelsPerPath)

with open(output_folder+"slice_"+str(vert_slice)+"_rotation_beampaths.pkl", 'wb') as f:
    pickle.dump(rotated_beampaths, f)
    
with open(output_folder+"slice_"+str(vert_slice)+"_svd_cake_params.pkl", 'wb') as f:
    pickle.dump(cake_params_chi, f)
    
chi_s_svd, chi1_svd, chi2_svd, n_chi_svd = chirange,chi1,chi2,nchi

chis_bin = np.asarray([item for sublist in chi_binned for item in sublist])

chi1_bin,chi2_bin,n_chi_bin = chi1_svd, chi2_svd, len(chis_bin)
                    
b_svd_arr = [0.0]*np.sum(masked_lens)
b_svd_bin = [0.0]*np.sum(bin_lens)
            
b_svd_arr,b_svd_bin  = np.asarray(b_svd_arr),np.asarray(b_svd_bin)
                
q0_m, wa_m, wMu_m, delta_m = initStruct["q0"]["m"], initStruct["wa"]["m"], initStruct["wMu"]["m"], initStruct["delta"]["m"]

params0 = {"q0": q0_m, "wa": wa_m, "wMu": 0.2, "delta": 0.5}

simu_peaks_lims = []
all_simu_peaks_lims = []

for ridx, r in enumerate(rot_range):
    # ridx goes from 1 to M
    print("ridx: ", ridx," angle: ", r)
    
    rot_beampath = rotated_beampaths[ridx]
    
    for i, pdict in enumerate(rot_beampath):
        """
        i goes from 1 to N
        """
        if pdict != None:
            if pdict["voxels"]!=None:
                """
                Only add the experimental Ichi for this path into it if the beam intersects at least one fibre
                """
                chi_lens = []
                chi_lens_bin = []
                
                if i in ichiExp[ridx] and type(ichiExp[ridx][i]["kapton"]) == bool:
                    
                    pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in pdict["voxels"]])
                    if len(pdict_vox_indxs)>1:
                        vox_repeats = [k for k in np.arange(1,len(pdict_vox_indxs),1) if pdict_vox_indxs[k]==pdict_vox_indxs[k-1]]
                        if len(vox_repeats)>0:
                            pdict["voxels"] = [pdict["voxels"][k] for k in np.arange(0,len(pdict["voxels"]),1) if k not in vox_repeats]
                            pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in pdict["voxels"]])
                    #sample_bps.append(i)
                                           

                    ichi=ichiExp[ridx][i]["ichi_smoothed"]
                    ichi_bin = ichiExp[ridx][i]["ichi_binned"]
                    
                    chi_len = len(ichi)   
                    bin_len = len(ichi_bin)   
                    
                    chi_lens.append(chi_len)
                    chi_lens_bin.append(bin_len)                            
                  
                    full_indx = int(ichiExp[ridx][i]["full_indx"])    
                    bin_indx = int(ichiExp[ridx][i]["bin_indx"])
                    
                    indexArr0 = np.linspace(0,chi_len-1,chi_len).astype(int)
                    binArr0 = np.linspace(0,bin_len-1,bin_len).astype(int)
                                        
                    indexArrTr0 = np.asarray([indexArr0]).T
                    binArrTr0 = np.asarray([binArr0]).T
                                            
                    ichiTr = np.asarray([ichi]).T
                    ichiBinTr = np.asarray(ichi_bin).T
                    
                    b_svd_tmp = np.zeros_like(b_svd_arr[full_indx:full_indx+chi_len])
                    np.put(b_svd_tmp,indexArrTr0,ichiTr)
                    b_svd_arr[full_indx:full_indx+chi_len] = b_svd_arr[full_indx:full_indx+chi_len]+b_svd_tmp
                    
                    b_bin_tmp = np.zeros_like(b_svd_bin[bin_indx:bin_indx+bin_len])
                    np.put(b_bin_tmp,binArrTr0,ichiBinTr)
                    b_svd_bin[bin_indx:bin_indx+bin_len] = b_svd_bin[bin_indx:bin_indx+bin_len]+b_bin_tmp
                    
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_b_svd_Arr.npy",b_svd_arr)
np.save(output_folder+"slice_"+str(vert_slice)+"_90-to-270_b_svd_Arr_bin.npy",b_svd_bin) 
