"""
Reading Guide
-------------
- PATHS & SLICE DISCOVERY: selects the tomoSAXS slice and prepares output dirs.
- PREREQUISITES: loads index matrices, mask/calibration, and configures pyFAI.
- OBJECT DISCOVERY: locates precomputed pickles/arrays needed for the run.
- RUNTIME CONFIG: flags for single/combi/multi fits, rotation range, etc.
- INITIALISATION: loads fibril library, beam paths, I(chi), and cake params.
- FIT BOUNDS: sets parameter ranges and thresholds for acceptance/relaxation.
- PER-SLICE SETUP: builds cake mask and sampling geometry.
- PHASE 1–5: fitting passes from strict single-fibre through relaxed loops with
  amplitude bootstrapping; each phase saves intermediate results.

Note
----
Most heavy lifting is delegated to `recon_library_final` (rec_lib) and
`threeDXRD_080923` (t3d). This file orchestrates iterations, IO, and thresholds.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:22:56 2025

Top-level script for tomoSAXS reconstruction of fibrillar properties (q0; wMu;wa;delta)
operates following:
    1. Isolate and fit all scattering instances that can be solved for single fibres by sampling regions of chi that are 
        independent for the respective fibre, saving fits to the "fibrilParams" and "rotated_beampaths" objects.
    2. Perform both single fibre fits and double fibre fits for all isolatable scattering instances over 10 loops of the 
        tomoSAXS scan.
    3. If there are still unfitted fibres remaining, relax the fit quality criteria and repeat step 2 (assuming that by now sufficient numbers 
        of fibres have been fitted to provide accurate estimation of the observed scatter to allow for low intensity scattering
        instances to be fitted).
    4. IF there are still unfitted fibres remaining - add fibres with no amplitude estimates (give them the mean fitted amplitude)
        and repeat relaxed estimation.

@author: Elis Newham
"""

# === IMPORTS ===
import sys

import numpy as np
import os,glob
import pickle
from time import time
import copy
import matplotlib.pyplot as plt
import re
from pathlib import Path
from h5py import File
from numpy import array, ones
import hdf5plugin

# === PATHS & SLICE DISCOVERY ===
input_folder = '/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded'

path = input_folder+"/"
output_path = path

saxs_slice = 1
subfolders = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
subfolders = [k for k in subfolders if k[-1].isnumeric()]
subfolders = [k for k in subfolders if "5x5" not in k]

folder = [k for k in subfolders if int(k.split("_")[-1]) == saxs_slice][0]

data_path = path+folder+"/"
output_path = data_path+"final_recon/"

# === OUTPUT FOLDERS (create if missing) ===
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
if os.path.isdir(output_path+"/full smooth test fits") == False:
    os.mkdir(output_path+"/full smooth test fits")


# (Additional result subfolders)
if os.path.isdir(output_path+"accepted_single_fits/") == False:
    os.mkdir(output_path+"accepted_single_fits/")
if os.path.isdir(output_path+"unaccepted_single_fits/") == False:
    os.mkdir(output_path+"unaccepted_single_fits/")
    
if os.path.isdir(output_path+"accepted_multi_fits/") == False:
    os.mkdir(output_path+"accepted_multi_fits/")
if os.path.isdir(output_path+"unaccepted_multi_fits/") == False:
    os.mkdir(output_path+"unaccepted_multi_fits/")

# === TIMER & RUN METADATA ===
recon_start_time = time()
np.save(output_path+"recon_start_time.npy",[recon_start_time])

# === CUSTOM LIBRARIES (recon + 3DXRD) ===
sys.path.append(r'/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/scripts/recon_final/')
import recon_library_final as rec_lib
sys.path.append(r'/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/scripts')
import threeDXRD_080923 as t3d

from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator

import logging
from datetime import datetime


# === REQUIRED FILES (indices, mask, calibration) ===
alpha_file = path+'/full_alpha_index_matrices.npy'
beta_file = path+'/full_beta_index_matrices.npy'
index_file = path+'/full_fibre_index_matrices.npy'

mask_file = '/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline/processing/SAXS_mask.nxs'
calib_file = '/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline/processing/SAXS_calibration.nxs'

binning = 1


     
# === UTILITY: find the index of a specific value in a sequence ===
def findindex(data,val):   #find index of a yexp value which equals localmin
    rval = -1
    for i in range(len(data)):
        if data[i] == val:
           rval = i
    return rval


# === LIVE PLOTTING (optional) ===
"""
Parameters for live plotting if enabled
"""
tick_label_size = 8
#set the tick frequency of X and y axes
tick_spacing_x, tick_spacing_y = 50, 50
width, height = 10.0,12.5


# === PREREQUISITES: load indices, mask/calibration; init pyFAI ===
"""
load prerequisite data
"""

alpha_data = np.load(alpha_file,allow_pickle = True)
index_data = np.load(index_file,allow_pickle = True)


MASK_PATH = Path(mask_file)
CALIBRANT_PATH = Path(calib_file)

with File(MASK_PATH) as maskFile:
    Mask = array(maskFile["entry/mask/mask"])
    maskFile.close()

with File(CALIBRANT_PATH) as calib_file:
    beam_center_x = array(calib_file["entry1/instrument/detector/beam_center_x"])
    beam_center_y = array(calib_file["entry1/instrument/detector/beam_center_y"])
    x_pixel_size = array(calib_file["entry1/instrument/detector/x_pixel_size"])
    y_pixel_size = array(calib_file["entry1/instrument/detector/y_pixel_size"])
    sample_detector_separation = array(calib_file["entry1/instrument/detector/distance"])
    Wavelength = np.copy(array(calib_file["entry1/calibration_sample/beam/incident_wavelength"]).tolist())/1e+10
    #Wavelength = 9.442870632672332e-11
    calib_file.close()

a1 = AzimuthalIntegrator(wavelength = Wavelength)        
a1.setFit2D(directDist = sample_detector_separation, centerX = beam_center_x.item()/x_pixel_size.item(),centerY = beam_center_y.item()/y_pixel_size.item(), pixelX = x_pixel_size.item()*1000,pixelY = y_pixel_size.item()*1000)
# (pyFAI integrator configured with beam/detector geometry)
 
#binning = int(coreg_files["-BINNING-"])


# === OBJECT DISCOVERY: pickles/arrays produced by earlier stages ===
dict_files = glob.glob(os.path.join(data_path, "*.pkl"))
npy_files = glob.glob(os.path.join(data_path, "*.npy"))
lhs_dict_files = [k for k in dict_files if "90-to-270" in k]
lhs_npy_files = [k for k in npy_files if "90-to-270" in k]
fibrilParams_file = [k for k in lhs_dict_files if "_fibril_params_svd_amps" in k][0]
#fibrilParams_file = [k for k in lhs_dict_files if "_ampvals" in k][0]
beampath_file = [k for k in dict_files if "rotation_beampaths_svd_amps" in k][0]
init_struct_file = [k for k in dict_files if "_init_structure" in k][0]
ichi_dict_file = [k for k in lhs_dict_files if "_ichi_dict_filled" in k][0]
cake_params_file = [k for k in dict_files if "svd_cake_params" in k][0]
    
# === RUNTIME CONFIG: fitting flags, angles, wavelength ===
"""
input varables and data
"""
single_solve = True
combi_solve = True
multi_solve = True
relax_fit = True

graph_display = False

single_solve = True
    
rot_range = np.linspace(-85,95,9)

wavelen = 0.08856

# === LOAD RUNTIME OBJECTS (fibril library, beam paths, I(chi), cake params, init structure) ===
with open(fibrilParams_file, 'rb') as f:
    fibrilParams = pickle.load(f)
    
with open(beampath_file, 'rb') as f:
    rotated_beampaths = pickle.load(f)
    
with open(ichi_dict_file, 'rb') as f:
    ichiExp = pickle.load(f)
    
with open(cake_params_file, 'rb') as f:
    cake_params_chi = pickle.load(f)
    
with open(init_struct_file, 'rb') as f:
    initStruct = pickle.load(f)

# === INITIALISATION: indices & parameter seeds; define sampling grids ===
fibril_idxs = [k["indx"] for k in fibrilParams]

q0_m,wMu_m = initStruct['q0']["m"],initStruct['wMu']["m"]
wa_m,delta_m = initStruct['wa']["m"],initStruct['delta']["m"]
amp_m = 1e6

sim_vals = [q0_m,wa_m,wMu_m,amp_m,delta_m]

q0m_low,q0m_high,nq0,wavelen = cake_params_chi["q1"],cake_params_chi["q2"],50,cake_params_chi["wavelen"]

chi1, chi2, nchi = 0, 180, 180
chirange = np.linspace(chi1, chi2, nchi)
gamma0, deltaGamma0 = np.radians(0.0), 0.01
mu = np.radians(0.0)

q_fitIq1, q_fitIq2, nq_fitIq = q0_m-0.02, q0_m+0.03, 120
#q_fitIq1, q_fitIq2, nq_fitIq = q0_m-0.02, q0_m+0.03, 60
dq_fitIq = (q_fitIq2-q_fitIq1)/nq_fitIq
q_fitIqr = np.arange(q_fitIq1, q_fitIq2, dq_fitIq)

qx, qy, qz, qxD, qyD, qxD_offset = t3d.calc_ewald_surface(0.08856,1.0, 1.0, 0.011)
ewald = (qx, qy, qz)

fitted_amps = np.asarray([k["initial_amp_est"] for k in fibrilParams if k["initial_amp_est"] !=0])

"""
FIT PARAMETERS:
    min/max values for parameter objects
    threshold values for fit testing
"""

# === FIT BOUNDS & THRESHOLDS (q0/wa/wMu/delta/amp; SNR/error gates) ===
chi_range_sample = 3 # chi range of cakes for sampling real data and simulating/fitting 
nslices = 3 # number of cakes to sample over



q0_min,q0_max = (2*np.pi)*3/74,(2*np.pi)*3/60 # min/max for q0 parameter estimation
wa_min,wa_max = 0.0005, 0.01 # min/max for wa parameter estimation
wMu_min,wMu_max = 0.02,0.8 # min/max for wMu parameter estimation
ab_var = 5 # variation permitted in estimated alpha and beta values (in degrees) for fitting
delta_min,delta_max = 0.001, 0.99 # min/max for delta parameter estimation
amp_min,amp_max = 1000,2e7*300 # min/max for amplitude estimation - mean is absolute value; max is multiplier of SVD amplitude estimate

max_amp_rat = 2 # threshold value for ratio between fitted peak intensity and measured intensity for discriminating fits that are too high/too low
min_abs_int = 0.5*1e8 # minimum intensity value for fits
min_error = 35 # minimum percentage standard error
min_error_relax = 40 # minimum percentage error for relaxed fitting
min_abs_relax = 3.5*1e6 # minimum intensity value for relaxed fits 

min_q0_fit = q0_min+0.001
max_q0_fit = q0_max-0.001

min_fit_wa = wa_min+0.0001
min_fit_wMu = 0.06
max_fit_wa = 0.0098 # threshold maximum fitted wa value for fits
max_fit_wMu = 0.799 # threshold maximum fitted wMu value for fits

single_fit_rat = 1.5 # lower threshold value for single fitting 
single_fit_error = 30 # lower standard error threshold value for single fitting 

min_fit_diameter = 5
max_fit_diameter = 200

single_min_snr = 4
min_snr = 3
relax_snr = 2

fit_params = {"q0_min": q0_min, "q0_max": q0_max, "wa_min": wa_min, "wa_max": wa_max,"wMu_min": wMu_min, "wMu_max": wMu_max,
               "ab_var": ab_var, "delta_min": delta_min, "delta_max": delta_max,"amp_min":amp_min,"amp_max":amp_max,
               "max_amp_rat":max_amp_rat,"min_abs_int": min_abs_int,"min_error": min_error,"min_abs_relax": min_abs_relax,
               "min_error_relax":min_error_relax,"min_error_single": single_fit_error,
               "min_fit_q0":min_q0_fit,"max_fit_q0":max_q0_fit,
               "min_fit_wa":min_fit_wa,"min_fit_wMu":min_fit_wMu,"max_fit_wa": max_fit_wa,"max_fit_wMu": max_fit_wMu,"single_fit_rat":single_fit_rat,
               "single_fit_error":single_fit_error,"diam_min":min_fit_diameter,"diam_max":max_fit_diameter,
               "single_min_snr":single_min_snr,"min_snr":min_snr,"relaxed_snr":2}

threshold_detection = min_abs_relax
threshold_detection_relxed = min_abs_relax/10
chiRefWindow = 10
threshold_interference = 10 
threshold_interference_multi = 20

# === CAKE PARAMS FOR RECONSTRUCTION (chi/q windows, background) ===
recon_cake_params = copy.deepcopy(cake_params_chi)
recon_cake_params["chi1"] = 0
recon_cake_params["chi2"] = 180

#recon_cake_params["q1"] = 0.27
#recon_cake_params["q2"] = 0.315


recon_cake_params["q2i"] = recon_cake_params['q1']
recon_cake_params["q1i"] = recon_cake_params['q1i']+0.01
recon_cake_params["q1o"] = recon_cake_params['q2']
recon_cake_params["q2o"] = recon_cake_params['q1o']+0.01
recon_cake_params['nq'] = 50
recon_cake_params['bg_start'] = 0.2
recon_cake_params['bg_end'] = 0.355

recon_cake_params['test'] = "full_smooth"


# === MULTI-FIBRE SOLVER SETTINGS ===
"""
multiple fibre solving parameters
"""
overlap_params = {"searchwindow": 10, "fitwindow": 10, "minseparation": 2}
overlap_params_triple = {"searchwindow": 10, "fitwindow": 10, "minseparation": 0.5}
overlap_thresh = {"thresh_detect": threshold_detection, "thresh_combined": 80,"thresh_individual": 10}

I3_tot = 50
I3_singleton = 20

triple_overlap_params = {"searchwindow": 10, "fitwindow": 10, "minseparation": 1, "nslices": 3}
triple_overlap_thresh = {"thresh_detect": threshold_detection, "I3_tot": I3_tot, "I3_singleton": I3_singleton, "angular_fraction": 3.0}
#overlap_thresh = {"thresh_detect": threshold_detection, "thresh_combined": 95,"thresh_individual": 5}

multi_params = [overlap_params,overlap_params_triple,overlap_thresh,I3_tot,I3_singleton,
                triple_overlap_params,triple_overlap_thresh]

# === PER-SLICE SETUP & MASKING ===
"""
run recon for each tomoSAXS slice
"""

slice_saxs_file = glob.glob(os.path.join(data_path, "*.h5"))[0]

vert_slice = slice_saxs_file.split("/")[-1].split(".")[0]
vert_slice = int(re.findall(r'\d+', vert_slice)[0])

slice_alpha_index = np.copy(np.asarray([k[vert_slice] for k in alpha_data],dtype=object))

slice_index_index = np.copy(np.asarray([k[vert_slice] for k in index_data],dtype=object))


Nx, Ny = slice_alpha_index[0].shape[0], 1
Nz = slice_alpha_index[0].shape[1]
samplex1, samplex2, Nxf = -.6,.6,Nx
sampley1, sampley2, Nyf = .1,.1,Ny
samplez1, samplez2, Nzf = -.6,.6,Nz
dxs, dys, dzs = (samplex2-samplex1)/Nx,(sampley2-sampley1)/Ny,(samplez2-samplez1)/Nz

recon_mask_chi = rec_lib.ichi_sample(0,Mask,0,0,recon_cake_params,a1,0,slice_saxs_file,[-180,0],
# (Derive reconstruction mask to exclude invalid chi regions)
                      fibre_chi = False,fit_chi=False,iq_plot = True)

recon_mask_chis = np.flip(recon_mask_chi[0][1])
recon_mask_pts = np.where(recon_mask_chis==0,True,False)

# === PHASE 1: SINGLE-FIBRE FITS (highest-intensity first) ===
"""
recon for single fibre solvable instances
    prioritising most intense instances first
"""

solve_params = [single_solve,combi_solve,multi_solve,relax_fit,"new"]

sampling_params = [chi1,chi2,chirange,nchi,qx,qy,qz,dxs,q0_m,q0m_low,q0m_high,nq0,threshold_interference,threshold_detection,q_fitIqr,binning,chi_range_sample,nslices]
print("parameters established: beginning recon")

# Skip if Phase 1 already completed (resumable)
if os.path.isfile(output_path+"fibrilParams_final_filled_1.pkl") == True:
    
    with open(output_path+"fibrilParams_final_filled_1.pkl", 'rb') as f:
        fibrilParams = pickle.load(f)
        
    with open(output_path+"rotated_beampaths_final_filled_1.pkl", 'rb') as f:
        rotated_beampaths = pickle.load(f)
    
else:

    solv_params = [False,False,False,False,"new"]
    
    fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(1,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                     recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,graph_display = False)
    
    
    bad_bps = []
    loop_index = 0
    fitted = 0
    
    while len(single_solves)>0:
        loop_index = loop_index+1
        if len(bad_bps)>0:
            bad_idxs = [[k for j in bad_bps if j[0:3] == single_solves[k][0:3]] for k in np.arange(0,len(single_solves),1)]
            bad_idxs = [k[0] for k in bad_idxs if len(k)>0]
            
            single_solves = [single_solves[k] for k in np.arange(0,len(single_solves)) if k not in bad_idxs]
    
        if len(single_solves)>0:
            solved_idxs = np.unique([k[2] for k in single_solves])
            
            solved_bps = [[[[k[0],k[1]],k[3]] for k in single_solves if k[2]== i] for i in solved_idxs]
            solved_ichis = [[[[k[4],k[5]],k[3]] for k in single_solves if k[2]== i] for i in solved_idxs]
            
            max_bps_idxs = [np.argmax([j[1] for j in k]) for k in solved_bps]
            
            max_bps = [k[np.argmax([j[1] for j in k])][0] for k in solved_bps]
            max_ichis = [k[np.argmax([j[1] for j in k])][0] for k in solved_ichis]
            
            fits,no_fits = [],[]
            
            std_errors = []
            
            for k in range(0,len(max_bps)):
                
                r,i = max_bps[k][0],max_bps[k][1]
                
                path_dict = rotated_beampaths[r][i]
                for j,voxel_test in enumerate(path_dict["voxels"]):
                    #voxel_test["fibril_param"]["solved"] = False
                    #voxel_test["fibril_param"]["fit"] = {}
                    
                    if voxel_test["fibril_param"]["indx"] == solved_idxs[k]:
                        voxel = copy.deepcopy(voxel_test)
                indx = np.where(fibril_idxs == solved_idxs[k])[0][0]
                
                ichi = max_ichis[k][0]
                Ichi1D_unsolved = max_ichis[k][1]
            
                fibrilParams,rotated_beampaths,vox_result,chi_sqr,std_error = rec_lib.singleSolve(max_bps[k][0],max_bps[k][1],fibrilParams,ichiExp,rotated_beampaths,
                                                                                solved_idxs[k],path_dict,voxel,indx,ichi,Ichi1D_unsolved,chirange,chiRefWindow,
                                                                                threshold_interference,threshold_detection,recon_cake_params,recon_mask_pts,recon_mask_chis,
                                                                                a1,Mask,slice_saxs_file,q_fitIqr,sim_vals,q0_m,fibril_idxs,output_path,binning,fit_params,chi_range_sample,nslices)
                if len(vox_result["fibril_param"]["fit"]) == 0:
                    no_fits.append(k)                    
                else:
                    fits.append(k)
                    std_errors.append([k,std_error])
            
            fitted = fitted+len(fits)
            
            
            
            if len(no_fits)==0:
                fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(1,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                                 recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,graph_display = False)   
            
            elif len(no_fits)>0:
                
                bad_bps = bad_bps+[[max_bps[k][0],max_bps[k][1],solved_idxs[k]] for k in no_fits]
                
                if len(no_fits) == len(max_bps):
                    single_solves = []
                else:    
                
                    second_bps,second_ichis = [],[]
                    second_idxs = []
                    
                    for k in no_fits:
                        if len(solved_bps[k])>1:
                            max_ints = np.asarray([j[1] for j in solved_bps[k]])
                            list2 = list(set(max_ints)) 
                            # Sorting the  list
                            list2.sort()
                            if len(list2)>1:
                                second_max = np.where(max_ints == list2[-2])[0][0]
                            else:
                                second_max = 1
                            
                            second_bps.append(solved_bps[k][second_max][0])
                            second_ichis.append(solved_ichis[k][second_max][0])
                            second_idxs.append(solved_idxs[k])
                    
                    no_fits_2 = []
                    fits_2 = []
                    
                    for k in range(0,len(second_bps)):
                        
                        r,i = second_bps[k][0],second_bps[k][1]
                        
                        idx = second_idxs[k]
                        
                        path_dict = rotated_beampaths[r][i]
                        for j,voxel_test in enumerate(path_dict["voxels"]):
                            if voxel_test["fibril_param"]["indx"] == idx:
                                voxel = copy.deepcopy(voxel_test)
                        indx = np.where(fibril_idxs == idx)[0][0]
                        
                        ichi = second_ichis[k][0]
                        Ichi1D_unsolved = second_ichis[k][1]
                    
                        fibrilParams,rotated_beampaths,vox_result,chi_sqr,std_error = rec_lib.singleSolve(second_bps[k][0],second_bps[k][1],fibrilParams,ichiExp,rotated_beampaths,
                                                                                        second_idxs[k],path_dict,voxel,indx,ichi,Ichi1D_unsolved,chirange,chiRefWindow,
                                                                                        threshold_interference,threshold_detection,recon_cake_params,recon_mask_pts,recon_mask_chis,
                                                                                        a1,Mask,slice_saxs_file,q_fitIqr,sim_vals,q0_m,fibril_idxs,output_path,binning,fit_params,chi_range_sample,nslices)
                        
                        if len(vox_result["fibril_param"]["fit"]) == 0:
                            no_fits_2.append(k)
                        else:
                            fits_2.append(k)
                            
                    fitted = fitted+len(fits)
                    
                    
                    
                    if len(no_fits_2)>0:
                        
                        bad_bps = bad_bps+[[second_bps[k][0],second_bps[k][1],second_idxs[k]] for k in no_fits_2]
                                                    
                if len(no_fits_2) == len(second_bps) and len(no_fits) == len(max_bps):
                    single_solves = []
                    
                else:
                    fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(1,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                                     recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,graph_display = False)   
                
    """
    recon for multiple fibre solvable instances
        prioritising most intense instances first
    """
    
    with open(output_path+"fibrilParams_final_filled_1.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_1.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)
        

# === PHASE 2: IN-LINE (STRICT) MULTI-ITERATION FITS ===
"""
2. in-line reconstruction - non-relaxed fits
"""    

if os.path.isfile(output_path+"loop_idx.npy") == True:
    loop_indx = np.load(output_path+"loop_idx.npy")
else:
    loop_indx = [2,0]

if os.path.isfile(output_path+"fibrilParams_final_filled_2.pkl") == True and loop_indx[0]>2:
    
    with open(output_path+"fibrilParams_final_filled_2.pkl", 'rb') as f:
        fibrilParams = pickle.load(f)
        
    with open(output_path+"rotated_beampaths_final_filled_2.pkl", 'rb') as f:
        rotated_beampaths = pickle.load(f)
    
else:
    
    loop_count = 10
    
    if os.path.isfile(output_path+"fibrilParams_final_filled_2.pkl") == True:
        with open(output_path+"fibrilParams_final_filled_2.pkl", 'rb') as f:
            fibrilParams = pickle.load(f)
            
        with open(output_path+"rotated_beampaths_final_filled_2.pkl", 'rb') as f:
            rotated_beampaths = pickle.load(f)
            
        loop_count = 10 - loop_indx[1]

    solv_params = [single_solve,combi_solve,multi_solve,relax_fit]
    solv_params = [True,True,True,False,"new"]
    #combi_solve
    
    percent_solved = len([k for k in fibrilParams if len(k["fit"])>0])
    
    
    
    fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(loop_count,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                     recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 2,graph_display = False)   
    
    with open(output_path+"fibrilParams_final_filled_2.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_2.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)
        

# === PHASE 3: RELAXED FITS (looser thresholds for unfitted fibres) ===
"""
3. in-line reconstruction - relaxed fits
"""     

if os.path.isfile(output_path+"loop_idx.npy") == True:
    loop_indx = np.load(output_path+"loop_idx.npy")
else:
    loop_indx = [3,0]

if os.path.isfile(output_path+"fibrilParams_final_filled_3.pkl") == True and loop_indx[0] >3:
    
    with open(output_path+"fibrilParams_final_filled_3.pkl", 'rb') as f:
        fibrilParams = pickle.load(f)
        
    with open(output_path+"rotated_beampaths_final_filled_3.pkl", 'rb') as f:
        rotated_beampaths = pickle.load(f)
    
else:

    if len([k for k in fibrilParams if len(k["fit"])>0])<len(fibrilParams):
        
        loop_count = 10
        
        if os.path.isfile(output_path+"fibrilParams_final_filled_3.pkl") == True: 
            with open(output_path+"fibrilParams_final_filled_3.pkl", 'rb') as f:
                fibrilParams = pickle.load(f)
                
            with open(output_path+"rotated_beampaths_final_filled_3.pkl", 'rb') as f:
                rotated_beampaths = pickle.load(f)
                
            loop_count = 10 - loop_indx[1]
        
        solv_params = [single_solve,combi_solve,multi_solve,relax_fit]
        solv_params = [True,True,True,True,"new"]
        #combi_solve
        
        percent_solved = len([k for k in fibrilParams if len(k["fit"])>0])
        
        
        
        
        fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(loop_count,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                         recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 3,graph_display = False) 
    
                    
    with open(output_path+"fibrilParams_final_filled_3.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_3.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)
        
    


# === PHASE 4: BOOTSTRAP UNFITTED FIBRES (mean amp) + WIDER BOUNDS ===
if os.path.isfile(output_path+"loop_idx.npy") == True:
    loop_indx = np.load(output_path+"loop_idx.npy")
else:
    loop_indx = [4,0]
    
if os.path.isfile(output_path+"fibrilParams_final_filled_4.pkl") == True:
    
    with open(output_path+"fibrilParams_final_filled_4.pkl", 'rb') as f:
        fibrilParams = pickle.load(f)
        
    with open(output_path+"rotated_beampaths_final_filled_4.pkl", 'rb') as f:
        rotated_beampaths = pickle.load(f)
    
else:

    if len([k for k in fibrilParams if len(k["fit"])>0])<len(fibrilParams):
    
        meanAmps = np.asarray([k["fit"]["amp"] for k in fibrilParams if len(k["fit"])>0])
        meanAmps = meanAmps[meanAmps<1e8]
        
        for vox in fibrilParams:
            if len(vox["fit"]) == 0 and vox["initial_amp_est"] == 0:
                vox["initial_amp_est"] = np.mean(meanAmps)
                
        for r in range(0,len(rot_range)):                            
            rot_beampath = np.asarray(rotated_beampaths[r])            
            for i, path_dict in enumerate(rot_beampath):
                if path_dict != None:                    
                    if path_dict["voxels"] != None: 
                        if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:
                            for j,voxel in enumerate(path_dict["voxels"]):
                                if voxel["fibril_param"]['initial_amp_est'] == 0:
                                    voxel["fibril_param"]['initial_amp_est'] = np.mean(meanAmps)
                
                
        solv_params = [True,True,True,True,"new"]
        
        fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(10,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                         recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 4,graph_display = False) 
            
    with open(output_path+"fibrilParams_final_filled_4.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_4.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)
        

if os.path.isfile(output_path+"loop_idx.npy") == True:
    loop_indx = np.load(output_path+"loop_idx.npy")
else:
    loop_indx = [5,0]
    
if os.path.isfile(output_path+"fibrilParams_final_filled_5.pkl") == True:
    
    with open(output_path+"fibrilParams_final_filled_5.pkl", 'rb') as f:
        fibrilParams = pickle.load(f)
        
    with open(output_path+"rotated_beampaths_final_filled_5.pkl", 'rb') as f:
        rotated_beampaths = pickle.load(f)
    
else:

    if len([k for k in fibrilParams if len(k["fit"])>0])<len(fibrilParams):
    
        meanAmps = np.asarray([k["fit"]["amp"] for k in fibrilParams if len(k["fit"])>0])
        meanAmps = meanAmps[meanAmps<1e8]
        
        for vox in fibrilParams:
            if len(vox["fit"]) == 0 and vox["initial_amp_est"] == 0:
                vox["initial_amp_est"] = np.mean(meanAmps)
                
        for r in range(0,len(rot_range)):                            
            rot_beampath = np.asarray(rotated_beampaths[r])            
            for i, path_dict in enumerate(rot_beampath):
                if path_dict != None:                    
                    if path_dict["voxels"] != None: 
                        if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:
                            for j,voxel in enumerate(path_dict["voxels"]):
                                if voxel["fibril_param"]['initial_amp_est'] == 0:
                                    voxel["fibril_param"]['initial_amp_est'] = np.mean(meanAmps)
                                    
        q0_min,q0_max = (2*np.pi)*3/76,(2*np.pi)*3/58 # min/max for q0 parameter estimation
        wa_min,wa_max = 0.0005, 0.01 # min/max for wa parameter estimation
        wMu_min,wMu_max = 0.005,0.9 # min/max for wMu parameter estimation
        ab_var = 5 # variation permitted in estimated alpha and beta values (in degrees) for fitting
        delta_min,delta_max = 0.001, 0.99 # min/max for delta parameter estimation
        amp_min,amp_max = 1000,2e7*300 # min/max for amplitude estimation - mean is absolute value; max is multiplier of SVD amplitude estimate

        max_amp_rat = 2 # threshold value for ratio between fitted peak intensity and measured intensity for discriminating fits that are too high/too low
        min_abs_int = 0.5*1e8 # minimum intensity value for fits
        min_error = 35 # minimum percentage standard error
        min_error_relax = 40 # minimum percentage error for relaxed fitting
        min_abs_relax = 3.5*1e6 # minimum intensity value for relaxed fits 

        min_q0_fit = q0_min+0.001
        max_q0_fit = q0_max-0.001

        min_fit_wa = wa_min+0.0001
        min_fit_wMu = 0.02
        max_fit_wa = 0.0099 # threshold maximum fitted wa value for fits
        max_fit_wMu = 0.899 # threshold maximum fitted wMu value for fits

        single_fit_rat = 1.5 # lower threshold value for single fitting 
        single_fit_error = 30 # lower standard error threshold value for single fitting 

        min_fit_diameter = 1
        max_fit_diameter = 220

        single_min_snr = 4
        min_snr = 3
        relax_snr = 2

        fit_params = {"q0_min": q0_min, "q0_max": q0_max, "wa_min": wa_min, "wa_max": wa_max,"wMu_min": wMu_min, "wMu_max": wMu_max,
                       "ab_var": ab_var, "delta_min": delta_min, "delta_max": delta_max,"amp_min":amp_min,"amp_max":amp_max,
                       "max_amp_rat":max_amp_rat,"min_abs_int": min_abs_int,"min_error": min_error,"min_abs_relax": min_abs_relax,
                       "min_error_relax":min_error_relax,"min_error_single": single_fit_error,
                       "min_fit_q0":min_q0_fit,"max_fit_q0":max_q0_fit,
                       "min_fit_wa":min_fit_wa,"min_fit_wMu":min_fit_wMu,"max_fit_wa": max_fit_wa,"max_fit_wMu": max_fit_wMu,"single_fit_rat":single_fit_rat,
                       "single_fit_error":single_fit_error,"diam_min":min_fit_diameter,"diam_max":max_fit_diameter,
                       "single_min_snr":single_min_snr,"min_snr":min_snr,"relaxed_snr":2}
                
                
        solv_params = [True,True,True,True,"new"]
        
        fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(10,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                         recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 5,graph_display = False) 
            
    with open(output_path+"fibrilParams_final_filled_5.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_5.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)









# === PHASE 5: CONTINUE RELAXED LOOPS UNTIL CONVERGENCE/ITER LIMIT ===
if loop_indx[0] == 5 and loop_indx[1] <9:
    
    if len([k for k in fibrilParams if len(k["fit"])>0])<len(fibrilParams):
    
        loop_count = 10 - loop_indx[1]    
    
        meanAmps = np.asarray([k["fit"]["amp"] for k in fibrilParams if len(k["fit"])>0])
        meanAmps = meanAmps[meanAmps<1e8]
        
        #solv_params = [single_solve,combi_solve,multi_solve,relax_fit]
        #solv_params = [True,True,True,True]
        
        for vox in fibrilParams:
            if len(vox["fit"]) == 0 and vox["initial_amp_est"] == 0:
                vox["initial_amp_est"] = np.mean(meanAmps)
                
        for r in range(0,len(rot_range)):                            
            rot_beampath = np.asarray(rotated_beampaths[r])            
            for i, path_dict in enumerate(rot_beampath):
                if path_dict != None:                    
                    if path_dict["voxels"] != None: 
                        if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:
                            for j,voxel in enumerate(path_dict["voxels"]):
                                if voxel["fibril_param"]['initial_amp_est'] == 0:
                                    voxel["fibril_param"]['initial_amp_est'] = np.mean(meanAmps)
                                    
        q0_min,q0_max = (2*np.pi)*3/76,(2*np.pi)*3/58 # min/max for q0 parameter estimation
        wa_min,wa_max = 0.0005, 0.01 # min/max for wa parameter estimation
        wMu_min,wMu_max = 0.005,0.9 # min/max for wMu parameter estimation
        ab_var = 5 # variation permitted in estimated alpha and beta values (in degrees) for fitting
        delta_min,delta_max = 0.001, 0.99 # min/max for delta parameter estimation
        amp_min,amp_max = 1000,2e7*300 # min/max for amplitude estimation - mean is absolute value; max is multiplier of SVD amplitude estimate

        max_amp_rat = 2 # threshold value for ratio between fitted peak intensity and measured intensity for discriminating fits that are too high/too low
        min_abs_int = 0.5*1e8 # minimum intensity value for fits
        min_error = 35 # minimum percentage standard error
        min_error_relax = 40 # minimum percentage error for relaxed fitting
        min_abs_relax = 3.5*1e6 # minimum intensity value for relaxed fits 

        min_q0_fit = q0_min+0.001
        max_q0_fit = q0_max-0.001

        min_fit_wa = wa_min+0.0001
        min_fit_wMu = 0.02
        max_fit_wa = 0.0099 # threshold maximum fitted wa value for fits
        max_fit_wMu = 0.899 # threshold maximum fitted wMu value for fits

        single_fit_rat = 1.5 # lower threshold value for single fitting 
        single_fit_error = 30 # lower standard error threshold value for single fitting 

        min_fit_diameter = 1
        max_fit_diameter = 220

        single_min_snr = 4
        min_snr = 3
        relax_snr = 2

        fit_params = {"q0_min": q0_min, "q0_max": q0_max, "wa_min": wa_min, "wa_max": wa_max,"wMu_min": wMu_min, "wMu_max": wMu_max,
                       "ab_var": ab_var, "delta_min": delta_min, "delta_max": delta_max,"amp_min":amp_min,"amp_max":amp_max,
                       "max_amp_rat":max_amp_rat,"min_abs_int": min_abs_int,"min_error": min_error,"min_abs_relax": min_abs_relax,
                       "min_error_relax":min_error_relax,"min_error_single": single_fit_error,
                       "min_fit_q0":min_q0_fit,"max_fit_q0":max_q0_fit,
                       "min_fit_wa":min_fit_wa,"min_fit_wMu":min_fit_wMu,"max_fit_wa": max_fit_wa,"max_fit_wMu": max_fit_wMu,"single_fit_rat":single_fit_rat,
                       "single_fit_error":single_fit_error,"diam_min":min_fit_diameter,"diam_max":max_fit_diameter,
                       "single_min_snr":single_min_snr,"min_snr":min_snr,"relaxed_snr":2}
                                    
        
        solv_params = [True,True,True,True,"new"]       
                
        fibrilParams,rotated_beampaths,single_solves,multi_solves = rec_lib.tomoSAXS_sim(loop_count,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                         recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 5,graph_display = False) 
            
    with open(output_path+"fibrilParams_final_filled_5.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)
    
    with open(output_path+"rotated_beampaths_final_filled_5.pkl", 'wb') as f:
        pickle.dump(rotated_beampaths, f)
