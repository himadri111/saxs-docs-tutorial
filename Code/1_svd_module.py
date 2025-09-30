# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:37:58 2024

@author: Himadri

"""

# === INPUT PATH DISCOVERY: locate folder and required files ===
# === RUNTIME CONFIG: rotations, beampaths, binning, plotting ===
# === DATA LOAD: fibril params, beam paths, I(chi), masks, cake params ===
# === MATRIX ALLOCATION: prepare A (simulated) and B (experimental) ===
# === A/B FILL: iterate rotations and beampaths; simulate & place segments ===
# === TRIMMING: clip A/B to peak regions to focus on informative bins ===
# === SOLVE: NNLS/SVD amplitude estimation for each fibril ===
# === WRITE-BACK: store amplitudes on fibril params and save arrays ===
# === PROPAGATE: update beam-path voxel dictionaries with amplitude estimates ===
# === REBUILD: amplitude-weighted simulation for diagnostic comparison ===
# === TRIMMING (AMP): focus on peaks in amplitude-adjusted A ===
# === DIAGNOSTICS: optional correlation/segment plots ===
"""
SVD-Based Fibre Amplitude Estimation (Annotated)

This annotated version adds SECTION-LEVEL comments to explain the flow and intent
without cluttering the code with line-by-line notes.

Purpose
-------
Given precomputed tomoSAXS assets (fibril parameter library, beam paths,
experimental I(chi), masking info, and integration parameters), this script:
1) Constructs an A-matrix of simulated I(chi) responses per fibril for each beam path.
2) Concatenates experimental I(chi) into a B-vector with the same indexing.
3) Uses non-negative least squares (NNLS) / SVD to estimate per-fibril amplitudes.
4) Writes amplitude estimates back to fibril dictionaries and beam-path objects.
5) Optionally rebuilds A using amplitudes to compare simulated vs experimental peaks.

Reading Guide
-------------
- IMPORTS & PATH DISCOVERY: libraries and logic to locate all prerequisite files.
- FORWARD MODEL: functions that simulate I(chi) from fibril parameters / geometry.
- OPTIMISATION: utilities for adjusting alpha/beta against data.
- RUNTIME CONFIG: user-tunable flags and ranges (rotations, binning, plotting).
- DATA LOAD: pickle/npy inputs (I(chi), masks, bounds, fibril params, beam paths).
- A/B MATRIX BUILD: fill A from simulations and B from experimental I(chi)).
- SVD/NNLS: solve for amplitudes, then persist results and update objects.
- DIAGNOSTICS: optional plots showing simulation/data agreement.

This file is intended as a drop-in replacement with commentary.
"""

"""
module for estimating fibre amplification via Single Value Decomposition (SVD)
prerequisites: series of data from preceeding modules for:
    - creating library of I(chi) data measured from real data
    - creating the necessary variables and datasets for SVD analysis
"""
# === IMPORTS ===
import pickle
import os,glob
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
sys.path.append(r'/dls/science/groups/i22/Himadri_Gupta/tomoSAXS_pipeline')
#import threeDXRD as t3d
import threeDXRD_080923 as t3d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from lmfit import Parameter, Parameters
from lmfit import minimize, report_fit
from scipy.stats import linregress

from scipy.optimize import nnls

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

# === RUNTIME SLICE SELECTION ===
saxs_slice = 1

"""
Isolate prerequisite data paths:
    - fibrilParams: dictionary of fibril parameters
    - beampaths: dictionaries of fibres interacting with each rotated beampath
    - cake params: dictionary of parameters used for I(chi) and I(q) integrations
    - init struct: dictionary of scattering parameters used as a global first guess for fibre scattering simulation
    - ichi dict: dictionary of I(chi) data for each REAL beampath
    - mask lens: list of angular ranges of masks
    - mask bounds: list of angular end-points of masks
    - masking lengths: list of angular widths of I(chi) integrations of each beampath after processing of masks
    - b_svd: saved b matrix (real I(chi) data) for the SVD 
    - other SVD files used here are for comparitive purposes for designing this module
      (i.e. generated from from previous scripts)
    
"""
input_folder = '/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded'
path = input_folder+"/"
output_path = path

subfolders = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
subfolders = [k for k in subfolders if k[-1].isnumeric()]
subfolders = [k for k in subfolders if k.split("_")[-1] != "5x5"]

folder = [k for k in subfolders if int(k.split("_")[-1]) == saxs_slice][0]


#for folder in subfolders:
    
data_path = path+folder+"/"
output_path = data_path


# Gather pickles/arrays in the selected data folder
dict_files = glob.glob(os.path.join(data_path, "*.pkl"))
npy_files = glob.glob(os.path.join(data_path, "*.npy"))
lhs_dict_files = [k for k in dict_files if "90-to-270" in k]
lhs_npy_files = [k for k in npy_files if "90-to-270" in k]
fibrilParams_file = [k for k in lhs_dict_files if "_fibril_params_raw" in k][0]
beampath_file = [k for k in dict_files if "rotation_beampaths.pkl" in k][0]
cake_params_file = [k for k in dict_files if "svd_cake_params" in k][0]
init_struct_file = [k for k in dict_files if "_init_structure" in k][0]
ichi_dict_file = [k for k in lhs_dict_files if "_ichi_dict_filled" in k][0]
mask_lens_file = [k for k in npy_files if "_90-to-270_masks.npy" in k][0]
mask_bounds_file = [k for k in npy_files if "90-to-270_mask_bounds.npy" in k][0]
masking_lens_file = [k for k in lhs_npy_files if "masked_lens.npy" in k][0]
mask_ichi_file = [k for k in lhs_npy_files if "mask_ichi.npy" in k][0]
b_svd_file = [k for k in lhs_npy_files if "_b_svd_Arr.npy" in k][0]
data_bounds_file = [k for k in lhs_npy_files if "_data_bounds.npy" in k][0]
just_data_file = [k for k in lhs_npy_files if "_unmasked_I.npy" in k][0]
just_data_chi_file = [k for k in lhs_npy_files if "_unmasked_chi.npy" in k][0]

bin_lens_file = [k for k in npy_files if "_90-to-270_masks_bin.npy" in k][0]
mask_bounds_bin_file = [k for k in npy_files if "90-to-270_mask_bounds_bin.npy" in k][0]
masking_lens_bin_file = [k for k in lhs_npy_files if "bin_lens.npy" in k][0]
mask_ichi_bin_file = [k for k in lhs_npy_files if "mask_ichi_bin.npy" in k][0]
b_svd_bin_file = [k for k in lhs_npy_files if "_b_svd_Arr_bin.npy" in k][0]
data_bounds_bin_file = [k for k in lhs_npy_files if "_bin_bounds.npy" in k][0]
just_data_bin_file = [k for k in lhs_npy_files if "_unmasked_binned_I.npy" in k][0]
just_data_chi_bin_file = [k for k in lhs_npy_files if "_unmasked_binned_chi.npy" in k][0]

# === UTILITY: simple 1D binning for I(chi) segments ===
def bin_data(data,width):
    result = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
    return result

#a_svd_ar,b_svd,svd_pos,pdict,params0,cake_params,mask_bounds,svd_amp_ests,svd_indxs,dataBounds,mask_ichi ,svd_amp = a_svd_arr,comb_ichi,[full_indx,indexArrTr0,chi_len],path_dict,params0,cake_params_chi,mask_bounds,0,0,data_bounds, False,False

# === FORWARD MODEL: assemble A rows from per-voxel simulated I(chi) ===
def a_svd(a_svd_ar,b_svd,svd_pos,pdict,params0,cake_params,mask_bounds,svd_amp_ests,svd_indxs,dataBounds,mask_ichi = False,svd_amp = False):
    
    """
    Modular function for storing simulated I(chi) data in the A-SVD matrix
    options are for applying mask and/or applying estimated amplitude
    """
    
    chi1,chi2,n_chi_svd,q0m_low,q0m_high,nq0,wavelen = list(cake_params.values())[0:7]
    
    full_idx = svd_pos[0]
    indexArrT0 = svd_pos[1]
    chiLen = svd_pos[2]
    
    comb_ichi = np.zeros_like(b_svd)
    bareichis = []
    for j, voxel in enumerate(pdict["voxels"]):
        
        alt_voxel = copy.deepcopy(voxel)
        alt_voxel["fibril_param"]["initial_amp_est"] = 1
        if len([alt_voxel["fibril_param"]["weight"]]) == 0:
            alt_voxel["fibril_param"]["weight"] = 1
        
        fib_idx = int(voxel["fibril_param"]["number"])
        
        ichi0,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile_neutral(alt_voxel,params0,chi1=chi1,chi2=chi2,nchi=n_chi_svd,q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen)
        masked_ichi = np.zeros_like(bareichi)
        masked_ichi[masked_ichi==0] = np.nan
        #mask_vals = [np.linspace(bareichi_mask_bounds[k][0],bareichi_mask_bounds[k][1],mask_lens[k]) for k in np.arange(0,len(mask_lens),1)]        
        if svd_amp != False:
            #bareichi = bareichi*voxel["fibril_param"]["initial_amp_est"]
            bareichi = bareichi*svd_amp_ests[np.where(svd_indxs==fib_idx)]
                
        data_masked = [np.asarray(bareichi[k]) for k in dataBounds]
        bareichi = np.asarray([item for sublist in data_masked for item in sublist]) 
        
        if len(b_svd)<n_chi_svd:
            """
            then we bin the data
            """        
            data_binned = np.asarray([bin_data(k,bin_val) for k in data_masked])
            ichi_binned = np.asarray([item for sublist in data_binned for item in sublist])
            
            smooth_bins = [savgol_filter(k,3,pol) for k in data_binned]
            bareichi = np.asarray([item for sublist in smooth_bins for item in sublist])
                    
        if mask_ichi != False:
            for k in range(0,len(dataBounds)):
                masked_ichi[dataBounds[k]] = data_masked[k]
                
            bareichi = masked_ichi
            
        
        
        comb_ichi = comb_ichi+bareichi
        bareichis.append([voxel["fibril_param"]["number"],bareichi])
        
        bareichiTr = np.asarray([bareichi]).T #change to column format
        a_svd_tmp = np.zeros_like(a_svd_ar[full_idx:full_idx+len(bareichiTr)]) #arrays which will be placed in the full arrays A and B
        np.put(a_svd_tmp[:,int(voxel["fibril_param"]["number"])],indexArrT0,bareichiTr)
        a_svd_ar[full_idx:full_idx+chiLen] = a_svd_ar[full_idx:full_idx+chiLen]+a_svd_tmp
        
    if len(np.where(comb_ichi>0.5)[0])>0:
        bp_peak_start = np.where(comb_ichi>0.5)[0][0]
        bp_peak_end = np.where(comb_ichi>0.5)[0][-1]
    else:
        bp_peak_start = 0
        bp_peak_end = 0
        
    return a_svd_ar,bareichis,comb_ichi,bp_peak_start,bp_peak_end


#get1DSAXSchiprofile_neutral(alt_voxel,params0,chi1=chi1,chi2=chi2,nchi=n_chi_svd,q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen)

#vox, params, chi1, chi2, nchi, q1, q2, nq, wavelen, amp = alt_voxel,params0,chi1,chi2,n_chi_svd,q0m_low,q0m_high,nq0,wavelen,False

# === MODEL WRAPPER: neutral (amp=1) I(chi) simulation ===
def get1DSAXSchiprofile_neutral(vox, params, chi1=0, chi2=360, nchi=90, q1=0.45, q2=.47, nq=10, wavelen=0.1, amp = False):
    q0  = params["q0"]
    wa  = params["wa"]
    wMu = params["wMu"]
    delta = params["delta"]
    
    vox2 = copy.deepcopy(vox)
    vox2["fibril_param"]["simu"]["q0"]=q0
    vox2["fibril_param"]["simu"]["wa"]=wa
    vox2["fibril_param"]["simu"]["wMu"]=wMu
    vox2["fibril_param"]["simu"]["delta"]=delta
    if amp != False:
        vox2["fibril_param"]["simu"]["amp"] = amp
    if  len([vox2["fibril_param"]["weight"]]) == 0:
        vox2["fibril_param"]["weight"] = 1
        
    if type(vox2["weight"]) == np.ndarray:
        vox2["fibril_param"]["weight"] = 1
        
    ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(vox2,chi1=chi1,chi2=chi2,nchi=nchi,\
                                                   q1=q1,q2=q2,nq=nq,wavelen=wavelen)
    
    return ichi, chir, ichimax, chimax, bareichi

# === CORE SCATTER MODEL: compute I(chi) for a fibril voxel (pyFAI/3DXRD) ===
def get1DSAXSchiprofile(vox, chi1=0, chi2=360, nchi=90, q1=0.45, q2=.47, nq=10, wavelen=0.1,simu_vals = False):
    fibril_chars = vox["fibril_param"]["simu"]
    
    if simu_vals == False:    
        q0  = fibril_chars["q0"]
        wa  = fibril_chars["wa"]
        wMu = fibril_chars["wMu"]
        amp = vox["fibril_param"]["initial_amp_est"]
        delta = fibril_chars["delta"]
        if type(vox["weight"]) == np.ndarray and len(vox["weight"]) >0:
            weight = vox["weight"]
        else:
            weight = 1
    else:
        if vox["fibril_param"]["solved"] == False or len(vox['fibril_param']["fit"]) == 0:              
            q0  = simu_vals[0]
            wa  = simu_vals[1]
            wMu = simu_vals[2]
            amp = vox["fibril_param"]["initial_amp_est"]
            delta = simu_vals[4]
            weight = vox["fibril_param"]['weight']
        else:                  
            q0  = vox["fibril_param"]["fit"]["q0"]
            wa  = vox["fibril_param"]["fit"]["wa"]
            wMu = vox["fibril_param"]["fit"]["wMu"]
            amp = vox["fibril_param"]["fit"]["amp"]
            delta = simu_vals[4]
            weight = vox["fibril_param"]['weight']
        
    flat = 1
    alpha = vox["fibril_param"]["alpha"]
    beta  = vox["fibril_param"]["beta"]
    alphaR, betaR = np.radians(alpha), np.radians(beta)
    normfac = 1
    integrate=False
    gamma0, deltaGamma0 = np.radians(0.0), 0.01
    mu = np.radians(0.0)
    
    dq = (q2-q1)/nq
    qr = np.arange(q1,q2,dq)
    chirange = np.linspace(chi1,chi2,nchi)
    ichi = np.zeros_like(chirange)
    qx,qy,qz=t3d.calc_ewald_trace_grid(wavelen,qr,chirange)
    iqchi = t3d.Iplanarfibril(qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,wa,normfac,\
                  alphaR,betaR,integrate,flat=flat,delta=delta)
    ichi = (np.average(iqchi,axis=1)*weight)*amp
    ichimax = ichi[np.argmax(ichi)]
    chimax  =  chirange[np.argmax(ichi)]

    bareichi = ichi/amp
    return ichi, chirange, ichimax, chimax, bareichi

# === UTILITY: simple 1D binning for I(chi) segments ===
def bin_data(data,width):
    result = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
    return result

# === OPTIMISATION: refine alpha/beta vs experimental I(chi) (optional) ===
def ab_opti(params,b_svd,pdict,params0,cake_params,mask_bounds,mask_ichi = False):
    #print(pdict["voxels"])
    fpars_idxs = [k["fibril_param"]["indx"] for k in pdict["voxels"]]
    
    params_alphas = [params['alpha1_' + str(int(k))].value for k in fpars_idxs]
    params_betas = [params['beta1_' + str(int(k))].value for k in fpars_idxs]
    for j, voxel in enumerate(pdict["voxels"]):
        voxel["fibril_param"]["alpha"] = params_alphas[j]
        voxel["fibril_param"]["beta"] = params_betas[j]
    
    chi1,chi2,n_chi_svd,q0m_low,q0m_high,nq0,wavelen = list(cake_params.values())[0:7]
    comb_ichi = np.zeros_like(b_svd)
    bareichis = []
    for j, voxel in enumerate(pdict["voxels"]):
        
        alt_voxel = copy.deepcopy(voxel)
        alt_voxel["fibril_param"]["initial_amp_est"] = 1
                
        ichi0,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile_neutral(alt_voxel,params0,chi1=chi1,chi2=chi2,nchi=n_chi_svd,q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen)
        bareichi_mask_bounds = [[np.mean(bareichi[k[0]-3:k[0]]),np.mean(bareichi[k[1]:k[1]+3])] for k in mask_bounds]
        #mask_vals = [np.linspace(bareichi_mask_bounds[k][0],bareichi_mask_bounds[k][1],mask_lens[k]) for k in np.arange(0,len(mask_lens),1)]
        if mask_ichi != False:
            mask_vals = [np.linspace(bareichi_mask_bounds[k][0],bareichi_mask_bounds[k][1],mask_lens[k]) for k in np.arange(0,len(mask_lens),1)]
            ichi_masked = np.copy(bareichi)
            for m in range(0,len(mask_vals)):
                ichi_masked[mask_bounds[m][0]:mask_bounds[m][1]] = mask_vals[m]
            bareichi = ichi_masked        
        comb_ichi = comb_ichi+bareichi
        
    resid = (comb_ichi/np.max(comb_ichi)) - (b_svd/np.max(b_svd))
    
    return resid

# === TOP-LEVEL EXECUTION ===
"""
""""""
START SCRIPT
""""""
"""
optimised_params = False

binning_data = True

"""
1. Input parameters
"""
optimise_ab = False
optimised_params = False

ichi_plot = False

#rot_range = [-90]
#rot_range = [-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]
rot_range = np.linspace(-85,95,9)

#bp_range = [91,92]
bp_range = "all"

window,pol = 9,1

#binning value for I(chi) data
bin_val = 4

"""
2. load saved objects/data
"""
with open(cake_params_file, 'rb') as f:
    cake_params_chi = pickle.load(f)

with open(ichi_dict_file, 'rb') as f:
    ichiExp = pickle.load(f)
    
with open(beampath_file, 'rb') as f:
    rotated_beampaths = pickle.load(f)
    
if optimise_ab == False:
    with open(fibrilParams_file, 'rb') as f:
        fibrilParams = pickle.load(f)
else:
    if os.path.isfile(fibrilParams_file.split(".pkl")[0]+"_opti_ab.pkl") == True:
        with open(fibrilParams_file.split(".pkl")[0]+"_opti_ab.pkl", 'wb') as f:
            fibrilParams = pickle.load(f)
            optimised_params = True
            
    
N_fibrils = len(fibrilParams)

with open(init_struct_file, 'rb') as f:
    init_Struct = pickle.load(f)
    
b_svd_arr = np.load(b_svd_file)
b_svd_bin = np.load(b_svd_bin_file)
#a_svd_arr_test = np.load(a_svd_file)
#a_peak_test = np.load(a_peaks_file)
#b_peak_test = np.load(b_peaks_file)

just_data =  np.load(just_data_file)
data_bounds = np.load(data_bounds_file,allow_pickle=True)
data_chi = np.load(just_data_chi_file,allow_pickle=True)

just_data_bin = np.load(just_data_bin_file)
data_bounds_bin = np.load(data_bounds_bin_file,allow_pickle=True)
data_chi_bin = np.load(just_data_chi_bin_file,allow_pickle=True)

mask_lens = np.load(mask_lens_file)
mask_bounds = np.load(mask_bounds_file)
masking_lens = np.load(masking_lens_file)
masked_len_sums = np.asarray([np.sum(masking_lens[0:k]) for k in np.arange(0,len(masking_lens),1)])
mask_ichi = np.load(mask_ichi_file)
mask_i = np.where(mask_ichi!=0,np.max(b_svd_arr),0)
#mask_i =  bin_data(mask_i,5)

mask_lens_bin = np.load(bin_lens_file)
mask_bounds_bin = np.load(mask_bounds_bin_file)
masking_lens_bin = np.load(masking_lens_bin_file)
masked_lens_sums_bin = np.asarray([np.sum(masking_lens_bin[0:k]) for k in np.arange(0,len(masking_lens_bin),1)])
mask_ichi_bin = np.load(mask_ichi_bin_file)
mask_i_bin = np.where(mask_ichi_bin!=0,np.max(b_svd_bin),0)


"""
Create an A-SVD matrix (for storing simulated I(chi) data)
"""
if binning_data != True:
    a_svd_arr = [None]*np.sum(masking_lens)
    b_svd_arr = [0.0]*np.sum(masking_lens)
else:
    a_svd_arr = [None]*np.sum(masking_lens_bin)
    b_svd_arr = [0.0]*np.sum(masking_lens_bin)
                
for i, val in enumerate(a_svd_arr):
    a_svd_arr[i]=np.zeros(N_fibrils)
    
a_svd_arr,b_svd_arr  = np.asarray(a_svd_arr),np.asarray(b_svd_arr)

a_svd_arr_test = copy.deepcopy(a_svd_arr)   
    
mask_svd_arr = copy.deepcopy(b_svd_arr)
         
q0_m, wa_m, wMu_m, delta_m = init_Struct["q0"]["m"], init_Struct["wa"]["m"], init_Struct["wMu"]["m"], init_Struct["delta"]["m"]

params0 = {"q0": q0_m, "wa": wa_m, "wMu": wMu_m, "delta": delta_m}

simu_peaks_lims = []
all_simu_peaks_lims = []

simu_peaks_lims_bin = []
all_simu_peaks_lims_bin = []

# === BEAM-PATH SELECTION: choose valid paths with usable experimental I(chi) ===
if bp_range == "all":
    
    bps = []
    
    for ridx, r in enumerate(rot_range):
        #print("ridx: ", ridx," angle: ", r)
            
        rot_beampath = np.asarray(rotated_beampaths[ridx])
        
        for i, path_dict in enumerate(rot_beampath):
            if path_dict != None:
                if path_dict["voxels"]!=None:                    
                    if i in ichiExp[ridx] and type(ichiExp[ridx][i]["kapton"]) == bool:
                        if i not in bps:
                            bps.append(i)
    
    bp_range = np.sort(bps)
    

"""
Fill A-svd matrix
"""
fib_indx = np.asarray([k["number"] for k in fibrilParams])

for ridx, r in enumerate(rot_range):
    print("ridx: ", ridx," angle: ", r)
        
    rot_beampath = np.asarray(rotated_beampaths[ridx])
    rot_simu_peaks = []
    
    #if bp_range != "all":
    
    bps = rot_beampath[bp_range]
    #else:
        #bps = rot_beampath
    
    for i, path_dict in enumerate(bps):
        if path_dict != None:
            if path_dict["voxels"]!=None:
                """
                Only add the experimental Ichi for this path into it if the beam intersects at least one fibre
                """
                if bp_range[i] in ichiExp[ridx] and type(ichiExp[ridx][bp_range[i]]["kapton"]) == bool:
                    
                    pdict_vox_indxs = np.asarray([k["fibril_param"]["number"] for k in path_dict["voxels"]])
                    if len(pdict_vox_indxs)>1:
                        vox_repeats = [k for k in np.arange(1,len(pdict_vox_indxs),1) if pdict_vox_indxs[k]==pdict_vox_indxs[k-1]]
                        if len(vox_repeats)>0:
                            path_dict["voxels"] = [path_dict["voxels"][k] for k in np.arange(0,len(path_dict["voxels"]),1) if k not in vox_repeats]
                            pdict_vox_indxs = np.asarray([k["fibril_param"]["number"] for k in path_dict["voxels"]])
                    
                    if binning_data != True:
                        ichi=ichiExp[ridx][bp_range[i]]["ichi_smoothed"] 
                        full_indx = int(ichiExp[ridx][bp_range[i]]["full_indx"]) 
                    else:
                        ichi=ichiExp[ridx][bp_range[i]]["ichi_binned"]  
                        full_indx = int(ichiExp[ridx][bp_range[i]]["bin_indx"]) 
                    
                    chi_len = len(ichi)  

                    for j,voxel in enumerate(path_dict["voxels"]):
                        if len([voxel["fibril_param"]["weight"]]) == 0:
                            voxel["fibril_param"]["weight"] = 1  
                        if len([voxel["weight"]]) == 0:
                            voxel["weight"] = 1
                        #if 
                                                                 
                  
                    #full_indx = int(ichiExp[ridx][bp_range[i]]["full_indx"]) 
                    
                    indexArr0 = np.linspace(0,chi_len-1,chi_len).astype(int)
                                        
                    indexArrTr0 = np.asarray([indexArr0]).T
                                                                
                    ichiTr = np.asarray([ichi]).T
                    mask_iTr = np.asarray([mask_i]).T
                    
                    b_svd_tmp = np.zeros_like(b_svd_arr[full_indx:full_indx+chi_len])
                    np.put(b_svd_tmp,indexArrTr0,ichiTr)
                    b_svd_arr[full_indx:full_indx+chi_len] = b_svd_arr[full_indx:full_indx+chi_len]+b_svd_tmp
                    #mask_svd_arr[full_indx:full_indx+chi_len] = mask_svd_arr[full_indx:full_indx+chi_len]+mask_i
                                                        
                    b_svd_clip = b_svd_arr[full_indx:full_indx+chi_len]
                                                        
                    comb_ichi = np.zeros_like(b_svd_clip)
                    
                    start_pts,end_pts = [],[]
                    
                    a_svd_arr_test,bareichis,comb_ichi_og,bp_peak_start,bp_peak_end = a_svd(a_svd_arr_test,comb_ichi,[full_indx,indexArrTr0,chi_len],
                                                                          path_dict,params0,cake_params_chi,mask_bounds,0,0,data_bounds,mask_ichi = False)
                    
                    if optimise_ab == True and optimised_params == False:
                    
                        params =  Parameters()                
                        for j, voxel in enumerate(path_dict["voxels"]):
                            label = str(int(voxel["fibril_param"]["number"]))
                            vox_alpha = voxel["fibril_param"]["alpha"]
                            vox_beta = voxel["fibril_param"]["beta"]
                            
                            params.add('idx_' + label, value = voxel["fibril_param"]["number"],vary = False)    
                            
                            fb_indx = fibrilParams[np.where(fib_indx==voxel["fibril_param"]["number"])[0][0]]
                            if fb_indx["intersected"] == False:
                                params.add('alpha1_' + label, value = vox_alpha,min = vox_alpha-20, max = vox_alpha+20)       
                                params.add('beta1_' + label, value = vox_beta,min = vox_beta-20, max = vox_beta+20)
                            else:
                                params.add('alpha1_' + label, value = vox_alpha,vary = False)       
                                params.add('beta1_' + label, value = vox_beta,vary = False)
                            voxel["fibril_param"]["intersected"] = True
                            fibrilParams[np.where(fib_indx==voxel["fibril_param"]["number"])[0][0]]["intersected"] = True
                            
                        print("######\optimising alpha/beta ",str(ridx),", beampath ",str(i),"\n######")   
                                                                        
                        ab_test = minimize(ab_opti,params,args=(b_svd_clip,path_dict,params0,cake_params_chi,mask_bounds),max_nfev=500)
                        
                        for j, voxel in enumerate(path_dict["voxels"]):
                            label = str(int(voxel["fibril_param"]["number"]))
                            voxel["fibril_param"]["alpha"] = ab_test.params["alpha1_"+label].value
                            voxel["fibril_param"]["beta"] = ab_test.params["beta1_"+label].value
                            fb_indx = fibrilParams[np.where(fib_indx==voxel["fibril_param"]["number"])[0][0]]
                            if voxel["fibril_param"]["number"] == fb_indx["number"]:
                                fb_indx["alpha"] = voxel["fibril_param"]["alpha"]
                                fb_indx["beta"]  = voxel["fibril_param"]["beta"]
                            else:
                                print("wrong")
                    
                    a_svd_arr,bareichis,comb_ichi,bp_peak_start,bp_peak_end = a_svd(a_svd_arr,comb_ichi,[full_indx,indexArrTr0,chi_len],
                                                                          path_dict,params0,cake_params_chi,mask_bounds,0,0,data_bounds,mask_ichi = False)
                    if optimise_ab == True:
                        #comb_ichi - np.asarray(comb_ichi)-90
                        plt.plot(comb_ichi/np.max(comb_ichi),label = "optimised fit\n(a/b +- 5 degrees)")
                        plt.plot(b_svd_clip/np.max(b_svd_clip),label = "real data")
                        plt.plot(comb_ichi_og/np.max(comb_ichi_og),label = "original simulation")
                        plt.title("rotation "+str(ridx)+", bp "+str(i))
                        plt.legend()
                        plt.show()
                        plt.close()
                    
                    a_svd_arr,bareichis,comb_ichi,bp_peak_start,bp_peak_end = a_svd(a_svd_arr,comb_ichi,[full_indx,indexArrTr0,chi_len],
                                                                          path_dict,params0,cake_params_chi,mask_bounds,0,0,data_bounds,mask_ichi = False)
                    
                    rot_simu_peaks.append([bp_range[i],bp_peak_start,bp_peak_end])
                    all_simu_peaks_lims.append([bp_range[i],bp_peak_start,bp_peak_end])
                    
    simu_peaks_lims.append(rot_simu_peaks)

#np.save(b_svd_file.split("_b_svd")[0]+"_a_svd_Arr.npy",a_svd_arr)

if optimise_ab == True:
    #opti_ab_params = fibrilParams_file.split(".pkl")[0]+"_opti_ab.pkl"
    with open(fibrilParams_file.split(".pkl")[0]+"_opti_ab.pkl", 'wb') as f:
        pickle.dump(fibrilParams, f)

"""
Clip off regions of A-svd matrix that do not comprise significant simulated scatter
"""



a_svd_sums = np.asarray([np.sum(a_svd_arr[k,:]) for k in np.arange(0,a_svd_arr.shape[0],1)])   

a_svd_pk_starts = [int(masked_len_sums[k]+all_simu_peaks_lims[k][1]) for k in np.arange(0,len(all_simu_peaks_lims),1)]
a_svd_pk_ends = [int(masked_len_sums[k]+all_simu_peaks_lims[k][2]) for k in np.arange(0,len(all_simu_peaks_lims),1)]

a_svd_peaks = [a_svd_arr[a_svd_pk_starts[k]:a_svd_pk_ends[k],:] for k in np.arange(0,len(a_svd_pk_starts),1)]    
a_svd_peaks = np.asarray([item for sublist in a_svd_peaks for item in sublist])

a_peak_sums = np.asarray([np.sum(a_svd_peaks[k,:]) for k in np.arange(0,a_svd_peaks.shape[0],1)])   

b_svd_peaks = [b_svd_arr[a_svd_pk_starts[k]:a_svd_pk_ends[k]] for k in np.arange(0,len(a_svd_pk_starts),1)]   
b_svd_peaks = np.asarray([item for sublist in b_svd_peaks for item in sublist])

mask_svd_peaks = [mask_svd_arr[a_svd_pk_starts[k]:a_svd_pk_ends[k]] for k in np.arange(0,len(a_svd_pk_starts),1)]   
mask_svd_peaks = np.asarray([item for sublist in mask_svd_peaks for item in sublist])

plt.plot(a_svd_sums/np.max(a_svd_sums)+1,label = "normalised (unamplified)\nsimulated (Ichi)")
plt.plot(b_svd_arr/np.max(b_svd_arr),label = "normalised real (Ichi)")
plt.ylabel("Normalised intensity")
plt.xlabel("Concatenated I(chi)")
plt.legend()
plt.show()
plt.close()

"""
Perform SVD estimation
"""

just_peaks = find_peaks(a_peak_sums[b_svd_peaks>0.5e8])[0]

ampval2 = nnls(a_svd_arr, b_svd_arr)[0]

#ampval2=np.linalg.lstsq(a_svd_peaks, b_svd_peaks, rcond=None)[0]
#ampval2=np.linalg.lstsq(a_svd_peaks[just_peaks], b_svd_peaks[just_peaks], rcond=None)[0]
#ampval2=np.linalg.lstsq(a_svd_peaks[b_svd_peaks>0.5e8], b_svd_peaks[b_svd_peaks>0.5e8], rcond=None)[0]


"""
Add initial amplitude estimates to fibril parameter dictionary
"""

svd_indxs = np.arange(0,len(ampval2),1)
          
fibril_idxs = [k["number"] for k in fibrilParams]

amp_test = ampval2[ampval2>0]

pop_stats = np.asarray([len(amp_test[amp_test>k])/len(amp_test) for k in np.logspace(1,10,10)])

ninety_pcnt = np.logspace(1,10,10)[np.where(pop_stats<0.1)[0][0]]

pop_mean = np.mean(amp_test[amp_test<ninety_pcnt])

#test = []
for i in range(0,len(fibrilParams)):
    fib_idx = fibrilParams[i]["number"]
    
    if len(np.where(svd_indxs==fib_idx)[0])>0:
        #if ampval2[np.where(svd_indxs==fib_idx)][0]>0:
        fibrilParams[i]["initial_amp_est"] = ampval2[np.where(svd_indxs==fib_idx)][0]
        #test.append(ampval2[np.where(svd_indxs==fib_idx)][0])
        #else:
            #fibrilParams[i]["initial_amp_est"] = np.mean(ampval2)
    else:
        fibrilParams[i]["initial_amp_est"] = pop_mean

with open(fibrilParams_file.split("_raw")[0]+"_svd_amps.pkl", 'wb') as f:
    pickle.dump(fibrilParams, f)
    
np.save(fibrilParams_file.split("_90_to")[0]+"_ampvals.npy",ampval2)


"""
Fill rotated_beampaths with amp estimates
"""
#test = []
for ridx, r in enumerate(rot_range):
    print("ridx: ", ridx," angle: ", r)
        
    rot_beampath = np.asarray(rotated_beampaths[ridx])
    rot_simu_peaks = []
    
    #if bp_range != "all":
    
    bps = rot_beampath[bp_range]
    #else:
        #bps = rot_beampath
    
    for i, path_dict in enumerate(bps):
        if path_dict != None:
            if path_dict["voxels"]!=None:
                """
                Only add the experimental Ichi for this path into it if the beam intersects at least one fibre
                """
                if bp_range[i] in ichiExp[ridx] and type(ichiExp[ridx][bp_range[i]]["kapton"]) == bool:
                    
                    for idx,voxel in enumerate(path_dict["voxels"]):
                        fib_idx = voxel["fibril_param"]["number"]
                        voxel["fibril_param"]["initial_amp_est"] = ampval2[np.where(svd_indxs==fib_idx)][0]

with open(beampath_file.split(".pkl")[0]+"_svd_amps.pkl", 'wb') as f:
    pickle.dump(rotated_beampaths, f)


"""
Fill new amplitude-adjusted A-SVD matrix with amplified I(chi) data
"""

unmasked_lens = []
    
for r in range(0,len(rot_range)):
    
    rot_mask_len = []
    rot_mask_max = []
    
    rot_bin_len = []
    rot_bin_max = []
    
    for i in range(0,len(rotated_beampaths[0])):
        if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:  
            ichiExp[r][i].update({"unmasked_indx": np.sum(unmasked_lens)})
            unmasked_lens.append(cake_params_chi["nchi"])


#if binning_data != True:
a_svd_arr_amps = [None]*np.sum(unmasked_lens)
b_svd_arr_amps = [0.0]*np.sum(unmasked_lens)
#else:
    #a_svd_arr_amps = [None]*np.sum(masking_lens_bin)

                
for i, val in enumerate(a_svd_arr_amps):
    a_svd_arr_amps[i]=np.zeros(N_fibrils)
a_svd_arr_amps,b_svd_arr_amps  = np.asarray(a_svd_arr_amps),np.asarray(b_svd_arr_amps)

simu_peaks_lims_amps = []
all_simu_peaks_lims_amps = []

for ridx, r in enumerate(rot_range):
    print("ridx: ", ridx," angle: ", r)
        
    rot_beampath = np.asarray(rotated_beampaths[ridx])
    rot_simu_peaks = []
    
    #if bp_range != "all":
    
    bps = rot_beampath[bp_range]
    #else:
        #bps = rot_beampath
    
    for i, path_dict in enumerate(bps):
        if path_dict != None:
            if path_dict["voxels"]!=None:
                """
                Only add the experimental Ichi for this path into it if the beam intersects at least one fibre
                """
                
                if bp_range[i] in ichiExp[ridx] and type(ichiExp[ridx][bp_range[i]]["kapton"]) == bool:
                    
                    pdict_vox_indxs = np.asarray([k["fibril_param"]["number"] for k in path_dict["voxels"]])
                    if len(pdict_vox_indxs)>1:
                        vox_repeats = [k for k in np.arange(1,len(pdict_vox_indxs),1) if pdict_vox_indxs[k]==pdict_vox_indxs[k-1]]
                        if len(vox_repeats)>0:
                            path_dict["voxels"] = [path_dict["voxels"][k] for k in np.arange(0,len(path_dict["voxels"]),1) if k not in vox_repeats]
                            pdict_vox_indxs = np.asarray([k["fibril_param"]["number"] for k in path_dict["voxels"]])
                    
                    for j,voxel in enumerate(path_dict["voxels"]):
                        if len([voxel["fibril_param"]["weight"]]) == 0:
                            voxel["fibril_param"]["weight"] = 1
                    
                    #if binning_data != True:
                    ichi=ichiExp[ridx][bp_range[i]]["ichi"] 
                    full_indx = int(ichiExp[ridx][bp_range[i]]["unmasked_indx"]) 
                    #else:
                        #ichi=ichiExp[ridx][bp_range[i]]["ichi_binned"]  
                        #full_indx = int(ichiExp[ridx][bp_range[i]]["bin_indx"]) 
                    
                    chi_len = len(ichi) 

                    data_masked = [ichi[k] for k in data_bounds]
                    #ichi_masked = np.asarray([item for sublist in data_masked for item in sublist])  
                    data_smooth = [savgol_filter(k,window,pol) for k in data_masked]
                    #smpl_smooth = np.asarray([item for sublist in data_smooth for item in sublist])
                    ichi_masked = np.zeros_like(ichi)  
                    ichi_masked[ichi_masked==0] = np.nan
                    for k in range(0,len(data_bounds)):
                        ichi_masked[data_bounds[k]] = data_smooth[k]              
                  
                    #full_indx = int(ichiExp[ridx][bp_range[i]]["full_indx"]) 
                    
                    indexArr0 = np.linspace(0,chi_len-1,chi_len).astype(int)
                                        
                    indexArrTr0 = np.asarray([indexArr0]).T
                                                                
                    ichiTr = np.asarray([ichi_masked]).T
                    mask_iTr = np.asarray([mask_i]).T
                    
                    b_svd_tmp = np.zeros_like(b_svd_arr_amps[full_indx:full_indx+chi_len])
                    np.put(b_svd_tmp,indexArrTr0,ichiTr)
                    b_svd_arr_amps[full_indx:full_indx+chi_len] = b_svd_arr_amps[full_indx:full_indx+chi_len]+b_svd_tmp
                    #mask_svd_arr[full_indx:full_indx+chi_len] = mask_svd_arr[full_indx:full_indx+chi_len]+mask_i
                                                        
                    b_svd_clip = b_svd_arr_amps[full_indx:full_indx+chi_len]
                                                        
                    comb_ichi = np.zeros_like(b_svd_clip)
                    
                    start_pts,end_pts = [],[]
                    
                    a_svd_arr_amps,bareichis,comb_ichi,bp_peak_start,bp_peak_end = a_svd(a_svd_arr_amps,comb_ichi,[full_indx,indexArrTr0,chi_len],
                                                                          path_dict,params0,cake_params_chi,mask_bounds,ampval2,svd_indxs,data_bounds,mask_ichi = True,svd_amp = True)                    

                    rot_simu_peaks.append([bp_range[i],bp_peak_start,bp_peak_end])
                    all_simu_peaks_lims_amps.append([bp_range[i],bp_peak_start,bp_peak_end])
                    
    simu_peaks_lims_amps.append(rot_simu_peaks)
    
np.save(b_svd_file.split("_b_svd")[0]+"_a_svd_Arr_amps.npy",a_svd_arr_amps)

"""
Clip non-significant scater from amplitude-adjusted matrix 
"""
a_svd_pk_amp_starts = [int(unmasked_lens[k]+all_simu_peaks_lims_amps[k][1]) for k in np.arange(0,len(all_simu_peaks_lims_amps),1)]
a_svd_pk_amp_ends = [int(unmasked_lens[k]+all_simu_peaks_lims_amps[k][2]) for k in np.arange(0,len(all_simu_peaks_lims_amps),1)]

a_svd_peaks_amps = [a_svd_arr_amps[a_svd_pk_amp_starts[k]:a_svd_pk_amp_ends[k],:] for k in np.arange(0,len(a_svd_pk_amp_starts),1)]    
a_svd_peaks_amps = np.asarray([item for sublist in a_svd_peaks_amps for item in sublist])

a_svd_peak_sum = np.asarray([np.sum(a_svd_peaks_amps[k,:]) for k in np.arange(0,a_svd_peaks_amps.shape[0],1)]) 
    
b_svd_peaks_amps = [b_svd_arr_amps[a_svd_pk_amp_starts[k]:a_svd_pk_amp_ends[k]] for k in np.arange(0,len(a_svd_pk_amp_starts),1)]   
b_svd_peaks_amps = np.asarray([item for sublist in b_svd_peaks_amps for item in sublist])

#mask_svd_peaks_amps = [mask_svd_arr[a_svd_pk_amp_starts[k]:a_svd_pk_amp_ends[k]] for k in np.arange(0,len(a_svd_pk_amp_starts),1)]   
#mask_svd_peaks_amps = np.asarray([item for sublist in mask_svd_peaks_amps for item in sublist])

#a_svd_peaks_masked = np.where(mask_svd_peaks_amps==0,a_svd_peak_sum,0)
#b_svd_peaks_masked = np.where(mask_svd_peaks_amps==0,b_svd_peaks_amps,0)

a_amp_sums = np.asarray([np.sum(a_svd_arr_amps[k,:]) for k in np.arange(0,a_svd_arr_amps.shape[0],1)])   
 
#a_amp_sums = a_amp_sums*(np.max(b_svd_arr)/np.max(a_amp_sums))

"""
plot results
"""

if ichi_plot == True:  
            
    a_svd_peaks_locs = find_peaks(a_amp_sums)[0]
    
    x,y = a_amp_sums[a_svd_peaks_locs],b_svd_arr_amps[a_svd_peaks_locs]
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k',label = "r^2: "+str(r_value)+"\np-val: "+str(p_value)) #'--k'=black dashed line, 'yo' = yellow circle marker
    plt.legend()
    
    
    plt.scatter(a_svd_peak_sum[a_svd_peaks_locs],b_svd_peaks_amps[a_svd_peaks_locs])
    plt.ylabel("real peak amplitudes")
    plt.xlabel("simulated peak amplitudes")
    plt.title("correlation between amplified simulated I(chi) peaks\nand real I(chi) peaks")
    plt.savefig(b_svd_file.split("_b_svd")[0]+"_SVD_correlation_plot.jpeg")
    plt.show()
    plt.close()    
    
    for k in range(0,int(len(a_amp_sums)/10)):
        chi = np.arange(k*1000,(k+1)*1000,1)
        plt.plot(chi,a_svd_peak_sum[k*1000:(k+1)*1000],label = "amp sim")
        plt.plot(chi,b_svd_peaks_amps[k*1000:(k+1)*1000],label = "real data")
        plt.ylim([np.min(a_svd_peak_sum),np.max(b_svd_peaks_amps)])
        plt.legend()
        plt.savefig("D:/SM29784-8/new_rotation_test/og_ab_svd_results/pos_svd/"+str(k*1000)+"-"+str((k+1)*1000)+".jpeg")    
        plt.show()
        plt.close() 
