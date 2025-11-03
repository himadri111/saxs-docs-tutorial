#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:57:52 2025

@author: via83767
"""

import numpy as np
import os,glob
import pickle
from time import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
#import PySimpleGUI as sg
import re
from pathlib import Path
from h5py import File
from numpy import array, ones
import hdf5plugin
#import cv2

from scipy.signal import savgol_filter
from scipy import signal

import statistics
from scipy import ndimage

from lmfit import Parameters
from lmfit.models import GaussianModel,ExponentialModel,LinearModel,VoigtModel,LorentzianModel
from lmfit.models import SkewedGaussianModel,SkewedVoigtModel,Model,ConstantModel,PowerLawModel
from lmfit import minimize, report_fit

from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations

import sys
sys.path.append(r'/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/scripts')
import threeDXRD_080923 as t3d
sys.path.append(r'/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/scripts')
from delta_method import delta_method, parametric_bootstrap
from numpy import exp, linspace
import scipy.optimize as opt

from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator

"""
data_path = 'D:/SM29784-8/new_rotation_test/'
output_path = data_path

dict_files = glob.glob(os.path.join(data_path, "*.pkl"))
npy_files = glob.glob(os.path.join(data_path, "*.npy"))
lhs_dict_files = [k for k in dict_files if "90-to-270" in k]
lhs_npy_files = [k for k in npy_files if "90-to-270" in k]
fibrilParams_file = [k for k in lhs_dict_files if "_fibril_params_raw" in k][0]
beampath_file = [k for k in dict_files if "rotation_beampaths" in k][0]
init_struct_file = [k for k in dict_files if "_init_structure" in k][0]
ichi_dict_file = [k for k in lhs_dict_files if "_ichi_dict_filled" in k][0]
cake_params_file = [k for k in dict_files if "svd_cake_params" in k][0]
"""

gamma0, deltaGamma0 = np.radians(0.0), 0.01
mu = np.radians(0.0)


def snr_test(q_data,i_data,cake_params,threshold):
    
    """
    Function to test the signal-to-noise ratio of I(q) samples
    (calculated as the ratio between the maximum peak intensity and the 
     standard deviation of the background).
    
    If the maximum SNR is above a certain threshold - pass true;
    if not - pass false
    """
    
    snr_passed = False
    
    q1 = cake_params["q1"]
    q2 = cake_params["q2"]
    
    peak_data = [i_data[k][(q_data[k] >= q1) & (q_data[k] <= q2)] for k in np.arange(0,len(q_data),1)]
    
    bg_indxs = [[k for k in np.arange(0,len(i),1) if i[k]<q1 or i[k]>q2] for i in q_data]
    
    bg_data = [[i_data[i][k] for k in bg_indxs[i]] for i in np.arange(0,len(i_data),1)]
            
    snr = [np.max(peak_data[k])/np.std(bg_data[k]) for k in np.arange(0,len(q_data),1)]
    
    if np.max(snr)> threshold:
        snr_passed  = True
        
    return snr_passed

# x, ichi_tot, ichi1, ichi2, ichi3, dx,thresh_detect,thresh_combined,thresh_individual,frac_threshold,ang_frac = chirange,Ichi_us,ichi_center,ichi_nbr1,ichi_nbr2,chiRefWindow,0.5,0.999,0.15,0.9,0.3

def solveTripleWrap(x, ichi_tot, ichi1, ichi2, ichi3, dx,
                    thresh_detect=0.5,
                    thresh_combined=0.8,
                    thresh_individual=0.15,
                    frac_threshold=0.9,
                    ang_frac=0.3,
                    plot_ratios=True,
                    plot_window=True):
    """
    Identify a window of width dx in x (with wraparound) where:
      1) The sum of Ichi1 + Ichi2 + Ichi3 dominates the total (i.e. triple ratio >= thresh_combined)
         for at least frac_threshold fraction of the points in that window.
      2) Each component is above thresh_detect in those points.
      3) Each individual component’s ratio (Ichi_i/ichi_tot) is >= thresh_individual for at least ang_frac fraction of the window.

    If such a window exists, the function shades it on the provided matplotlib axis (ax) and returns:
      solved (bool): True if a valid window is found.
      best_window (tuple): (start_x, end_x) of the best window. For a wrapped window, end_x < start_x.
      best_ratio (float): Average triple ratio in that window.
      ax: The matplotlib axis with the shaded region added.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot and shade the region.
    x : 1D numpy array
        x-values (assumed sorted in ascending order, e.g. [0,1]).
    ichi_tot : 1D array
        Total intensity.
    ichi1, ichi2, ichi3 : 1D arrays
        Intensities for the three components.
    dx : float
        Width of the window in the same units as x.
    thresh_detect : float, optional
        Absolute detection threshold (default 0.5).
    thresh_combined : float, optional
        Minimum triple ratio (Ichi1+Ichi2+Ichi3)/ichi_tot required (default 0.8).
    thresh_individual : float, optional
        Minimum ratio each component must have relative to ichi_tot (default 0.15).
    frac_threshold : float, optional
        Fraction of points in the window that must satisfy the global condition (default 0.9).
    ang_frac : float, optional
        Fraction of points in the window that must satisfy the individual condition for each component (default 0.3).
    plot_ratios : bool, optional
        If True, plot the ratios I1/Itot, I2/Itot, I3/Itot, and their sum on ax before shading (default True).

    Returns
    -------
    solved : bool
        True if a valid window is found.
    best_window : tuple or None
        (start_x, end_x) of the best window. In wraparound, end_x < start_x.
    best_ratio : float
        Average triple ratio in that window.
    ax : matplotlib.axes.Axes
        The axis with the ratio plots and shaded region added.
    """
    domain_min, domain_max = x[0], x[-1]
    domain_size = domain_max - domain_min
    n_points = len(x)
    
    solved = False
    best_window = None
    best_ratio = -1.0
    
    eps = 1e-12
    
    # Compute triple intensity and ratio
    triple_intensity = ichi1 + ichi2 + ichi3 + eps
    triple_ratio = triple_intensity / (ichi_tot + eps)
        

    # Iterate over all possible window start positions (sample each point)
    for start_idx in range(n_points):
        start_val = x[start_idx]
        end_val = start_val + dx
        
        # Get indices in the current window, taking wrap into account
        idx_in_window = get_window_indices(x, start_val, dx, domain_min, domain_max)
        if len(idx_in_window) == 0:
            continue
        
        # Global condition per point: each component above thresh_detect and triple ratio >= thresh_combined.
        cond1 = (ichi1[idx_in_window] >= thresh_detect)
        cond2 = (ichi2[idx_in_window] >= thresh_detect)
        cond3 = (ichi3[idx_in_window] >= thresh_detect)
        cond4 = (triple_ratio[idx_in_window] >= thresh_combined)
        global_condition = cond1 & cond2 & cond3 & cond4
        frac_satisfied = np.mean(global_condition)
        
        # Individual condition: each component's ratio (Ichi_i/ichi_tot) >= thresh_individual
        ratio1_w = ichi1[idx_in_window] / (ichi_tot[idx_in_window] + eps)
        ratio2_w = ichi2[idx_in_window] / (ichi_tot[idx_in_window] + eps)
        ratio3_w = ichi3[idx_in_window] / (ichi_tot[idx_in_window] + eps)
        frac1 = np.mean(ratio1_w >= thresh_individual)
        frac2 = np.mean(ratio2_w >= thresh_individual)
        frac3 = np.mean(ratio3_w >= thresh_individual)
        individual_condition = (frac1 >= ang_frac) and (frac2 >= ang_frac) and (frac3 >= ang_frac)
        
        if (frac_satisfied >= frac_threshold) and individual_condition:
            avg_ratio = np.mean(triple_ratio[idx_in_window])
            if avg_ratio > best_ratio:
                solved = True
                best_ratio = avg_ratio
                # Determine best window boundaries (account for wrap)
                if end_val <= domain_max:
                    best_window = (start_val, end_val)
                else:
                    best_window = (start_val, end_val - domain_size)
    
    
        
    return solved, best_window, best_ratio


def get_window_indices(x, start_val, dx, domain_min, domain_max):
    """
    Return the indices of points in x that lie within [start_val, start_val+dx],
    taking into account wraparound.
    """
    domain_size = domain_max - domain_min
    end_val = start_val + dx
    if end_val <= domain_max:
        return np.where((x >= start_val) & (x <= end_val))[0]
    else:
        # Wraparound case: select points in [start_val, domain_max] and [domain_min, end_val - domain_size]
        idx1 = np.where((x >= start_val) & (x <= domain_max))[0]
        idx2 = np.where((x >= domain_min) & (x <= end_val - domain_size))[0]
        return np.concatenate((idx1, idx2))

def checkSolvableTriple(chi,ichi_cen, ichi_nbr, ichi_nbr2, ichi_all,TripleOverlapParams,TripleOverlapThresh):
    threshwindow    = TripleOverlapParams["fitwindow"] #because threshold window serves same role as fitwindow
    thresh_detect = TripleOverlapThresh["thresh_detect"]
    thresh_combined = TripleOverlapThresh["I3_tot"]
    thresh_individual = TripleOverlapThresh["I3_singleton"]
    thresh_angular = TripleOverlapThresh["angular_fraction"]
    
    ichi_triple = ichi_cen+ichi_nbr+ichi_nbr2
    ichi1 = ichi_cen
    ichi2 = ichi_nbr
    ichi3 = ichi_nbr2
    chi_z = chi - chi[0]
    chiwin_length = len(chi_z[chi_z<=threshwindow ])
    #print("chiwin_length: ", chiwin_length)
    #chiwin_length_23rds = int((thresh_angular/100)*chiwin_length) #integer form of 1/3ds of window length
    solved=False
    solved_window = None
    
    ichi1_ratio = ichi1/ichi_all
    ichi2_ratio = ichi2/ichi_all
    ichi3_ratio = ichi3/ichi_all
    ichi_triple_ratio = ichi_triple/ichi_all #-> ichi_pair_win OK
    """
    using stride
    """
    chi_sw       = sliding_window_view(chi, chiwin_length)
    #print("chi_sw: ", chi_sw)
    ichi1_sw     = sliding_window_view(ichi1, chiwin_length)
    ichi2_sw     = sliding_window_view(ichi2, chiwin_length)
    ichi3_sw     = sliding_window_view(ichi3, chiwin_length)
    ichi_triple_sw = sliding_window_view(ichi_triple, chiwin_length)
    
    ichi1_ratio_sw     = sliding_window_view(ichi1_ratio, chiwin_length)
    ichi2_ratio_sw     = sliding_window_view(ichi2_ratio, chiwin_length)
    ichi3_ratio_sw     = sliding_window_view(ichi3_ratio, chiwin_length)
    ichi_triple_ratio_sw = sliding_window_view(ichi_triple_ratio, chiwin_length)
    """
    what stride returns is a numpy array of length chi on axis 0, with the slices of the chi, ratio, Ichi etc hanging "down" on axis 1
    This is why applying np.all() or np.average() on axis=1 just means a flattened 1D array along axis 0, all of same length
    and therefore comparable
    
    The below array is a Boolean selecting where the triplet intensity, and the individual components 1,2,3]
    are all above absolute detector threshold AND where the ratio of triplet intensity to total intensity>95%,
    i.e. dominates the scattering
    """
    ichi_allconds = np.all((ichi_triple_sw>=thresh_detect) & (ichi1_sw>=thresh_detect) 
                           & (ichi2_sw>=thresh_detect)   & (ichi3_sw>=thresh_detect) 
                           & (ichi_triple_ratio_sw>=(thresh_combined/100)),axis=1)
    # ichi_allconds = np.all((ichi_pair_sw>=thresh_detect) 
    #                        & (ichi_pair_ratio_sw>=(thresh_combined/100)),axis=1)
    # # ichi_allconds = np.all((ichi_pair_sw>=thresh_detect) & (ichi1_sw>=thresh_detect) 
    #                        & (ichi2_sw>=thresh_detect) 
    #                        & (ichi_pair_ratio_sw>=(thresh_combined/100)),axis=1)
    #print("in triple: thresh_detect: ", thresh_detect, ", thresh_combined: ", thresh_combined)
    #print(ichi_allconds)
    """
    define Boolean array where each of ichi1, ichi2, ichi3 are greater than the threshold 
    for at least 1/3rd of the window
    """
    threshold_frac = thresh_individual/100  # intensity ratio of ichi1 (and 2,3) to ichi_all. 
                                            # E.g. thresh_individual = 90, threshold_frac = 0.90
                                            
    angular_frac   = thresh_angular/chiwin_length   # angular fraction over chiwin where this holds
                                                    # e.g. if thresh_angular = 5, and chiwin_length = 10, angular_frac = 0.5
    
    """
    This is a 360-degree 1D array, where each element is 
    
    count_satisfying = (v >= threshold).sum(axis=1) counts, for each row, how many elements are >= threshold
    fraction_satisfying = count_satisfying / v.shape[1] calculates the fraction of elements that meet that condition for each row.
    mask1...: creates a boolean mask of rows whose fraction exceeds angular_frac.
              so mask1 is a 1D 360 degree array, basically a set of angular windows, which says True or False 
              if that angular window has intensity from that fibre over the threshold over a critical angular fraction
              
              0.2 or 20%
              
              ichi1 
              0(0-10) 1(1-11) 2(2-12)...
              3/10    8/10    9/10
              0.3     0.8     0.8
                              [2  3    4   5    6    7   8    9    10  11 ]
                              0.1 0.15 0.2 0.22 0.23 0.4 0.5  0.7 0.8 0.9 
                              
              
              False  True     True
              
              ich2
              0(0-10) 1(1-11) 2(2-12)...
              1/10    2/10    5/10
                              0.5
                              [2  3    4   5    6    7   8     9    10  11]
                              0.8 0.7 0.4  0.3  0.2 0.1  0.05 0.02 0.01 0.01
                              
    """
    count_satisfying1 = (ichi1_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying1 = count_satisfying1 / ichi1_ratio_sw.shape[1]
    mask1 = fraction_satisfying1 > angular_frac #1D 360 degree length array with points satisfying angualr_frac
    
    count_satisfying2 = (ichi2_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying2 = count_satisfying2 / ichi2_ratio_sw.shape[1]
    mask2 = fraction_satisfying2 > angular_frac
    
    count_satisfying3 = (ichi3_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying3 = count_satisfying3 / ichi3_ratio_sw.shape[1]
    mask3 = fraction_satisfying3 > angular_frac
    
    """
    mask takes the intersection of masks from each of the three fibres to find the angular windows where
    the condition is satisfied for all the three fibres.
    """
    mask = mask1 & mask2 & mask3

    ichi_triple_ratio_av = np.average(ichi_triple_ratio_sw,axis=1)
    
    """
    the "final" Boolean array which combines all the conditions
    """
    # ichi_pair_ratio_av_sel = ichi_pair_ratio_av[(ichi1_2_overthresh==True)&(ichi_allconds==True)]
    # chi_sw_sel                         = chi_sw[(ichi1_2_overthresh==True)&(ichi_allconds==True)]
    
    # ichi_pair_ratio_av_sel = ichi_pair_ratio_av[ (mask==True) & (ichi_allconds==True) ]
    # chi_sw_sel                         = chi_sw[ (mask==True) & (ichi_allconds==True) ]
    ichi_triple_ratio_av_sel = ichi_triple_ratio_av[ (mask==True) & (ichi_allconds==True) ]
    chi_sw_sel                         = chi_sw[ (mask==True) & (ichi_allconds==True) ]
    
    if len(chi_sw_sel>0):    
        """
        at least one element in the array. find the "best" value via argmax()
        """
        idx = np.argmax(ichi_triple_ratio_av_sel)
        solved_window = chi_sw_sel[idx]
        solved=True     
    return solved, solved_window

#chi,ichi_cen, ichi_nbr, ichi_all,overlapParams,overlapThresh = chirange,ichi_center,ichi_nbr,Ichi_us,overlap_params,overlap_thresh

def checkSolvablePair(chi,ichi_cen, ichi_nbr, ichi_all,overlapParams,overlapThresh):
    threshwindow    = overlapParams["fitwindow"] #because threshold window serves same role as fitwindow
    thresh_detect = overlapThresh["thresh_detect"]
    thresh_combined = overlapThresh["thresh_combined"]
    thresh_individual = overlapThresh["thresh_individual"]
    
    ichi_pair = ichi_cen+ichi_nbr
    ichi1 = ichi_cen
    ichi2 = ichi_nbr
    chi_z = chi - chi[0]
    chiwin_length = len(chi_z[chi_z<=threshwindow ])
    solved=False
    solved_window = None
    
    ichi1_ratio = ichi1/ichi_all
    ichi2_ratio = ichi2/ichi_all
    ichi_pair_ratio = ichi_pair/ichi_all #-> ichi_pair_win OK
    #print("ichi_all: ", ichi_all)
    """
    using stride
    """
    chi_sw       = sliding_window_view(chi, chiwin_length)
    ichi1_sw     = sliding_window_view(ichi1, chiwin_length)
    ichi2_sw     = sliding_window_view(ichi2, chiwin_length)
    ichi_pair_sw = sliding_window_view(ichi_pair, chiwin_length)
    
    ichi1_ratio_sw     = sliding_window_view(ichi1_ratio, chiwin_length)
    ichi2_ratio_sw     = sliding_window_view(ichi2_ratio, chiwin_length)
    ichi_pair_ratio_sw = sliding_window_view(ichi_pair_ratio, chiwin_length)
    """
    what stride returns is a numpy array of length chi on axis 0, with the slices of the chi, ratio, Ichi etc hanging "down" on axis 1
    This is why applying np.all() or np.average() on axis=1 just means a flattened 1D array along axis 0, all of same length
    and therefore comparable
    """
    ichi_allconds = np.all((ichi_pair_sw>=thresh_detect) & (ichi1_sw>=thresh_detect) 
                           & (ichi2_sw>=thresh_detect) & (ichi_pair_ratio_sw>=(thresh_combined/100)),axis=1)
    """
    note ichi_allconds here deals with ABSOLUTE intensity values, while ichi1_ratio... ichi_pair_ratio...
    all deal with RATIOS of intensity values. They are then combined into the final Boolean array 
    ichi_pair_ratio_av_sel which combines the individual ratio (ichi1_2_overthresh) tests with the absolute 
    intensity test (ichi_allconds)
    """
    ichi1_ratio_av  = np.average(ichi1_ratio_sw,axis=1)
    ichi2_ratio_av  = np.average(ichi2_ratio_sw,axis=1)
    ichi1_2_overthresh = ((ichi1_ratio_av>(thresh_individual/100)) & (ichi2_ratio_av>(thresh_individual/100)))
    
    ichi_pair_ratio_av = np.average(ichi_pair_ratio_sw,axis=1)
    
    """
    the "final" Boolean array which combines all the conditions
    """
    ichi_pair_ratio_av_sel = ichi_pair_ratio_av[(ichi1_2_overthresh==True)&(ichi_allconds==True)]
    chi_sw_sel                         = chi_sw[(ichi1_2_overthresh==True)&(ichi_allconds==True)]

    
    if len(chi_sw_sel>0):    
        """
        at least one element in the array. find the "best" value via argmax()
        """
        idx = np.argmax(ichi_pair_ratio_av_sel)
        solved_window = chi_sw_sel[idx]
        solved=True     
    return solved, solved_window

#i,index,pdict,voxList,chimaxList,ichimaxList,overlapPars,threshold = idx,idx,path_dict["voxels"],copy.deepcopy(path_dict["voxels"]),IchiCptsChiMax,IchiCptsIChiMax,overlap_params,threshold_detection

def FindNbrIndices(i,index,pdict,voxList,chimaxList,ichimaxList,overlapPars,threshold):
    """
    * overlap_params = {"searchwindow": 10, "fitwindow": 10, "minseparation": 4}
    * search the pdict to find points within the searchwindow and above threshold
    * 
    * note this does not make any estimate of the combined nbr+main voxel intensity
    * ratio over the others. It only checks if there are some adjacent voxels 
    * which are neighbours. Testing the ratio will be the next function
    """
    searchwindow = overlapPars["searchwindow"]
    minseparation = overlapPars["minseparation"]
    #pdict_sorted = sorted(pdict, key=lambda x: x['chimax'])
    """
    find the position of i in new list
    """
    voxel_centre = pdict[i]
    chimax_centre = chimaxList[index]
    nbrlist = []
    for i,pval in enumerate(pdict):
        # need to check that it is also unsolved
        if pval["fibril_param"]["solved"]==False:
            indx = pval["fibril_param"]["indx"]
            chimax  = chimaxList[i]
            ichimax = ichimaxList[i]
            if (np.abs(chimax-chimax_centre)<=searchwindow) and (np.abs(chimax-chimax_centre)>=minseparation) \
                and (ichimax>=threshold):
                nbrlist.append(i)
    return nbrlist


def checkSolvableTriple(chi,ichi_cen, ichi_nbr, ichi_nbr2, ichi_all,TripleOverlapParams,TripleOverlapThresh):
    threshwindow    = TripleOverlapParams["fitwindow"] #because threshold window serves same role as fitwindow
    thresh_detect = TripleOverlapThresh["thresh_detect"]
    thresh_combined = TripleOverlapThresh["I3_tot"]
    thresh_individual = TripleOverlapThresh["I3_singleton"]
    thresh_angular = TripleOverlapThresh["angular_fraction"]
    
    ichi_triple = ichi_cen+ichi_nbr+ichi_nbr2
    ichi1 = ichi_cen
    ichi2 = ichi_nbr
    ichi3 = ichi_nbr2
    chi_z = chi - chi[0]
    chiwin_length = len(chi_z[chi_z<=threshwindow ])
    #print("chiwin_length: ", chiwin_length)
    #chiwin_length_23rds = int((thresh_angular/100)*chiwin_length) #integer form of 1/3ds of window length
    solved=False
    solved_window = None
    
    ichi1_ratio = ichi1/ichi_all
    ichi2_ratio = ichi2/ichi_all
    ichi3_ratio = ichi3/ichi_all
    ichi_triple_ratio = ichi_triple/ichi_all #-> ichi_pair_win OK
    """
    using stride
    """
    chi_sw       = sliding_window_view(chi, chiwin_length)
    #print("chi_sw: ", chi_sw)
    ichi1_sw     = sliding_window_view(ichi1, chiwin_length)
    ichi2_sw     = sliding_window_view(ichi2, chiwin_length)
    ichi3_sw     = sliding_window_view(ichi3, chiwin_length)
    ichi_triple_sw = sliding_window_view(ichi_triple, chiwin_length)
    
    ichi1_ratio_sw     = sliding_window_view(ichi1_ratio, chiwin_length)
    ichi2_ratio_sw     = sliding_window_view(ichi2_ratio, chiwin_length)
    ichi3_ratio_sw     = sliding_window_view(ichi3_ratio, chiwin_length)
    ichi_triple_ratio_sw = sliding_window_view(ichi_triple_ratio, chiwin_length)
    """
    what stride returns is a numpy array of length chi on axis 0, with the slices of the chi, ratio, Ichi etc hanging "down" on axis 1
    This is why applying np.all() or np.average() on axis=1 just means a flattened 1D array along axis 0, all of same length
    and therefore comparable
    
    The below array is a Boolean selecting where the triplet intensity, and the individual components 1,2,3]
    are all above absolute detector threshold AND where the ratio of triplet intensity to total intensity>95%,
    i.e. dominates the scattering
    """
    ichi_allconds = np.all((ichi_triple_sw>=thresh_detect) & (ichi1_sw>=thresh_detect) 
                           & (ichi2_sw>=thresh_detect)   & (ichi3_sw>=thresh_detect) 
                           & (ichi_triple_ratio_sw>=(thresh_combined/100)),axis=1)
    # ichi_allconds = np.all((ichi_pair_sw>=thresh_detect) 
    #                        & (ichi_pair_ratio_sw>=(thresh_combined/100)),axis=1)
    # # ichi_allconds = np.all((ichi_pair_sw>=thresh_detect) & (ichi1_sw>=thresh_detect) 
    #                        & (ichi2_sw>=thresh_detect) 
    #                        & (ichi_pair_ratio_sw>=(thresh_combined/100)),axis=1)
    #print("in triple: thresh_detect: ", thresh_detect, ", thresh_combined: ", thresh_combined)
    #print(ichi_allconds)
    """
    define Boolean array where each of ichi1, ichi2, ichi3 are greater than the threshold 
    for at least 1/3rd of the window
    """
    threshold_frac = thresh_individual/100  # intensity ratio of ichi1 (and 2,3) to ichi_all. 
                                            # E.g. thresh_individual = 90, threshold_frac = 0.90
                                            
    angular_frac   = thresh_angular/chiwin_length   # angular fraction over chiwin where this holds
                                                    # e.g. if thresh_angular = 5, and chiwin_length = 10, angular_frac = 0.5
    
    """
    This is a 360-degree 1D array, where each element is 
    
    count_satisfying = (v >= threshold).sum(axis=1) counts, for each row, how many elements are >= threshold
    fraction_satisfying = count_satisfying / v.shape[1] calculates the fraction of elements that meet that condition for each row.
    mask1...: creates a boolean mask of rows whose fraction exceeds angular_frac.
              so mask1 is a 1D 360 degree array, basically a set of angular windows, which says True or False 
              if that angular window has intensity from that fibre over the threshold over a critical angular fraction
              
              0.2 or 20%
              
              ichi1 
              0(0-10) 1(1-11) 2(2-12)...
              3/10    8/10    9/10
              0.3     0.8     0.8
                              [2  3    4   5    6    7   8    9    10  11 ]
                              0.1 0.15 0.2 0.22 0.23 0.4 0.5  0.7 0.8 0.9 
                              
              
              False  True     True
              
              ich2
              0(0-10) 1(1-11) 2(2-12)...
              1/10    2/10    5/10
                              0.5
                              [2  3    4   5    6    7   8     9    10  11]
                              0.8 0.7 0.4  0.3  0.2 0.1  0.05 0.02 0.01 0.01
                              
    """
    count_satisfying1 = (ichi1_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying1 = count_satisfying1 / ichi1_ratio_sw.shape[1]
    mask1 = fraction_satisfying1 > angular_frac #1D 360 degree length array with points satisfying angualr_frac
    
    count_satisfying2 = (ichi2_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying2 = count_satisfying2 / ichi2_ratio_sw.shape[1]
    mask2 = fraction_satisfying2 > angular_frac
    
    count_satisfying3 = (ichi3_ratio_sw  >= threshold_frac).sum(axis=1)
    fraction_satisfying3 = count_satisfying3 / ichi3_ratio_sw.shape[1]
    mask3 = fraction_satisfying3 > angular_frac
    
    """
    mask takes the intersection of masks from each of the three fibres to find the angular windows where
    the condition is satisfied for all the three fibres.
    """
    mask = mask1 & mask2 & mask3

    ichi_triple_ratio_av = np.average(ichi_triple_ratio_sw,axis=1)
    
    """
    the "final" Boolean array which combines all the conditions
    """
    # ichi_pair_ratio_av_sel = ichi_pair_ratio_av[(ichi1_2_overthresh==True)&(ichi_allconds==True)]
    # chi_sw_sel                         = chi_sw[(ichi1_2_overthresh==True)&(ichi_allconds==True)]
    
    # ichi_pair_ratio_av_sel = ichi_pair_ratio_av[ (mask==True) & (ichi_allconds==True) ]
    # chi_sw_sel                         = chi_sw[ (mask==True) & (ichi_allconds==True) ]
    ichi_triple_ratio_av_sel = ichi_triple_ratio_av[ (mask==True) & (ichi_allconds==True) ]
    chi_sw_sel                         = chi_sw[ (mask==True) & (ichi_allconds==True) ]
    
    if len(chi_sw_sel>0):    
        """
        at least one element in the array. find the "best" value via argmax()
        """
        idx = np.argmax(ichi_triple_ratio_av_sel)
        solved_window = chi_sw_sel[idx]
        solved=True     
    return solved, solved_window



#def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    
    #"""
    #Function for resizing an image using cv2 to retain as much of the original 
    #per-pixel values as possible.
    #"""
    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    #dim = None
    #(h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    #if width is None and height is None:
        #return image

    # check to see if the width is None
    #if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        #r = height / float(h)
        #dim = (int(w * r), height)

    # otherwise, the height is None
    #else:
        # calculate the ratio of the width and construct the
        # dimensions
        #r = width / float(w)
        #dim = (width, int(h * r))

    # resize the image
    #resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    #return resized



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def getRotatedFibrilPoints(fibrilParams,r,rot,scan_data,fib_sample_data):
    
    """
    Modification of simulation function that reads in real data for respective
    tomoSAXS orientation and processes into library for being read by other functions
    and stages of the tomoSAXS reconstruction process.
    """
    
    slice_scan_Data = np.copy(scan_data)
    
    slice_alpha_index,slice_beta_index,slice_index_index,slice_counts_index,slice_weights_index = slice_scan_Data[0:5]
    
    dxs,samplex1,dzs,samplez1 = fib_sample_data[0:4]
    
    fibre_index = [[slice_index_index[rot][i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != "int" 
                    and np.max(slice_index_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]

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
    
    r_alpha_index = [[slice_alpha_index[rot][i,k][0:len(slice_alpha_index[rot][i,k])] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != "int" and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]
    r_alpha_voxs = [[[i,k] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != "int" and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]

    r_beta_index = [[slice_beta_index[rot][i,k][0:len(slice_beta_index[rot][i,k])] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != "int" and np.max(slice_beta_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]
    r_beta_voxs = [[[i,k] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != "int" and np.max(slice_beta_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]

    r_fibre_index = [[slice_index_index[rot][i,k][0:len(slice_index_index[rot][i,k])] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != "int" and np.max(slice_index_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]
    r_fibre_voxs = [[[i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != "int" and np.max(slice_index_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]

    r_count_index = [[slice_counts_index[rot][i,k][0:len(slice_counts_index[rot][i,k])] for k in np.arange(0,slice_counts_index[rot].shape[1],1) if type(slice_counts_index[rot][i,k]) != "int" and np.max(slice_counts_index[rot][i,k]) !=0] for i in np.arange(0,slice_counts_index[rot].shape[0],1)]
    r_weight_index = [[slice_weights_index[rot][i,k][0:len(slice_weights_index[rot][i,k])] for k in np.arange(0,slice_weights_index[rot].shape[1],1) if type(slice_weights_index[rot][i,k]) != "int" and np.max(slice_weights_index[rot][i,k]) !=0] for i in np.arange(0,slice_weights_index[rot].shape[0],1)]    
    r_vox_index = [[[i,k] for k in np.arange(0,slice_index_index[rot].shape[1],1) if type(slice_index_index[rot][i,k]) != "int" and np.max(slice_index_index[rot][i,k]) !=0] for i in np.arange(0,slice_index_index[rot].shape[0],1)]
    
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
                
        r_alpha_index = [[slice_alpha_index[rot][i,k][0:len(slice_alpha_index[rot][i,k])] for k in np.arange(0,slice_alpha_index[rot].shape[1],1) if type(slice_alpha_index[rot][i,k]) != "int" and np.max(slice_alpha_index[rot][i,k]) !=0] for i in np.arange(0,slice_alpha_index[rot].shape[0],1)]
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
                
        r_beta_index = [[slice_beta_index[rot][i,k][0:len(slice_beta_index[rot][i,k])] for k in np.arange(0,slice_beta_index[rot].shape[1],1) if type(slice_beta_index[rot][i,k]) != "int" and np.max(slice_beta_index[rot][i,k]) !=0] for i in np.arange(0,slice_beta_index[rot].shape[0],1)]

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
    r_fibre_betas = r_fibre_betas-20
    
    og_fibre_idxs = np.asarray([k["indx"] for k in fibrilParams])
    
    """
    fill rotated fibril dictionary
    """

    rotatedFibrilParams = [None]*len(r_fibre_indexes)
    test = []
    r_xTest,r_zTest = [],[]
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
            
            weight = np.mean([k[0] for k in r_countIndex if k[1] == idx])
            count = np.mean([k[0] for k in r_weightIndex if k[1] == idx])
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
                    
    return rotatedFibrilParams,[r_xTest,r_zTest]


def get2DSAXSmap(vox,q1,q2,q3,simu_vals = False):
    
    """
    Simulation function for generating 2D SAXS map -
    needs checking for use with real data.
    """
    
    if simu_vals == False:
        fibril_chars = vox["fibril_param"]["fit"]
        q0  = fibril_chars["q0"]
        wa  = fibril_chars["wa"]
        wMu = fibril_chars["wMu"]
        amp = fibril_chars["amp"]
        delta = fibril_chars["delta"]
    else:
        fibril_chars = vox["fibril_param"]["fit"]
        q0  = simu_vals[0]
        wa  = simu_vals[1]
        wMu = simu_vals[2]
        if type(vox["fibril_param"]["weight"]) == np.ndarray and len(vox["fibril_param"]["weight"]) == 0:
            vox["fibril_param"]["weight"] = 1
        #else:
            #weight = vox["weight"]
        amp = vox["fibril_param"]["weight"]
        delta = simu_vals[4]
        
    flat = 1
    alpha = vox["fibril_param"]["alpha"]
    beta  = vox["fibril_param"]["beta"]
    alphaR, betaR = np.radians(alpha), np.radians(beta)
    normfac = 1
    integrate=False
    iqchi = amp*t3d.Iplanarfibril(q1,q2,q3,gamma0,deltaGamma0,mu,q0,wMu,wa,normfac,\
                  alphaR,betaR,integrate,flat=flat,delta=delta)
    return iqchi


#ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(s_voxel,chi1=chi1,chi2=chi2,nchi=nchi,
                                               #q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen,simu_vals = sim_vals)

#vox, chi1, chi2, nchi, q1, q2, nq, wavelen,simu_vals = voxel,chi1,chi2,nchi,q0m_low,q0m_high,nq0,wavelen,sim_vals

def get1DSAXSchiprofile(vox, chi1=0, chi2=360, nchi=90, q1=0.45, q2=.47, nq=10, wavelen=0.1,simu_vals = False):
    
    """
    Function for simulating I(chi) profile for a single fibre
    """
    
    fibril_chars = vox["fibril_param"]["simu"]
    
    if simu_vals == False:    
        q0  = fibril_chars["q0"]
        wa  = fibril_chars["wa"]
        wMu = fibril_chars["wMu"]
        amp = vox["fibril_param"]["initial_amp_est"]
        delta = fibril_chars["delta"]
        if type(vox["weight"]) == np.ndarray and len(vox["weight"]) == 0:
            vox["weight"] = 0
        else:
            weight = vox["weight"]
            if type(weight) == np.ndarray:
                weight = weight[0]
    else:
        if vox["fibril_param"]["solved"] == False or len(vox['fibril_param']["fit"]) == 0:              
            q0  = simu_vals[0]
            wa  = simu_vals[1]
            wMu = simu_vals[2]
            if vox["fibril_param"]["initial_amp_est"] == 0:
                amp = simu_vals[3]
            else:
                amp = vox["fibril_param"]["initial_amp_est"]
             
            delta = simu_vals[4]
            if type(vox["fibril_param"]["weight"]) == np.ndarray and len(vox["fibril_param"]["weight"]) == 0:
                vox["fibril_param"]["weight"] = 1
            #else:
                #weight = vox["weight"]
            weight = vox["fibril_param"]['weight']
            if type(weight) == np.ndarray:
                if len(weight)>0:
                    weight = weight[0]
                else:
                    weight = 0
        else:                  
            q0  = vox["fibril_param"]["fit"]["q0"]
            wa  = vox["fibril_param"]["fit"]["wa"]
            wMu = vox["fibril_param"]["fit"]["wMu"]
            amp = vox["fibril_param"]["fit"]["amp"]
            delta = simu_vals[4]
            if type(vox["fibril_param"]["weight"]) == np.ndarray and len(vox["fibril_param"]["weight"]) == 0:
                vox["fibril_param"]["weight"] = 1
            weight = vox["fibril_param"]['weight']
            if type(weight) == np.ndarray:
                if len(weight)>0:
                    weight = weight[0]
                else:
                    weight = 0
        
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
    """
    chimax, ichimax = 0, -1
    for idx, chi in enumerate(chirange):
        qx,qy,qz,qxD,qyD=t3d.calc_ewald_trace_radial(wavelen,chi,q1,q2,dq=dq)
        iqchi = amp*t3d.Iplanarfibril(qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,wa,normfac,\
                      alphaR,betaR,integrate,flat=flat,delta=delta)
        iqchiav = np.average(iqchi)
        #print("lenchirange: ", len(chirange), "; idx is: ",idx, " shape iqchi: ", len(iqchi))
        #print("average: ",iqchiav)
        #ichi[idx] = ichi[idx] + iqchiav
        ichi[idx] = iqchiav
        if iqchiav > ichimax:
            ichimax = iqchiav
            chimax = chi
    ichi = ichi * vox["weight"]
    """
    bareichi = ichi/amp
    return ichi, chirange, ichimax, chimax, bareichi


#ichi0,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile_neutral(alt_voxel,params0,chi1=chi1,chi2=chi2,nchi=n_chi_svd,q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen)

def get1DSAXSchiprofile_neutral(vox, params, chi1=0, chi2=360, nchi=90, q1=0.45, q2=.47, nq=10, wavelen=0.1, amp = False):
    
    """
    Function for simulating I(chi) profile for a single fibre of unknown scattering properties.
    """
    
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
            
    ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(vox2,chi1=chi1,chi2=chi2,nchi=nchi,\
                                                   q1=q1,q2=q2,nq=nq,wavelen=wavelen)
    
    return ichi, chir, ichimax, chimax, bareichi


def getAllSAXS(pdict,axis,I2D,I1D,qx1,qx2,qx3,nvox,nvoxs,fP,chi1=0,chi2=180,nchi=100,\
                q1=0.45,q2=0.47,ymax=1000,nq=10,wavelen=0.08856,threshold_detection=50,simu_vals = False):
    
    """
    Simulation function for simulating and convolving I(chi) profiles for all fibres along a single 
    beampath
    Untested for real data.
    """
    
    vox_amps = []
    
    for idx, voxel in enumerate(pdict["voxels"]):
        vox_amps.append(voxel['fibril_param']['initial_amp_est'])
        
    for idx, voxel in enumerate(pdict["voxels"]):    
        voxel = copy.deepcopy(voxel)
        if voxel['fibril_param']['initial_amp_est']<0:
            voxel['fibril_param']['initial_amp_est'] = np.mean(np.asarray(vox_amps)[np.asarray(vox_amps)>0])
        if type(voxel["weight"]) == np.ndarray and len(voxel["weight"]) == 0:
            voxel["weight"] = 1
        
        
        if len(voxel["fibril_param"]["fit"])>0:
            iqchi = get2DSAXSmap(voxel,qx1,qx2,qx3)
        else:
            iqchi = get2DSAXSmap(voxel,qx1,qx2,qx3,simu_vals=simu_vals)
        #iqchi = np.zeros_like(I2D)
        I2D=I2D+iqchi*voxel["weight"]
        if simu_vals == False:
            ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(voxel,chi1=chi1,chi2=chi2,nchi=nchi,\
                                       q1=q1,q2=q2,nq=nq,wavelen=wavelen)
        else:
            ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(voxel,chi1=chi1,chi2=chi2,nchi=nchi,\
                                       q1=q1,q2=q2,nq=nq,wavelen=wavelen,simu_vals=simu_vals)
            
                
        #ichi = ichi*voxel["weight"]
        I1D = I1D + ichi
        nvox = nvox+1
            
        chimax = chir[np.argmax(ichi)]
        ichimax = ichi[np.argmax(ichi)]
        """
        if it is a solved voxel, plot it with a different color
            black = unsolved
            grey = solved
        """
        #indx = int(voxel["fibril_param"]["number"])
        #if fP[indx]["solved"]==False:
            #axis[1].plot(chir,ichi,"k--", lw=1)
            #pass
       # else:
            #axis[1].plot(chir,ichi,color="grey", linestyle="--", lw=1)
           # pass
        
        if ichimax>=threshold_detection:
            #print("threshold_detection: ", threshold_detection)
            axis.plot([chimax,chimax],[0,ichimax],color="red",linestyle="--",
                         lw=0.5)
            nvoxs = nvoxs+1  
            nvox = nvox + 1
        else:
            axis.plot([chimax,chimax],[0,ichimax],color="cyan",
                         linestyle="-.",lw=0.5)
            nvox = nvox + 1
    return axis, I2D, I1D, nvox, nvoxs


#ichi_vox,ichi_all,chi,mask_pts,chiwin,threshold_interference,threshold_detection = ichi,Ichi1D_unsolved,chirange,recon_mask_pts,chiRefWindow,threshold_interference,threshold_detection

#findChiWindow(ichi,Ichi1D_unsolved,chirange,recon_mask_pts,chiwin=chiRefWindow,threshold_interference=threshold_interference,threshold_detection=threshold_detection)

def findChiWindow(ichi_vox,ichi_all,chi,mask_pts,chiwin=10,\
                  threshold_interference=10,threshold_detection=50):
    """
    ichi_vox = ichi to test for separability
    ichi_all = total intensity of all unsolved voxels
    chiwin = the angular range over which it is separable
    threshold_interference = the PERCENTAGE value that (ichi_all-ichi_vox)/ichi_all must be <=
                             to class as noninterfering
    threshold_detection = minumum ABSOLUTE level of intensity that ichi_vox and ichi_all
                          must have over the chi window (so that we are not measuring noise)
    """
    
    """
    Adding masking 
    """
    ichi_vox = ichi_vox[mask_pts==True]
    ichi_all = ichi_all[mask_pts==True]
    chi = chi[mask_pts==True]
    
    solved = False
    meanval = 0
    #solved_window = np.zeros_like(chiwin)
    solved_window = None
    solved_windows = []
    chimin,chimax = chi[0], chi[len(chi)-1]
    opti_window = None
    for chival in chi:
        #if chival<155.4317548746518:
        chi1 = chival
        chi2 = chival + chiwin
        
        """
        exclude wraparound for now
        """
        if chi2<chimax:
            #solved_window = chi[(chi>=chi1)&(chi<=chi2)]
            #print("solved window :", solved_window)
            
            ichi_vox_win = ichi_vox[(chi>=chi1)&(chi<=chi2)]
            ichi_all_win = ichi_all[(chi>=chi1)&(chi<=chi2)]
            if np.all(ichi_vox_win>=threshold_detection)==True:
                ichi_ratio = ichi_vox_win/ichi_all_win
                if np.all(ichi_ratio>=(1-(threshold_interference/100))):
                    if np.round(np.average(ichi_ratio),3)>=meanval:
                        solved=True
                        #print(np.average(ichi_ratio),(1-(threshold_interference/100)))
                        meanval = np.round(np.average(ichi_ratio),3)
                        solved_window = chi[(chi>=chi1)&(chi<=chi2)]
                        solved_windows.append(chi[(chi>=chi1)&(chi<=chi2)])
                        
    if len(solved_windows)>0:
        opti_intensities = [[ichi_vox[find_nearest(chi,j)[1]] for j in k] for k in solved_windows]
        opti_window = solved_windows[np.argmax([np.sum(k) for k in opti_intensities])]        
        #opti_window = solved_windows[np.argmax([len(k) for k in solved_windows])]
    
    return solved_window,solved,opti_window,solved_windows


def fit_gauss(xData,yData,model = "gauss"):
    """
    Function for fitting gaussian model to a data profile.
    """    
    if model == "gauss":
        mod = GaussianModel()
    else:
        mod = VoigtModel()
    pars = mod.guess(yData,xData)
    result = mod.fit(yData, pars, x=xData)
    model = mod.eval(result.params,x=xData)
    params = result.params
    
    return model,params

#ichi_sample(rot,bp_idx,0,0,cake_params,a1,Mask,slice_saxs_file,[chislice[i]-1,chislice[i]+1],fibre_chi = False,fit_chi=False,iq_plot = True)
def ichi_sample(rotation,frameIndex,ichiSim,chiSim,cake_params_chi,a1,Mask,slice_saxs_file,chiRange,fibre_chi = False,fit_chi=False,iq_plot = False):
    
    """
    function for ichi sampling (normalised against background intensity )    
    """
    """
    1. Use peak and background parameters from dictionary 
    """    
    nchi = cake_params_chi["nchi"]
    nq0 = cake_params_chi["nq"]
    
    q0i_low,q0i_high = cake_params_chi["q1i"],cake_params_chi["q2i"]
    q0o_low,q0o_high = cake_params_chi["q1o"],cake_params_chi["q2o"]
    
    q0c_low,q0c_high = cake_params_chi["q1"],cake_params_chi["q2"]
     
    """
    2. If selected, load 2D SAXS frame
    """
    
    if type(frameIndex) == int:
        SAMPLE_PATH = Path(slice_saxs_file)
        
        with File(SAMPLE_PATH,'r') as sample_file:
            entry = list(sample_file.keys())[rotation]
            frame = array(sample_file[entry][frameIndex])
            sample_file.close()
            
        frame = (frame+1)*-1
    else:
        frame = frameIndex
    
    """
    3. If selected, sample I(chi) and estimate range of peak along chi
    """
    
    if fibre_chi == True and ichiSim !=0:
        mod = GaussianModel()
        pars = mod.guess(ichiSim,chiSim)
        result = mod.fit(ichiSim, pars, x=chiSim)
        model = mod.eval(result.params,x=chiSim)
        params = result.params
        
        chiRange = [180-params["center"].value - (params["sigma"].value*5),180-params["center"].value + (params["sigma"].value*5)]
    
    """
    4. Perform I(chi) integration
    """
    
    if type(Mask) != int:
        #print("incl mask")
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
    
    """
    5. If selected, perform I(q) integration
    """
                
    chi_model = None
    peak_iq = None
    peak_iq = None
    
    if fit_chi==True:
        chi_model = fit_gauss(inner_bg[0],norm_chi)
        
        
    if iq_plot==True:
        if type(Mask) != int:
            #print("q incl. mask")
            mask_iq = a1.integrate1d(Mask,nq0,radial_range=(q0i_low,q0o_high),
                                     azimuth_range=(chiRange[0],chiRange[1]))
            
            peak_iq = a1.integrate1d(frame,nq0,radial_range=(q0i_low,q0o_high),
                                     azimuth_range=(chiRange[0],chiRange[1]),mask=Mask)
        else:
            peak_iq = a1.integrate1d(frame,nq0,radial_range=(q0i_low,q0o_high),
                                     azimuth_range=(chiRange[0],chiRange[1]))
        
    peak_iq = [np.copy(peak_iq[0]),np.copy(peak_iq[1])]
        
    return [peak_chi[0],norm_chi],chi_model,peak_iq,frame,[inner_bg[1],outer_bg[1],peak_chi[1]]


def plot2D_1DSAXS(ax_0,ax_1,chir,I2D,I1D,thresh,y_max):
    
    """
    Simulation function for plotting 1D I(chi) integration
    Untested for real data. 
    """
    
    ax_1.plot([0,180],[(thresh/100.0)*y_max,(thresh/100.0)*y_max],
              linestyle="-",color = 'b', linewidth=0.2)
    
    #maxIchi = np.max(I1D)
            
    # https://stackoverflow.com/questions/31401812/matplotlib-rotate-image-file-by-x-degrees
    rotated_img = ndimage.rotate(I2D, 180)
    
    #ax_0.imshow(rotated_img)
    #ax_0.imshow(I2D.T,origin="lower")
    ax_0.imshow(np.log(I2D.T+0.01),origin="lower")
    """
    plot the q-averaged I(chi) data in ax[1]
    """
    ax_1.plot(chir,I1D,'g-')
    ax_1.set_xlim((0,180))
    ax_1.set_yscale("log")
    ax_1.set_ylim((1,np.max(I1D)*1.1))
    
    return ax_0, ax_1

    
#params,profileData,unmasked_mask_iqs,cake_params,straight_profiles,straight = params,unmasked_iqs,unmasked_mask_iqs,cake_params,0,False

def iq_bg(params,profileData,unmasked_mask_iqs,cake_params,straight_profiles,straight=False):
    """
    Function for performing background correction for I(q) data
    """
    """
    1. Load peak parameters from library and find closest sampled q values for 
        sampled data.
    """
    
    peak_start = cake_params["q1"]
    peak_end = cake_params["q2"]
    #peak_end = 0.315
    
    q_start = find_nearest(profileData[0][0],peak_start)[1]
    q_end = find_nearest(profileData[0][0],peak_end)[1]
    
    results,models = [],[]
    
    for k in range(0,len(profileData)):
        
        """
        2. for each sampled chi range - remove regins represented by the mask
        and by the peak
        """
        
        if len(np.where(unmasked_mask_iqs[k][1]==0)[0])>0:
            un_masked = np.where(unmasked_mask_iqs[k][1]==0,True,False)
        else:
            un_masked = np.where(unmasked_mask_iqs[k][1]<0.3,True,False)
        
        IQProfile = np.copy(profileData[k][1])
        IQXaxis = np.copy(profileData[k][0])
        
        IQProfile = np.asarray(IQProfile[0:q_start].tolist()+IQProfile[q_end:len(IQProfile)].tolist())
        IQXaxis = np.asarray(IQXaxis[0:q_start].tolist()+IQXaxis[q_end:len(IQXaxis)].tolist())
        
        un_masked = np.asarray(un_masked[0:q_start].tolist()+un_masked[q_end:len(un_masked)].tolist())
        
        #peak_plane = np.linspace(IQProfile[q_start],IQProfile[q_end],(q_end-q_start))
        #IQProfile[q_start:q_end] = peak_plane
                                
        if straight != False:
            
            """
            If the straight option is selected, then fit a linear function to the remaining 
            background data
            """
                        
            straight_profile = np.asarray(straight_profiles[k][0:q_start].tolist()+straight_profiles[k][q_end:len(straight_profiles[k])].tolist())
            IQProfile = straight_profile[un_masked==True]
            IQXaxis = IQXaxis[un_masked==True]    
            
            model = LinearModel()
            modelPars = model.guess(IQProfile,IQXaxis)
            result = model.fit(IQProfile, modelPars, x=IQXaxis)
            modelData = model.eval(result.params,x=profileData[k][0])
            
            results.append(result.params)
            models.append(modelData)
            
        else:
            
            """
            If the straight option is NOT selected, then fit an expontential function + a constant to the remaining 
            background data
            """
            IQProfile = IQProfile[un_masked==True]
            IQXaxis = IQXaxis[un_masked==True]        
            
            if len(IQProfile)>2:               
                model = ExponentialModel()+ConstantModel()                
                result = model.fit(IQProfile, params, x=IQXaxis)
                modelData = model.eval(result.params,x=profileData[k][0])
                
                results.append(result.params)
                models.append(modelData)
        
        #plt.scatter(IQXaxis,IQProfile)
        #plt.plot(profileData[k][0],modelData)
        #plt.plot(np.ones((10))*0.315,np.linspace(1e8,7e8,10))
        #plt.plot(np.ones((10))*0.27,np.linspace(1e8,7e8,10))
        #plt.plot(profileData[k][0],profileData[k][1])
        
        """
        save model parameters and background-removed data
        """
        
        #results.append(result.params)
        #models.append(modelData)
        
    return results,models


def bg_test(params,profileData,unmasked_mask_iqs,cake_params):
    
    """
    function for optimising shared fitting parameters
    """
    
    
    peak_start = cake_params["q1"]
    peak_end = cake_params["q2"]
    
    q_start = find_nearest(profileData[0][0],peak_start)[1]
    q_end = find_nearest(profileData[0][0],peak_end)[1]
    
    resids = []
    
    for k in range(0,len(profileData)):
        
        un_masked = np.where(unmasked_mask_iqs[k][1]==0,True,False)
        
        IQProfile = np.copy(profileData[k][1])
        IQXaxis = np.copy(profileData[k][0])
        
        IQProfile = np.asarray(IQProfile[0:q_start].tolist()+IQProfile[q_end:len(IQProfile)].tolist())
        IQXaxis = np.asarray(IQXaxis[0:q_start].tolist()+IQXaxis[q_end:len(IQXaxis)].tolist())
        
        un_masked = np.asarray(un_masked[0:q_start].tolist()+un_masked[q_end:len(un_masked)].tolist())
        
        #peak_plane = np.linspace(IQProfile[q_start],IQProfile[q_end],(q_end-q_start))
        #IQProfile[q_start:q_end] = peak_plane
                        
        IQProfile = IQProfile[un_masked==True]
        IQXaxis = IQXaxis[un_masked==True]        
                       
        model = ExponentialModel()+ConstantModel()                
        #result = model.fit(IQProfile, params, x=IQXaxis)
        modelData = model.eval(params,x=profileData[k][0])
                        
        resids.append(profileData[k][1]-modelData)
        
    residuals = (resids[0],resids[1],resids[2])
    residuals = np.concatenate(residuals)
        
    residuals = np.asarray(residuals)
    
    return residuals.flatten()


def iq_sim(waveLen,q_range,chis,q0,wa,wMuVal,deltaVal,alphaTest,betaTest,amp,weight):
    
    flat=1
    integrate=False
    Nfac = t3d.normFac(0,wMuVal,wa)
    wfuncInt = 1
    iqs = []
    for chi in chis:
        #print(chi)
        qx, qy, qz = t3d.calc_ewald_trace_singular(waveLen,q_range,chi)
        #Iq = constant+amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
        #Iq = amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
        Iq = t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
        #chi_avr = chi_avr+Iq
        iqs.append(Iq)
    iqs = np.asarray(iqs)
            
    return (np.average(iqs,0)*weight)*amp


def sampleIqOnChiWin_old(rot,bp_idx,cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,chiwin,qr,solved_windows,nslices=3,nq0=50,to_smooth = False):
    """
    new method:
        a. from FULL (instead of optimal) independent chi range - perform I(q) integration over mask 
           between 0.25-0.35 nm-1 for each chi value +/- 2.5 degrees (5 degree chi window)
        b. find which chi values have no mask for extended I(chi) q window (i.e. between q1i and q2o)
        c. Isolate closest of these values to the start, midpoint, and end of their range
        d. Perform shared background correction on these three chi values to get optimal background correction parameters
    """
    
    straightIq,qrange,chislice,masked_q,frame,q_sigma,success_test = None,None,None,None,None,None,None
    
    chi_range_sample = 3
    
    """
    Load 2D SAXS frame
    """
    
    SAMPLE_PATH = Path(slice_saxs_file)
    
    if type(bp_idx) == int:   
        with File(SAMPLE_PATH,'r') as sample_file:
            entry = list(sample_file.keys())[rot]
            frame = array(sample_file[entry][bp_idx])
            sample_file.close()
            
        frame = (frame+1)*-1
    else:
        
        if len(bp_idx) == 1:
            with File(SAMPLE_PATH,'r') as sample_file:
                entry = list(sample_file.keys())[rot]
                frame = array(sample_file[entry][bp_idx[0]])
                sample_file.close()
                
            frame = (frame+1)*-1
            
        else:
            frames = []
            for idx in bp_idx:
                with File(SAMPLE_PATH,'r') as sample_file:
                    entry = list(sample_file.keys())[rot]
                    frame = array(sample_file[entry][idx])
                    sample_file.close()
                    
                frame = (frame+1)*-1
                frames.append(frame)
            frame = np.sum(np.asarray(frames),0)
            
    
    """
    create testing vectors
    """
    
    #nq0 = cake_params["nq"]
    nq0 = 50
    #qs = np.linspace(0.25,0.35,nq0)
    mask_qs = np.linspace(0.255,0.4,nq0)
    peak_qs = np.where((mask_qs >= cake_params["q1"]) & (mask_qs <= cake_params["q2"]))[0]
    
    chi_start = np.min([np.min(k) for k in solved_windows])
    chi_end = np.max([np.max(k) for k in solved_windows])
    
    mask_tests = 360-np.arange(chi_start,chi_end,2)
    
    """
    2. perform I(q) integration of mask for all chi values within the range estimated as independent for the respective 
        scattering object(s)
    """
    
    mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(0.255,0.4),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in mask_tests])
    
    """
    3. Isolate chi values that do not have mask over the estimated range of the I(q) peak
    """
    
    unmasked_chis = np.asarray([mask_tests[k] for k in np.arange(0,len(mask_iqs),1) if np.max(np.abs(mask_iqs[k][1][peak_qs])) ==0])
    #unmasked_chis = [mask_tests[k] for k in np.arange(0,len(mask_iqs),1) if len(np.where(mask_iqs[k][1]==0)[0])>100]
    
    if len(unmasked_chis) > 0:
        
        """
        If at least one chi value has no mask over the I(q) peak - calculate a uniform sampling increment between the optimal estimated
        independent chi range
        """
            
        #unmasked_chi_int = (unmasked_chis[-1]-unmasked_chis[0])/(nslices+1)
        unmasked_chi_int = ((360-chiwin)[-1]-(360-chiwin)[0])/(nslices+1)
        #unmasked_chi_ints = [(360-chiwin)[0]+(unmasked_chi_int*k) for k in [1,2,3]]
                
        #chislice = [find_nearest(unmasked_chis,unmasked_chis[0]+(unmasked_chi_int*k))[0] for k in [1,2,3]]
        #first_slice = find_nearest(unmasked_chis[unmasked_chis>unmasked_chi_ints[0]],unmasked_chi_ints[0])[0]
        #second_slice = 
        
        """
        find nearest unmasked chis for each increment  
        """
        
        chislice = [find_nearest(unmasked_chis,(360-chiwin)[0]+(unmasked_chi_int*k))[0] for k in np.arange(1,nslices+1,1)]
        if chislice[0]==chislice[1]:
            if len(unmasked_chis[unmasked_chis>chislice[0]])>0:
                chislice[0] = unmasked_chis[unmasked_chis>chislice[0]][-1]
            else:
                chislice[0] = chislice[1] + 2
        if chislice[2]==chislice[1]:
            if len(unmasked_chis[unmasked_chis<chislice[2]])>0:
                chislice[2] = unmasked_chis[unmasked_chis<chislice[2]][0]
            else:
                chislice[0] = chislice[1] - 2
        
        if np.max(np.unique(chislice,return_counts=True)[1])>1:
            chislice = np.linspace(chislice[0],chislice[-1],nslices)
            
                        
        """
        Perform I(q) integration over the SAXS frame and mask for the isolated chi range
        """
        
        unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(0.255,0.4),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
        unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(0.255,0.4),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])
        
        #unmasked_regions = [unmasked_iqs[0][0][np.where(k[1]==0)[0]] for k in unmasked_mask_iqs]
        #unmasked_rhs_0 = [k[np.where(k>cake_params["q2o"])[0]] if len(np.where(k>cake_params["q2o"])[0])>0 else k[-2:len(k)] for k in unmasked_regions]
        #unmasked_rhs = [k[0:np.where(np.diff(k)>(k[1]-k[0])*2)[0][0]] if len(np.where(np.diff(k)>(k[1]-k[0])*2)[0])>0 else k for k in unmasked_rhs_0]
        
        #unmasked_lhs = [k[np.where(k<cake_params["q1"])[0]] if len(np.where(k<cake_params["q1"])[0])>0 else k[0:2] for k in unmasked_regions]
        #unmasked_lhs = [k[np.where(np.diff(k)>(k[1]-k[0])*2)[0][0]:len(k)] if len(np.where(np.diff(k)>(k[1]-k[0])*2)[0])>0 else k for k in unmasked_lhs]
            
        """
        perform background correction (exponential followed by linear)
        """
        
        params = Parameters()
        params.add('amplitude', value = 1e10,min = 1e8,max=1e13)
        params.add('decay', value = 0.05,min = 0.02,max = 0.1)
        params.add('c', value = 1e8,min = 0,max=3e8)
                    
        #est_bg_params = minimize(bg_test, params, args=(unmasked_iqs,unmasked_mask_iqs,cake_params))
        
        #bins = np.linspace(cake_params["q1"],cake_params["q2"],cake_params["nq"])
        #iq_bins = [(unmasked_rhs[k][-1] - cake_params["q1"])/np.mean(np.diff(bins)) for k in np.arange(0,len(chislice),1)]
            
        #iqs = np.asarray([a1.integrate1d(frame,cake_params["nq"],radial_range=(cake_params["q1"],unmasked_rhs[k][-1]),
                                         #azimuth_range=(chislice[k]-2.5,chislice[k]+2.5)) for k in np.arange(0,len(chislice),1)])
        
        #iqs = np.asarray([a1.integrate1d(frame,iq_bins[k],radial_range=(cake_params["q1"],unmasked_rhs[k][-1]),
                                         #azimuth_range=(chislice[k]-2.5,chislice[k]+2.5)) for k in np.arange(0,len(chislice),1)])
        
        #iqs = np.asarray([a1.integrate1d(frame,iq_bins[k],radial_range=(unmasked_lhs[k][0],unmasked_rhs[k][-1]),
                                         #azimuth_range=(chislice[k]-2.5,chislice[k]+2.5)) for k in np.arange(0,len(chislice),1)])
        
        iq_params,iq_models = iq_bg(params,unmasked_iqs,unmasked_mask_iqs,cake_params,0)
        
        if len(iq_params) == nslices:
            
            model = ExponentialModel()+ConstantModel()   
            #iq_flat_mods = [model.fit(iqs[k][1], params, x=iqs[k][0]) for k in np.arange(0,len(iqs),1)]
            #iqs_flat = np.asarray([model.eval(est_bg_params.params,x=iqs[k][0]) for k in np.arange(0,len(iqs),1)])
            #iqs_flat = np.asarray([model.eval(iq_params[k],x=iqs[k][0]) for k in np.arange(0,len(iqs),1)])
            iqs_flat = np.asarray([model.eval(iq_params[k],x=unmasked_iqs[k][0]) for k in np.arange(0,len(unmasked_iqs),1)])
            
            #sampledIq = np.asarray([iqs[k][1] - iqs_flat[k] for k in np.arange(0,len(iqs),1)])
            sampledIq = np.asarray([unmasked_iqs[k][1] - iqs_flat[k] for k in np.arange(0,len(unmasked_iqs),1)])
            
            #straight_params,straight_models = iq_bg(params,unmasked_iqs,unmasked_mask_iqs,cake_params,sampledIq,straight=True)
                            
            #iqs_straight = np.asarray([model.eval(straight_params[k],x=unmasked_iqs[k][0]) for k in np.arange(0,len(unmasked_iqs),1)])
            #straightIq = np.asarray([sampledIq[k] - straight_models[k] for k in np.arange(0,len(unmasked_iqs),1)])
            
            straightIq = copy.deepcopy(sampledIq)
            
            if to_smooth == True:
                straightIq = [savgol_filter(k,3,1) for k in straightIq]
            
            qrange = np.asarray([k[0] for k in unmasked_iqs])
            masked_q = np.asarray([k[1] for k in unmasked_mask_iqs])
            q_sigma = np.asarray([k[2] for k in unmasked_iqs])
            
            success_test = 1
    
    return straightIq,qrange,chislice,masked_q,frame,q_sigma,success_test

#single
#rot,bp_idx,cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,chiwin,qr,solved_windows,voxelsInPath,voxel_to_slv,simu_vals,waveLen,threshold_interference,nslices,nq0,chi_range_sample = r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,opti_chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],voxel,sim_vals,wavelen,threshold_interference,nslices,50,chi_range_sample

#multi
#rot,bp_idx,cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,chiwin,qr,solved_windows,voxelsInPath,voxel_to_slv,simu_vals,waveLen,threshold_interference,nslices,nq0,chi_range_sample = r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],comb_indx_path,sim_vals,wavelen,threshold_interference,nslices,50,chi_range_sample

def sampleIqOnChiWin(rot,bp_idx,cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,chiwin,qr,solved_windows,voxelsInPath,voxel_to_slv,simu_vals,waveLen,threshold_interference,nslices=3,nq0=50,to_smooth = False,chi_range_sample = 3):
    """
    new method:
        a. from FULL (instead of optimal) independent chi range - perform I(q) integration over mask 
           between 0.25-0.35 nm-1 for each chi value +/- 2.5 degrees (5 degree chi window)
        b. find which chi values have no mask for extended I(chi) q window (i.e. between q1i and q2o)
        c. Isolate closest of these values to the start, midpoint, and end of their range
        d. Perform shared background correction on these three chi values to get optimal background correction parameters
    """
    
    straightIq,qrange,chislice,masked_q,frame,q_sigma,success_test = None,None,None,None,None,None,None
        
    
    """
    Load 2D SAXS frame
    """
    
    SAMPLE_PATH = Path(slice_saxs_file)
    
    if type(bp_idx) == int:   
        with File(SAMPLE_PATH,'r') as sample_file:
            entry = list(sample_file.keys())[rot]
            frame = array(sample_file[entry][bp_idx])
            sample_file.close()
            
        frame = (frame+1)*-1
    else:
        
        if len(bp_idx) == 1:
            with File(SAMPLE_PATH,'r') as sample_file:
                entry = list(sample_file.keys())[rot]
                frame = array(sample_file[entry][bp_idx[0]])
                sample_file.close()
                
            frame = (frame+1)*-1
            
        else:
            frames = []
            for idx in bp_idx:
                with File(SAMPLE_PATH,'r') as sample_file:
                    entry = list(sample_file.keys())[rot]
                    frame = array(sample_file[entry][idx])
                    sample_file.close()
                    
                frame = (frame+1)*-1
                frames.append(frame)
            frame = np.sum(np.asarray(frames),0)
            
    
    """
    create testing vectors
    """
    
    bg_start,bg_end = cake_params["bg_start"],cake_params["bg_end"]
    
    mask_qs = np.linspace(bg_start,bg_end ,nq0)
    peak_qs = np.where((mask_qs >= cake_params["q1"]) & (mask_qs <= cake_params["q2"]))[0]
    
    chi_start = np.min([np.min(k) for k in solved_windows])
    chi_end = np.max([np.max(k) for k in solved_windows])
    
    mask_tests = 360-np.arange(chi_start,chi_end,2)
    
    """
    2. perform I(q) integration of mask for all chi values within the range estimated as independent for the respective 
        scattering object(s)
    """
    
    mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),
                                          azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in mask_tests])
    
    """
    3. Isolate chi values that do not have mask over the estimated range of the I(q) peak
    """
    
    unmasked_chis = np.asarray([mask_tests[k] for k in np.arange(0,len(mask_iqs),1) if np.max(np.abs(mask_iqs[k][1][peak_qs])) ==0])
    
    if len(unmasked_chis) > 0:
        
        """
        If at least one chi value has no mask over the I(q) peak - calculate a uniform sampling increment between the optimal estimated
        independent chi range
        """
            
        unmasked_chi_int = ((360-chiwin)[-1]-(360-chiwin)[0])/(nslices+1)
        
        """
        find nearest unmasked chis for each increment  
        """

        unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),
                                                  azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in unmasked_chis])
        unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),
                                                       azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in unmasked_chis])

        mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
        max_iqs = [np.max(k[1]) for k in unmasked_iqs]

        sim_chi = 360 - np.asarray(unmasked_chis)

        """
        Check if the fibril to solve is prominent in I(q) relative to unfitted fibrils
        """   

        indexs = np.asarray([k["fibril_param"]["indx"] for k in voxelsInPath])
        
        if type(voxel_to_slv) == dict:
            to_slv_indxs = [np.where(indexs == voxel_to_slv["fibril_param"]["indx"])[0][0]]
        else:
            to_slv_indxs = [k for k in np.arange(0,len(voxelsInPath),1) if 
                           voxelsInPath[k]["fibril_param"]["indx"] in voxel_to_slv]
        
        sum_vox_raw = []
        sum_vox_metrics = []
        for to_slv_indx in to_slv_indxs:
                
            to_slv_alpha = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["alpha"])
            to_slv_beta = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["beta"])
            to_slv_q0 = simu_vals[0]
            to_slv_wa = simu_vals[1]
            to_slv_wMu = simu_vals[2]
            #to_slv_amp = simu_vals[3]
            if voxelsInPath[to_slv_indx]["fibril_param"]["initial_amp_est"] != 0:
                to_slv_amp = voxelsInPath[to_slv_indx]["fibril_param"]["initial_amp_est"]
            else:
                to_slv_amp = simu_vals[3]
            to_slv_delta = simu_vals[4]
            to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
            if type(to_slv_weight)== np.ndarray:
                to_slv_weight = to_slv_weight[0]
                
            vox_metrics = [to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight]
            
            sum_vox_metrics.append(vox_metrics)    
                    
            voxel_sims_raw = [[iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,1,to_slv_weight)]
                          for i in np.arange(0,len(unmasked_chis),1)]
            
            sum_vox_raw.append(voxel_sims_raw)

        sum_vox_comb = [[k[i][0] for k in sum_vox_raw] for i in np.arange(0,len(sum_vox_raw[0]),1)]

        voxel_sims_raw = [np.sum(np.asarray(k),0) for k in sum_vox_comb]

        voxel_sims_ints = np.asarray([np.max(k) for k in voxel_sims_raw])
        
        sim_chi = sim_chi[np.where(voxel_sims_ints>1)]
        
        unmasked_chis = unmasked_chis[np.where(voxel_sims_ints>1)]
        
        unmasked_iqs = unmasked_iqs[np.where(voxel_sims_ints>1)]
        
        if len(unmasked_chis)>1:
            
            if len(unmasked_chis)<(nslices-1):
                
                unmasked_chis = np.linspace(unmasked_chis[0],unmasked_chis[-1],6).astype(int)  
                
                sim_chi = 360 - np.asarray(unmasked_chis)
                
                unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),
                                                          azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in unmasked_chis])
                unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),
                                                               azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in unmasked_chis])

                mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                
            sum_vox_raw = []
            
            for to_slv_indx,vox_metrics in zip(to_slv_indxs,sum_vox_metrics):
                
                to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight = vox_metrics[0:len(vox_metrics)]
            
                voxel_sims_raw = [[iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                         to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                              for i in np.arange(0,len(unmasked_chis),1)]
                
                sum_vox_raw.append(voxel_sims_raw)
                                
            sum_vox_comb = [[k[i][0] for k in sum_vox_raw] for i in np.arange(0,len(sum_vox_raw[0]),1)]

            voxel_sims_raw = [np.sum(np.asarray(k),0) for k in sum_vox_comb]
            
            alphas = np.radians(np.asarray([k["fibril_param"]["alpha"] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]))
    
            betas = np.radians(np.asarray([k["fibril_param"]["beta"] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]))
    
            #indexs = np.asarray([k["fibril_param"]["indx"] for k in voxelsInPath if k["fibril_param"]["fit"] == {}])
            #to_slv_indx = np.where(indexs == voxel_to_slv["fibril_param"]["indx"])[0][0]
    
            slvd_indxs = np.asarray([k["fibril_param"]["indx"] for k in voxelsInPath if "fit" in k])
            simu_vals[0]
            q0s = [simu_vals[0] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]
            #q0s = [k["fibril_param"]["simu"]["q0"] if k["fibril_param"]["fit"] == {} else k["fibril_param"]["fit"]["q0"] for k in voxelsInPath]    
            was = [simu_vals[1] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]
            #was = [k["fibril_param"]["simu"]["wa"] if k["fibril_param"]["fit"] == {} else k["fibril_param"]["fit"]["wa"] for k in voxelsInPath]    
            wMus = [simu_vals[2] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]
            #wMus = [k["fibril_param"]["simu"]["wMu"] if k["fibril_param"]["fit"] == {} else k["fibril_param"]["fit"]["wMu"] for k in voxelsInPath]    
            amps = []
            for k in voxelsInPath:
                if k["fibril_param"]["fit"] == {}:
                    if k["fibril_param"]["initial_amp_est"] == 0:
                        amps.append(simu_vals[3])
                    else:
                        amps.append(k["fibril_param"]["initial_amp_est"])
            #amps = [k["fibril_param"]["simu"]["amp"] if k["fibril_param"]["fit"] == {} else k["fibril_param"]["fit"]["amp"] for k in voxelsInPath]    
            deltas = [simu_vals[4]for k in voxelsInPath if k["fibril_param"]["fit"] == {}]
            #deltas = [k["fibril_param"]["simu"]["delta"] if k["fibril_param"]["fit"] == {} else k["fibril_param"]["fit"]["delta"] for k in voxelsInPath]    
            weights = [k["weight"][0] if type(k["weight"])== np.ndarray and np.sum(k["weight"])>0 else k["weight"] for k in voxelsInPath if k["fibril_param"]["fit"] == {}]
            weights = [k if type(k)!= np.ndarray else 0 for k in weights]            
                                     
            iq_sims_raw = [np.sum(np.asarray([iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     q0s[k],was[k],wMus[k],deltas[k],alphas[k],betas[k],amps[k],weights[k]) for k in np.arange(0,len(alphas),1)]),0) 
             for i in np.arange(0,len(unmasked_chis),1)]
    
            voxel_mods = [fit_gauss(np.arange(0,len(voxel_sims_raw[k]),1),voxel_sims_raw[k]) for k in np.arange(0,len(voxel_sims_raw),1)]
            
            vox_peak_range = [[int(k[1]["center"].value-(k[1]["sigma"].value*3)),int(k[1]["center"].value+(k[1]["sigma"].value*3))]
                         for k in voxel_mods]
            
            vox_peaks = [voxel_sims_raw[k][vox_peak_range[k][0]:vox_peak_range[k][1]] for k in np.arange(0,len(voxel_sims_raw),1)]
            
            iq_peaks = [iq_sims_raw[k][vox_peak_range[k][0]:vox_peak_range[k][1]] for k in np.arange(0,len(voxel_sims_raw),1)]
            
            ratios = [(np.sum(vox_peaks[k])/np.sum(iq_peaks[k]))*100 for k in np.arange(0,len(unmasked_chis),1)]
            unmasked_chis = np.asarray([unmasked_chis[k] for k in np.arange(0,len(unmasked_chis),1) if str(ratios[k])!="nan"])
            unmasked_iqs = np.asarray([unmasked_iqs[k] for k in np.arange(0,len(unmasked_iqs),1) if str(ratios[k])!="nan"])
            sim_chi = np.asarray([sim_chi[k] for k in np.arange(0,len(sim_chi),1) if str(ratios[k])!="nan"])
            ratios = np.asarray([k for k in ratios if str(k) !="nan"])
            
            unmasked_chis = unmasked_chis[ratios>(100-threshold_interference)]
            unmasked_iqs = unmasked_iqs[ratios>(100-threshold_interference)]
            sim_chi = sim_chi[ratios>(100-threshold_interference)]
            ratios = ratios[ratios>(100-threshold_interference)]
            
            mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
            max_iqs = [np.max(k[1]) for k in unmasked_iqs]
            
            if len(ratios) < 2:    
                print("simulation failed iq raw")
                print("max ratio = ", np.max([(np.sum(voxel_sims_raw[k][0])/np.sum(iq_sims_raw[k]))*100 for k in np.arange(0,len(voxel_sims_raw),1)]))
    
            else:
                                                                
                print("simulation passed iq raw")
                
                iq_sims = [np.sum(np.asarray([iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                         q0s[k],was[k],wMus[k],deltas[k],alphas[k],betas[k],amps[k],weights[k]) for k in np.arange(0,len(alphas),1)]),0) 
                 for i in np.arange(0,len(sim_chi),1)]
                
                sum_vox = []
                
                for to_slv_indx,vox_metrics in zip(to_slv_indxs,sum_vox_metrics):
                    
                    to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight = vox_metrics[0:len(vox_metrics)]
                
                    voxel_sims = [[iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                             to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                                  for i in np.arange(0,len(sim_chi),1)]
                    
                    sum_vox.append(voxel_sims)
                                    
                sum_vox_comb = [[k[i][0] for k in sum_vox] for i in np.arange(0,len(sum_vox[0]),1)]

                voxel_sims = [np.sum(np.asarray(k),0) for k in sum_vox_comb]
    
                """
                Check if the fibril to solve is prominent in I(q) relative to fitted fibrils
                """        
                
                sig_fib = True
                
                if len([k for k in voxelsInPath if k["fibril_param"]["fit"] != {}]) <2:
                
                    sorted_ratios = np.sort(ratios)
                
                    if np.min(sorted_ratios) == 100:
                        
                        chislice = [unmasked_chis[0],unmasked_chis[int(len(unmasked_chis)/2)],unmasked_chis[-1]]
                        
                        sim_chi = 360 - np.asarray(chislice)
                        
                        unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                        unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])

                        mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                        max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                        
                    else:
                        
                        if len(sorted_ratios)> (nslices-1):       

                            max_ratios = sorted_ratios[(nslices*-1):len(iq_sims_raw)]
                            
                            max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])

                            #chislice = unmasked_chis[max_ratio_idx]
                            
                            chislice = [unmasked_chis[k] for k in [0,int(len(unmasked_chis)/2),-1]]

                            sim_chi = 360 - np.asarray(chislice)
                            
                            unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                            unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])

                            mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                            max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                        
                        else:
                            
                            sorted_ratios = np.sort(ratios)
                            
                            if len(sorted_ratios)>3:
                                max_ratios = sorted_ratios[-3:len(iq_sims_raw)]
                                max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])
                                #chislice = unmasked_chis[max_ratio_idx]
                                chislice = [unmasked_chis[k] for k in [0,int(len(unmasked_chis)/2),-1]]
                                if len(chislice)<nslices:
                                    chislice = np.linspace(chislice[0],chislice[-1]+1,nslices)
                            else:
                                max_ratios = sorted_ratios[0:len(iq_sims_raw)] 
                                max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])
                                #chislice = unmasked_chis[max_ratio_idx]
                                chislice = [unmasked_chis[k] for k in [0,int(len(unmasked_chis)/2),-1]]
                                chislice = np.linspace(chislice[0],chislice[-1]+1,nslices)
                                
                        sim_chi = 360 - np.asarray(chislice)
                        
                        unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                        unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])

                        mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                        max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                                                               
                else:
                    
                    alphas = np.radians(np.asarray([k["fibril_param"]["alpha"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]))
                    
                    betas = np.radians(np.asarray([k["fibril_param"]["beta"] for k in voxelsInPath if k["fibril_param"]["fit"] != {}]))
                    
                    indexs = np.asarray([k["fibril_param"]["indx"] for k in voxelsInPath])
                    
                    
                    slvd_indxs = np.asarray([k["fibril_param"]["indx"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ])
                    
                    q0s = [k["fibril_param"]["fit"]["q0"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]
    
                    was = [k["fibril_param"]["fit"]["wa"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]
                    
                    wMus = [k["fibril_param"]["fit"]["wMu"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]
                    
                    amps = [k["fibril_param"]["fit"]["amp"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]
                    
                    deltas = [k["fibril_param"]["fit"]["delta"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]
    
                    weights = [k["weight"] for k in voxelsInPath if k["fibril_param"]["fit"] != {} ]       
                    weights = [k[0] if type(k)== np.ndarray else k for k in weights]
                    
                    #sim_chi = 360 - np.asarray(chislice)
                    
                                        
                    iq_sims = [np.sum(np.asarray([iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-chi_range_sample,sim_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                             q0s[k],was[k],wMus[k],deltas[k],alphas[k],betas[k],amps[k],weights[k]) for k in np.arange(0,len(alphas),1)]),0) 
                     for i in np.arange(0,len(sim_chi),1)]
                    
                    ratios = [(np.sum(voxel_sims[k])/np.sum(iq_sims[k]))*100 for k in np.arange(0,len(unmasked_chis),1)]
                    unmasked_chis = np.asarray([unmasked_chis[k] for k in np.arange(0,len(unmasked_chis),1) if str(ratios[k])!="nan"])
                    unmasked_iqs = np.asarray([unmasked_iqs[k] for k in np.arange(0,len(unmasked_iqs),1) if str(ratios[k])!="nan"])
                    sim_chi = np.asarray([sim_chi[k] for k in np.arange(0,len(sim_chi),1) if str(ratios[k])!="nan"])
                    ratios = np.asarray([k for k in ratios if str(k) !="nan"])
                    
                    if np.min([(np.sum(voxel_sims[k])/np.sum(iq_sims[k]))*100 for k in np.arange(0,len(voxel_sims),1)])<30:
                        
                        sig_fib = False
                        
                        print("simulation failed fitted IQ test")
                        
                    else:
                        
                        sorted_ratios = np.sort(ratios)

                        #sorted_ratios = sorted_ratios[sorted_ratios>(100-threshold_interference)]

                        if len(sorted_ratios) < 2:    
                            print("simulation failed iq raw")
                            print("max ratio = ", np.max([(np.sum(voxel_sims_raw[k][0])/np.sum(iq_sims_raw[k]))*100 for k in np.arange(0,len(voxel_sims_raw),1)]))

                        else:
                            
                            if len(sorted_ratios)> (nslices-1):       

                                max_ratios = sorted_ratios[(nslices*-1):len(iq_sims_raw)]
                                
                                max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])

                                chislice = unmasked_chis[max_ratio_idx]

                                sim_chi = 360 - np.asarray(chislice)
                                
                                unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                                unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])

                                mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                                max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                            
                            else:
                                
                                sorted_ratios = np.sort(ratios)
                                
                                if len(sorted_ratios)>3:
                                    max_ratios = sorted_ratios[-3:len(iq_sims_raw)]
                                    max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])
                                    chislice = unmasked_chis[max_ratio_idx]
                                    if len(chislice)<nslices:
                                        chislice = np.linspace(chislice[0],chislice[-1]+1,nslices)
                                else:
                                    max_ratios = sorted_ratios[0:len(iq_sims_raw)] 
                                    max_ratio_idx = np.asarray([np.where(ratios==k)[0][0] for k in max_ratios])
                                    chislice = unmasked_chis[max_ratio_idx]
                                    chislice = np.linspace(chislice[0],chislice[-1]+1,nslices)
                            
                            if np.max(np.unique(chislice,return_counts = True)[1])>1:
                                chislice = np.linspace(chislice[0],chislice[-1]+1,nslices)
                            
                            sim_chi = 360 - np.asarray(chislice)
                            
                            unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                            unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])

                            mean_iqs = [np.mean(k[1]) for k in unmasked_iqs]
                            max_iqs = [np.max(k[1]) for k in unmasked_iqs]
                
                            
                if sig_fib == True:
                        
                    """
                    Perform I(q) integration over the SAXS frame and mask for the isolated chi range
                    """
                    
                    unmasked_iqs = np.asarray([a1.integrate1d(frame,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample),error_model="poisson") for k in chislice])
                    unmasked_mask_iqs = np.asarray([a1.integrate1d(Mask,nq0,radial_range=(bg_start,bg_end),azimuth_range=(k-chi_range_sample,k+chi_range_sample)) for k in chislice])
                       
                    """
                    perform background correction (exponential followed by linear)
                    """
                    
                    params = Parameters()
                    params.add('amplitude', value = 1e10,min = 1e8,max=1e13)
                    params.add('decay', value = 0.05,min = 0.02,max = 0.1)
                    params.add('c', value = 1e8,min = 0,max=3e8)
                    
                    iq_params,iq_models = iq_bg(params,unmasked_iqs,unmasked_mask_iqs,cake_params,0)
                    
                    if len(iq_params) == nslices:
                        
                        model = ExponentialModel()+ConstantModel()               
                        iqs_flat = np.asarray([model.eval(iq_params[k],x=unmasked_iqs[k][0]) for k in np.arange(0,len(unmasked_iqs),1)])
                        
                        sampledIq = np.asarray([unmasked_iqs[k][1] - iqs_flat[k] for k in np.arange(0,len(unmasked_iqs),1)])
                        
                        #if to_smooth == True:
                            #sampledIq = [savgol_filter(k,3,1) for k in sampledIq]
                        
                        straight_params,straight_models = iq_bg(params,unmasked_iqs,unmasked_mask_iqs,cake_params,sampledIq,straight=True)
                                        
                        qs_straight = np.asarray([model.eval(straight_params[k],x=unmasked_iqs[k][0]) for k in np.arange(0,len(unmasked_iqs),1)])
                        straightIq = np.asarray([sampledIq[k] - straight_models[k] for k in np.arange(0,len(unmasked_iqs),1)])
                        
                        straightIq = copy.deepcopy(sampledIq)
                        
                        if to_smooth == True:
                            straightIq = [savgol_filter(k,3,1) for k in straightIq]
                        
                        qrange = np.asarray([k[0] for k in unmasked_iqs])
                        masked_q = np.asarray([k[1] for k in unmasked_mask_iqs])
                        q_sigma = np.asarray([k[2] for k in unmasked_iqs])
                        
                        success_test = 1
                        
    return straightIq,qrange,chislice,masked_q,frame,q_sigma,success_test

    
def single_amp_test(iq,chi_slices,qRange,voxelsInPath,voxel_to_slv,min_amp,max_amp,mean_amp,waveLen,simu_vals):

    
    chi_range_sample = 3
    
    amp_imbalance,high_amp = None,None   

    xr_chi = 360-np.asarray(chi_slices)
    
    to_slv_indxs = [k for k in np.arange(0,len(voxelsInPath),1) if 
                   voxelsInPath[k]["fibril_param"]["number"] in voxel_to_slv]
    
    #min_amp,max_amp,mean_amp = 10000,np.mean(amp_sample) + (np.std(amp_sample)*2),np.mean(amp_sample)
    
    test_amps = [min_amp,max_amp,mean_amp]
    
    sum_vox_raw = []
    sum_vox_min,sum_vox_max, sum_vox_mean = [],[],[]
    sum_vox_metrics = []
    for idx,to_slv_indx in enumerate(to_slv_indxs):
            
        to_slv_vals = voxelsInPath[to_slv_indx]["fibril_param"]["simu"]
        to_slv_alpha = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["alpha"])
        to_slv_beta = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["beta"])
        
        if voxelsInPath[to_slv_indx]["fibril_param"]["fit"] == {}:
            
            to_solv_idx = idx
            
            amp_min,amp_max,amp_mean = min_amp,max_amp,mean_amp
        
            to_slv_q0 = simu_vals[0]
            #to_slv_q0 = to_slv_vals["q0"]
            to_slv_wa = simu_vals[1]
            #to_slv_wa = to_slv_vals["wa"]
            to_slv_wMu = simu_vals[2]
            #to_slv_wMu = 0.4
            to_slv_amp = simu_vals[3]
            #to_slv_amp = known_amps[idx]
            to_slv_delta = simu_vals[4]
            to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
            if type(to_slv_weight)== np.ndarray and len(to_slv_weight)>0:
                to_slv_weight = to_slv_weight[0]
                
                
        else:
            
            to_slv_q0 = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["q0"]
            #to_slv_q0 = to_slv_vals["q0"]
            to_slv_wa = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wa"]
            #to_slv_wa = to_slv_vals["wa"]
            to_slv_wMu = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wMu"]
            #to_slv_wMu = 0.4
            to_slv_amp = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["amp"]
            #to_slv_amp = known_amps[idx]
            to_slv_delta = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["delta"]
            to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
            if type(to_slv_weight)== np.ndarray and len(to_slv_weight)>0:
                to_slv_weight = to_slv_weight[0]
                
            amp_min,amp_max,amp_mean = to_slv_amp,to_slv_amp,to_slv_amp
            
        vox_metrics = [to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight]
        
        sum_vox_metrics.append(vox_metrics)    
                
        voxel_sims_raw = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_min = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_min,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_mean = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_mean,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_max = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_max,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        sum_vox_raw.append(voxel_sims_raw)
        sum_vox_min.append(voxel_sims_min)
        sum_vox_max.append(voxel_sims_max)
        sum_vox_mean.append(voxel_sims_mean)
        
    min_test = [np.sum([sum_vox_min[j][k][0] for j in np.arange(0,len(sum_vox_min),1)],0) for k in np.arange(0,len(iq),1)]
    min_test_diff = np.sum([np.abs(np.max(min_test[k]) - np.max(iq[k])) for k in np.arange(0,len(iq),1)])
    
    mean_test = [np.sum([sum_vox_mean[j][k][0] for j in np.arange(0,len(sum_vox_mean),1)],0) for k in np.arange(0,len(iq),1)]
    mean_test_diff = np.sum([np.abs(np.max(mean_test[k]) - np.max(iq[k])) for k in np.arange(0,len(iq),1)])
    
    max_test = [np.sum([sum_vox_max[j][k][0] for j in np.arange(0,len(sum_vox_max),1)],0) for k in np.arange(0,len(iq),1)]
    max_test_diff = np.sum([np.abs(np.max(max_test[k]) - np.max(iq[k])) for k in np.arange(0,len(iq),1)])
    
    amp_test = np.argmin([min_test_diff,mean_test_diff,max_test_diff])
    
    return test_amps[amp_test]            


#iq,chi_slices,qRange,voxelsInPath,voxel_to_slv,min_amp,max_amp,mean_amp,waveLen,simu_vals = iq,chi_slices,qRange,path_dict["voxels"],comb_indxs,10000,1500000,600000,wavelen,sim_vals 
   
def amp_balance_test(iq,chi_slices,qRange,voxelsInPath,voxel_to_slv,min_amp,max_amp,mean_amp,waveLen,simu_vals):

    
    chi_range_sample = 3
    
    amp_imbalance,high_amp = None,None   

    xr_chi = 360-np.asarray(chi_slices)
    
    to_slv_indxs = [k for k in np.arange(0,len(voxelsInPath),1) if 
                   voxelsInPath[k]["fibril_param"]["number"] in voxel_to_slv]
    
    slvd_indxs = [k for k in np.arange(0,len(voxelsInPath),1) if 
                   voxelsInPath[k]["fibril_param"]["fit"] != {}]
        
    solved_vox = []
    if len(slvd_indxs)>0:
        for idx in slvd_indxs:
            to_slv_vals = voxelsInPath[idx]["fibril_param"]["fit"]
            to_slv_alpha = np.radians(voxelsInPath[idx]["fibril_param"]["alpha"])
            to_slv_beta = np.radians(voxelsInPath[idx]["fibril_param"]["beta"])
            to_slv_q0 = to_slv_vals["q0"]
            to_slv_wa = to_slv_vals["wa"]
            to_slv_wMu = to_slv_vals["wMu"]
            to_slv_amp = to_slv_vals["amp"]
            to_slv_delta = to_slv_vals["delta"]
            to_slv_weight = voxelsInPath[idx]["weight"]
            if type(to_slv_weight)== np.ndarray:
                to_slv_weight = to_slv_weight[0]
                
            voxel_sims_raw = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                          for i in np.arange(0,len(xr_chi),1)]
            solved_vox.append(voxel_sims_raw)
            
        solved_iq = [np.sum([solved_vox[j][k][0] for j in np.arange(0,len(solved_vox),1)],0) for k in np.arange(0,len(iq),1)]
                        
    known_amps = 88108.91719440738, 11761.7813414807
    
    sum_vox_raw = []
    sum_vox_min,sum_vox_max, sum_vox_mean = [],[],[]
    sum_vox_metrics = []
    
    for idx,to_slv_indx in enumerate(to_slv_indxs):
            
        to_slv_vals = voxelsInPath[to_slv_indx]["fibril_param"]["simu"]
        print(to_slv_vals["wMu"])
        print(to_slv_vals["amp"])
        to_slv_alpha = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["alpha"])
        to_slv_beta = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["beta"])
        
        if voxelsInPath[to_slv_indx]["fibril_param"]["fit"] == {}:
            
            amp_min,amp_max,amp_mean = min_amp,max_amp,mean_amp
        
            to_slv_q0 = simu_vals[0]
            #to_slv_q0 = to_slv_vals["q0"]
            to_slv_wa = simu_vals[1]
            #to_slv_wa = to_slv_vals["wa"]
            to_slv_wMu = simu_vals[2]
            #to_slv_wMu = 0.4
            to_slv_amp = simu_vals[3]
            #to_slv_amp = known_amps[idx]
            to_slv_delta = simu_vals[4]
            to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
            if type(to_slv_weight)== np.ndarray:
                to_slv_weight = to_slv_weight[0]
                
        else:
            
            to_slv_q0 = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["q0"]
            #to_slv_q0 = to_slv_vals["q0"]
            to_slv_wa = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wa"]
            #to_slv_wa = to_slv_vals["wa"]
            to_slv_wMu = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wMu"]
            #to_slv_wMu = 0.4
            to_slv_amp = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["amp"]
            #to_slv_amp = known_amps[idx]
            to_slv_delta = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["delta"]
            to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
            if type(to_slv_weight)== np.ndarray:
                to_slv_weight = to_slv_weight[0]
                
            amp_min,amp_max,amp_mean = to_slv_amp,to_slv_amp,to_slv_amp
            
        vox_metrics = [to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight]
        
        sum_vox_metrics.append(vox_metrics)    
            
        #voxel_sims = [[iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-2.5,sim_chi[i]+2.5,5),
                 #to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                      #for i in np.arange(0,len(chislice),1)]
    
        voxel_sims_raw = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_min = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_min,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_mean = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_mean,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        voxel_sims_max = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                 to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_max,to_slv_weight)]
                      for i in np.arange(0,len(xr_chi),1)]
        
        sum_vox_raw.append(voxel_sims_raw)
        sum_vox_min.append(voxel_sims_min)
        sum_vox_max.append(voxel_sims_max)
        sum_vox_mean.append(voxel_sims_mean)
        
    
    raw1_raw2 = [sum_vox_raw[0][k][0] + sum_vox_raw[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]  
    
    if len(slvd_indxs) == 0:
        solved_iq = [np.zeros_like(k) for k in raw1_raw2]
    
    raw1_raw2 = [raw1_raw2[k] - solved_iq[k] for k in np.arange(0,len(iq),1)]    
    
    raw_mods = [fit_gauss(np.arange(0,len(raw1_raw2[k]),1),raw1_raw2[k],model = "voigt") for k in np.arange(0,len(raw1_raw2),1)]
    
    raw_peak_range = [[int(k[1]["center"].value-(k[1]["sigma"].value*3)),int(k[1]["center"].value+(k[1]["sigma"].value*3))]
                 for k in raw_mods]
    
    raw_ranges = np.asarray([k[1]-k[0] for k in raw_peak_range])
            
    real_mods = [fit_gauss(np.arange(0,25,1),iq[k][0:25]) for k in np.arange(0,len(iq),1)]
    
    real_peak_range = [[int(k[1]["center"].value-(k[1]["sigma"].value*3)),int(k[1]["center"].value+(k[1]["sigma"].value*3))]
                 for k in real_mods]
    
    real_ranges = np.asarray([k[1]-k[0] for k in real_peak_range])
    
    if np.mean(real_ranges - raw_ranges)<2:
        
        imbalance_test = 0
        
    else:
        
        sum_vox_raw = []
        sum_vox_min,sum_vox_max, sum_vox_mean = [],[],[]
        sum_vox_metrics = []
    
        for idx,to_slv_indx in enumerate(to_slv_indxs):
                
            to_slv_vals = voxelsInPath[to_slv_indx]["fibril_param"]["simu"]
            to_slv_alpha = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["alpha"])
            to_slv_beta = np.radians(voxelsInPath[to_slv_indx]["fibril_param"]["beta"])
            
            if voxelsInPath[to_slv_indx]["fibril_param"]["fit"] == {}:
                
                amp_min,amp_max,amp_mean = min_amp,max_amp,mean_amp
            
                to_slv_q0 = simu_vals[0]
                #to_slv_q0 = to_slv_vals["q0"]
                to_slv_wa = simu_vals[1]
                #to_slv_wa = to_slv_vals["wa"]
                to_slv_wMu = 0.4
                #to_slv_wMu = 0.4
                to_slv_amp = simu_vals[3]
                #to_slv_amp = known_amps[idx]
                to_slv_delta = simu_vals[4]
                to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
                if type(to_slv_weight)== np.ndarray:
                    to_slv_weight = to_slv_weight[0]
                    
            else:
                
                to_slv_q0 = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["q0"]
                #to_slv_q0 = to_slv_vals["q0"]
                to_slv_wa = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wa"]
                #to_slv_wa = to_slv_vals["wa"]
                to_slv_wMu = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["wMu"]
                #to_slv_wMu = 0.4
                to_slv_amp = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["amp"]
                #to_slv_amp = known_amps[idx]
                to_slv_delta = voxelsInPath[to_slv_indx]["fibril_param"]["fit"]["delta"]
                to_slv_weight = voxelsInPath[to_slv_indx]["weight"]
                if type(to_slv_weight)== np.ndarray:
                    to_slv_weight = to_slv_weight[0]
                    
                amp_min,amp_max,amp_mean = to_slv_amp,to_slv_amp,to_slv_amp
                
            vox_metrics = [to_slv_alpha,to_slv_beta,to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_amp,to_slv_delta,to_slv_weight]
            
            sum_vox_metrics.append(vox_metrics)    
                
            #voxel_sims = [[iq_sim(waveLen,unmasked_iqs[i][0],np.linspace(sim_chi[i]-2.5,sim_chi[i]+2.5,5),
                     #to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                          #for i in np.arange(0,len(chislice),1)]
        
            voxel_sims_raw = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,to_slv_amp,to_slv_weight)]
                          for i in np.arange(0,len(xr_chi),1)]
            
            voxel_sims_min = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_min,to_slv_weight)]
                          for i in np.arange(0,len(xr_chi),1)]
            
            voxel_sims_mean = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_mean,to_slv_weight)]
                          for i in np.arange(0,len(xr_chi),1)]
            
            voxel_sims_max = [[iq_sim(waveLen,qRange[i],np.linspace(xr_chi[i]-chi_range_sample,xr_chi[i]+chi_range_sample,int(chi_range_sample*2)),
                     to_slv_q0,to_slv_wa,to_slv_wMu,to_slv_delta,to_slv_alpha,to_slv_beta,amp_max,to_slv_weight)]
                          for i in np.arange(0,len(xr_chi),1)]
            
            sum_vox_raw.append(voxel_sims_raw)
            sum_vox_min.append(voxel_sims_min)
            sum_vox_max.append(voxel_sims_max)
            sum_vox_mean.append(voxel_sims_mean)
        
        
    max1_mean2 = [sum_vox_max[0][k][0] + sum_vox_mean[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    max1_mean2 = [max1_mean2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]  
    max1_mean2_diff = np.sum([np.abs(np.max(max1_mean2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    mean1_max2 = [sum_vox_mean[0][k][0] + sum_vox_max[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    mean1_max2 = [mean1_max2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]  
    mean1_max2_diff = np.sum([np.abs(np.max(mean1_max2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    max1_max2 = [sum_vox_max[0][k][0] + sum_vox_max[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    mean1_max2 = [mean1_max2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    max1_max2_diff = np.sum([np.abs(np.max(max1_max2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    mean1_mean2 = [sum_vox_mean[0][k][0] + sum_vox_mean[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    mean1_mean2 = [mean1_mean2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    mean1_mean2_diff = np.sum([np.abs(np.max(mean1_mean2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    min1_min2 = [sum_vox_min[0][k][0] + sum_vox_min[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    min1_min2 = [min1_min2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    min1_min2_diff = np.sum([np.abs(np.max(min1_min2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    min1_max2 = [sum_vox_min[0][k][0] + sum_vox_max[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    min1_max2 = [min1_max2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    min1_max2_diff = np.sum([np.abs(np.max(min1_max2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    max1_min2 = [sum_vox_max[0][k][0] + sum_vox_min[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    max1_min2 = [max1_min2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    max1_min2_diff = np.sum([np.abs(np.max(max1_min2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    min1_mean2 = [sum_vox_min[0][k][0] + sum_vox_mean[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    min1_mean2 = [min1_mean2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    min1_mean2_diff = np.sum([np.abs(np.max(min1_mean2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    mean1_min2 = [sum_vox_mean[0][k][0] + sum_vox_min[1][k][0] for k in np.arange(0,len(sum_vox_max[0]),1)]
    mean1_min2 = [mean1_min2[k][0:25] - solved_iq[k][0:25] for k in np.arange(0,len(iq),1)]
    mean1_min2_diff = np.sum([np.abs(np.max(mean1_min2[k]) - np.max(iq[k][0:25])) for k in np.arange(0,len(iq),1)])
    
    imbalance_test = np.argmin([max1_mean2_diff,mean1_max2_diff,max1_max2_diff,mean1_mean2_diff,min1_min2_diff,
                                min1_max2_diff,max1_min2_diff,min1_mean2_diff,mean1_min2_diff])
        
    amp_vals = [[max_amp,mean_amp],[mean_amp,max_amp],[max_amp,max_amp],[max_amp,max_amp],[min_amp,min_amp],
                [min_amp,max_amp],[max_amp,min_amp],[min_amp,mean_amp],[mean_amp,min_amp]]
    
    if imbalance_test>4:
        
        amp_imbalance = 1
        high_amp = imbalance_test
        
    return amp_imbalance,imbalance_test,amp_vals[imbalance_test]

def lhs_sample(data,qr,right_thresh = 0.10):
    
    """
    Simulation function for isolating lefthand-side of I(q) peak
    """
    
    qr_r = []
    data_r = []
    for i in range(len(data)):
        localdata = data[i]
        maxindx = np.argmax(data[i])
        maxval  = data[i][maxindx]
        yslice1 = data[i][0:maxindx]
        xslice1 = qr[0:maxindx]
        rightThreshval = right_thresh*maxval
        foundRight = False
        rightindx = 0
        for j in range(maxindx,len(data[i])):
            if data[i][j]<=rightThreshval and foundRight==False:
                foundRight=True
                rightindx = j
            pass
        yslice2 = data[i][maxindx:rightindx]
        xslice2 = qr[maxindx:rightindx]
        data_r.append(np.concatenate((yslice1,yslice2)))
        qr_r.append(np.concatenate((xslice1,xslice2)))
        
    return qr_r,data_r

def optiIchiPlot_single(params,profileData):
    
    """
    Fitting function for I(q) data
    """
                
    residuals = []
    
    chi_range_sample = 3
          
    #qRange = profileData[-1]
    
    #cutoffIqs = np.copy(profileData[1][1])
    
    #cutoffQs = np.copy(profileData[1][0])
    
    #model_IQs = [np.zeros((len(k))) for k in profileData[1][0]]
    model_IQs = [np.zeros((len(k))) for k in profileData[1]]
    
    #model_IQs = np.zeros((len(cutoffIqs),len(cutoffIqs[0])))
    
    chiWindow = profileData[3]
    
    """
    1. generate detector using parameters for each voxel along the beampath, and normalise 
        observed intensity values to highest observed intensity
        
        REMOVE CONSTANT
    """
    for i in range(0,len(profileData[0])):
                        
        label = profileData[0][i]
        #index = profileData[1][i]
        q0 = params["q0_" + label].value 
        wa = params["wa_" + label].value 
        wMuVal = params["wMu_" + label].value 
          
 
        alphaTest = np.radians(params["alpha1_" + label].value)
        #alphaTest = np.radians(profileData[7][i])
 
        betaTest = np.radians(params["beta1_" + label].value)
        #betaTest = np.radians(profileData[8][i])
        
        #constant = params["c_" + label].value  
        amp = params["amp_" + label].value
        #print(amp)
        
        waveLen = params["wavelength"].value  
        
        deltaVal = params["delta_" + label].value  
        
        weight = params["weight_" + label].value  
                        
        """
        2. Measure simulated i(q) values across i(chi) sectors measured in observed data, and normalise
            simulated intensity values 
        """ 
        
        flat=1
        integrate=False
        Nfac = t3d.normFac(0,wMuVal,wa)
        wfuncInt = 1
        
        chi_avrs = []
                                   
        for l in range(0,len(profileData[3])):
            #q_range = profileData[1][0][l]
            q_range = profileData[4][l]
            chi_avr = np.zeros_like(q_range)            
            chi = profileData[3][l]
            chi_avr = np.zeros_like(q_range)
            chis = np.linspace(chi-chi_range_sample,chi+chi_range_sample,int(chi_range_sample*2))
            for chi in chis:
                #print(chi)
                qx, qy, qz = t3d.calc_ewald_trace_singular(waveLen,q_range,chi)
                #Iq = constant+amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                Iq = amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                chi_avr = chi_avr+Iq
            model_IQs[l] = model_IQs[l]+(chi_avr/int(chi_range_sample*2))*weight            
            #model_IQs[l] = model_IQs[l]+Iq*weight
        
            
    """
    Clip out masked regions
    """
    real_masked_iqs = []
    model_masked_iqs = []
    
    for i in range(0,len(model_IQs)):   
        masks = np.where(profileData[-1][i]==0,True,False)
        real_masked_iqs.append(np.asarray(profileData[1][i])[masks==True])
        model_masked_iqs.append(model_IQs[i][masks==True])
                                            
    """
    3. compare observed and simulated values
    """
    resids = []
           
    #obs = np.copy(cutoffIqs)
    #exp = np.copy(model_IQs)
            
    #xAxis = np.arange(0,len(obs[0]),1)
    
    for k in range(0,len(model_IQs)):
        resid1 = []
        #obs1 = np.copy(profileData[1][1][k])
        #obs1 = np.copy(profileData[1][k])
        obs1 = np.copy(real_masked_iqs[k])
        obs1[np.isnan(obs1)] = 0
        
        exp1 = np.copy(model_masked_iqs[k])
        exp1[np.isnan(exp1)] = 0
        if exp1.size>obs1.size:
            obs1=np.append(obs1,np.zeros(exp1.size-obs1.size))
        
        if obs1.size>exp1.size:
            obs1 = obs1[0:obs1.size-(obs1.size-exp1.size)]
            
        #plt.plot(obs1)
        #plt.plot(exp1,color="r",linestyle = "dashed")
                    
        resid1 = obs1-exp1

        resid1[np.isnan(resid1)] = 0

        resids.append(resid1)
   
    residuals = (resids[0],resids[1],resids[2])
    residuals = np.concatenate(residuals)
        
    residuals = np.asarray(residuals)
    
    return residuals.flatten()


# paramTester_single(est_fibrilParams.params,profileData)

def paramTester_single(params,profileData):
    
    """
    Function for producing data for plotting fitted I(q) profiles
    """
    
    chi_range_sample = 3
                
    residuals = []
          
    qRange = profileData[-1]
    
    #cutoffIqs = np.copy(profileData[1][1])
    
    #cutoffQs = np.copy(profileData[1][0])
    
    #model_IQs = [np.zeros((len(k))) for k in profileData[1][0]]
    model_IQs = [np.zeros((len(k))) for k in profileData[1]]
        
    #model_IQs = np.zeros((len(cutoffIqs),len(cutoffIqs[0])))
    
    #chiWindow = profileData[3]
    
    indi_chis = []
    bare_indi_chis = []
    
    """
    1. generate detector using parameters for each voxel along the beampath, and normalise 
        observed intensity values to highest observed intensity
    """
    for i in range(0,len(profileData[0])):
                        
        label = profileData[0][i]
        #index = profileData[1][i]
        q0 = params["q0_" + label].value 
        wa = params["wa_" + label].value 
        wMuVal = params["wMu_" + label].value 
          
 
        alphaTest = np.radians(params["alpha1_" + label].value)
        #alphaTest = np.radians(profileData[7][i])
 
        betaTest = np.radians(params["beta1_" + label].value)
        #betaTest = np.radians(profileData[8][i])
        
        #constant = params["c_" + label].value  
        amp = params["amp_" + label].value  
        
        waveLen = params["wavelength"].value  
        
        deltaVal = params["delta_" + label].value  
        
        weight = params["weight_" + label].value  
                        
        """
        2. Measure simulated i(q) values across i(chi) sectors measured in observed data, and normalise
            simulated intensity values 
        """ 
        
        flat=1
        integrate=False
        Nfac = t3d.normFac(0,wMuVal,wa)
        wfuncInt = 1
        
        chi_avrs = []
        bare_chi_avrs = []
                                   
        for l in range(0,len(profileData[3])):
            #q_range = profileData[1][0][l]
            q_range = profileData[4][l]
            chi_avr = np.zeros_like(q_range)            
            chi = profileData[3][l]
            chi_avr = np.zeros_like(q_range)
            bare_avr = np.zeros_like(q_range)
            chis = np.linspace(chi-chi_range_sample,chi+chi_range_sample,int(chi_range_sample*2))
            for chi in chis:
                #print(chi)
                qx, qy, qz = t3d.calc_ewald_trace_singular(waveLen,q_range,chi)
                #Iq = constant+amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                Iq = amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                bare_avr = bare_avr+(Iq/amp)
                chi_avr = chi_avr+Iq
            chi_avrs.append(((chi_avr/int(chi_range_sample*2))*weight))
            bare_chi_avrs.append(bare_avr)
            model_IQs[l] = model_IQs[l]+((chi_avr/int(chi_range_sample*2))*weight)
            
        indi_chis.append(chi_avrs)
        bare_indi_chis.append(bare_chi_avrs)
                                        
                   
    return profileData[1][1],model_IQs,indi_chis,bare_indi_chis


def paramTester_comb(params,profileData):
    
    """
    Function for producing data for plotting fitted I(q) profiles
    probably redundant
    """
                
    residuals = []
          
    qRange = profileData[-1]
    
    cutoffIqs = np.copy(profileData[1])
    
    model_IQs = np.zeros((len(cutoffIqs),len(cutoffIqs[0])))
    
    chiWindow = profileData[2]
    
    """
    1. generate detector using parameters for each voxel along the beampath, and normalise 
        observed intensity values to highest observed intensity
    """
    for i in range(0,len(profileData[0])):
                        
        label = profileData[0][i]
        #index = profileData[1][i]
        q0 = params["q0_" + label].value 
        wa = params["wa_" + label].value 
        wMuVal = params["wMu_" + label].value 
          
 
        alphaTest = np.radians(params["alpha1_" + label].value)
        #alphaTest = np.radians(profileData[7][i])
 
        betaTest = np.radians(params["beta1_" + label].value)
        #betaTest = np.radians(profileData[8][i])
        
        constant = params["c_" + label].value  
        amp = params["amp_" + label].value  
        
        waveLen = params["wavelength"].value  
        
        deltaVal = params["delta_" + label].value  
        
        weight = params["weight_" + label].value  
                        
        """
        2. Measure simulated i(q) values across i(chi) sectors measured in observed data, and normalise
            simulated intensity values 
        """ 
        
        flat=1
        integrate=False
        Nfac = t3d.normFac(0,wMuVal,wa)
        wfuncInt = 1
                                   
        for l in range(0,len(profileData[2])):
            chi = profileData[2][l]
            qx, qy, qz = t3d.calc_ewald_trace_singular(waveLen,qRange,chi)
            Iq = constant+amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)      
            model_IQs[l] = model_IQs[l]+Iq*weight
                                        
    """
    3. compare observed and simulated values
    """
    resids = []
           
    obs = np.copy(cutoffIqs)
    exp = np.copy(model_IQs)
            
    xAxis = np.arange(0,len(obs[0]),1)
    
    for k in range(0,exp.shape[0]):
        resid1 = []
        obs1 = np.copy(obs[k])
        obs1[np.isnan(obs1)] = 0
        
        exp1 = np.copy(exp[k,:])
        exp1[np.isnan(exp1)] = 0
        if exp1.size>obs1.size:
            obs1=np.append(obs1,np.zeros(exp1.size-obs1.size))
        
        if obs1.size>exp1.size:
            obs1 = obs1[0:obs1.size-(obs1.size-exp1.size)]
                
        #plt.plot(obs1,color = "blue")
        #plt.plot(exp1,color = "red",linestyle = "dashed")
        
        resid1 = obs1-exp1

        resid1[np.isnan(resid1)] = 0

        resids.append(resid1)
                            
    return cutoffIqs,model_IQs,resids


def optiIchiPlot_comb(params,profileData):
    
    """
    Function for fitting I(q) profiles
    probably redundant
    """
    
    chi_range_sample = 3
    
    residuals = []
          
    qRange = profileData[-1]
    
    #cutoffIqs = np.copy(profileData[1][1])
    
    #cutoffQs = np.copy(profileData[1][0])
    
    #model_IQs = [np.zeros((len(k))) for k in profileData[1][0]]
    model_IQs = [np.zeros((len(k))) for k in profileData[1]]
    
    #model_IQs = np.zeros((len(cutoffIqs),len(cutoffIqs[0])))
    
    chiWindow = profileData[3]
    
    """
    1. generate detector using parameters for each voxel along the beampath, and normalise 
        observed intensity values to highest observed intensity
        
        REMOVE CONSTANT
    """
    for i in range(0,len(profileData[0])):
                        
        label = profileData[0][i]
        #index = profileData[1][i]
        q0 = params["q0_" + label].value 
        wa = params["wa_" + label].value 
        wMuVal = params["wMu_" + label].value 
          
    
        alphaTest = np.radians(params["alpha1_" + label].value)
        #alphaTest = np.radians(profileData[7][i])
    
        betaTest = np.radians(params["beta1_" + label].value)
        #betaTest = np.radians(profileData[8][i])
        
        #constant = params["c_" + label].value  
        amp = params["amp_" + label].value
        #print(amp)
        
        waveLen = params["wavelength"].value  
        
        deltaVal = params["delta_" + label].value  
        
        weight = params["weight_" + label].value  
                        
        """
        2. Measure simulated i(q) values across i(chi) sectors measured in observed data, and normalise
            simulated intensity values 
        """ 
        
        flat=1
        integrate=False
        Nfac = t3d.normFac(0,wMuVal,wa)
        wfuncInt = 1
                                   
        for l in range(0,len(profileData[3])):
            #q_range = profileData[1][0][l]
            q_range = profileData[4][l]
            chi_avr = np.zeros_like(q_range)            
            chi = profileData[3][l]
            chi_avr = np.zeros_like(q_range)
            chis = np.linspace(chi-chi_range_sample,chi+chi_range_sample,int(chi_range_sample*2))
            for chi in chis:
                #print(chi)
                qx, qy, qz = t3d.calc_ewald_trace_singular(waveLen,q_range,chi)
                #Iq = constant+amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                Iq = amp*t3d.Iplanarfibril(qx,qy,qz,0,0.01,0,q0,wMuVal,wa,Nfac,alphaTest,betaTest,integrate,flat=flat,delta=deltaVal)  
                chi_avr = chi_avr+Iq*weight
            model_IQs[l] = model_IQs[l]+((chi_avr/int(chi_range_sample*2)))            
            #model_IQs[l] = model_IQs[l]+Iq*weight
            
    """
    Clip out masked regions
    """
    real_masked_iqs = []
    model_masked_iqs = []
    
    for i in range(0,len(model_IQs)):   
        masks = np.where(profileData[-1][i]==0,True,False)
        real_masked_iqs.append(np.asarray(profileData[1][i])[masks==True])
        model_masked_iqs.append(model_IQs[i][masks==True])
                                            
    """
    3. compare observed and simulated values
    """
    resids = []
           
    #obs = np.copy(cutoffIqs)
    #exp = np.copy(model_IQs)
            
    #xAxis = np.arange(0,len(obs[0]),1)
    
    for k in range(0,len(model_IQs)):
        resid1 = []
        #obs1 = np.copy(profileData[1][1][k])
        #obs1 = np.copy(profileData[1][k])
        obs1 = np.copy(real_masked_iqs[k])
        obs1[np.isnan(obs1)] = 0
        
        exp1 = np.copy(model_masked_iqs[k])
        exp1[np.isnan(exp1)] = 0
        if exp1.size>obs1.size:
            obs1=np.append(obs1,np.zeros(exp1.size-obs1.size))
        
        if obs1.size>exp1.size:
            obs1 = obs1[0:obs1.size-(obs1.size-exp1.size)]
            
        #plt.plot(obs1)
        #plt.plot(exp1,color="r",linestyle = "dashed")
                    
        resid1 = obs1-exp1
    
        resid1[np.isnan(resid1)] = 0
    
        resids.append(resid1)
    
    residuals = (resids[0],resids[1],resids[2])
    residuals = np.concatenate(residuals)
        
    residuals = np.asarray(residuals)
    
    return residuals.flatten()

def padder(y):
    
    hlf_pts = (y[1:] + y[:-1]) / 2
    
    test = []
    for j in range(0,len(hlf_pts)):
        test.append(y[j])
        test.append(hlf_pts[j])
    test.append(y[-1])
    
    return np.asarray(test)


def cluster(data,indexes, maxgap):
    '''
    https://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python
    Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    indexes = [x for _, x in sorted(zip(data, indexes))]
    data.sort()
    groups = [[data[0]]]
    index_groups = [[indexes[0]]]
    for x,idx in zip(data[1:],indexes[1:]):
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
            index_groups[-1].append(idx)
        else:
            groups.append([x])
            index_groups.append([idx])
    return groups,index_groups

#loop_range,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params = 10,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params

def tomoSAXS_sim(loop_range,rot_range,rotated_beampaths,ichiExp,fibrilParams,sampling_params,recon_cake_params,recon_mask_chis,sim_vals,fibril_idxs,
                 recon_mask_pts,chiRefWindow,solv_params,a1,Mask,slice_saxs_file,output_path,fit_params,multi_params,param_save = 1,graph_display = False):
    
    """
    Function for isolating independent chi regions for single and/or multiple scattering objects, sampling these regions in the respective SAXS frame, and 
    fitting threeDXRD scattering models to sampled data, for each beampath of each orientation in a SAXS scan.
    """
    
    recon_start_time = np.load(output_path+"recon_start_time.npy")[0]
    
    recon_times = []
    
    wake_time = time()
    
    q0_min,q0_max = fit_params["q0_min"],fit_params["q0_max"] # min/max for q0 parameter estimation
    wa_min,wa_max = fit_params["wa_min"],fit_params["wa_max"]  # min/max for wa parameter estimation
    wMu_min,wMu_max = fit_params["wMu_min"],fit_params["wMu_max"] # min/max for wMu parameter estimation

    ab_var = fit_params["ab_var"] # variation permitted in estimated alpha and beta values (in degrees) for fitting
    delta_min,delta_max = fit_params["delta_min"],fit_params["delta_max"] # min/max for delta parameter estimation
    amp_min,amp_max = fit_params["amp_min"],fit_params["amp_max"] # min/max for amplitude estimation - mean is absolute value; max is multiplier of SVD amplitude estimate

    max_amp_rat = fit_params["max_amp_rat"] # threshold value for ratio between fitted peak intensity and measured intensity for discriminating fits that are too high/too low
    min_abs_int = fit_params["min_abs_int"]  # minimum intensity value for fits
    min_error = fit_params["min_error"] # minimum percentage standard error
    min_abs_relax = fit_params["min_abs_relax"] # minimum intensity value for relaxed fits 
    min_fit_wa = fit_params["min_fit_wa"] # threshold minimum fitted wa value for fits
    max_fit_wa = fit_params["max_fit_wa"] # threshold maximum fitted wa value for fits
    min_fit_wMu = fit_params["min_fit_wMu"] # threshold minimum fitted wMu value for fits
    max_fit_wMu = fit_params["max_fit_wMu"] # threshold maximum fitted wMu value for fits
    
    min_fit_q0 = fit_params["min_fit_q0"] # threshold minimum fitted q0 value for fits
    max_fit_q0 = fit_params["max_fit_q0"] # threshold maximum fitted q0 value for fits
            
    diameter_min = fit_params["diam_min"] # threshold maximum fitted wa value for fits
    diameter_max = fit_params["diam_max"] # threshold maximum fitted wMu value for fits
    
    snr_min = fit_params["min_snr"]
    
    if recon_cake_params["test"]!=None:
        if recon_cake_params["test"]=="full_smooth":
            int_smooth = True
    else:
        int_smooth = False
    
    
    chi1,chi2,chirange,nchi,qx,qy,qz,dxs,q0_m,q0m_low,q0m_high,nq0,threshold_interference,threshold_detection,q_fitIqr,binning,chi_range_sample,nslices = sampling_params[0:len(sampling_params)]
    
    single_solve,combi_solve,multi_solve,relax_fit = solv_params[0:4]
    
    if relax_fit == True:
        min_error = fit_params["min_error_relax"]
        snr_min = fit_params["relaxed_snr"]
    
    wavelen = recon_cake_params["wavelen"]
    
    single_solves = []
    multi_solves = []
    
    fitted_sum = len([k for k in fibrilParams if k["fit"] !={}])
    
    overlap_params,overlap_params_triple,overlap_thresh,I3_tot,I3_singleton,triple_overlap_params,triple_overlap_thresh = multi_params
    
    
    threshold_interference_multi = 100 - overlap_thresh["thresh_combined"]
    
    kill_loop = False
    
    while kill_loop == False:
    
        for loopIndex in range(0,loop_range):
            
            fitted_sum = len([k for k in fibrilParams if k["fit"] !={}])
            
            for r in range(0,len(rot_range)):
                
                print("angle: ", rot_range[r])
                    
                rot_beampath = np.asarray(rotated_beampaths[r])
                
                og_bps = np.arange(0,len(rot_beampath)*binning,binning)
                                
                Ichi1D_full = [None]*len(rot_beampath)
                
                sample_bps = []
                
                for i, path_dict in enumerate(rot_beampath):
                    if path_dict != None:
                        
                        """
                        for all beam paths
                        """
                        Ichi1D_full[i]=np.zeros_like(chirange)
                        allVoxelsSolved = True
                        if path_dict["voxels"] != None:
                            for vox in path_dict["voxels"]:
                                if vox["fibril_param"]["solved"]==False and allVoxelsSolved==True:
                                    allVoxelsSolved = False
                            
                        
                        if path_dict["voxels"] != None and allVoxelsSolved==False:
                            
                            if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:

                                pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in path_dict["voxels"]])
                                if len(pdict_vox_indxs)>1:
                                    vox_repeats = [k for k in np.arange(1,len(pdict_vox_indxs),1) if pdict_vox_indxs[k]==pdict_vox_indxs[k-1]]
                                    vox_repeats = [np.unique(pdict_vox_indxs)[k] for k in np.arange(0,len(np.unique(pdict_vox_indxs)),1) 
                                                   if np.unique(pdict_vox_indxs,return_counts = True)[1][k]>1]
                                    vox_singles = [np.unique(pdict_vox_indxs)[k] for k in np.arange(0,len(np.unique(pdict_vox_indxs)),1) 
                                                   if np.unique(pdict_vox_indxs,return_counts = True)[1][k]==1]
                                    if len(vox_repeats)>0:
                                        
                                        path_dict_singles = [path_dict["voxels"][np.where(pdict_vox_indxs ==k)[0][0]] for k in vox_singles]
                                        path_dict_repeats = [path_dict["voxels"][np.where(pdict_vox_indxs ==k)[0][0]] for k in vox_repeats]
                                        path_dict["voxels"] = path_dict_singles+path_dict_repeats
                                        pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in path_dict["voxels"]])
                                sample_bps.append(i)
                                
                                print("beampath: ",og_bps[i])
                               
                                """
                                Intensity2D and Intensity1D are the predicted 2D and 1D
                                intensity patterns on the detector for the i^th beam_path
                                """
                                Intensity2D = np.zeros_like(qx)
                                Ichi1D = np.zeros_like(chirange)
                        
                                t1 = time()
                                nv = 0
                                nvstrong = 0
                                
                                if graph_display == False and r == 0 and i == sample_bps[0]:
                                    scalefac = 0.5
                                    fig = plt.figure(layout="constrained")
                                    gs = GridSpec(2, 2, figure=fig)
                                    ax1 = fig.add_subplot(gs[1, 0])
                                    ax0 = fig.add_subplot(gs[0, 0])
                                    ax2 = fig.add_subplot(gs[0, 1])
                                    #ax3 has the I(q) plots
                                    ax3 = fig.add_subplot(gs[1, 1])
                                elif graph_display == True:
                                    scalefac = 0.5
                                    fig = plt.figure(layout="constrained")
                                    gs = GridSpec(2, 2, figure=fig)
                                    ax1 = fig.add_subplot(gs[1, 0])
                                    ax0 = fig.add_subplot(gs[0, 0])
                                    ax2 = fig.add_subplot(gs[0, 1])
                                    #ax3 has the I(q) plots
                                    ax3 = fig.add_subplot(gs[1, 1])
                                    
                                ax1.autoscale(True)
                                tgas1 = time()
                                ax1, Intensity2D, Ichi1D, nv, nvstrong = getAllSAXS(path_dict,ax1,Intensity2D,Ichi1D,qx,qy,qz,nv,nvstrong,fibrilParams,
                                                                                    chi1=chi1,chi2=chi2,nchi=nchi,q1=q0m_low,q2=q0m_high,ymax=100,nq=nq0,
                                                                                    wavelen=wavelen,threshold_detection=threshold_detection,simu_vals = sim_vals)
                                
                                tgas2 = time()
                                tgas = tgas2-tgas1
                                
                                Ichi1D_full[i]=Ichi1D_full[i]+Ichi1D
                                
                                """
                                Plot the total 2D and 1D SAXS pattern
                                """
                                if graph_display == True:
                                    tp21S1 = time()
                                    ax0, ax1 = plot2D_1DSAXS(ax0,ax1,chirange,Intensity2D,Ichi1D,threshold_detection,ymax=100)
                                    tp21S2 = time()
                                    tp21S = tp21S2-tp21S1
                                    t2 = time()
                                               
                                
                                """
                                next - scan this profile to find any isolatable voxels. The criteria should be that
                                the intensity over this range is >90% (e.g.) from that voxel. e.g. if
                                threshold = 10%, then intensity over the range should be >(100-threshold)
                                """
                                rot_beampath[i]["ichi"]=Ichi1D
                                """
                                first run through to calculate the total intensity of unsolved voxels
                                """
                                trun11 = time()
                                Ichi1D_unsolved = np.zeros_like(Ichi1D)
                                Ichi1D_solved = np.zeros_like(Ichi1D)
                                max_chis,slv_max_chis = [],[]
                                chi_indxs,slv_indxs = [],[]
                                ichis,slv_ichis = [],[]
                                x_locs,y_locs,z_locs = [],[],[]
                                for idx, voxel in enumerate(path_dict["voxels"]):
                                    vox_indx = voxel["fibril_param"]["indx"]
                                    #print(voxel["fibril_param"]["initial_amp_est"])
                                    indx = np.where(fibril_idxs == vox_indx)[0][0]
                                    fibrilParams[indx]["intersected"]=True
                                    ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(voxel,chi1=chi1,chi2=chi2,nchi=nchi,
                                                                                   q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen,simu_vals = sim_vals)
                                    
                                    if fibrilParams[indx]["solved"]==False:
                                        x_locs.append(voxel["fibril_param"]["x"])
                                        y_locs.append(voxel["fibril_param"]["y"])
                                        z_locs.append(voxel["fibril_param"]["z"])
                                        chi_indxs.append(indx)
                                        max_chis.append(chimax)
                                        ichis.append(ichi)
                                        Ichi1D_unsolved = Ichi1D_unsolved + ichi  
                                                                                
                                    else:
                                        #print(idx)
                                        slv_indxs.append(indx)
                                        slv_max_chis.append(chimax)
                                        slv_ichis.append(ichi)
                                        Ichi1D_solved = Ichi1D_solved + ichi  
                                                                
                                max_chis = np.asarray(max_chis)
                                chi_indxs = np.asarray(chi_indxs)
                                ichis = np.asarray(ichis)
                                
                                slv_indxs = np.asarray(slv_indxs)
                                
                                trun12 = time()
                                trun1 = trun12-trun11
                                
                                """
                                second run through to calculate if any of the unsolved voxels 
                                can be solved
                                """
                                trun21 = time()
                                                               
                                bp_solves = 0
                                last_solve = bp_solves
                                
                                for idx, voxel in enumerate(path_dict["voxels"]):
                                    
                                    std_errors = [100]
                                    
                                    vox_indx = voxel["fibril_param"]["indx"]
                                    indx = np.where(fibril_idxs == vox_indx)[0][0]
                                    
                                    if type(voxel["weight"]) == np.ndarray and len(voxel["weight"])>0 or type(voxel["weight"]) != np.ndarray:
                                    
                                        if fibrilParams[indx]["solved"]==False:
                                        
                                            
                                            Ichi1D_unsolved = np.zeros_like(Ichi1D)
                                            Ichi1D_solved = np.zeros_like(Ichi1D)
                                            max_chis,slv_max_chis = [],[]
                                            chi_indxs,slv_indxs = [],[]
                                            ichis,slv_ichis = [],[]
                                            x_locs,y_locs,z_locs = [],[],[]
                                            bareichis = []
                                            IchiCptsChiMax = []
                                            IchiCptsIChiMax = []
                                            IchiCpts = []
                                            Ichi_us = np.zeros_like(chirange)
                                            
                                            for s_idx, s_voxel in enumerate(path_dict["voxels"]):
                                                s_vox_indx = s_voxel["fibril_param"]["indx"]
                                                s_indx = np.where(fibril_idxs == s_vox_indx)[0][0]
                                                #s_indx = s_voxel["fibril_param"]["indx"]
                                                fibrilParams[s_indx]["intersected"]=True
                                                ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(s_voxel,chi1=chi1,chi2=chi2,nchi=nchi,
                                                                                               q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen,simu_vals = sim_vals)
                                                
                                                IchiCptsChiMax.append(chirange[np.argmax(ichi)])
                                                IchiCptsIChiMax.append(np.max(ichi))
                                                IchiCpts.append(ichi)
                                                #Ichi_us = Ichi_us+ichi
                                                
                                                if fibrilParams[s_indx]["solved"]==False:
                                                    x_locs.append(s_voxel["fibril_param"]["x"])
                                                    y_locs.append(s_voxel["fibril_param"]["y"])
                                                    z_locs.append(s_voxel["fibril_param"]["z"])
                                                    chi_indxs.append(s_indx)
                                                    max_chis.append(chimax)
                                                    ichis.append(ichi)
                                                    Ichi1D_unsolved = Ichi1D_unsolved + ichi 
                                                    bareichis.append(bareichi)
                                                    Ichi_us = Ichi_us+ichi
                                                else:
                                                    #print(i)
                                                    slv_indxs.append(s_indx)
                                                    slv_max_chis.append(chimax)
                                                    slv_ichis.append(ichi)
                                                    Ichi1D_solved = Ichi1D_solved + ichi  
                                                                   
                                            max_chis = np.asarray(max_chis)
                                            chi_indxs = np.asarray(chi_indxs)
                                            ichis = np.asarray(ichis)
                                            max_ichis = [chirange[np.argmax(k)] for k in ichis]
                                            slv_indxs = np.asarray(slv_indxs)
                                            
                                            #indx = voxel["fibril_param"]["indx"]
                                            est_amp = voxel["fibril_param"]["initial_amp_est"]
                                            thisalpha, thisbeta = voxel["fibril_param"]["alpha"],\
                                                voxel["fibril_param"]["beta"]
                                            thisweight = voxel["weight"]
                                            #if fibrilParams[indx]["solved"]==False:
                                                
                                            ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(voxel,chi1=chi1,chi2=chi2,nchi=nchi,
                                                                                           q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen,simu_vals = sim_vals)
                                            
                                            if np.max(bareichi)>1:
                                                solvable = False
                                                chiwindow, solvable,opti_chiwindow,solved_windows = findChiWindow(ichi,Ichi1D_unsolved,chirange,recon_mask_pts,
                                                                                                                  chiwin=chiRefWindow,threshold_interference=threshold_interference,
                                                                                                                  threshold_detection=threshold_detection)
                                                                                                                                                                                        
                                                if solvable == True and len(opti_chiwindow)>1:                                                    
                                                                                                                                                    
                                                    print("analysing real data for ",r,i,idx,indx)
                                                    
                                                    if solv_params[-1] == "new":
                                                        iq,qRange,chi_slices,masked_q,frame,sig,succ_test = sampleIqOnChiWin(r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,
                                                                                                                             recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,
                                                                                                                             opti_chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],voxel,sim_vals,wavelen,
                                                                                                                             threshold_interference,nslices=nslices,nq0=nq0,to_smooth = int_smooth,chi_range_sample = chi_range_sample)
                                                    else:
                                                        
                                                        iq,qRange,chi_slices,masked_q,frame,sig,succ_test = sampleIqOnChiWin(r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,
                                                                                                                             recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,
                                                                                                                             opti_chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],voxel,sim_vals,wavelen,
                                                                                                                             threshold_interference,nslices=3,nq0=nq0,to_smooth = int_smooth,chi_range_sample = chi_range_sample)
                                                        #if succ_test == None:
                                                        
                                                            #iq,qRange,chi_slices,masked_q,frame,sig,succ_test = sampleIqOnChiWin_old(r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,
                                                                                                                             #recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,opti_chiwindow,q_fitIqr,solved_windows,nslices=3,to_smooth = int_smooth)
                                                            
                                                                                                    
                                                    comb_indxs = np.append(slv_indxs,indx)
                                                
                                                    
                                                    if succ_test != None and np.min([np.min(k[0:25]) for k in iq])>-3e8 and np.max([np.max(k[5:20])-np.min(k[5:20]) for k in iq])>min_abs_int:
                                                        
                                                        amp_test = single_amp_test(iq,chi_slices,qRange,path_dict["voxels"],comb_indxs,10000,1500000,600000,wavelen,sim_vals) 
                                                        
                                                    
                                                        print("single solving ",r,i,idx,indx) 

                                                        if np.min([np.max(k) for k in iq])<min_abs_int: 
                                                            print("FAILED MIN AMP THRESHOLD")
                                                        else:
                                                            
                                                            if single_solve == True:
                                                                high_snr = snr_test(qRange,iq,recon_cake_params,snr_min)
                                                            else:
                                                                high_snr = snr_test(qRange,iq,recon_cake_params,fit_params["single_min_snr"])
                                                            
                                                            if high_snr == False:
                                                                print("FAILED SNR THRESHOLD")
                                                            else:
                                                                
                                                                single_solves.append([r,i,vox_indx,np.max(ichi),ichi,Ichi1D_unsolved])
                                                                                                
                                                                if single_solve == True:
                                                                    
                                                                    og_indx = indx
                                                                                                        
                                                                    if graph_display == True:
                                                                    
                                                                        xsol, ysol, zsol = voxel["fibril_param"]["x"],voxel["fibril_param"]["y"],voxel["fibril_param"]["z"]                                                                      
                                                                        linewidth=0.5
                                                                        circle = patches.Circle([xsol,zsol],radius=0.6*(dxs/2.0),\
                                                                                                      fc="yellow",alpha=1.0,ec="red",lw=linewidth,\
                                                                                                          angle=r)
                                                                        ax2.add_patch(circle)
                                                                    """
                                                                    fit I(q) profile to independent sector and update
                                                                    the "fit" entry for fibrilParams[indx]
                                                                    """
                                                                    
                                                                    """
                                                                    simulating the data with the real parameters; adding noise at end
                                                                    """
                                                                            
                                                                    params = Parameters()
                                                                    vox_alpha = voxel["fibril_param"]["alpha"]
                                                                    vox_beta = voxel["fibril_param"]["beta"]
                                                                        
                                                                    labels = []                                                                                       
                                                                    
                                                                    labels.append(str(int(vox_indx)))
                                                                    params.add('q0_' + labels[0], value = sim_vals[0], min = q0_min, max = q0_max)      
                                                                    params.add('wa_' + labels[0], value = sim_vals[1], min = wa_min, max = wa_max)        
                                                                    params.add('wMu_' + labels[0], value = sim_vals[2], min = wMu_min, max = wMu_max)
                                                                    #params.add('alpha1_' + labels[0], value = vox_alpha,min = vox_alpha-ab_var, max = vox_alpha+ab_var)       
                                                                    #params.add('beta1_' + labels[0], value = vox_beta,min = vox_beta-ab_var, max = vox_beta+ab_var)
                                                                    params.add('alpha1_' + labels[0], value = vox_alpha,vary = False)       
                                                                    params.add('beta1_' + labels[0], value = vox_beta,vary = False)
                                                                    params.add('wavelength', value = wavelen,vary = False)
                                                                    params.add('delta_'+ labels[0], value = sim_vals[4],min = delta_min, max = delta_max)
                                                                    
                                                                    #params.add("c_"+labels[0],value = 0,min = -1e8, max = 1e8)
                                                                    #params.add("amp_"+labels[0],value = voxel["fibril_param"]["initial_amp_est"],min = amp_min, max = 1e6)
                                                                    params.add("amp_"+labels[0],value = amp_test,min = amp_min, max = amp_max)
                                                                    params.add("weight_"+labels[0],value = voxel["fibril_param"]["weight"],vary = False)
                                                                    for solv_indx in slv_indxs:
                                                                        solv_vox = path_dict["voxels"][np.where(pdict_vox_indxs==fibrilParams[solv_indx]["indx"])[0][0]]
                                                                        if solv_vox["fibril_param"]["fit"] != {}:
                                                                            solv_label = str(int(solv_vox["fibril_param"]["indx"]))
                                                                            labels.append(solv_label)
                                                                            params.add('q0_' + solv_label, value = solv_vox['fibril_param']["fit"]["q0"],vary = False)      
                                                                            params.add('wa_' + solv_label, value = solv_vox['fibril_param']["fit"]["wa"],vary = False)        
                                                                            params.add('wMu_' + solv_label, value = solv_vox['fibril_param']["fit"]["wMu"],vary = False)
                                                                            params.add('alpha1_' + solv_label, value = solv_vox['fibril_param']["alpha"],vary = False)       
                                                                            params.add('beta1_' + solv_label, value = solv_vox['fibril_param']["beta"],vary = False)
                                                                            #params.add('wavelength', value = wavelen,vary = False)
                                                                            params.add('delta_'+ solv_label, value = solv_vox['fibril_param']["fit"]["delta"],vary = False)
                                                                            #params.add('c_'+ solv_label, value = solv_vox['fibril_param']["fit"]["c"],vary = False)
                                                                            params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],min = solv_vox['fibril_param']["fit"]["amp"]/10, max = solv_vox['fibril_param']["fit"]["amp"]*10)
                                                                            #params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],vary = False)
                                                                            #params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],min = solv_vox['fibril_param']["fit"]["amp"]/10, max = solv_vox['fibril_param']["fit"]["amp"]*10)
                                                                            params.add('weight_'+ solv_label, value = solv_vox['fibril_param']["weight"],vary = False)
                                                                                  
                                                                    profileData = [labels,iq,chimax,360-np.asarray(chi_slices),qRange,masked_q]
                                                                    
                                                                    t1 = time()
                                                                    est_fibrilParams = minimize(optiIchiPlot_single, params, method = "Nelder",args=([profileData]),max_nfev=50000)
                                                                    t2 = time()
                                                                    print("computation time: ",t2-t1)
                                                                    
                                                                    ax3_data = paramTester_single(est_fibrilParams.params,profileData)
                                                                    
                                                                    if np.min([np.argmax(ax3_data[1][k]) for k in np.arange(0,3,1)])>2:
                                                                    
                                                                   
                                                                        cols = ["blue","green","orange","purple","orange","yellow"]
                                                                        
                                                                        masked_qs = [np.where(masked_q[k]==0,qRange[k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                        masked_real = [np.where(masked_q[k]==0,iq[k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                        masked_fits = [np.where(masked_q[k]==0,ax3_data[1][k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                        
                                                                        esti_amp = est_fibrilParams.params['amp_' + labels[0]].value 
                                                                        esti_q0 = est_fibrilParams.params['q0_' + labels[0]].value 
                                                                        esti_wa = est_fibrilParams.params['wa_' + labels[0]].value 
                                                                        esti_wMu = est_fibrilParams.params['wMu_' + labels[0]].value 
                                                                        esti_delta = est_fibrilParams.params['delta_' + labels[0]].value 
                                                                        
                                                                        esti_wp = esti_q0*np.tan(esti_wMu)
                                                                        
                                                                        esti_diam = (2.857*(1/esti_wp))+10
                                                                                                                                                                                               
                                                                        bad_fit = False
                                                                        
                                                                        if esti_q0 == sim_vals[0] or esti_wa == sim_vals[1] or esti_wMu == sim_vals[2]:
                                                                            
                                                                            bad_fit = False
                                                                        
                                                                        else: 
                                                                        
                                                                            sim_peaks = [ np.mean(ax3_data[1][k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
                                                                            real_peaks = [ np.mean(iq[k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
                                                                            
                                                                            peak_ratios = [sim_peaks[k]/real_peaks[k] for k in np.arange(0,len(iq),1)]                                                            
                                                                            
                                                                            if relax_fit == True:
                                                                                
                                                                                for k in range(nslices):
                                                                                    plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                    plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                solve_time = time() - wake_time
                                                                                print("solve time = ",solve_time)
                                                                                                                                       
                                                                                if np.max([np.max(k)-np.min(k) for k in ax3_data[1]])<min_abs_relax:
                                                                                    print("amp too low")
                                                                                    plt.title(("solving ",r,i,idx,og_indx,": AMP INACCURATE, ","solve time = ",str(solve_time)))
                                                                                    bad_fit = True
                                                                                    plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                    plt.show()
                                                                                    plt.close()
                                                                                else:
                                                                                    if esti_wa>max_fit_wa or esti_wa<min_fit_wa:
                                                                                        print("wa inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": WA INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_wMu>max_fit_wMu or esti_wMu<min_fit_wMu:
                                                                                        print("wMu inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": wMu INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_q0>max_fit_q0 or esti_q0<min_fit_q0:
                                                                                        print("q0 inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": q0 INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_diam>diameter_max or esti_diam<diameter_min:
                                                                                        print("diameter inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": diameter INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                if bad_fit == True:
                                                                                    plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                    
                                                                                else:
                                                                                    print("accuracy check passed - adding fit to dictionary")
                                                                                    plt.title(("solving ",r,i,idx,og_indx))
                                                                                    #plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                plt.show()
                                                                                plt.close()
                                                                            
                                                                            else:
                                                                            
                                                                                for k in range(nslices):
                                                                                    plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                    plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                solve_time = time() - wake_time
                                                                                print("solve time = ",solve_time)
                                                                                                                                                    
                                                                                
                                                                                if np.max([np.max(k)-np.min(k) for k in ax3_data[1]])<min_abs_int: 
                                                                                    print("amp too low")
                                                                                    plt.title(("solving ",r,i,idx,og_indx,": AMP INACCURATE, ","solve time = ",str(solve_time)))
                                                                                    bad_fit = True
                                                                                    plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                    plt.show()
                                                                                    plt.close()
                                                                                else:
                                                                                    if esti_wa>max_fit_wa or esti_wa<min_fit_wa:
                                                                                        print("wa inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": WA INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_wMu>max_fit_wMu or esti_wMu<min_fit_wMu:
                                                                                        print("wMu inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": wMu INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_q0>max_fit_q0 or esti_q0<min_fit_q0:
                                                                                        print("q0 inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": q0 INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                    if esti_diam>diameter_max or esti_diam<diameter_min:
                                                                                        print("diameter inaccurate")
                                                                                        plt.title(("solving ",r,i,idx,og_indx,": diameter INACCURATE, ","solve time = ",str(solve_time)))
                                                                                        bad_fit = True
                                                                                if bad_fit == True:
                                                                                    plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                else:
                                                                                    #print("accuracy check passed - adding fit to dictionary")
                                                                                    plt.title(("solving ",r,i,idx,og_indx))
                                                                                    title = ("solving ",r,i,idx,og_indx)
                                                                                        
                                                                                        #plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                plt.show()
                                                                                plt.close()
                                                                                                                                                        
                                                                        if bad_fit == False:                                                                                                                                                                                                                                                                                                                                                 
                                                            
                                                                            std_errors,xs,ys,new_xs,new_ys = [],[],[],[],[]

                                                                            for chi_test in range(0,len(chi_slices)):

                                                                                chi = 360-np.asarray(chi_slices[chi_test])                                                                                
                                                                                                                                                                
                                                                                fitted_iq = np.sum(np.asarray([ax3_data[2][k][chi_test] for k in np.arange(1,len(ax3_data[2]),1)]),0) 
                                                                                
                                                                                if len(labels) == 1:
                                                                                    fitted_iq = np.zeros_like(iq[chi_test])
                                                                                                                                               
                                                                                nan_mask = ~np.isnan(masked_fits[chi_test])
                                                                                
                                                                                i_masked = masked_fits[chi_test][nan_mask]
                                                                                q_masked = qRange[chi_test][nan_mask]
                                                                                
                                                                                if len(q_masked)==0:
                                                                                    
                                                                                    std_errors.append(100)
                                                                                    xs.append(qRange[chi_test])
                                                                                    ys.append(masked_fits[chi_test])
                                                                                    new_ys.append(masked_fits[chi_test])
                                                                                
                                                                                else:
                                                                                    peak_mod = fit_gauss(q_masked,i_masked)
                                                                                    
                                                                                    q_lim = [peak_mod[1]["center"].value - (peak_mod[1]["sigma"].value*3),
                                                                                             peak_mod[1]["center"].value + (peak_mod[1]["sigma"].value*3)]
                                                                                    
                                                                                    qRange_lim = [find_nearest(qRange[chi_test],q_lim[0])[1],
                                                                                                  find_nearest(qRange[chi_test],q_lim[1])[1]]
                                                                                    
                                                                                    if np.max(qRange_lim) == 0:
                                                                                        
                                                                                        std_errors.append(100)
                                                                                        xs.append(qRange[chi_test])
                                                                                        ys.append(masked_fits[chi_test])
                                                                                        new_ys.append(masked_fits[chi_test])
                                                                                        
                                                                                    else:
                                                                                    
                                                                                        if qRange_lim[1] - qRange_lim[0]<2:
                                                                                           qRange_lim[0] = qRange_lim[0]-2 
                                                                                           qRange_lim[1] = qRange_lim[1]+3
                                                                                        
                                                                                        fitted_iq = fitted_iq[qRange_lim[0]:qRange_lim[1]]
                                                                                        
                                                                                        x = qRange[chi_test][qRange_lim[0]:qRange_lim[1]]
                                                                                        
                                                                                        y = masked_real[chi_test][qRange_lim[0]:qRange_lim[1]]            
                                                                                        
                                                                                        y2 = masked_fits[chi_test][qRange_lim[0]:qRange_lim[1]]
                                                                                        
                                                                                        while len(y)<10:
                                                                                            y = np.asarray(padder(y))
                                                                                            y2 = np.asarray(padder(y2))
                                                                                            
                                                                                        x = np.linspace(x[0],x[-1],len(y))
                                                                                        
                                                                                        max_calc = np.max([np.max(k) for k in [y,y2]])
                                                                                        
                                                                                        y2 = (y2/max_calc)*100
                                                                                        y = (y/max_calc)*100
                                                                                        
                                                                                        mse = ((y - y2)**2).mean(axis=0)
                                                                                        syx = np.sqrt(mse)
                                                                                        if "nan" in str(syx):
                                                                                            std_errors.append(100)
                                                                                            new_ys.append(y)
                                                                                        else:    
                                                                                            std_errors.append(int(syx))
                                                                                            new_ys.append(y2)
                                                                                        
                                                                                        xs.append(x)
                                                                                        ys.append(y)
                                                                                
                                                                            if np.min(std_errors)<=min_error:
                                                                                for k in range(0,len(std_errors)):
                                                                                    plt.plot(xs[k],ys[k],'.', color = cols[k], label='chi range '+str(k)+' observations')
                                                                                    plt.plot(xs[k], new_ys[k], '-', color = cols[k], label='chi range '+str(k)+' fit') 
                                                                                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                title = ("solving ",r,i,idx,og_indx)
                                                                                plt.title(("solving ",r,i,idx,og_indx,", solve time = ",str(solve_time)))                                   
                                                                                plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
                                                                                plt.show()
                                                                                plt.close()   
                                                                                
                                                                                for k in range(nslices):
                                                                                    plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                    plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                print("accuracy check passed - adding fit to dictionary")
                                                                                solve_time = time() - wake_time
                                                                                print("solve time = ",solve_time)
                                                                                plt.title(("solving ",r,i,idx,og_indx,", solve time = ",str(solve_time)))
                                                                                plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                plt.show()
                                                                                plt.close()
                                                                                
                                                                            if np.min(std_errors)<=min_error:
                                                                                for idx, voxel in enumerate(fibrilParams):
                                                                                    indx = str(int(voxel["indx"]))
                                                                                    if indx in labels: 

                                                                                        voxel["fit"]["q0"] = est_fibrilParams.params['q0_' + str(int(indx))].value
                                                                                        voxel["fit"]["amp"] = est_fibrilParams.params['amp_' + str(int(indx))].value
                                                                                        voxel["fit"]["wa"] = est_fibrilParams.params['wa_' + str(int(indx))].value
                                                                                        voxel["fit"]["wMu"] = est_fibrilParams.params['wMu_' + str(int(indx))].value
                                                                                        voxel["fit"]["delta"] = est_fibrilParams.params['delta_' + str(int(indx))].value
                                                                                        #voxel["fit"]["c"] = est_fibrilParams.params['c_' + str(int(indx))].value
                                                                                        voxel["alpha"] = est_fibrilParams.params['alpha1_' + str(int(indx))].value
                                                                                        voxel["beta"] = est_fibrilParams.params['beta1_' + str(int(indx))].value
                                                                                        voxel["solved"]=True
                                                                                        
                                                                                for bp in rotated_beampaths:                                                        
                                                                                    for j, pdict in enumerate(bp):
                                                                                        if pdict != None:
                                                                                            if pdict["voxels"] != None:                                                                    
                                                                                                if j in ichiExp[r] and type(ichiExp[r][j]["kapton"]) == bool:                                                    
                                                                                                    for idx, voxel in enumerate(pdict["voxels"]):
                                                                                                        indx = str(int(voxel['fibril_param']["indx"]))
                                                                                                        if indx in labels: 
                                                                                                            #add_test.append([idx,indx])
                                                                                                            voxel['fibril_param']['solved'] = True
                                                                                                            voxel['fibril_param']['fit']['q0'] = est_fibrilParams.params['q0_' + str(indx)].value
                                                                                                            voxel['fibril_param']['fit']['wa'] = est_fibrilParams.params['wa_' + str(indx)].value
                                                                                                            voxel['fibril_param']['fit']['wMu'] = est_fibrilParams.params['wMu_' + str(indx)].value
                                                                                                            voxel['fibril_param']['fit']['delta'] = est_fibrilParams.params['delta_' + str(indx)].value
                                                                                                            #voxel['fibril_param']['fit']['c'] = est_fibrilParams.params['c_' + str(indx)].value
                                                                                                            voxel['fibril_param']['fit']['amp'] = est_fibrilParams.params['amp_' + str(indx)].value
                                                                                                            voxel["fibril_param"]["alpha"] = est_fibrilParams.params['alpha1_' + str(indx)].value
                                                                                                            voxel["fibril_param"]["beta"] = est_fibrilParams.params['beta1_' + str(indx)].value
                                                                                                            voxel["solved"]=True
                                                                                                                                                                                           
                                                                                now = datetime.now()
                                                                                year,month,day = now.strftime("%Y"),now.strftime("%m"),now.strftime("%d")
                                                                                
                                                                                time_finsihed = time()
                                                                                
                                                                                fit_time = time_finsihed - recon_start_time
                                                                                
                                                                                
                                                                                fitfile = output_path+day+"-"+month+"-"+year+"_"+"fitfile.txt"
                                                                                file = open(fitfile,"a")
                                                                                str2write = str(vox_indx)
                                                                                params2write = ["amp","q0","wa","wMu","delta"]
                                                                                for param in params2write:
                                                                                    str2write = str2write + " " + str(fibrilParams[og_indx]["fit"][param])
                                                                                #str2write = str2write+" "
                                                                                #for param in params2write:
                                                                                    #str2write = str2write + " " + str(fibrilParams[indx]["simu"][param])
                                                                                str2write = str2write+" "+str(ichimax)+"\n"
                                                                                str2write = str2write+" "+str(fit_time)+"\n"
                                                                                #filestring = str("indx: " + str(indx) + ", q0: " + str(fibrilParams[indx]["fit"]["q0"])+"\n")
                                                                                file.write(str2write)
                                                                                file.close()
                                           
                                                                            last_solve = bp_solves
                                                                            bp_solves = bp_solves +1
                                                                            
                                                                            with open(output_path+"fibrilParams_final_filled_"+str(param_save)+".pkl", 'wb') as f:
                                                                                pickle.dump(fibrilParams, f)
                                                                         
                                                                            with open(output_path+"rotated_beampaths_final_filled_"+str(param_save)+".pkl", 'wb') as f:
                                                                                pickle.dump(rotated_beampaths, f)
                                                                                
                                                                            #np.save(output_path+"loop_idx.npy",[param_save,loopIndex])
                                                                
                                                else:
                                                    if combi_solve == True:
                                                        
                                                        og_indx = indx
                                                        
                                                        searchingNbrs=True
                                                        sweep_complete = False
                                                        
                                                        nbrs = FindNbrIndices(idx,idx,path_dict["voxels"],copy.deepcopy(path_dict["voxels"]),IchiCptsChiMax,\
                                                                                   IchiCptsIChiMax,overlap_params,threshold_detection)
                                                        
                                                        """
                                                        all_comb_indxs = [og_indx,nbr_indx]
                                                        all_comb_indxs = [this voxels fibfrilParams index,neighbouring voxels fibrilParams index]
                                                        indx = np.where(fibril_idxs == vox_indx)[0][0]
                                                        og_indx = indx
                                                        
                                                        """
                                                        
                                                            
                                                        all_comb_indxs,all_chiWindows,all_poss_chiWindows = [],[],[]
                                                        all_comb_ichis = []
                                                            
                                                        if nbrs:
                                                            j=0
                                                            solved_overlap,scanned_all_nbrs,solvable=False,False,False
                                                            while solved_overlap==False and scanned_all_nbrs==False: #FUTURE IMPROVEMENT: iterate over all nbrs till find the "best" one to do overlap fit
                                                                nbrindx=path_dict["voxels"][nbrs[j]]["fibril_param"]["indx"]
                                                                #print("pair fit: ", indx, nbrindx)
                                                                ichi_center, ichi_nbr = IchiCpts[idx], IchiCpts[nbrs[j]]
                                                                """
                                                                pass only the nonzero part of Ichi_uys
                                                                """
                                                                solvable, overlapwindow = checkSolvablePair(chirange,ichi_center,ichi_nbr,\
                                                                                                                 Ichi_us,overlap_params,overlap_thresh)
                                                                #print("out of checkSolvablePair")
                                                                if solvable == True: #plot, fit and mark pair as solved
                                                                
                                                                    nbr_indx = path_dict["voxels"][nbrs[j]]["fibril_param"]["indx"]
                                                                    nbr_s_indx = np.where(fibril_idxs == nbr_indx)[0][0]
                                                                    all_comb_indxs.append([og_indx,nbr_s_indx])
                                                                    all_chiWindows.append(np.linspace(overlapwindow[0],overlapwindow[-1],10))
                                                                    all_poss_chiWindows.append(np.linspace(overlapwindow[0],overlapwindow[-1],10))
                                                                    all_comb_ichis.append(ichi_center+ichi_nbr)
                                                                
                                                                    if sweep_complete==True:
                                                                        sweep_complete = False # since a voxel was solved
                                                                
                                                                    solved_overlap=True
                                                                j=j+1
                                                                if j>=len(nbrs):
                                                                    scanned_all_nbrs=True
                                                                    
                                                        if len(all_comb_indxs) == 0:
                                                            
                                                            nbrs3 = FindNbrIndices(idx,idx,path_dict["voxels"],copy.deepcopy(path_dict["voxels"]),IchiCptsChiMax,IchiCptsIChiMax,overlap_params_triple,threshold_detection)
                                                    
                                                            if len(nbrs3)>1:
                                                                j=0
                                                                pairs = list(map(list, combinations(nbrs3, 2)))
                                                                print(pairs)
                                                                solved_triple_overlap,scanned_all_triple_nbrs,solvable=False,False,False
                                                                while scanned_all_triple_nbrs==False:
                                                                    """
                                                                    search all 2-ples of nbrs, add centre voxel and test
                                                                    """
                                                                    for pair in pairs:
                                                                        nbr1 = path_dict["voxels"][pair[0]]["fibril_param"]["indx"]
                                                                        nbr2 = path_dict["voxels"][pair[1]]["fibril_param"]["indx"]                                                                                                                                               
                                                                        
                                                                        ichi_center, ichi_nbr1, ichi_nbr2 = IchiCpts[idx], IchiCpts[pair[0]], IchiCpts[pair[1]]
                                                                                                                         
                                                                        plt.plot(ichi_center)
                                                                        plt.plot(ichi_nbr1)
                                                                        plt.plot(ichi_nbr2)
    
                                                                        plt.plot(Ichi_us)
                                                                        plt.plot(ichi_center+ichi_nbr1+ichi_nbr2,linestyle = "dashed")
                                                                        plt.title(str(r)+","+str(i)+","+str(idx)+","+str(indx))
                                                                        plt.show()
                                                                        plt.close()
                                                                        
                                                                        solvable, best_window, best_ratios = solveTripleWrap(chirange,Ichi_us,ichi_center,ichi_nbr1,ichi_nbr2,chiRefWindow)
                                                                        
                                                                        print("triple overlap: ", nbr1, nbr2, solvable, overlapwindow)
                                                                        if solvable == True:
                                                                            """
                                                                            plot, fit and mark triple as solved
                                                                            """
                                                                            
                                                                            nbr_indx_1 = path_dict["voxels"][pair[0]]["fibril_param"]["indx"]
                                                                            nbr_s_indx_1 = np.where(fibril_idxs == nbr_indx_1)[0][0]
                                                                            
                                                                            nbr_indx_2 = path_dict["voxels"][pair[1]]["fibril_param"]["indx"]
                                                                            nbr_s_indx_2 = np.where(fibril_idxs == nbr_indx_2)[0][0]
                                                                            
                                                                            all_comb_indxs.append([og_indx,nbr_s_indx_1,nbr_s_indx_2])
                                                                            all_chiWindows.append(np.linspace(best_window[0],best_window[-1],10))
                                                                            all_poss_chiWindows.append(np.linspace(best_window[0],best_window[-1],10))
                                                                            all_comb_ichis.append(ichi_center+ichi_nbr1+ichi_nbr1)
                                                                            
                                                                            pass
                                                                          
                                                                    scanned_all_triple_nbrs = True
                                                                    print("exiting triple overlap")
                                                                    #sys.exit()
                                                                    pass
                                                        
                                                        
                                                        if len(all_comb_indxs)==0:
                                                            print("NO NEIGHBOUR SOLUTIONS FOUND")   
                                                        else:
                                                                        
                                                            if len(all_comb_indxs)>0:
                                                                
                                                                for comb_indxs,chiwindow,solved_windows,comb_ichi in zip(all_comb_indxs,all_chiWindows,all_poss_chiWindows,all_comb_ichis):
                                                                    
                                                                    #if len(comb_indxs)<3:                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                       
                                                                    comb_indx_path = [fibrilParams[k]["indx"] for k in comb_indxs]
                                                                    
                                                                    print("analysing real data for ",r,i,idx,indx,str(comb_indxs))
                                                                    
                                                                    iq,qRange,chi_slices,masked_q,frame,sig,succ_test = sampleIqOnChiWin(r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,
                                                                                                                                         recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,
                                                                                                                                         chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],comb_indx_path,sim_vals,wavelen,
                                                                                                                                         threshold_interference_multi,nslices=nslices,nq0=nq0,to_smooth = int_smooth,chi_range_sample = chi_range_sample)
                                                                                                                                                                                            
                                                                    
                                                                
                                                                    if succ_test != None and np.min([np.min(k[0:25]) for k in iq])>-3e8:
                                                                        
                                                                        if np.min([np.max(k) for k in iq])<min_abs_int:
                                                                            print("MULTI-FIBRE SOLVE FAILED MIN AMP THRESHOLD")
                                                                            
                                                                        else:
                                                                            
                                                                            high_snr = snr_test(qRange,iq,recon_cake_params,snr_min)
                                                                            
                                                                            if high_snr == False:
                                                                                print("MULTI-FIBRE SOLVE FAILED SNR THREHSOLD")
                                                                    
                                                                            else:
                                                                                
                                                                                print("combined solving ",r,i,idx,indx,str(comb_indxs))
                                                                                
                                                                                
                                                                                og_comb_indxs = np.copy(comb_indxs)
                                                                                                                                                                
                                                                                multi_solves.append([r,i,comb_indxs,np.max(comb_ichi),comb_ichi,Ichi1D_unsolved])
                                                                                
                                                                                if multi_solve == True and len(chiwindow)>0:
                                                                                
                                                                                    if graph_display == True:
                                                                                        for comb_idx in comb_indxs:
                                                                                            
                                                                                            xsol, ysol, zsol = x_locs[np.where(chi_indxs == comb_idx)[0][0]],y_locs[np.where(chi_indxs == comb_idx)[0][0]],z_locs[np.where(chi_indxs == comb_idx)[0][0]]
                                                                                            
                                                                                            linewidth=0.5
                                                                                            circle = patches.Circle([xsol,zsol],radius=0.6*(dxs/2.0),\
                                                                                                                          fc="yellow",alpha=1.0,ec="red",lw=linewidth,\
                                                                                                                              angle=r)
                                                                                            ax2.add_patch(circle)
                                                                                                                                                                       
                                                                                                                                                                                 
                                                                                    if np.min([len(k[k<0]) for k in iq])>20:
                                                                                        print("low intensity")                                                                                        
                                                                                                                                                                                                                                                                                   
                                                                                    params = Parameters()
                                                                                    
                                                                                    #voxAlpha,voxBeta = 
                        
                                                                                    labels = []
                                                                                    for k in range(0,len(comb_indxs)):
                                                                                        label = str(int(fibrilParams[comb_indxs[k]]["indx"]))
                                                                                        fib_voxel = path_dict["voxels"][np.where(pdict_vox_indxs==fibrilParams[comb_indxs[k]]["indx"])[0][0]]
                                                                                        labels.append(label)
                                                                                        #if int(label) not in np.asarray(solved_indxs) and fibrilParams[int(label)]["solved"]==False and solved_params[k] ==0:
                                                                                        #if int(label) not in np.asarray(solved_indxs) or solved_params[k] ==0:    
                                                                                        params.add('q0_' + label, value = sim_vals[0], min = q0_min, max = q0_max)  
                                                                                        #voxParams.add('q0_' + label, value = fitQ-0.002, min = 0.19, max = fitQ+0.02)  
                                                                                        params.add('wa_' + label, value = sim_vals[1], min = wa_min, max = wa_max)        
                                                                                        #if imbalance_test[1] == 1:
                                                                                            #params.add('wMu_' + label, value = 0.4, min = wMu_min, max = wMu_max)
                                                                                        #else:
                                                                                        params.add('wMu_' + label, value = sim_vals[2], min = wMu_min, max = wMu_max)
                                                                                        #params.add('alpha1_' + label, value = fibrilParams[comb_indxs[k]]["alpha"],min = fibrilParams[comb_indxs[k]]["alpha"]-ab_var, max = fibrilParams[comb_indxs[k]]["alpha"]+ab_var)       
                                                                                        #params.add('beta1_' + label, value = fibrilParams[comb_indxs[k]]["beta"],min = fibrilParams[comb_indxs[k]]["beta"]-ab_var, max = fibrilParams[comb_indxs[k]]["beta"]+ab_var)
                                                                                        params.add('alpha1_' + label, value = fibrilParams[comb_indxs[k]]["alpha"],vary = False)       
                                                                                        params.add('beta1_' + label, value = fibrilParams[comb_indxs[k]]["beta"],vary = False)                                                                                    
                                                                                        params.add('wavelength', value = wavelen,vary = False)
                                                                                        params.add('delta_'+ label, value = sim_vals[4],min = delta_min, max = delta_max)
                                                                                        
                                                                                        #params.add("c_"+label,value = 0,min = -1e8, max = 1e8)
                                                                                        params.add("amp_"+label,value = sim_vals[3],min = amp_min, max = amp_max)
                                                                                        params.add("weight_"+label,value = fib_voxel['fibril_param']["weight"],vary = False)
                                                                                    for solv_indx in slv_indxs:
                                                                                        solv_vox = path_dict["voxels"][np.where(pdict_vox_indxs==fibrilParams[solv_indx]["indx"])[0][0]]
                                                                                        solv_label = str(int(solv_vox["fibril_param"]["indx"]))
                                                                                        labels.append(solv_label)
                                                                                        solv_amp = solv_vox['fibril_param']["fit"]["amp"]
                                                                                        params.add('q0_' + solv_label, value = solv_vox['fibril_param']["fit"]["q0"],vary = False)      
                                                                                        params.add('wa_' + solv_label, value = solv_vox['fibril_param']["fit"]["wa"],vary = False)        
                                                                                        params.add('wMu_' + solv_label, value = solv_vox['fibril_param']["fit"]["wMu"],vary = False)
                                                                                        params.add('alpha1_' + solv_label, value = solv_vox['fibril_param']["alpha"],vary = False)       
                                                                                        params.add('beta1_' + solv_label, value = solv_vox['fibril_param']["beta"],vary = False)
                                                                                        #params.add('wavelength', value = wavelen,vary = False)
                                                                                        params.add('delta_'+ solv_label, value = solv_vox['fibril_param']["fit"]["delta"],vary = False)
                                                                                        #params.add('c_'+ solv_label, value = 0,min = -1e8, max = 1e8)
                                                                                        #params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],min=amp_min,max=solv_vox['fibril_param']["fit"]["amp"]*amp_max)
                                                                                        params.add('amp_'+ solv_label, value = solv_amp,min = solv_amp/10,max = solv_amp*10)
                                                                                        #params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],min = solv_vox['fibril_param']["fit"]["amp"]/10, max = solv_vox['fibril_param']["fit"]["amp"]*10)
                                                                                        params.add('weight_'+ solv_label, value = solv_vox['fibril_param']["weight"],vary = False)    
                                                                                    
                                                                                    profileData = [labels,iq,chimax,360-np.asarray(chi_slices),qRange,masked_q]
                                                                                    
                                                                                    t1 = time()
                                                                                    est_fibrilParams = minimize(optiIchiPlot_comb, params, args=([profileData]),max_nfev=50000)
                                                                                    t2 = time() 
                                                                                    print("computation time: ", t2-t1)                                                                                    
                                                                                    
                                                                                    ax3_data = paramTester_single(est_fibrilParams.params,profileData)
                                                                                                                                                                            
                                                                                    cols = ["blue","green","orange","purple","orange","yellow"]
                                                                                    
                                                                                    masked_qs = [np.where(masked_q[k]==0,qRange[k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                                    masked_real = [np.where(masked_q[k]==0,iq[k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                                    masked_fits = [np.where(masked_q[k]==0,ax3_data[1][k],np.nan) for k in np.arange(0,len(iq),1)]
                                                                                    
                                                                                    esti_amps = np.asarray([est_fibrilParams.params['amp_' + k].value for k in labels[0:len(comb_indxs)]])
                                                                                    esti_q0s = np.asarray([est_fibrilParams.params['q0_' + k].value for k in labels[0:len(comb_indxs)]])
                                                                                    esti_was = np.asarray([est_fibrilParams.params['wa_' + k].value for k in labels[0:len(comb_indxs)]])
                                                                                    esti_wMus = np.asarray([est_fibrilParams.params['wMu_' + k].value for k in labels[0:len(comb_indxs)]])
                                                                                    esti_deltas = np.asarray([est_fibrilParams.params['delta_' + k].value for k in labels[0:len(comb_indxs)]])
                                                                                    
                                                                                    esti_wps = esti_q0s*np.tan(esti_wMus)
                                                                                    
                                                                                    esti_diams = (2.857*(1/esti_wps))+10
                                                                                    
                                                                                    bad_fit = False
                                                                                    
                                                                                    sim_peaks = [ np.mean(ax3_data[1][k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
                                                                                    real_peaks = [ np.mean(iq[k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
                                                                                    
                                                                                    peak_ratios = [sim_peaks[k]/real_peaks[k] for k in np.arange(0,len(iq),1)]            
                                                                                    
                                                                                    if relax_fit == True:
                                                                                        
                                                                                        for k in range(nslices):
                                                                                            plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                            plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                        solve_time = time() - wake_time
                                                                                        print("solve time = ",solve_time)
                                                                                        
                                                                                        if np.max([np.max(k)-np.min(k) for k in ax3_data[1]])<min_abs_relax:
                                                                                            print("amp too low")
                                                                                            plt.title(("solving ",r,i,idx,og_indx,": AMP INACCURATE, ","solve time = ",str(solve_time)))
                                                                                            bad_fit = True
                                                                                            plt.savefig(output_path+"unaccepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            plt.show()
                                                                                            plt.close()
                                                                                        else:
                                                                                            if esti_was[0]>max_fit_wa or esti_was[0]<min_fit_wa:
                                                                                                print("wa inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": WA INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if esti_wMus[0]>max_fit_wMu or esti_wMus[0]<min_fit_wMu:
                                                                                                print("wMu inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": wMu INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if esti_q0s[0]>max_fit_q0 or esti_q0s[0]<min_fit_q0:
                                                                                                print("q0 inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": q0 INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if esti_diams[0]>diameter_max or esti_diams[0]<diameter_min:
                                                                                                print("diameter inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": diameter INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                        if bad_fit == True:
                                                                                            plt.savefig(output_path+"unaccepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            
                                                                                        else:
                                                                                            print("accuracy check passed - adding fit to dictionary")
                                                                                            plt.title(("solving ",r,i,idx,og_indx))
                                                                                        #plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                        plt.show()
                                                                                        plt.close()
                                                                                    
                                                                                    else:
                                                                                    
                                                                                        for k in range(nslices):
                                                                                            plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                            plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                        solve_time = time() - wake_time
                                                                                        print("solve time = ",solve_time)
                                                                                        
                                                                                        if np.max([np.max(k)-np.min(k) for k in ax3_data[1]])<min_abs_int: 
                                                                                            print("amp too low")
                                                                                            plt.title(("solving ",r,i,idx,og_indx,": AMP INACCURATE, ","solve time = ",str(solve_time)))
                                                                                            bad_fit = True
                                                                                            plt.savefig(output_path+"unaccepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            plt.show()
                                                                                            plt.close()
                                                                                        else:
                                                                                            if np.max(esti_was)>max_fit_wa or np.min(esti_was)<min_fit_wa:
                                                                                                print("wa inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": WA INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if np.max(esti_wMus)>max_fit_wMu or np.min(esti_wMus)<min_fit_wMu:
                                                                                                print("wMu inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": wMu INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if np.max(esti_q0s)>max_fit_q0 or np.min(esti_q0s)<min_fit_q0:
                                                                                                print("q0 inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": q0 INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                            if np.max(esti_diams)>diameter_max or np.min(esti_diams)<diameter_min:
                                                                                                print("diameter inaccurate")
                                                                                                plt.title(("solving ",r,i,idx,og_indx,": diameter INACCURATE, ","solve time = ",str(solve_time)))
                                                                                                bad_fit = True
                                                                                        if bad_fit == True:
                                                                                            plt.savefig(output_path+"unaccepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            
                                                                                        else:
                                                                                            print("accuracy check passed - adding fit to dictionary")
                                                                                            plt.title(("solving ",r,i,idx,og_indx))
                                                                                        #plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                        plt.show()
                                                                                        plt.close()
                                                                                    
                                                                                                
                                                                                    if bad_fit == False:
                                                                                        
                                                                                        std_errors,xs,ys,new_xs,new_ys = [],[],[],[],[]
                                                                                        
                                                                                        max_q_cond = (esti_q0s<max_fit_q0)
                                                                                        min_q_cond = (esti_q0s>min_fit_q0)
                                                                                        max_wa_cond = (esti_was<max_fit_wa)
                                                                                        min_wa_cond = (esti_was>min_fit_wa)
                                                                                        max_wMu_cond = (esti_wMus<max_fit_wMu)
                                                                                        min_wMu_cond = (esti_wMus>min_fit_wMu)
                                                                                        max_diam_cond = (esti_diams<diameter_max)
                                                                                        min_diam_cond = (esti_diams>diameter_min)

                                                                                        global_condition = max_q_cond & min_q_cond & max_wa_cond & min_wa_cond & max_wMu_cond & min_wMu_cond & max_diam_cond & min_diam_cond
                                                                                        
                                                                                        fitted_labels = [labels[0:len(comb_indxs)][k] for k in np.arange(0,len(global_condition),1) if 
                                                                                                         global_condition[k]==True]                                                                                                                                                                                

                                                                                        just_fitted_iq = np.asarray([ax3_data[2][k] for k in np.arange(0,len(global_condition),1) if 
                                                                                                                global_condition[k]==True])
                                                                                        
                                                                                        fitted_iq_tot = np.sum(just_fitted_iq,0)
                                                                                        
                                                                                        
                                                                                        fitted_iq_tot_masked = [np.where(masked_q[k]==0,fitted_iq_tot[k],np.nan) for k in np.arange(0,len(iq),1)]

                                                                                        for chi_test in range(0,len(chi_slices)):

                                                                                            chi = 360-np.asarray(chi_slices[chi_test])
                                                                                                                                                                                       
                                                                                            
                                                                                            fitted_iq = np.sum(np.asarray([ax3_data[2][k][chi_test] for k in np.arange(1,len(ax3_data[2]),1)]),0) 
                                                                                            
                                                                                            if len(labels) == 1:
                                                                                                fitted_iq = np.zeros_like(iq[chi_test])
                                                                                                                                                           
                                                                                            nan_mask = ~np.isnan(masked_fits[chi_test])
                                                                                            
                                                                                            i_masked = masked_fits[chi_test][nan_mask]
                                                                                            q_masked = qRange[chi_test][nan_mask]
                                                                                            
                                                                                            if len(q_masked)==0:
                                                                                                
                                                                                                std_errors.append(100)
                                                                                                xs.append(qRange[chi_test])
                                                                                                ys.append(masked_fits[chi_test])
                                                                                                new_ys.append(masked_fits[chi_test])
                                                                                            
                                                                                            else:
                                                                                                peak_mod = fit_gauss(q_masked,i_masked)
                                                                                                
                                                                                                q_lim = [peak_mod[1]["center"].value - (peak_mod[1]["sigma"].value*3),
                                                                                                         peak_mod[1]["center"].value + (peak_mod[1]["sigma"].value*3)]
                                                                                                
                                                                                                qRange_lim = [find_nearest(qRange[chi_test],q_lim[0])[1],
                                                                                                              find_nearest(qRange[chi_test],q_lim[1])[1]]
                                                                                                
                                                                                                if np.max(qRange_lim) == 0:
                                                                                                    
                                                                                                    std_errors.append(100)
                                                                                                    xs.append(qRange[chi_test])
                                                                                                    ys.append(masked_fits[chi_test])
                                                                                                    new_ys.append(masked_fits[chi_test])
                                                                                                    
                                                                                                else:
                                                                                                
                                                                                                    if qRange_lim[1] - qRange_lim[0]<2:
                                                                                                       qRange_lim[0] = qRange_lim[0]-2 
                                                                                                       qRange_lim[1] = qRange_lim[1]+3
                                                                                                    
                                                                                                    fitted_iq = fitted_iq[qRange_lim[0]:qRange_lim[1]]
                                                                                                    
                                                                                                    x = qRange[chi_test][qRange_lim[0]:qRange_lim[1]]
                                                                                                    
                                                                                                    y = masked_real[chi_test][qRange_lim[0]:qRange_lim[1]]            
                                                                                                    
                                                                                                    #y2 = masked_fits[chi_test][qRange_lim[0]:qRange_lim[1]]
                                                                                                    y2 = fitted_iq_tot_masked[chi_test][qRange_lim[0]:qRange_lim[1]]
                                                                                                    
                                                                                                    while len(y)<10:
                                                                                                        y = np.asarray(padder(y))
                                                                                                        y2 = np.asarray(padder(y2))
                                                                                                        
                                                                                                    x = np.linspace(x[0],x[-1],len(y))
                                                                                                    
                                                                                                    max_calc = np.max([np.max(k) for k in [y,y2]])
                                                                                                    
                                                                                                    y2 = (y2/max_calc)*100
                                                                                                    y = (y/max_calc)*100
                                                                                                    
                                                                                                    mse = ((y - y2)**2).mean(axis=0)
                                                                                                    syx = np.sqrt(mse)
                                                                                                    if "nan" in str(syx):
                                                                                                        std_errors.append(100)
                                                                                                        new_ys.append(y)
                                                                                                    else:    
                                                                                                        std_errors.append(int(syx))
                                                                                                        new_ys.append(y2)
                                                                                                    
                                                                                                    xs.append(x)
                                                                                                    ys.append(y)
                                                                                            
                                                                                        if np.min(std_errors)<=min_error:
                                                                                            for k in range(0,len(std_errors)):
                                                                                                plt.plot(xs[k],ys[k],'.', color = cols[k], label='chi range '+str(k)+' observations')
                                                                                                plt.plot(xs[k], new_ys[k], '-', color = cols[k], label='chi range '+str(k)+' fit') 
                                                                                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                            plt.title(("solving ",r,i,idx,og_indx,", solve time = ",str(solve_time)))                                  
                                                                                            plt.savefig(output_path+"accepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
                                                                                            plt.show()
                                                                                            plt.close()   
                                                                                            
                                                                                            for k in range(nslices):
                                                                                                plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                                plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                            print("accuracy check passed - adding fit to dictionary")
                                                                                            solve_time = time() - wake_time
                                                                                            print("solve time = ",solve_time)
                                                                                            plt.title(("solving ",r,i,idx,og_indx,", solve time = ",str(solve_time)))
                                                                                            plt.savefig(output_path+"accepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            plt.show()
                                                                                            plt.close()
                                                                                            
                                                                                        else:
                                                                                            
                                                                                            for k in range(nslices):
                                                                                                plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                                                                                                plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                                                                                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                                                                            print("fit not accepted: std eror>30")
                                                                                            plt.title(("solving ",r,i,idx,og_indx,": high std error"))
                                                                                            title = ("solving ",r,i,idx,og_indx,": high std error")
                                                                                            plt.savefig(output_path+"unaccepted_multi_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                                                                            plt.show()
                                                                                            plt.close()
                                                                                    
                                                                                        if np.min(std_errors)<=min_error:
                                                                                        
                                                                                            add_test = []
                                                                                            #ax1, Ichi1D_unsolved, fibrilParams = addSolvedProfile_comb(ax1,Ichi1D_unsolved,fibrilParams,[og_indx,nbr[1]],comb_ichi,chiwindow,chirange) 
                                                                                                                                                                                                                                                                                                               
                                                                                            for idx, voxel in enumerate(fibrilParams):
                                                                                                indx = str(int(voxel["indx"]))
                                                                                                #if indx in labels: 
                                                                                                if indx in fitted_labels:
                                                                                                    add_test.append([idx,indx])
                                                                                                    voxel["fit"]["q0"] = est_fibrilParams.params['q0_' + str(int(indx))].value
                                                                                                    voxel["fit"]["amp"] = est_fibrilParams.params['amp_' + str(int(indx))].value
                                                                                                    voxel["fit"]["wa"] = est_fibrilParams.params['wa_' + str(int(indx))].value
                                                                                                    voxel["fit"]["wMu"] = est_fibrilParams.params['wMu_' + str(int(indx))].value
                                                                                                    voxel["fit"]["delta"] = est_fibrilParams.params['delta_' + str(int(indx))].value
                                                                                                    #voxel["fit"]["c"] = est_fibrilParams.params['c_' + str(int(indx))].value
                                                                                                    voxel["alpha"] = est_fibrilParams.params['alpha1_' + str(int(indx))].value
                                                                                                    voxel["beta"] = est_fibrilParams.params['beta1_' + str(int(indx))].value
                                                                                                    voxel["solved"]=True                                                                                                                                                                                                                
                                                                                                        
                                                                                            for bp in rotated_beampaths:                                                        
                                                                                                for j, pdict in enumerate(bp):
                                                                                                    if pdict != None:
                                                                                                        if pdict["voxels"] != None and allVoxelsSolved==False:                                                                    
                                                                                                            if j in ichiExp[r] and type(ichiExp[r][j]["kapton"]) == bool:                                                                             
                                                                                                                for idx, voxel in enumerate(pdict["voxels"]):
                                                                                                                    indx = str(int(voxel['fibril_param']["indx"]))
                                                                                                                    #if indx in labels: 
                                                                                                                    if indx in fitted_labels:
                                                                                                                        add_test.append([idx,indx])
                                                                                                                        voxel['fibril_param']['solved'] = True
                                                                                                                        voxel['fibril_param']['fit']['q0'] = est_fibrilParams.params['q0_' + str(indx)].value
                                                                                                                        voxel['fibril_param']['fit']['wa'] = est_fibrilParams.params['wa_' + str(indx)].value
                                                                                                                        voxel['fibril_param']['fit']['wMu'] = est_fibrilParams.params['wMu_' + str(indx)].value
                                                                                                                        voxel['fibril_param']['fit']['delta'] = est_fibrilParams.params['delta_' + str(indx)].value
                                                                                                                        #voxel['fibril_param']['fit']['c'] = est_fibrilParams.params['c_' + str(indx)].value
                                                                                                                        voxel['fibril_param']['fit']['amp'] = est_fibrilParams.params['amp_' + str(indx)].value
                                                                                                                        voxel["fibril_param"]["alpha"] = est_fibrilParams.params['alpha1_' + str(indx)].value
                                                                                                                        voxel["fibril_param"]["beta"] = est_fibrilParams.params['beta1_' + str(indx)].value
                                                                                                                        voxel["solved"]=True
                                                                                                                        
                                                                                                                        now = datetime.now()
                                                                                                                        year,month,day = now.strftime("%Y"),now.strftime("%m"),now.strftime("%d")
                                                                                                                        
                                                                                                                        time_finsihed = time()
                                                                                                                        
                                                                                                                        fit_time = time_finsihed - recon_start_time
                                                                                                                        
                                                                                                                        fitfile = output_path+day+"-"+month+"-"+year+"_"+"fitfile.txt"
                                                                                                                        file = open(fitfile,"a")
                                                                                                                        str2write = str(voxel['fibril_param']["indx"])
                                                                                                                        params2write = ["amp","q0","wa","wMu","delta"]
                                                                                                                        #str2write = str2write + " " + str(voxel['fibril_param']["indx"])
                                                                                                                        for param in params2write:
                                                                                                                            str2write = str2write + " " + str(voxel['fibril_param']["fit"][param])
                                                                                                                        #str2write = str2write+" "
                                                                                                                        #for param in params2write:
                                                                                                                            #str2write = str2write + " " + str(fibrilParams[indx]["simu"][param])
                                                                                                                        str2write = str2write+" "+str(ichimax)+"\n"
                                                                                                                        str2write = str2write+" "+str(fit_time)+"\n"
                                                                                                                        #filestring = str("indx: " + str(indx) + ", q0: " + str(fibrilParams[indx]["fit"]["q0"])+"\n")
                                                                                                                        file.write(str2write)
                                                                                                                        file.close()
                                                                                                                        # sys.exit()
                                                                                                                        # fits the data (full data or expdata) and gets fibril parameters for "indx"
                                                                                                                        # adds these to the fibrilParams[indx]
                        
                                                                                        with open(output_path+"fibrilParams_final_filled_"+str(param_save)+".pkl", 'wb') as f:
                                                                                            pickle.dump(fibrilParams, f)
                                                                                     
                                                                                        with open(output_path+"rotated_beampaths_final_filled_"+str(param_save)+".pkl", 'wb') as f:
                                                                                            pickle.dump(rotated_beampaths, f)
                                                                                            
            np.save(output_path+"loop_idx.npy",[param_save,loopIndex])
            new_fit_sum = len([k for k in fibrilParams if k["fit"] !={}])
            
            if new_fit_sum == fitted_sum or loopIndex == (loop_range-1):
                kill_loop = True
                
                                                                                                                                                        
    return fibrilParams,rotated_beampaths,single_solves,multi_solves


#r,i,fibrilParams,ichiExp,rotated_beampaths,idx,path_dict,voxel,indx,ichi,Ichi1D_unsolved,chirange,chiRefWindow,threshold_interference,threshold_detection,recon_cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,q_fitIqr,sim_vals,q0_m,fibril_idxs,output_path,binning,chi_range_sample,nslices = max_bps[k][0],max_bps[k][1],fibrilParams,ichiExp,rotated_beampaths,solved_idxs[k],path_dict,voxel,indx,ichi,Ichi1D_unsolved,chirange,chiRefWindow,threshold_interference,threshold_detection,recon_cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,q_fitIqr,sim_vals,q0_m,fibril_idxs,output_path,binning,chi_range_sample,nslices

def singleSolve(r,i,fibrilParams,ichiExp,rotated_beampaths,idx,path_dict,voxel,indx,ichi,Ichi1D_unsolved,chirange,
                chiRefWindow,threshold_interference,threshold_detection,recon_cake_params,recon_mask_pts,recon_mask_chis,a1,Mask,
                slice_saxs_file,q_fitIqr,sim_vals,q0_m,fibril_idxs,output_path,binning,fit_params,chi_range_sample,nslices):
        
    recon_start_time = np.load(output_path+"recon_start_time.npy")[0]
    
    q0_min,q0_max = fit_params["q0_min"],fit_params["q0_max"] # min/max for q0 parameter estimation
    wa_min,wa_max = fit_params["wa_min"],fit_params["wa_max"]  # min/max for wa parameter estimation
    wMu_min,wMu_max = fit_params["wMu_min"],fit_params["wMu_max"] # min/max for wMu parameter estimation
    
    ab_var = fit_params["ab_var"] # variation permitted in estimated alpha and beta values (in degrees) for fitting
    delta_min,delta_max = fit_params["delta_min"],fit_params["delta_max"] # min/max for delta parameter estimation
    amp_min,amp_max = fit_params["amp_min"],fit_params["amp_max"] # min/max for amplitude estimation - mean is absolute value; max is multiplier of SVD amplitude estimate

    max_amp_rat = fit_params["max_amp_rat"] # threshold value for ratio between fitted peak intensity and measured intensity for discriminating fits that are too high/too low
    min_abs_int = fit_params["min_abs_int"]  # minimum intensity value for fits
    min_error = fit_params["min_error"] # minimum percentage standard error
    min_abs_relax = fit_params["min_abs_relax"] # minimum intensity value for relaxed fits 
    min_fit_wa = fit_params["min_fit_wa"] # threshold minimum fitted wa value for fits
    max_fit_wa = fit_params["max_fit_wa"] # threshold maximum fitted wa value for fits
    min_fit_wMu = fit_params["min_fit_wMu"] # threshold minimum fitted wMu value for fits
    max_fit_wMu = fit_params["max_fit_wMu"] # threshold maximum fitted wMu value for fits
    
    min_fit_q0 = fit_params["min_fit_q0"] # threshold minimum fitted q0 value for fits
    max_fit_q0 = fit_params["max_fit_q0"] # threshold maximum fitted q0 value for fits
            
    diameter_min = fit_params["diam_min"] # threshold maximum fitted wa value for fits
    diameter_max = fit_params["diam_max"] # threshold maximum fitted wMu value for fits
    
    snr_min = fit_params["single_min_snr"]
    
    single_fit_rat = fit_params["single_fit_rat"]
    single_fit_error = fit_params["single_fit_error"]
        
    vox_result = copy.deepcopy(voxel)
    ichis = []
    ichimax = np.max(ichi)
    chimax  =  chirange[np.argmax(ichi)]
    
    rot_beampath = np.asarray(rotated_beampaths[r])
    
    og_bps = np.arange(0,len(rot_beampath)*binning,binning)
            
    
    """
    for integration testing
    """
    if recon_cake_params["test"]=="full":
        Nq0 = 50
        int_smooth = False
    else:
        if recon_cake_params["test"]=="short":
            Nq0 = 25
            int_smooth = False
        if recon_cake_params["test"]=="full_smooth":
            Nq0 = 50
            int_smooth = True
        if recon_cake_params["test"]=="short_smooth":
            Nq0 = 25
            int_smooth = True
    
    
    chi_indxs,slv_indxs = [],[]
    for idx, test_voxel in enumerate(path_dict["voxels"]):
        vox_indx = test_voxel["fibril_param"]["indx"]
        test_indx = np.where(fibril_idxs == vox_indx)[0][0]
        fibrilParams[indx]["intersected"]=True
        #ichi,chir,ichimax,chimax, bareichi = get1DSAXSchiprofile(voxel,chi1=chi1,chi2=chi2,nchi=nchi,
                                                       #q1=q0m_low,q2=q0m_high,nq=nq0,wavelen=wavelen,simu_vals = sim_vals)[0]
        
        if fibrilParams[test_indx]["solved"]==False:

            chi_indxs.append(test_indx)
                                                   
        else:
            #print(idx)
            slv_indxs.append(test_indx) 
                                        
    slv_indxs = np.asarray(slv_indxs)
    
    #print("analysing scan ",str(loopindx)," rotation ",str(r)," beampath ",str(i), " voxels: ",len(path_dict["voxels"]))
    pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in path_dict["voxels"]])
    if len(pdict_vox_indxs)>1:
        vox_repeats = [k for k in np.arange(1,len(pdict_vox_indxs),1) if pdict_vox_indxs[k]==pdict_vox_indxs[k-1]]
        vox_repeats = [np.unique(pdict_vox_indxs)[k] for k in np.arange(0,len(np.unique(pdict_vox_indxs)),1) 
                       if np.unique(pdict_vox_indxs,return_counts = True)[1][k]>1]
        vox_singles = [np.unique(pdict_vox_indxs)[k] for k in np.arange(0,len(np.unique(pdict_vox_indxs)),1) 
                       if np.unique(pdict_vox_indxs,return_counts = True)[1][k]==1]
        if len(vox_repeats)>0:
            
            path_dict_singles = [path_dict["voxels"][np.where(pdict_vox_indxs ==k)[0][0]] for k in vox_singles]
            path_dict_repeats = [path_dict["voxels"][np.where(pdict_vox_indxs ==k)[0][0]] for k in vox_repeats]
            path_dict["voxels"] = path_dict_singles+path_dict_repeats
            #path_dict["voxels"] = [path_dict["voxels"][k] for k in np.arange(0,len(path_dict["voxels"]),1) if k not in vox_repeats]
            pdict_vox_indxs = np.asarray([k["fibril_param"]["indx"] for k in path_dict["voxels"]])
    
    vox_indx = voxel["fibril_param"]["indx"]
    
    wavelen = recon_cake_params["wavelen"]
            
    solvable = False
    chiwindow, solvable,opti_chiwindow,solved_windows = findChiWindow(ichi,Ichi1D_unsolved,chirange,recon_mask_pts,chiwin=chiRefWindow,threshold_interference=threshold_interference,threshold_detection=threshold_detection)
    
    if np.sum(np.abs(ichi - Ichi1D_unsolved)) == 0:
        opti_chiwindow = np.linspace(chimax-5,chimax+5,10)
        solved_windows = solved_windows+[opti_chiwindow.tolist()]
        
    std_errors = []
            
    if solvable == True and len(opti_chiwindow)>1:
                                                                                                                
        iq,qRange,chi_slices,masked_q,frame,sig,succ_test = sampleIqOnChiWin(r,np.arange(og_bps[i],og_bps[i+1],1).astype(int),recon_cake_params,
                                                                             recon_mask_pts,recon_mask_chis,a1,Mask,slice_saxs_file,
                                                                             opti_chiwindow,q_fitIqr,solved_windows,path_dict["voxels"],voxel,sim_vals,wavelen,
                                                                             threshold_interference,nslices=nslices,nq0=Nq0,to_smooth = int_smooth,chi_range_sample = chi_range_sample)
                                                        
        comb_indxs = np.append(slv_indxs,indx)
    
        
        if succ_test != None and np.min([np.min(k[0:int(len(k)/2)]) for k in iq])>-3e8:
            
            amp_test = single_amp_test(iq,chi_slices,qRange,path_dict["voxels"],comb_indxs,10000,1500000,600000,wavelen,sim_vals) 
            
            og_indx = indx
                                
            if np.max([np.max(k) for k in iq])<1e8:
                print("low intensity")
            #else:
                #lhs_data = lhs_sample(iq,qRange,right_thresh=0.6)                                                                                                                                                           
                        
            params = Parameters()
            vox_alpha = voxel["fibril_param"]["alpha"]
            vox_beta = voxel["fibril_param"]["beta"]
                
            labels = []
            
            labels.append(str(int(vox_indx)))
            params.add('q0_' + labels[0], value = sim_vals[0], min = q0_min, max = q0_max)      
            params.add('wa_' + labels[0], value = sim_vals[1], min = wa_min, max = wa_max)        
            params.add('wMu_' + labels[0], value = sim_vals[2], min = wMu_min, max = wMu_max)
            #params.add('alpha1_' + labels[0], value = vox_alpha,min = vox_alpha-ab_var, max = vox_alpha+ab_var)       
            #params.add('beta1_' + labels[0], value = vox_beta,min = vox_beta-ab_var, max = vox_beta+ab_var)
            params.add('alpha1_' + labels[0], value = vox_alpha,vary = False)       
            params.add('beta1_' + labels[0], value = vox_beta,vary = False)
            params.add('wavelength', value = wavelen,vary = False)
            params.add('delta_'+ labels[0], value = sim_vals[4],min = delta_min, max = delta_max)
            
            #params.add("c_"+labels[0],value = 0,min = -1e8, max = 1e8)
            params.add("amp_"+labels[0],value = voxel["fibril_param"]["initial_amp_est"],min = amp_min, max = amp_max)
            params.add("weight_"+labels[0],value = voxel["fibril_param"]["weight"],vary = False)
            for solv_indx in slv_indxs:
                solv_vox = path_dict["voxels"][np.where(pdict_vox_indxs==fibrilParams[solv_indx]["indx"])[0][0]]
                if solv_vox["fibril_param"]["fit"] != {}:
                    solv_label = str(int(solv_vox["fibril_param"]["indx"]))
                    labels.append(solv_label)
                    solv_amp = solv_vox['fibril_param']["fit"]["amp"]
                    params.add('q0_' + solv_label, value = solv_vox['fibril_param']["fit"]["q0"],vary = False)      
                    params.add('wa_' + solv_label, value = solv_vox['fibril_param']["fit"]["wa"],vary = False)        
                    params.add('wMu_' + solv_label, value = solv_vox['fibril_param']["fit"]["wMu"],vary = False)
                    params.add('alpha1_' + solv_label, value = solv_vox['fibril_param']["alpha"],vary = False)       
                    params.add('beta1_' + solv_label, value = solv_vox['fibril_param']["beta"],vary = False)
                    #params.add('wavelength', value = wavelen,vary = False)
                    params.add('delta_'+ solv_label, value = solv_vox['fibril_param']["fit"]["delta"],vary = False)
                    #params.add('c_'+ solv_label, value = solv_vox['fibril_param']["fit"]["c"],vary = False)
                    #params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],vary = False)
                    params.add('amp_'+ solv_label, value = solv_vox['fibril_param']["fit"]["amp"],min = solv_amp/10,max = solv_amp*10)
                    params.add('weight_'+ solv_label, value = solv_vox['fibril_param']["weight"],vary = False)
                          
            profileData = [labels,iq,chimax,360-np.asarray(chi_slices),qRange,masked_q]
        
            est_fibrilParams = minimize(optiIchiPlot_single, params, method = "Nelder", args=([profileData]))
            
            """
            test_params = Parameters()
            test_params.add('q0_' + labels[0], value = est_fibrilParams.params['q0_' + labels[0]].value, min = q0_m - 0.01, max = q0_m + 0.01)      
            test_params.add('wa_' + labels[0], value = 0.02, min = 0.001, max = 0.05)        
            test_params.add('wMu_' + labels[0], value = est_fibrilParams.params['wMu_' + labels[0]].value, min = 0.09, max = 0.6)
            test_params.add('alpha1_' + labels[0], value = vox_alpha,min = vox_alpha-5, max = vox_alpha+5)       
            test_params.add('beta1_' + labels[0], value = vox_beta,min = vox_beta-5, max = vox_beta+5)
            test_params.add('wavelength', value = wavelen,vary = False)
            test_params.add('delta_'+ labels[0], value = est_fibrilParams.params['delta_' + labels[0]].value,min = 0.001, max = 0.99)
            
            #params.add("c_"+labels[0],value = 0,min = -1e8, max = 1e8)
            test_params.add("amp_"+labels[0],value = est_fibrilParams.params['amp_' + labels[0]].value,min = 0.001, max = voxel["fibril_param"]["initial_amp_est"]*100)
            test_params.add("weight_"+labels[0],value = voxel["fibril_param"]["weight"],vary = False)
            """
            
            ax3_data = paramTester_single(est_fibrilParams.params,profileData)
            cols = ["blue","green","orange","purple","orange","yellow"]
            
            masked_qs = [np.where(masked_q[k]==0,qRange[k],np.nan) for k in np.arange(0,len(iq),1)]
            masked_real = [np.where(masked_q[k]==0,iq[k],np.nan) for k in np.arange(0,len(iq),1)]
            masked_fits = [np.where(masked_q[k]==0,ax3_data[1][k],np.nan) for k in np.arange(0,len(iq),1)]
            
            esti_amp = est_fibrilParams.params['amp_' + labels[0]].value 
            esti_q0 = est_fibrilParams.params['q0_' + labels[0]].value 
            esti_wa = est_fibrilParams.params['wa_' + labels[0]].value 
            esti_wMu = est_fibrilParams.params['wMu_' + labels[0]].value 
            esti_delta = est_fibrilParams.params['delta_' + labels[0]].value 
            
            esti_wp = esti_q0*np.tan(esti_wMu)
            
            esti_diam = (2.857*(1/esti_wp))+10
            
            bad_fit = False
            
            sim_peaks = [ np.mean(ax3_data[1][k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
            real_peaks = [ np.mean(iq[k][np.argmax(ax3_data[1][k])-1:np.argmax(ax3_data[1][k])+1]) for k in np.arange(0,len(iq),1)]
            
            peak_ratios = [sim_peaks[k]/real_peaks[k] for k in np.arange(0,len(iq),1)]                       
            
            for k in range(nslices):
                plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            #elif np.max([np.max(k) for k in ax3_data[1]])<np.max([np.max(k) for k in iq])/3:
            if np.max([np.max(k)-np.min(k) for k in ax3_data[1]])<min_abs_int: 
                print("amp too low")
                plt.title(("solving ",r,i,idx,og_indx,": AMP INACCURATE"))
                title = ("solving ",r,i,idx,og_indx,": AMP INACCURATE")
                bad_fit = True
                plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                plt.show()
                plt.close()
            else:
                if esti_wa>max_fit_wa or esti_wa<min_fit_wa:
                    print("wa inaccurate")
                    plt.title(("solving ",r,i,idx,og_indx,": WA INACCURATE, "))
                    bad_fit = True
                if esti_wMu>max_fit_wMu or esti_wMu<min_fit_wMu:
                    print("wMu inaccurate")
                    plt.title(("solving ",r,i,idx,og_indx,": wMu INACCURATE, "))
                    bad_fit = True
                if esti_q0>max_fit_q0 or esti_q0<min_fit_q0:
                    print("q0 inaccurate")
                    plt.title(("solving ",r,i,idx,og_indx,": q0 INACCURATE, "))
                    bad_fit = True
                if esti_diam>diameter_max or esti_diam<diameter_min:
                    print("diameter inaccurate")
                    plt.title(("solving ",r,i,idx,og_indx,": diameter INACCURATE, "))
                    bad_fit = True
            if bad_fit == True:
                plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
            else:
                title = ("solving ",r,i,idx,og_indx)
                plt.title((title))
            
            plt.show()
            plt.close()
            
                        
            if bad_fit == False:
                                                   
                std_errors,xs,ys,new_xs,new_ys = [],[],[],[],[]

                for chi_test in range(0,len(chi_slices)):

                    chi = 360-np.asarray(chi_slices[chi_test])                                    
                    
                    fitted_iq = np.sum(np.asarray([ax3_data[2][k][chi_test] for k in np.arange(1,len(ax3_data[2]),1)]),0) 
                    
                    if len(labels) == 1:
                        fitted_iq = np.zeros_like(iq[chi_test])
                                                                                   
                    nan_mask = ~np.isnan(masked_fits[chi_test])
                    
                    i_masked = masked_fits[chi_test][nan_mask]
                    q_masked = qRange[chi_test][nan_mask]
                    
                    if len(q_masked)==0:
                        
                        std_errors.append(100)
                        xs.append(qRange[chi_test])
                        ys.append(masked_fits[chi_test])
                        new_ys.append(masked_fits[chi_test])
                    
                    else:
                        peak_mod = fit_gauss(q_masked,i_masked)
                        
                        q_lim = [peak_mod[1]["center"].value - (peak_mod[1]["sigma"].value*3),
                                 peak_mod[1]["center"].value + (peak_mod[1]["sigma"].value*3)]
                        
                        qRange_lim = [find_nearest(qRange[chi_test],q_lim[0])[1],
                                      find_nearest(qRange[chi_test],q_lim[1])[1]]
                        
                        if np.max(qRange_lim) == 0:
                            
                            std_errors.append(100)
                            xs.append(qRange[chi_test])
                            ys.append(masked_fits[chi_test])
                            new_ys.append(masked_fits[chi_test])
                            
                        else:
                        
                            if qRange_lim[1] - qRange_lim[0]<2:
                               qRange_lim[0] = qRange_lim[0]-2 
                               qRange_lim[1] = qRange_lim[1]+3
                            
                            fitted_iq = fitted_iq[qRange_lim[0]:qRange_lim[1]]
                            
                            x = qRange[chi_test][qRange_lim[0]:qRange_lim[1]]
                            
                            y = masked_real[chi_test][qRange_lim[0]:qRange_lim[1]]            
                            
                            y2 = masked_fits[chi_test][qRange_lim[0]:qRange_lim[1]]
                            
                            while len(y)<10:
                                y = np.asarray(padder(y))
                                y2 = np.asarray(padder(y2))
                                
                            x = np.linspace(x[0],x[-1],len(y))
                            
                            max_calc = np.max([np.max(k) for k in [y,y2]])
                            
                            y2 = (y2/max_calc)*100
                            y = (y/max_calc)*100
                            
                            mse = ((y - y2)**2).mean(axis=0)
                            syx = np.sqrt(mse)
                            if "nan" in str(syx):
                                std_errors.append(100)
                                new_ys.append(y)
                            else:    
                                std_errors.append(int(syx))
                                new_ys.append(y2)
                            
                            xs.append(x)
                            ys.append(y)
                    
                if np.min(std_errors)<=single_fit_error:
                    for k in range(0,len(std_errors)):
                        plt.plot(xs[k],ys[k],'.', color = cols[k], label='chi range '+str(k)+' observations')
                        plt.plot(xs[k], new_ys[k], '-', color = cols[k], label='chi range '+str(k)+' fit') 
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.title(title)                                    
                    
                    if "test" not in recon_cake_params or recon_cake_params["test"]=="full_smooth":
                        plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
                        plt.savefig(output_path+"full smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")

                    else:
                        if "delta_test" not in recon_cake_params or recon_cake_params["delta_test"] =={}:
                            if recon_cake_params["test"]=="full":
                                plt.savefig(output_path+"full test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="short":
                                plt.savefig(output_path+"short test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="full_smooth":
                                plt.savefig(output_path+"full smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="short_smooth":
                                plt.savefig(output_path+"short smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")

                        else:
                            if recon_cake_params["test"]=="full":
                                plt.savefig(output_path+"full test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="short":
                                plt.savefig(output_path+"short test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="full_smooth":
                                plt.savefig(output_path+"full smooth test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")
    
                            if recon_cake_params["test"]=="short_smooth":
                                plt.savefig(output_path+"short smooth test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+"_std_error.png")


                    plt.show()
                    plt.close()   
                    
                    for k in range(nslices):
                        plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                        plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    print("accuracy check passed - adding fit to dictionary")
                    plt.title(("solving ",r,i,idx,og_indx))
                    if "test" not in recon_cake_params or recon_cake_params["test"]=="full_smooth":
                        plt.savefig(output_path+"accepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                        plt.savefig(output_path+"full smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")

                    else:
                        if "delta_test" not in recon_cake_params or recon_cake_params["delta_test"] =={}:
                            if recon_cake_params["test"]=="full":
                                plt.savefig(output_path+"full test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
    
                            if recon_cake_params["test"]=="short":
                                plt.savefig(output_path+"short test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
    
                            if recon_cake_params["test"]=="full_smooth":
                                plt.savefig(output_path+"full smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
    
                            if recon_cake_params["test"]=="short_smooth":
                                plt.savefig(output_path+"short smooth test fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                        else:
                            if recon_cake_params["test"]=="full":
                                plt.savefig(output_path+"full test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                                
                            if recon_cake_params["test"]=="short":
                                plt.savefig(output_path+"short test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
    
                            if recon_cake_params["test"]=="full_smooth":
                                plt.savefig(output_path+"full smooth test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
    
                            if recon_cake_params["test"]=="short_smooth":
                                plt.savefig(output_path+"short smooth test fits_d/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")

                    plt.show()
                    plt.close()
                    
                    
                    
                                                                                                                                      
                    vox_result["fibril_param"]["fit"]["q0"] = est_fibrilParams.params['q0_' + labels[0]].value
                    vox_result["fibril_param"]["fit"]["amp"] = est_fibrilParams.params['amp_' + labels[0]].value
                    vox_result["fibril_param"]["fit"]["wa"] = est_fibrilParams.params['wa_' + labels[0]].value
                    vox_result["fibril_param"]["fit"]["wMu"] = est_fibrilParams.params['wMu_' + labels[0]].value
                    vox_result["fibril_param"]["fit"]["delta"] = est_fibrilParams.params['delta_' + labels[0]].value
                    vox_result["fibril_param"]["fit"]["std_error"] = std_errors

                    #vox_result["fibril_param"]["fit"]["c"] = est_fibrilParams.params['c_' + labels[0]].value
                    vox_result["fibril_param"]["alpha"] = est_fibrilParams.params['alpha1_' + labels[0]].value
                    vox_result["fibril_param"]["beta"] = est_fibrilParams.params['beta1_' + labels[0]].value
                    vox_result["fibril_param"]["solved"]=True
                            
                    #not_found = False
                    for idx, voxel in enumerate(fibrilParams):
                        indx = str(int(voxel["indx"]))
                        if indx in labels: 
                            #not_found = False
                            #add_test.append([idx,indx])
                            voxel["fit"]["q0"] = est_fibrilParams.params['q0_' + str(int(indx))].value
                            voxel["fit"]["amp"] = est_fibrilParams.params['amp_' + str(int(indx))].value
                            voxel["fit"]["wa"] = est_fibrilParams.params['wa_' + str(int(indx))].value
                            voxel["fit"]["wMu"] = est_fibrilParams.params['wMu_' + str(int(indx))].value
                            voxel["fit"]["delta"] = est_fibrilParams.params['delta_' + str(int(indx))].value
                            voxel["fit"]["std_error"] = est_fibrilParams.params['delta_' + str(int(indx))].value
                            #voxel["fit"]["c"] = est_fibrilParams.params['c_' + str(int(indx))].value
                            voxel["alpha"] = est_fibrilParams.params['alpha1_' + str(int(indx))].value
                            voxel["beta"] = est_fibrilParams.params['beta1_' + str(int(indx))].value
                            voxel["solved"]=True
                            
                    for bp in rotated_beampaths:                                                        
                        for j, pdict in enumerate(bp):
                            if pdict != None:
                                if pdict["voxels"] != None:                                                                    
                                    if j in ichiExp[r] and type(ichiExp[r][j]["kapton"]) == bool:                                                    
                                        for idx, voxel in enumerate(pdict["voxels"]):
                                            indx = str(int(voxel['fibril_param']["indx"]))
                                            if indx in labels: 
                                                #add_test.append([idx,indx])
                                                voxel['fibril_param']['solved'] = True
                                                voxel['fibril_param']['fit']['q0'] = est_fibrilParams.params['q0_' + str(indx)].value
                                                voxel['fibril_param']['fit']['wa'] = est_fibrilParams.params['wa_' + str(indx)].value
                                                voxel['fibril_param']['fit']['wMu'] = est_fibrilParams.params['wMu_' + str(indx)].value
                                                voxel['fibril_param']['fit']['delta'] = est_fibrilParams.params['delta_' + str(indx)].value
                                                #voxel['fibril_param']['fit']['c'] = est_fibrilParams.params['c_' + str(indx)].value
                                                voxel['fibril_param']['fit']['amp'] = est_fibrilParams.params['amp_' + str(indx)].value
                                                voxel["fibril_param"]["alpha"] = est_fibrilParams.params['alpha1_' + str(indx)].value
                                                voxel["fibril_param"]["beta"] = est_fibrilParams.params['beta1_' + str(indx)].value
                                                voxel["solved"]=True
                                                                                                                               
                    now = datetime.now()
                    year,month,day = now.strftime("%Y"),now.strftime("%m"),now.strftime("%d")
                    
                    time_finsihed = time()
                    
                    fit_time = time_finsihed - recon_start_time
                    
                    fitfile = output_path+day+"-"+month+"-"+year+"_"+"fitfile.txt"
                    file = open(fitfile,"a")
                    str2write = str(vox_indx)
                    #str2write = str2write + " " + str(voxel['fibril_param']["indx"])
                    params2write = ["amp","q0","wa","wMu","delta"]
                    for param in params2write:
                        str2write = str2write + " " + str(fibrilParams[og_indx]["fit"][param])
                    #str2write = str2write+" "
                    #for param in params2write:
                        #str2write = str2write + " " + str(fibrilParams[indx]["simu"][param])
                    str2write = str2write+" "+str(ichimax)+"\n"
                    str2write = str2write+" "+str(fit_time)+"\n"
                    #filestring = str("indx: " + str(indx) + ", q0: " + str(fibrilParams[indx]["fit"]["q0"])+"\n")
                    file.write(str2write)
                    file.close()
                    # sys.exit()
                    # fits the data (full data or expdata) and gets fibril parameters for "indx"
                    # adds these to the fibrilParams[indx]
                        
                    return fibrilParams,rotated_beampaths,vox_result,est_fibrilParams.chisqr,std_errors
                
                else:
                    
                    for k in range(nslices):
                        plt.plot(masked_qs[k],masked_real[k],color = cols[k],label="real data")
                        plt.plot(masked_qs[k],masked_fits[k],linestyle="dashed",color = cols[k],lw=1,label = "tomoSAXS recon")
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    print("fit not accepted: std eror>30")
                    plt.title(("solving ",r,i,idx,og_indx,": high std error"))
                    title = ("solving ",r,i,idx,og_indx,": high std error")
                    plt.savefig(output_path+"unaccepted_single_fits/"+str(r)+"_"+str(i)+"_"+labels[0]+".png")
                    plt.show()
                    plt.close()
                    
                    return fibrilParams,rotated_beampaths,vox_result,0,std_errors
            else:
                
                return fibrilParams,rotated_beampaths,vox_result,0,std_errors
                
        else:
            return fibrilParams,rotated_beampaths,vox_result,0,std_errors
    else:
        return fibrilParams,rotated_beampaths,vox_result,0,std_errors
