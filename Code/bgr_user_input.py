# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 06:51:28 2023

Script for applying "adcorr" background correction library for tomoSAXS data,
applying a GUI to select data, output folders, and scanning parameters 
and type of scan/background correction to apply. 

PREREQUISITES:

a.    The script uses a "thickness" file - a text/numpy file that should be previously 
      generated for fibre tracing CT data using the coregistration script.

b.    A "background" file - comprising data from a linescan across an empty kepton tube

c.    A "dispersant" file - comprising data from a linescan across a kapton tube
      containing only the hydrating medium used for the sample undergoing background 
      correction
      
d.    The background and dispersant data should originate from:
        i.  A kapton tube the same width as the tube used for the sample undergoing 
            background correction.
        ii. Frames created with the same exposure period, X-ray wavelength, and detector
            distance as the sample undergoing background correction

Frames collected between the two kapton edges is distinguished for the dispersant and 
background linescans. 

The script runs sequentially for each individual rotational scan in a tomoSAXS scan. 
Each horizontal slice of the respective scan, the space between the two
edges of the kapton tube is isolated. 

The x-axis position of frames within this space are then saved, and the closest frames to them
(relative to the kapton edges) are found for the: background; dispersant; and thickness data.

The width of the kapton tube is then estimated for each frame as its position along a coord between
the two edges.

The relative displacement of hydration medium can then be estimated for each frame, and used in the 
"tomoSAXS_disp_multiproc" multiprocessing script. This script simultaneously background corrects each 
horizontal slice in the respective scan. 

The resultant background corrected frames are saved in a new .hdf5 file in the specified output
folder. 


@author: elisn
"""
import os,sys
import time
import PySimpleGUI as sg
import numpy as np
from pathlib import Path

from h5py import File
from numpy import array, ones
import hdf5plugin

from adcorr.corrections import mask_frames
from adcorr.sequences import pauw_dispersed_sample_sequence, pauw_simple_sample_sequence

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, subplots
from matplotlib.colors import LogNorm

from scipy.signal import find_peaks

from multiprocessing import Pool, cpu_count
from time import time
#import fabio

from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from scipy.interpolate import UnivariateSpline

input_folder = "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded"

saxs_slice = 0

def fit_spline(slice_thickness,Deg):
    test_thickness = slice_thickness[np.where(slice_thickness>0)[0][0]:np.where(slice_thickness>0)[0][-1]]
    bg_zeros = np.zeros_like(slice_thickness)
    
    
    thickness_peaks = find_peaks(test_thickness)[0]
    peak_thickness = test_thickness[find_peaks(test_thickness)[0]]
    
    x = np.arange(0,len(test_thickness),1)
    
    spl = UnivariateSpline(thickness_peaks, peak_thickness)
    #spl.set_smoothing_factor(Deg)
    
    poly = spl(x)

    poly[poly<0] = 0
    
    bg_zeros[np.where(slice_thickness>0)[0][0]:np.where(slice_thickness>0)[0][-1]] = poly
    
    return bg_zeros

def fit_poly(slice_thickness,Deg):
    
    """
    Function for fitting a polynomial (degree controlled by "Deg") to the peaks found in sample
    thickness data 
    """
    
    #Isolate region where sample is found
    test_thickness = slice_thickness[np.where(slice_thickness>0)[0][0]:np.where(slice_thickness>0)[0][-1]]
    bg_zeros = np.zeros_like(slice_thickness)
    
    #find peaks using "scipy.signal.find_peaks"
    thickness_peaks = find_peaks(test_thickness)[0]
    peak_thickness = test_thickness[find_peaks(test_thickness)[0]]
    
    x = np.arange(0,len(test_thickness),1)
    
    #fit polynomial to peaks
    poly = np.polyfit(thickness_peaks, peak_thickness, deg=Deg)
    
    poly_model = np.polyval(poly, x)
    poly_model[poly_model<0] = 0
    
    bg_zeros[np.where(slice_thickness>0)[0][0]:np.where(slice_thickness>0)[0][-1]] = poly_model

    return bg_zeros

def find_kapton(slice_sums):
    
    """
    Function for finding edges of kapton tube in sum SAXS data
    
    check if the first set of frames is:
        background - typically minus sum WAXS radiation values
        kapton - high sum WAXS rad values followed by positive values
        artefact - high values followed by minus values
    """

    if len(np.where(slice_sums>np.abs(slice_sums[0])*10)[0])>0:
        first_x_kapton = np.where(slice_sums>np.abs(slice_sums[0])*10)[0][0]
    else:
        if np.min(slice_sums[0:50])>0:
            first_x_kapton = 0
        else:
            first_frame = np.where(slice_sums<0)[0][0]
            first_x_kapton = np.where(slice_sums[first_frame:-1]>np.abs(slice_sums[first_frame])*10)[0][0]
            
    return first_x_kapton

def figSave(figure,maxMin,filepath,filename):

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    fig.suptitle(filename)
    if np.max(maxMin) >0:
        ax.imshow(figure,vmin=maxMin[0],vmax = maxMin[1])
    else:
        ax.imshow(figure)
    fig.savefig(filepath)   # save the figure to file
    plt.close(fig)    # close the figure window 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def dirFromfile(fname):
    """
    from a filename with full direcory path, removes filname at end 
    and returns the directory
    """
    flist = fname.split("/")
    dirList = flist.pop()
    dirList = "/".join(flist)
    dirList = dirList+"/"
    return dirList,fname.split("/")[-1]


def collapse(layout, key, visible):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this section visible / invisible
    :param visible: visible determines if section is rendered visible or invisible on initialization
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible, pad=(0,0)))

def collectThicknessParameters():
    
    sg.theme('Light Blue 2')
            
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Input sample thickness (mm)')],
               [sg.InputText(key="-THICKTEXT-",size=(5,1))]]
    
    layout = [[sg.Frame('Sample thickness', frame_1, pad=(0, 5))],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('Testing Window', layout)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
            
    window.close() 
#section1 = [[sg.Text('Dispersant file', size=(8, 1)), sg.Input(), sg.FileBrowse(key="-DISPFILE-")]]

def collectFileParameters():
    
    sg.theme('Light Blue 2')
    
    # hidden input for dispersant file
    section1 = [[sg.Text('Dispersant file', size=(12, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-DISPFILE-")]]
    section2 = [[sg.Text('sample width file', size=(12, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-THICKFILE-")]]
    section3 = [[sg.Text('sample width (mm)', size=(20, 1)), sg.InputText(key="-STHICKTEXT-",size=(5,1))],
                [sg.Text('sample holder width (mm)', size=(20, 1)), sg.InputText(key="-SHTEXT-",size=(5,1))],
                [sg.Text('kapton width (mm)', size=(20, 1)), sg.InputText(key="-KTEXT-",size=(5,1))]]
    
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Select files')],
               [sg.Text('SAXS data folder', size=(12, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-FOLDERLIST-")],               
               [sg.Text('Mask file', size=(12, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-MASKFILE-")],
               [sg.Text('Calibration file', size=(12, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-CALIBFILE-")],
               [sg.Text('Scan name', size=(12, 1)), sg.InputText(key="-SCANNAME-")]]##
    """
    Second frame: background input (dispersent and sample thickness input hidden until tickboxes are checked) and output folder selection
    """
    
    frame_2 = [[sg.Text('Select backgrounds')],
               [sg.Text('Background file', size=(12, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-BGFILE-")],
               [sg.Checkbox('Dispersant used?',key='-DISPCHECK-',enable_events=True)],
               [sg.Checkbox('Sample thickness file?',key='-THICKCHECK-',enable_events=True)],
               #[sg.Text('Dispersant file', size=(8, 1)), sg.Input(), sg.FileBrowse(key="-DISPFILE-")]]
               [collapse(section1,'sec_1',False)],
               [collapse(section2,'sec_2',False)],  
               [sg.Text('Script folder', size=(12, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-SCRIPTFOLDER-")],
               [sg.Text('Output folder', size=(12, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-OUTFOLDER-")]]
    """
    Third frame: scan info.
        single background - single frame used for background
        sample background - several frames used for background 
        line-scan background - tomoSAXS-specific background using horizontal axis position to determine displaced volume
    """
    
    frame_3 = [[sg.Text('Scan info')],
               [sg.Listbox(values=["2D scanning SAXS", "tomoSAXS"], s=(30,2), key='-SCANTYPE-')],
               [sg.Listbox(values=["Correct background and disperant", "Correct just background"], s=(30,2), key='-CORRTYPE-')],
               [sg.Listbox(values=["Single background", "Sample background", "Line-scan background"], s=(30,3), key='-BGTYPE-')],
               [sg.Listbox(values=["kapton tube", "kapton cuboid", "no chamber"], s=(30,3), key='-ENVITYPE-',enable_events=True)],
               [collapse(section3,'sec_3',False)]]
    
    
    layout = [[sg.Frame('Files, parameters, and options', frame_1, pad=(0, 5)),
               sg.Frame('', frame_2, pad=(0, (10, 5)), key='Hide'),
               sg.Frame('', frame_3, pad=(0, (10, 5)), key='Hide')],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('TomoSAXS background correction', layout)
    
    toggle_sec1 = False
    toggle_sec2 = False
    toggle_sec3 = False
    
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            print(values[0])
            return values
        if event == '-DISPCHECK-':
            toggle_sec1 = not toggle_sec1
            window['sec_1'].update(visible=toggle_sec1)
        if event == '-THICKCHECK-':
            values['-ENVITYPE-'] == ['kapton cuboid']
            toggle_sec2 = not toggle_sec2
            window['sec_2'].update(visible=toggle_sec2)
        if event == '-ENVITYPE-':
            if values['-ENVITYPE-'] == ['kapton cuboid'] or values['-ENVITYPE-'] == ['kapton tube']:
            
                toggle_sec3 = not toggle_sec3
                window['sec_3'].update(visible=toggle_sec3)
                                
    window.close()   


"""
Begin script
"""                      

import pickle

if __name__ == "__main__":
    params = collectFileParameters()
    
    with open(params["-OUTFOLDER-"]+"/bgcorr_info.pkl", "wb") as f:
        pickle.dump(params, f)
               
    """
    1. find names for all scans within selected interval
    """        
    os.chdir(params["-FOLDERLIST-"])
    files = sg.popup_get_file('Unique File select', multiple_files=True)
    if len(files)>100:
        files = files.split(";")
    np.save(params["-OUTFOLDER-"]+"/registration_scan_files.npy",files)
