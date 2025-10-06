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

saxs_slice = 1

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
            
    if first_x_kapton>(len(slice_sums)/2):
        first_x_kapton = 0
            
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
    #params = collectFileParameters()
    
    with open(input_folder+"/bgcorr_info.pkl", "rb") as f:
        params = pickle.load(f)
    
    t1 = time()
    """
    beamline-specific information: https://diamondlightsource.github.io/adcorr/main/tutorials/i22_corrections.html
    """
    MINIMUM_PULSE_SEPARATION = 2e-6
    MINIMUM_ARRIVAL_SEPARATION = 3e-6
    BASE_DARK_CURRENT = 0.0
    TEMPORAL_DARK_CURRENT = 0.0
    FLUX_DEPENDANT_DARK_CURRENT = 0.0
    SENSOR_ABSORPTION_COEFFICIENT = 0.85
    SENSOR_THICKNESS = 1e-3
    BEAM_POLARIZATION = 0.5
    DISPLACED_FRACTION = 0.8
    
    
    """
    1. find names for all scans within selected interval
    """        
    os.chdir(params["-FOLDERLIST-"])
    #files = sg.popup_get_file('Unique File select', multiple_files=True)
    files = np.load(input_folder+"/registration_scan_files.npy")
    scanNumbers = files
    scan_name = params['-SCANNAME-']
    
    firstName = scanNumbers[0]
    firstNumber = int(firstName.split("-")[-1].split(".")[0][0])
    
    outputFolder = params["-OUTFOLDER-"]
    
    """
    Load dispersant and background data 
    """
    
    if params["-DISPCHECK-"] == False:
        
        """
        If a dispersant file is not provided - kill the script with an eror message
        """
        
        print("Background correction aborted: no dispersant background")
        print("Please provide dispersant background for tomoSAXS background correction")
        sys.exit()
        
    else:
        DISPERSANT_PATH = Path(params["-DISPFILE-"])            
                    
        with File(DISPERSANT_PATH) as dispersant_file:
            entry = list(dispersant_file.keys())[0]
            if entry == 'I0_data':
                dispersants = array(dispersant_file["data"])
                dispersant_sums = array(dispersant_file["sum"])
                dispersant_xAxis = array(dispersant_file["base_x_value_set"])
                dispersants_count_times = array(dispersant_file["count_time"]).tolist()
                dispersants_incident_flux = array(dispersant_file["I0_data"])
                dispersants_transmitted_flux = array(dispersant_file["OAV_data"])
            else:
                if entry+"/SAXS/data" in dispersant_file:
                    dispersants = array(dispersant_file[entry+"/SAXS/data"])
                    dispersant_sums = array(dispersant_file[entry+"/SAXS_sum/sum"])
                    dispersant_xAxis = array(dispersant_file[entry+"/SAXS_sum/base_x_value_set"])
                    dispersants_count_times = array(dispersant_file[entry+"/instrument/SAXS/count_time"]).tolist()
                    dispersants_incident_flux = array(dispersant_file[entry+"/I0/data"])
                    dispersants_transmitted_flux =  array(dispersant_file[entry+"/BSDIODES/data"])
                else:
                    dispersants = array(dispersant_file[entry+"/detector/data"])[0,0,:,:]
                    dispersant_sums = np.sum(dispersants)
                    dispersants_incident_flux = array(dispersant_file[entry+"/I0/data"])
                    dispersants_transmitted_flux = array(dispersant_file[entry+"/bsdiodes/data"])
                    dispersants_count_times = array(dispersant_file[entry+"/instrument/detector/count_time"])[0]
            dispersant_file.close()
                                    
    if len(dispersant_sums.shape)>1 and dispersant_sums.shape[0]>1:
        dispersant_sums = dispersant_sums[0,:]
        dispersants = dispersants[0]
        
    disp_vox = dispersant_xAxis[1]-dispersant_xAxis[0]
    dispRange = [find_kapton(dispersant_sums),len(dispersant_sums) - (find_kapton(np.flip(dispersant_sums))+1)]
        
    
    BACKGROUND_PATH = Path(params["-BGFILE-"])
    with File(BACKGROUND_PATH) as background_file:
        entry = list(background_file.keys())[0]
        if entry+"/SAXS/data" in background_file:
            backgrounds = background_file[entry+"/SAXS/data"]
            background_sums = array(background_file[entry+"/SAXS_sum/sum"])
            background_xAxis = array(background_file[entry+"/SAXS_sum/base_x_value_set"])
            backgrounds_count_times = array(background_file[entry+"/instrument/SAXS/count_time"]).tolist()
            backgrounds_incident_flux = array(background_file[entry+"/I0/data"])
            backgrounds_transmitted_flux =  array(background_file[entry+"/BSDIODES/data"])
        else:
            backgrounds = array(background_file[entry+"/detector/data"])[0,0,:,:]
            background_sums = np.sum(backgrounds)
            backgrounds_incident_flux = array(background_file[entry+"/I0/data"])
            backgrounds_transmitted_flux = array(background_file[entry+"/bsdiodes/data"])
            backgrounds_count_times = array(background_file[entry+"/instrument/detector/count_time"])[0]
        background_file.close()

    if len(background_sums.shape)>1 and background_sums.shape[0]>1:
        background_sums = background_sums[0,:] 
        backgrounds = backgrounds[0]

    bgRange = [find_kapton(background_sums),len(background_sums) - (find_kapton(np.flip(background_sums))+1)]
      
    if len(backgrounds.shape)<3:
        
        """
        If you have incorrectly chosen a single background file - abort
        """
        
        print("Background correction aborted: incorrect background file")
        print("Please provide linescan background file for tomoSAXS background correction")
        sys.exit()
                    
    if params["-THICKCHECK-"] == True:
        
        """
        Kept previous method available for saving thickness data as .hdf5 file for now
        although this will be removed in later versions
        """
                
        if params["-THICKFILE-"][-3:len(params["-THICKFILE-"])] == "npy":
            
            width_entries = np.load(params["-THICKFILE-"],allow_pickle = True)
            
        else:
            width_entries = np.loadtxt(params["-THICKFILE-"])
            
    else:
        print("Background correction aborted: no sample thickness file")
        sys.exit()
    
    """
    Load mask and calibration data
    """    
    MASK_PATH = Path(params["-MASKFILE-"])
    CALIBRANT_PATH = Path(params["-CALIBFILE-"])
    
    with File(MASK_PATH) as mask_file:
        Mask = array(mask_file["entry/mask/mask"])
        mask_file.close()
    
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
        
    """
    2. Find number of tomoSAXS slices in scan
    """
    SAMPLE_PATH = Path(scanNumbers[0])
               
    MASK_PATH = Path(params["-MASKFILE-"])
    CALIBRANT_PATH = Path(params["-CALIBFILE-"])
    
    with File(SAMPLE_PATH) as sample_file:
        entry = list(sample_file.keys())[0]
        if entry+"/SAXS/data" in sample_file:
            frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])
        else:
            frames = array(sample_file[entry+"/detector/data"])
            frames_sum = np.sum(frames)
        sample_file.close()
    
    slice_count = np.arange(0,frames_sum.shape[0],1)
    
    """
    3. Run adcorr for each tomoSAXS slice
    """
    
    t2 = time()
    print("Time spent: ",str(t2 - t1))
    
    plt.plot(background_sums[100:150])
    plt.plot(dispersant_sums[100:150])
    
    #for saxs_slice in slice_count[0:10]:
        
    t3 = time()
    print("Background correction running for slice: ",str(saxs_slice))
    print("Total time spent: ",str(t3 - t2))
    
    """
    Create subfolder in output folder
    """
    scanFolder = outputFolder+"/"+scan_name+"_"+str(saxs_slice)
    if os.path.isdir(scanFolder) == False:
        os.mkdir(scanFolder)
        
    """
    Create empy H5 file for slice
    """
    outputHf = File(scanFolder+"/"+scan_name+"_"+str(saxs_slice)+"_processed.h5", 'w')
        
    """
    Load slice from the data for each tomoSAXS orientation
    """
    
    for scan in scanNumbers:
        
        t3 = time()
        print("Background correction running for slice ",str(saxs_slice)," scan: ",scan.split("-")[-1])
        print("Total time spent: ",str(t3 - t2))
        
    
        """
        Load sample, mask, and calibration data views 
        """
        SAMPLE_PATH = Path(scan)
                               
        with File(SAMPLE_PATH) as sample_file:
            entry = list(sample_file.keys())[0]
            if entry+"/SAXS/data" in sample_file:
                frames = sample_file[entry+'/SAXS/data'][saxs_slice,:,:,:]   
                frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])[saxs_slice,:]
                if entry+"/SAXS_sum/base_x_value_set" in sample_file:
                    frames_xAxis = array(sample_file[entry+"/SAXS_sum/base_x_value_set"])
                else:
                    frames_xAxis = array(sample_file[entry+"/SAXS_sum/p1xy_x_value_set"])
                                    
                frames_count_times = array(sample_file[entry+"/instrument/SAXS/count_time"]).tolist()            
                frames_incident_flux = array(sample_file[entry+"/I0/data"])[saxs_slice,:,:,:]
                frames_transmitted_flux = array(sample_file[entry+"/BSDIODES/data"])[saxs_slice,:,:,:]
                #sample_thickness = array(sample_file[entry+"/sample/thickness"])
            else:
                frames = array(sample_file[entry+"/detector/data"])
                frames_sum = np.sum(frames)
                frames_incident_flux = array(sample_file[entry+"/I0/data"])[saxs_slice,:,:,:]
                frames_transmitted_flux = array(sample_file[entry+"/bsdiodes/data"])[saxs_slice,:,:,:]
            #sample_file.close()
            
        """
        IF you have a file for sample thickness 
        """
        #if params["-THICKCHECK-"] == True:                           
        scan_no = np.where(np.asarray(scanNumbers) == scan)[0][0]
        if params["-THICKFILE-"][-3:len(params["-THICKFILE-"])] != "txt":      
            sample_thickness =  width_entries[scan_no,saxs_slice,:]
        else:
            sample_thickness =  width_entries[:,scan_no,:]
            
        beam_center_pixels = (
            beam_center_y.item(),
            beam_center_x.item()
        )
        pixel_sizes = (
            x_pixel_size.item(),
            y_pixel_size.item()
        )
        
        """
        Split scan into its consecutive rows to minimise memory load 
        """
               
        thick_vals, displacement_vals = [],[] #sample thickness values and amount of dispersant displaced
        
        info_array = [] #beamline-specific information for correction
            
        t4 = time()
        print("Collating data for scan: ",str(scan))
        print("Total time spent: ",str(t4 - t2))
                                        
        """
        Load frames, incident flux, transmitted flux, and count times for sample,
        background, and dispersant data.                
        """
    
        slice_frames = np.copy(frames)
        slice_sums = np.copy(frames_sum)
        if len(slice_sums.shape) >2:
            slice_sums = [slice_sums[k][0][0] for k in np.arange(0,slice_sums.shape[0],1)]
        
        slice_frames_count_times = np.full(slice_frames.shape[0],(frames_count_times))
        slice_frames_incident_flux = np.asarray([np.mean(frames_incident_flux[k,:,5]) for k in np.arange(0,slice_frames.shape[0],1)])                
        slice_frames_transmitted_flux = np.asarray([np.mean(frames_transmitted_flux[k,:,1]) for k in np.arange(0,slice_frames.shape[0],1)])
        
        if len(backgrounds_incident_flux.shape)>3:
            slice_backgrounds_incident_flux = np.asarray([np.mean(backgrounds_incident_flux[0,k,:,5]) for k in np.arange(0,dispersants_incident_flux.shape[1],1)])
            slice_backgrounds_transmitted_flux = np.asarray([np.mean(backgrounds_transmitted_flux[0,k,:,1]) for k in np.arange(0,dispersants_transmitted_flux.shape[1],1)])
            slice_backgrounds_count_times =  np.full(len(slice_backgrounds_incident_flux),(backgrounds_count_times))  
        else:
            slice_backgrounds_incident_flux = np.asarray([np.mean(backgrounds_incident_flux[k,:,5]) for k in np.arange(0,backgrounds_incident_flux.shape[0],1)])
            slice_backgrounds_transmitted_flux = np.asarray([np.mean(backgrounds_transmitted_flux[k,:,1]) for k in np.arange(0,backgrounds_transmitted_flux.shape[0],1)])
            slice_backgrounds_count_times =  np.full(len(slice_backgrounds_incident_flux),(backgrounds_count_times)) 
          
            
        if len(slice_frames)>0:
            
            FLATFIELD = ones(slice_frames.shape[-2:])            
                                              
            sample_vox = frames_xAxis[1]-frames_xAxis[0]
              
            slice_thickness =  np.copy(sample_thickness)*sample_vox
            thickness_limits = np.where(slice_thickness == np.max(slice_thickness))[0] 
            
            """
            double check if sample thickness values are stored as mm or m
            if mm convert to m
            """
            
            if np.max(slice_thickness[slice_thickness>0])>4:
                slice_thickness = slice_thickness*1e-3
                
            
            #slice_thickness = fit_poly(slice_thickness,3)
            
            if params["-ENVITYPE-"][0] == "kapton tube":
                
                """
                Find frames represented by kapton edges
                isolate frames between these edges
                """
                                        
                tube_limits = [find_kapton(slice_sums),len(slice_sums) - (find_kapton(np.flip(slice_sums))+1)]
                                                
                sample_axis = np.arange(frames_xAxis[tube_limits[0]],frames_xAxis[tube_limits[1]],sample_vox)
                #sample_axis = np.linspace(frames_xAxis[tube_limits[0]],frames_xAxis[tube_limits[1]],len(slice_thickness))
            else:
                print("Background correction aborted: incorrect sample environment")
                sys.exit()
            
            #if params["-DISPCHECK-"] == True:
            if len(backgrounds_incident_flux.shape)>3:
                slice_dispersants_incident_flux = np.asarray([np.mean(dispersants_incident_flux[0,k,:,5]) for k in np.arange(0,dispersants_incident_flux.shape[1],1)])
                slice_dispersants_transmitted_flux = np.asarray([np.mean(dispersants_transmitted_flux[0,k,:,1]) for k in np.arange(0,dispersants_transmitted_flux.shape[1],1)])
                slice_dispersants_count_times =  np.full(len(slice_dispersants_incident_flux),(dispersants_count_times)) 
            else:
                slice_dispersants_incident_flux = np.asarray([np.mean(dispersants_incident_flux[k,:,5]) for k in np.arange(0,dispersants_incident_flux.shape[0],1)])
                slice_dispersants_transmitted_flux = np.asarray([np.mean(dispersants_transmitted_flux[k,:,1]) for k in np.arange(0,dispersants_transmitted_flux.shape[0],1)])
                slice_dispersants_count_times =  np.full(len(slice_dispersants_incident_flux),(dispersants_count_times)) 
                
            if params["-BGTYPE-"] == ['Line-scan background']:
                
                """
                For dispersant sample (treated first as an optional step for other background 
                correction that may not need a dispersant dataset)
                dispersant and background data may be a different dynamic range to sample data 
                (i.e. different voxel size)
                so need to resample frames, flux, and count times of dispersant and background data
                based on the voxel size of the sample data.
                
                To do this, find the dispersant frames that represent the kapton edges, and how
                many frames are between this at what voxel size.
                Then resample these frames at the same voxel resolution as the sample                         
                """
                                                        
                disp_sample_axis = np.arange(dispersant_xAxis[dispRange[0]],dispersant_xAxis[dispRange[1]],sample_vox)
                if len(disp_sample_axis)>len(sample_axis):
                    disp_sample_axis = disp_sample_axis[0:len(sample_axis)]
                
                disp_samples = np.asarray([dispersants[find_nearest(dispersant_xAxis,disp_sample_axis[k])[1],:,:] for k in np.arange(0,len(disp_sample_axis),1)])
                slice_dispersants_incident_flux = np.asarray([slice_dispersants_incident_flux[find_nearest(dispersant_xAxis,disp_sample_axis[k])[1]] for k in np.arange(0,len(disp_sample_axis),1)])
                slice_dispersants_transmitted_flux = np.asarray([slice_dispersants_transmitted_flux[find_nearest(dispersant_xAxis,disp_sample_axis[k])[1]] for k in np.arange(0,len(disp_sample_axis),1)])
                slice_dispersants_count_times =  np.full(len(slice_dispersants_incident_flux),(dispersants_count_times))
            else:
                print("Background correction aborted: incorrect background file")
                print("Background file must be Line-scan background")
                sys.exit()
                
            """
            IF you are correcting with a line scan - register the dispersant and background scans with the sample scan using suitable landmark
            (i.e. Kapton tube limits)
            """
            
            """
            isolate the sample frames within the kapton and repeat the same resampling 
            process for the empty kapton background.
            """
                
            #a1 = AzimuthalIntegrator(wavelength = Wavelength)
            
            #a1.setFit2D(directDist = sample_detector_separation, centerX = beam_center_x.item()/x_pixel_size.item(),centerY = beam_center_y.item()/y_pixel_size.item(), pixelX = x_pixel_size.item()*1000,pixelY = y_pixel_size.item()*1000)
                                                                                     
            slice_samples = np.asarray([slice_frames[find_nearest(frames_xAxis,sample_axis[k])[1],:,:] for k in np.arange(0,len(sample_axis),1)])
            slice_idxs = np.asarray([find_nearest(frames_xAxis,sample_axis[k])[1] for k in np.arange(0,len(sample_axis),1)])
            slice_sums = np.asarray([slice_sums[find_nearest(frames_xAxis,sample_axis[k])[1]] for k in np.arange(0,len(sample_axis),1)])
            slice_sample_incident_flux = np.asarray([slice_frames_incident_flux[find_nearest(frames_xAxis,sample_axis[k])[1]] for k in np.arange(0,len(sample_axis),1)])
            slice_sample_transmitted_flux = np.asarray([slice_frames_transmitted_flux[find_nearest(frames_xAxis,sample_axis[k])[1]] for k in np.arange(0,len(sample_axis),1)])
            slice_sample_count_times =  np.asarray([slice_frames_count_times[find_nearest(frames_xAxis,sample_axis[k])[1]] for k in np.arange(0,len(sample_axis),1)])                     
            
            #bgRange = [find_peaks(background_sums,height =  np.max(background_sums)-(np.max(background_sums)/2))[0][0],find_peaks(background_sums,height =  np.max(background_sums)-(np.max(background_sums)/2))[0][-1]]                
            bg_sample_axis = np.arange(background_xAxis[bgRange[0]],background_xAxis[bgRange[1]],sample_vox)                                
            if len(bg_sample_axis)>len(sample_axis):
                bg_sample_axis = bg_sample_axis[0:len(sample_axis)]
            
            bg_samples = np.asarray([backgrounds[find_nearest(background_xAxis,bg_sample_axis[k])[1],:,:] for k in np.arange(0,len(bg_sample_axis),1)])                
            slice_backgrounds_incident_flux = np.asarray([slice_backgrounds_incident_flux[find_nearest(background_xAxis,bg_sample_axis[k])[1]] for k in np.arange(0,len(bg_sample_axis),1)]) 
            slice_backgrounds_transmitted_flux = np.asarray([slice_backgrounds_transmitted_flux[find_nearest(background_xAxis,bg_sample_axis[k])[1]] for k in np.arange(0,len(bg_sample_axis),1)]) 
            slice_backgrounds_count_times =  np.full(len(slice_backgrounds_incident_flux),(backgrounds_count_times))
            

            """
            Estimate the width of the kapton tube at each sample frame
            """
                                                               
            disp_sample_range = sample_axis[-1]-sample_axis                
            disp_dist_frm_ctr = np.sqrt((disp_sample_range-(disp_sample_range[0]/2))**2)                
            choord_len = [((disp_dist_frm_ctr[0]**2)-(disp_dist_frm_ctr[k]**2))*1000 for k in np.arange(0,len(disp_dist_frm_ctr),1)]
            choord_len[choord_len == 0.0] = choord_len[1]
            choord_len[-1] = choord_len[1]
            #slice_thickness =  sample_thickness[slice,:]
            choord_len = np.asarray(choord_len)*1e-3
            
            thickness_axis = np.arange(0,len(slice_thickness),1)
            thick_smpl_axis = np.round(np.linspace(0,len(slice_thickness)-1,len(choord_len))).astype(int)
            #slice_thicknesses = np.asarray([slice_thickness[k] for k in thick_smpl_axis]) 
            slice_thicknesses = np.asarray([slice_thickness[find_nearest(frames_xAxis,sample_axis[k])[1]] if find_nearest(frames_xAxis,sample_axis[k])[1] <(len(slice_thickness)) else 0 for k in np.arange(0,len(sample_axis),1)])
                                                                            
            thick_vals.append(slice_thicknesses) 
            displacement_vals.append(choord_len)
            
            infoMatrix = [beam_center_pixels,pixel_sizes,sample_detector_separation,Mask]  
            info_array = np.repeat(np.asarray(infoMatrix)[None,:],len(slice_thicknesses),axis = 0)
            
            #sf_full.append(slice_frames) 
            
            t5 = time()
            
            print("Data collated for scan: ",scan.split("-")[-1])
            print("Running background correction")
            print("Total time spent: ",str(t5 - t2))
            
            slice_thicknesses[slice_thicknesses>0.175] = 0
           
            """
            Apply background correction for slice using multiprocessing
            """
            
            if params["-CORRTYPE-"][0] == "Correct just background":
                
                ncpus = cpu_count()
                if params["-DISPCHECK-"] == True:
                    if __name__ == '__main__':
                        os.chdir(params['-SCRIPTFOLDER-']+"/")
                        from adcorr_multiFuncs import tomoSAXS_simple_multiproc 
                        
                        corrected_frames = []
                        tl1 = time()
                        for i in range(0,np.min([len(slice_idxs),len(bg_samples),len(disp_samples)])):
                            slice_sample = slice_samples[i].reshape((1,slice_samples[i].shape[0], slice_samples[i].shape[1]))
                            disp_sample = disp_samples[i].reshape((1,disp_samples[i].shape[0], disp_samples[i].shape[1]))
                            bg_sample = bg_samples[i].reshape((1,bg_samples[i].shape[0], bg_samples[i].shape[1]))

                            corrected_frame = tomoSAXS_simple_multiproc(slice_idxs[i],slice_sample,
                                                                        disp_sample,bg_sample,
                                                                        slice_sample_count_times[i],
                                                                        slice_dispersants_count_times[i],
                                                                        slice_backgrounds_count_times[i],
                                                                        slice_sample_incident_flux[i],
                                                                        slice_sample_transmitted_flux[i],
                                                                        slice_dispersants_incident_flux[i],
                                                                        slice_dispersants_transmitted_flux[i],
                                                                        slice_backgrounds_incident_flux[i],
                                                                        slice_backgrounds_transmitted_flux[i],
                                                                        slice_thicknesses[i],
                                                                        choord_len[i],
                                                                        info_array[i])
                            
                            print("slice ", str(i)," corrected")
                            
                            corrected_frames.append(corrected_frame)
                        
                        tl2 = time() 
                        
                        print("total multi background correction time (MP) was ", tl2-tl1," seconds")
                            
                            
            elif params["-CORRTYPE-"][0] == "Correct background and disperant":
                
                ncpus = cpu_count()
                if params["-DISPCHECK-"] == True:
                    if __name__ == '__main__':
                        os.chdir(params['-SCRIPTFOLDER-']+"/")
                        from adcorr_multiFuncs import tomoSAXS_disp_multiproc 
                        
                        with Pool(ncpus) as p:
                            tl1 = time()
                            corrected_frames = p.starmap(tomoSAXS_disp_multiproc, zip(slice_idxs,slice_samples,
                                                                                      disp_samples,bg_samples,
                                                                                      slice_sample_count_times,
                                                                                      slice_dispersants_count_times,
                                                                                      slice_backgrounds_count_times,
                                                                                      slice_sample_incident_flux,
                                                                                      slice_sample_transmitted_flux,
                                                                                      slice_dispersants_incident_flux,
                                                                                      slice_dispersants_transmitted_flux,
                                                                                      slice_backgrounds_incident_flux,
                                                                                      slice_backgrounds_transmitted_flux,
                                                                                      slice_thicknesses,
                                                                                      choord_len,
                                                                                      info_array))
                            tl2 = time()
                            tmp = tl2-tl1
                            print("total multi background correction time (MP) was ", tl2-tl1," seconds")
                            
            
            """
            Copy the background corrected frame into the original data for the 
            respective slice
            """
            
            for i in range(0,len(corrected_frames)):
                corrected_idx = corrected_frames[i][1]
                slice_frames[corrected_idx] = corrected_frames[i][0]
                
            """
            Save in individual row of output hdf5 file
            """    
                               
            outputHf.create_dataset("scan "+scan.split("-")[-1].split(".")[0],data = slice_frames)
            
    """
    Save all data for slice and close file.
    """  
    t6 = time()
    print("Background correction finished for slice ",str(saxs_slice)," scan: ",scan.split("-")[-1].split(".")[0])
    print("Total time spent: ",str(t6 - t2))
    
    outputHf.close()
