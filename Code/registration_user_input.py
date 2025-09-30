# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:15:44 2025

@author: Himadri
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:50:20 2024

@author: Elis Newham

Script for registering tomoSAXS and CT data. 

Steps are:

1. Register vertical axis between resliced CT dataset and coarse mapping of tomoSAXS dataset.
Registration uses the offset between vertebrae end-points surrounding the respective IVD.

The offset is provided with absolute values from coarse SAXS map.

This calibration also allows estimation of which CT slices represent the regions investigated in the tomoSAXS scan.

2. Pad fibre tracing data.
The Fibre tracing technique sumsamples and downsamples original CT data. This process adds empty voxels to fibre tracing data so that the fibre tracing data returns to the same dimensions as the original CT data.

Padded data can then be registered with SAXS data both vertically and horizontally.

The padded fibre tracing slices that vertically correspond to the slices of the tomoSAXS scan are then isolated and saved.

The kapton data that correspond to these slices are also isolated and saved.

3. Register sample axes (-90/90 degree; 0/180 degree) of padded fibre tracing data using the kapton segmentation, and preprocess fibre data.
This step pre-processes the padded fibre tracing data to align it with the major axes (0/180; -90/90) of the tomoSAXS scan, before registering voxels for each tomoSAXS orientation (step 4). The coarse WAXS maps are conducted at 0 degreess. The CT reconstructions are also oriented with the Y axis corresponding to -90/90 degrees, and X axis corresponding to 0/180 degrees. Thus, the Y axis of the kapton segmentation corresponds to the first orientation of the tomoSAXS scan, and the X axis to the midpoint orientation. The offsets for each axis are calculated by comparing the positions of the left-hand-side (lhs) kapton edge along both axes in the CT and SAXS data. The fibre tracing data and kapton data is then padded accordingly to overcome these offsets.

The preprocessing step here uses K-means clustering to assess the heterogeneity of orientation values for fibres sampled within each tomoSAXS voxel. In a single tomoSAXS voxel, fibres with values within 5 degrees of each other are determined to be insufficiently independent in terms of their angular orientation to be able to seperate using tomoSAXS. So they are given the same mean value and indexed as the same scattering object.

4. Register horizontal axis between padded fibre tracing data and SAXS data for each orientation in tomoSAXS scan, and subsample across tomoSAXS beampaths.
This process repeats the kapton edge offset calculation from step 3 to register the fibre tracing and SAXS axes (only x axis for this step as this is the axis that the SAXS mapping is conducted over), but for each orientation in the tomoSAXS scan. For the first orientation, the alpha and beta values for fibres are subsampled into the tomoSAXS voxels within-which they are found. The fibre data within these subsampled voxels are then indexed. Index data is then rotated alongside fibre tracing data forthe rest of the scan, allowing the identity of traced fibres to be consistently identified.

5. Save data.
The data outputted from this process is saved, both as python objects and legible formats for humans.


"""

import os,glob
import PySimpleGUI as sg
import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from io import BytesIO
from PIL import Image
from multiprocessing import Pool, cpu_count 

from h5py import File
from numpy import array, ones
import hdf5plugin

import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from collections import Counter
from skimage import measure
from sklearn import cluster
import random

from scipy.ndimage import rotate
from skg import nsphere_fit

import time

import cv2


def roi_inspect(data):
    width, height = 640, 480

    layout = [[sg.Graph(
        canvas_size=(500, 450),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key="-GRAPH-",
        change_submits=True,  # mouse click events
        background_color='lightblue',
        drag_submits=True), ],
        [sg.Text('Correct ROI selected?', size=(40, 1)), 
                      sg.Listbox(values=["YES", "NO"], s=(8,2),
                                 key='-correct-',enable_events=True)],
        [sg.Submit(),sg.Cancel()]]
    window = sg.Window("Inspect selected ROI", layout, finalize=True)
    graph = window["-GRAPH-"]
    graph.draw_image(data=data, location=(80, 450))
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
    
    window.close()

def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

def figSave(figure,filepath,filename):

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    fig.suptitle(filename)
    ax.imshow(figure)
    fig.savefig(filepath)   # save the figure to file
    plt.close(fig)    # close the figure window 

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

def collectFileParameters():
    
    Font = ("Arial", 11)
    
    sg.theme('Light Blue 2')
            
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Scan name (check against excel file)', size=(53, 1)), 
                sg.InputText(default_text = "QIVD X/FIVD Y",key="-SCANNAME-", size=(13, 1))],
                [sg.Text('Original CT data', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-CTFOLDER-")],
                [sg.Text('Inverted resliced CT map', size=(39, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-CTMAP-")],
                [sg.Text('Kapton CT dataset', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-KAPTONFOLDER-")],
                [sg.Text('Beta/phi fibre tracing data', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-BETAFOLDER-")],
                [sg.Text('Alpha/theta fibre tracing data', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-ALPHAFOLDER-")],
                [sg.Text('WAXS map data', size=(39, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-WAXSFILE-")],
                [sg.Text('Output folder', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-OUTFOLDER-")],
                [sg.Text('Script folder', size=(39, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-SCRIPTFOLDER-")],
                [sg.Text('Fibre tracing padding file', size=(39, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-PADFILE-")],
                [sg.Text('Original CT voxel size (um)', size=(58, 1)), sg.InputText(default_text = "1.625",key="-xyCTvox-", size=(8, 1))],
                [sg.Text('Inverted CT voxel size (um)', size=(58, 1)), sg.InputText(default_text = "6.5",key="-invertCTvox-", size=(8, 1))],
                [sg.Text('Kapton data voxel size (um)', size=(58, 1)), sg.InputText(default_text = "6.5",key="-kaptonvox-", size=(8, 1))],
                [sg.Text('Fibre tracing voxel scale', size=(58, 1)), sg.InputText(default_text = "5",key="-fibTraceVox-", size=(8, 1))],
                [sg.Text('Kapton tube diameter (um)', size=(58, 1)), sg.InputText(default_text = "4000",key="-tube-", size=(8, 1))],
                [sg.Text('SAXS rotational direction', size=(51, 1)), sg.Listbox(values=["clockwise", "anti-clockwise"], s=(14,2), 
                                                                                key='-rotation-',enable_events=True)],
                [sg.Text('tomoSAXS binning', size=(58, 1)), sg.InputText(default_text = "1",key="-saxsScale-", size=(8, 1))]]
    
    layout = [[sg.Frame('Folder selection', frame_1, pad=(0, 5))],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('3D registration: folder selection', layout,font=Font)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
            
    window.close()

def collectScanParameters():
    
    Font = ("Arial", 14)
    
    sg.theme('Light Blue 2')
            
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Number of rotational angles in tomoSAXS scan', size=(58, 1)), sg.InputText(default_text = "9",key="-nr-", size=(8, 1))],
               [sg.Text('start angle', size=(58, 1)), sg.InputText(default_text = "-90",key="-r0-", size=(8, 1))],
               [sg.Text('end angle', size=(58, 1)), sg.InputText(default_text = "90",key="-r-1-", size=(8, 1))],
               [sg.Text('angle of WAXS map', size=(58, 1)), sg.InputText(default_text = "0",key="-mapR-", size=(8, 1))]]
    
    layout = [[sg.Frame('TomoSAXS scan parameters', frame_1, pad=(0, 5))],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('3D registration: TomoSAXS scan parameters', layout,font = Font)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
            
    window.close()




import pickle

"""
Begin script
"""                      
if __name__ == "__main__":
        
    """
    Collect folder and scanning parameters
    """

    calib_paths = collectFileParameters()
    tomoSAXS_params = collectScanParameters()
    
    output_path = calib_paths['-OUTFOLDER-']
    
    with open(output_path+"/registration_scan_info.pkl", 'wb') as f:
        pickle.dump(calib_paths, f)
        
    with open(output_path+"/bgcorr_info.pkl", 'wb') as f:
        pickle.dump(tomoSAXS_params, f)
        
    with open(output_path+"/registration_info.pkl", 'wb') as f:
        pickle.dump(tomoSAXS_params, f)
    
    tomosaxs_files = sg.popup_get_file('Select files in tomoSAXS scan', multiple_files=True)
    tomosaxs_paths = tomosaxs_files.split(";")
    
    np.save(output_path+"/registration_scan_files.npy",tomosaxs_paths)
            
    scan_name = calib_paths["-SCANNAME-"]
    
    ct_vox = float(calib_paths['-xyCTvox-'])
    inv_ct_vox = float(calib_paths['-invertCTvox-'])
    fibtrac_scale = float(calib_paths['-fibTraceVox-'])
    fibre_vox = ct_vox*fibtrac_scale
    kapton_vox = float(calib_paths['-kaptonvox-'])
    kapt_w = float(calib_paths['-tube-'])
    
    scan_count = int(tomoSAXS_params["-nr-"])
    scan_start = float(tomoSAXS_params["-r0-"])
    scan_end = float(tomoSAXS_params["-r-1-"])
    map_rot = float(tomoSAXS_params["-mapR-"])
    
    scan_angles = np.linspace(scan_start,scan_end,scan_count)
    
    saxsScale = int(calib_paths["-saxsScale-"])
    
    scan_angles = np.linspace(-90,90,9)
    
    print("######\nStarting tomoSAXS registration program for scan\n###### ",calib_paths["-SCANNAME-"])
    
    len_checker = False
    
    if len_checker == True:
        if len(tomoSAXS_params["-nr-"])>len(tomosaxs_paths):
            
            redo_tomoSAXS = False
            
            while redo_tomoSAXS == False:
                print("######\nINSUFFICIENT NUMBER OF TOMOSAXS FILES SELECTED: SELECT AGAIN\n######")
                tomosaxs_files = sg.popup_get_file('Select files in tomoSAXS scan', multiple_files=True)
                tomosaxs_paths = tomosaxs_files.split(";")
                if len(tomoSAXS_params["-nr-"]) == len(tomosaxs_paths):
                    redo_tomoSAXS == True
                
    
    ct_path = calib_paths["-CTMAP-"]    
    waxs_path = calib_paths["-WAXSFILE-"]
    kapton_path = calib_paths["-KAPTONFOLDER-"]
    
    output_path = calib_paths["-OUTFOLDER-"]
    
    script_path = calib_paths["-SCRIPTFOLDER-"]
    
    os.chdir(script_path+"/")
    #import tomoSAXS_outputs_and_figs as outp
    
    """
    Kapton files created in imageJ may be created at a set number of significant figures.
    So we need to isolate this formatting:
    """
    
    kapton_file = glob.glob(os.path.join(kapton_path, "*.tif"))[0]
    kapton_SF = len(kapton_file.split('\\')[-1].split(".")[0])
    
    SAMPLE_PATH = Path(waxs_path)
    
    with File(SAMPLE_PATH) as sample_file:
        entry = list(sample_file.keys())[0]
        if entry+"/WAXS/data" in sample_file:
            frames_sum = array(sample_file[entry+"/WAXS_sum/sum"])
            frames_xVals = array(sample_file[entry+"/WAXS_sum/base_x_value"])
            frames_yVals = array(sample_file[entry+"/WAXS_sum/base_y_value"])
        sample_file.close()
    frames_sum = frames_sum-np.min(frames_sum)  
    
    figSave(frames_sum,output_path+"/WAXS map original.png","Original WAXS map")
         
    normalized_frames = (frames_sum - np.min(frames_sum)) * 255.0 / (np.max(frames_sum) - np.min(frames_sum)) 
    
    figSave(normalized_frames,output_path+"/WAXS map 8bit.png","8bit WAXS map")
            
    """
    Select endpoint of top vertebra
    GUI system:
        a. full image first shown in full GUI
        b. zoom in and select end point
        c. hit esc twice
        d. this brings up new gui with zoomed in detail with crosshair showing selection
        e. if this is correct hit "yes"
        f. if incorrect hit "no" - and steps a-d will repeat
            
    """
    map_frames = image_resize(normalized_frames, width = normalized_frames.shape[1]*10, height =  normalized_frames.shape[0]*10, inter = cv2.INTER_AREA).astype(np.uint8)

    figSave(map_frames,output_path+"/WAXS_map_scaled.png","Scaled WAXS map")
            
    # Select ROI 
    roi_selected = False
    
    print("######## \nSelect endpoint of top vertebra in WAXS map: \n Zoom into IVD region in GUI and click on vertebral endpoint. \nSelection shown by white crosshair. \n Once selected, hit Esc until window disappears. \n########")

    while roi_selected == False:
        map_coords = cv2.selectROI("Zoom in to select endpoint of top vertebra: once selected hit Esc until window disappears", map_frames) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        cropped_image = np.copy(map_frames)
        cropped_image[map_coords[1]-5:map_coords[1]+6,map_coords[0]] = 0
        cropped_image[map_coords[1],map_coords[0]-5:map_coords[0]+6] = 0
        cropped_image = cropped_image[np.max([0,map_coords[1]-100]):np.min([cropped_image.shape[0],map_coords[1]+100]),np.max([0,map_coords[0]-100]):np.min([cropped_image.shape[1],map_coords[0]+100])]
        
        cropped_map = image_resize(cropped_image, width = cropped_image.shape[1]*2, height =  cropped_image.shape[0]*2, inter = cv2.INTER_AREA).astype(np.uint8)
        
        data = array_to_data(cropped_map)
        
        selection_params = roi_inspect(data)
        
        if selection_params["-correct-"][0] == "YES":
            roi_selected = True
                
    figSave(cropped_map,output_path+"/WAXS_map_top_vert_endpoint.png","Chosen endpoint of top vertebra in WAXS data")
     
    top_vert_end = int(map_coords[1]/10) 
    
    top_vert_end_abs = frames_yVals[top_vert_end,int(435/10)]
    
    print("Upper vertebral endpoint in WAXS map.png",str(top_vert_end_abs)," mm")
    
    """
    NOW you can perform a similar technique with CT data
    """
    ct_img = cv2.cvtColor(cv2.imread(ct_path), cv2.COLOR_BGR2GRAY)
        
    figSave(ct_img,output_path+"/CT map original.png","Original CT map")
    
    print("######## \nSelect endpoint of top vertebra in CT map: \n Zoom into IVD region in GUI and click on vertebral endpoint. \nSelection shown by white crosshair. \n Once selected, hit Esc until window disappears. \n########")
        
    roi_selected = False

    while roi_selected == False:
        map_coords = cv2.selectROI("select endpoint of top vertebra", ct_img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        cropped_image = np.copy(ct_img)
        cropped_image[map_coords[1]-5:map_coords[1]+6,map_coords[0]] = 0
        cropped_image[map_coords[1],map_coords[0]-5:map_coords[0]+6] = 0
        cropped_image = cropped_image[np.max([0,map_coords[1]-100]):np.min([cropped_image.shape[0],map_coords[1]+100]),np.max([0,map_coords[0]-100]):np.min([cropped_image.shape[1],map_coords[0]+100])]
        
        cropped_map = image_resize(cropped_image, width = cropped_image.shape[1]*2, height =  cropped_image.shape[0]*2, inter = cv2.INTER_AREA).astype(np.uint8)
        
        data = array_to_data(cropped_map)
        
        selection_params = roi_inspect(data)
        
        if selection_params["-correct-"][0] == "YES":
            roi_selected = True
        
    figSave(cropped_map,output_path+"/Upper_vertebral_endpoint_in CT_map.png","Chosen endpoint of top vertebra in CT data")
    
    ct_top_vert_end = map_coords[1]
    ct_top_vert_end_abs = ct_top_vert_end*inv_ct_vox
    
    top_vert_data = [[scan_name,top_vert_end_abs,ct_top_vert_end_abs]]
    
    np.save(output_path+"/vertical_calibration.npy",top_vert_data)
