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

input_folder = "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/"

fib_thresh = 2000

#scatter_indexer(indexSlices[slice],fibreIndexData[0][slice],rot_alpha_maps[0][slice])

def scatter_indexer(rot_index_data,og_index_data,og_scatter_data):
    
    fibre_index = [[rot_index_data[i,k] for k in np.arange(0,rot_index_data.shape[1],1) if rot_index_data[i,k] !=0]
                   for i in np.arange(0,rot_index_data.shape[0],1)]
    
    dbl_index = [[[j,l] for l in np.arange(0,rot_index_data.shape[1],1) if type(rot_index_data[j,l]) == list] 
                 for j in np.arange(0,rot_index_data.shape[0],1)]
    dbl_index = [k for k in dbl_index if len(k)>0]
    dbl_index = [item for sublist in dbl_index for item in sublist]
    fibreIndex = [[item] if type(item) !=list else item for sublist in fibre_index for item in sublist]
    fibreIndex = np.asarray([item for sublist in fibreIndex for item in sublist])
    uniqueFibres = np.unique(fibreIndex)
    
    rot_scatter_data = np.zeros_like(rot_index_data,dtype=np.object_)
    
    for index in uniqueFibres:
        if len(np.where(rot_index_data == index)[0])>0:
            scatter_val = np.mean(og_scatter_data[np.where(og_index_data == index)])
            scatter_vox = np.where(rot_index_data == index)
            for vox in range(0,len(scatter_vox[0])):
                rot_scatter_data[scatter_vox[0][vox],scatter_vox[1][vox]] = scatter_val
            
    for vox in dbl_index:
        index_vox = rot_index_data[vox[0],vox[1]]
        scatter_vals = [np.mean(og_scatter_data[np.where(og_index_data == k)]) for k in index_vox]
        rot_scatter_data[vox[0],vox[1]] = scatter_vals
        
    return rot_scatter_data


def rot_gridder(data,rot_angle):

    #fibTest = np.copy(fibreIndexData[0][slice])

    dilate_kernel = np.ones((5, 5), np.uint8) 
    erode_kernel = np.ones((4, 4), np.uint8) 
    
    
    fibre_index = [[data[i,k] for k in np.arange(0,data.shape[1],1) if data[i,k] !=0]
                   for i in np.arange(0,data.shape[0],1)]
    
    fibreIndex = [item for sublist in fibre_index for item in sublist]
    fibreIndex = np.unique(fibreIndex)
    
    #if len(fibreIndex)<1000:
    dilation_imgs = [cv2.dilate(np.where(data == k,1,0).astype(np.uint8), dilate_kernel, iterations=1) for  k in fibreIndex]
    
    rotation_imgs = [rotate(dilation_imgs[k],rot_angle,reshape=False,mode='nearest',order=0)
     for k in np.arange(0,len(dilation_imgs),1)]
      
    erosion_imgs = np.asarray([cv2.erode(rotation_imgs[k].astype(np.uint8), erode_kernel, iterations=1)*fibreIndex[k] for k in np.arange(0,len(dilation_imgs),1)])
    
    """
    else:
        idxs = np.arange(0,len(fibreIndex),100).tolist()
        idxs.append(len(fibreIndex))
        
        dilation_img,rotation_imgs = [],[]
        
        for idx in range(0,len(idxs)-1):
            #print(idx)
            
            indexes_idx = fibreIndex[idxs[idx]:idxs[idx+1]]
            
            dilation_imgs_idx = [cv2.dilate(np.where(data == k,1,0).astype(np.uint8), dilate_kernel, iterations=1) for  k in indexes_idx]
            
            if rot_angle>0:
                rotation_imgs_idx = []
                for k in np.arange(0,len(dilation_imgs_idx),1):
                    rotation_imgs_idx.append(rotate(dilation_imgs_idx[k],rot_angle,reshape=False,mode='nearest',order=0))
            else:
                rotation_imgs_idx = dilation_imgs_idx
            
            erosion_imgs_idx = np.asarray([cv2.erode(rotation_imgs_idx[k].astype(np.uint8), erode_kernel, iterations=1)*indexes_idx[k] for k in np.arange(0,len(dilation_imgs_idx),1)])
    """        
                                    
    pixel_comp = np.asarray([len(np.where(data==fibreIndex[k])[0]) - len(np.where(erosion_imgs[k]==fibreIndex[k])[0]) for k in np.arange(0,len(erosion_imgs),1)])
    
    correct_rots = np.where(np.asarray(pixel_comp) ==0)
    pixels_to_fix = fibreIndex[np.where(np.asarray(pixel_comp) !=0)]
    pixels_to_fix_scores = pixel_comp[np.where(np.asarray(pixel_comp) !=0)]
    
    rotate_test = np.asarray([rotate(np.where(data == k,1,0),rot_angle,reshape=False,mode='nearest',order=0)*k
     for k in pixels_to_fix])
    
    rot_pixel_comp = [len(np.where(data==pixels_to_fix[k])[0]) - len(np.where(rotate_test[k]==pixels_to_fix[k])[0]) for k in np.arange(0,len(pixels_to_fix),1)]
    
    final_rot_comp = np.where(np.abs(rot_pixel_comp)<=np.abs(pixels_to_fix_scores)[0])
    missing_rots = np.where(np.abs(rot_pixel_comp)>=np.abs(pixels_to_fix_scores)[0])
    
    alt_rots_final = np.asarray(rotate_test[np.where(np.abs(rot_pixel_comp)<=np.abs(pixels_to_fix_scores))])
    
    erosion_rots_final = np.asarray([erosion_imgs[k] for k in correct_rots[0]])
    if len(erosion_rots_final)>0 and len(alt_rots_final)>0:
        full_rots = np.concatenate((erosion_rots_final,alt_rots_final),axis=0)
    elif len(erosion_rots_final)>0 and len(alt_rots_final)==0:
        full_rots = erosion_rots_final
    elif len(erosion_rots_final)==0 and len(alt_rots_final)>0:
        full_rots = alt_rots_final    
    
    binary_test = np.sum(np.where(full_rots !=0,1,0),axis=0)
    
    binary_doubles = np.where(binary_test>1)
    
    full_rotation = np.sum(full_rots,axis=0).astype(np.object_)
    
    if len(binary_doubles[0])>0:
        dbl_index_vals = [[k[binary_doubles[0][i],binary_doubles[1][i]] for k in full_rots if k[binary_doubles[0][i],binary_doubles[1][i]] !=0] 
         for i in np.arange(0,len(binary_doubles[0]),1)]
        
        for i in range(0,len(dbl_index_vals)):
            full_rotation[binary_doubles[0][i],binary_doubles[1][i]] = dbl_index_vals[i]
        
    return full_rotation


def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

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


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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

def figSave(figure,filepath,filename):

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    fig.suptitle(filename)
    ax.imshow(figure)
    fig.savefig(filepath)   # save the figure to file
    plt.close(fig)    # close the figure window 
    

def singlePlotSave(xAxis,yAxis,axisLabels,folder,filename):
    fig, (ax1) = plt.subplots(1)
    fig.suptitle(filename)
    ax1.plot(xAxis,yAxis) 
    ax1.set(xlabel=axisLabels[0], ylabel=axisLabels[0])  
    ax1.set_ylim([0,np.max(yAxis)+np.std(yAxis)])   
    #ax1.set_xlim([0,1])
    plt.tight_layout() 
    plt.savefig(folder+filename+'.png')

def paddingParameters():
    
    sg.theme('Light Blue 2')
            
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Fibre tracing padding file', size=(16, 1)), sg.Input(size=(20,1)), sg.FileBrowse(key="-PADFILE-")],                
                [sg.Text('Fibre tracing output folder', size=(16, 1)), sg.Input(size=(20,1)), sg.FolderBrowse(key="-OUTFOLDER-")]]
    
    layout = [[sg.Frame('Folder selection', frame_1, pad=(0, 5))],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('3D registration: folder selection', layout)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
            
    window.close()

def map_cluster(img,clusters):
    
    if len(np.shape(img))<3:
        img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    
    # Convert MxNx3 image into Kx3 where K=MxN
    img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN
    
    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    img2 = np.float32(img2)
    
    #Define criteria, number of clusters and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Number of clusters - 5 seems to work best
    k = clusters
    
    # Number of attempts, number of times algorithm is executed using different initial labelings.
    attempts = 10
    
    ret,label,center=cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    
    #cv2.kmeans outputs 2 parameters.
    center = np.uint8(center) 
    
    #Next, we have to access the labels to regenerate the clustered image
    res = center[label.flatten()]
    res2 = res.reshape((img.shape)) #Reshape labels to the size of original image
    
    return(cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY))

#alpha_slice,beta_slice,scan_params,voxel_params = newAlphaMaps[slice],newBetaMaps[slice],scan_params,voxel_params

#alpha_slice,beta_slice,scan_params,voxel_params = newAlphaMaps[slice],newBetaMaps[slice],scan_params,voxel_params

def ab_gridder(alpha_slice,beta_slice,scan_params,voxel_params):

    """
    First create empty maps of tomoSAXS voxels
    """
    #ct_zAxis_SAXS_voxels = np.arange(0,kapton_pad.shape[0],(saxs_vox/fibre_vox))
    ct_xAxis_SAXS_voxels = voxel_params[0]
    ct_yAxis_SAXS_voxels = voxel_params[2]
        
    indexRefMatrix = np.zeros((len(ct_xAxis_SAXS_voxels)-1,len(ct_yAxis_SAXS_voxels)-1), dtype=np.object_)
    voxIndexRefMatrix = np.zeros((len(ct_xAxis_SAXS_voxels)-1,len(ct_yAxis_SAXS_voxels)-1), dtype=np.object_)
    
    indexMatrix = np.zeros((alpha_slice.shape))
    voxIndexMatrix = np.zeros((alpha_slice.shape))
    
    alphaValTestMatrix = np.zeros((alpha_slice.shape),dtype = "float32")
    alphaValTestMatrix = np.zeros((alpha_slice.shape),dtype = object)
    betaValTestMatrix = np.copy(alphaValTestMatrix)
    
    alphaValMatrix = np.zeros((len(ct_xAxis_SAXS_voxels)-1,len(ct_yAxis_SAXS_voxels)-1), dtype=np.object_)
    alphaCountMatrix = np.zeros((len(ct_xAxis_SAXS_voxels)-1,len(ct_yAxis_SAXS_voxels)-1), dtype=np.object_)
    
    betaValMatrix = np.copy(alphaValMatrix)
    betaCountMatrix = np.copy(alphaCountMatrix)
    
    voxels = [ct_xAxis_SAXS_voxels[1]-ct_xAxis_SAXS_voxels[0],ct_yAxis_SAXS_voxels[1]-ct_yAxis_SAXS_voxels[0]]
    
    voxelIndex = 0
    elementIndex = 0
    
    """
    Now detect and label the alpha and beta values of fibres that they capture
    """
    
    for i in range(0,len(ct_xAxis_SAXS_voxels)-1):
        xAxisBoundaries = np.round([ct_xAxis_SAXS_voxels[i],ct_xAxis_SAXS_voxels[i+1]]).astype(int)
                
        for k in range(0,len(ct_yAxis_SAXS_voxels)-1):
            yAxisBoundaries = np.round([ct_yAxis_SAXS_voxels[k],ct_yAxis_SAXS_voxels[k+1]]).astype(int)
                           
            #alpha_sample = np.ceil(np.copy(alpha_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]]).astype(float))
            #beta_sample = np.ceil(np.copy(beta_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]]).astype(float))                                       
            
            alpha_sample = np.copy(alpha_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]])
            beta_sample = np.copy(beta_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]])                                       
            
            list_idx_alpha,list_idx_beta = [],[]
            
            for j in range(0,alpha_sample.shape[0]):
                for l in range(0,alpha_sample.shape[1]):
                    if type(alpha_sample[j,l])==list:                        
                        list_idx_alpha.append([j,l,alpha_sample[j,l]])
                        alpha_sample[j,l] = 0
                        
            for j in range(0,beta_sample.shape[0]):
                for l in range(0,beta_sample.shape[1]):
                    if type(beta_sample[j,l])==list:                        
                        list_idx_beta.append([j,l,beta_sample[j,l]])
                        beta_sample[j,l] = 0
            
            alpha_sample = alpha_sample.astype(float)
            beta_sample = beta_sample.astype(float)
            
            #voxIndexSample = np.ceil(np.copy(alpha_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]]).astype(float))
            voxIndexSample = np.copy(alpha_sample)

            #voxIndexSample = voxIndexSample*beta_sample
            #if np.min(voxIndexSample)<0:
                #test.append([i,k,len(voxIndexSample[voxIndexSample<0])])
                        
            #betaIndexSample = np.ceil(np.copy(beta_slice[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]]).astype(float))
            
            betaIndexSample = np.copy(beta_sample)

            
            indexSample = (np.where(voxIndexSample!=0,1,0))*voxelIndex
            voxelIndex=voxelIndex+1
            
            indexMatrix[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]] = voxelIndex
            indexRefMatrix[i,k] = voxelIndex
                            
            if len(np.unique(voxIndexSample))>1 or np.max(voxIndexSample)!=0:
                vox_labels = []
                for voxIndexSample in ([alpha_sample,beta_sample]):
                    if len(np.unique(measure.label(voxIndexSample)))>2 and (int(np.round((np.max(np.unique(voxIndexSample))-np.min(np.unique(voxIndexSample[voxIndexSample!=0])))/5))-1) > 1:
                        if np.min(voxIndexSample)<0:                       
                            voxForClusters = voxIndexSample-(np.min(voxIndexSample)-10)
                            voxForClusters[voxForClusters == 0-(np.min(voxIndexSample)-10)] = 0
                            voxClusters = clusters(voxForClusters,5)      
                        else:
                            voxClusters = clusters(voxIndexSample,5)                
                        labels = voxClusters[1] 
                    else:   
                        voxIndexSample[voxIndexSample>0] = np.mean(voxIndexSample[voxIndexSample>0])
                        labels = measure.label(voxIndexSample)
                        voxClusters = measure.label(voxIndexSample)
                        
                    vox_labels.append(labels)
                    
                if len(np.where(vox_labels[0] != vox_labels[1])[0])>0:
                    labels = vox_labels[np.argmax([len(np.unique(k)) for k in vox_labels])]
                    
                    
                if len(np.unique(labels))>1 and np.min(indexSample)==0:
                    labels[labels==np.min(labels)] = 0
                labels[labels>0] = labels[labels>0] + elementIndex   
                elementIndex = np.max(labels)   
                
                voxIndexMatrix[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]] = labels
                                                 
                labelBW = np.where(labels!=0,1,0)
                alphaBW = np.where(alpha_sample!=0,1,0)
                
                if len(np.where((labelBW-alphaBW)>0)[0])>0:
                    fill_x,fill_y =np.where((labelBW-alphaBW)>0)
                    for fill_indx in range(0,len(fill_x)):
                        alpha_sample[fill_x[fill_indx],fill_y[fill_indx]] = 1
                        
                            
                
                alphaLabels = data_label(np.copy(labels),alpha_sample)
                betaLabels = data_label(np.copy(labels),beta_sample)
                
                if len(np.unique(betaLabels))!=len(np.unique(alphaLabels)):
                    print(i,k)
                    
                if len(list_idx_alpha)>0: 
                    alphaLabels = alphaLabels.astype(object)
                    for idx in list_idx_alpha:
                        alphaLabels[idx[0],idx[1]] = idx[2]
                if len(list_idx_beta)>0: 
                    betaLabels = betaLabels.astype(object)
                    for idx in list_idx_beta:
                        betaLabels[idx[0],idx[1]] = idx[2]
                
                alphaValTestMatrix[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]] = alphaLabels
                betaValTestMatrix[xAxisBoundaries[0]:xAxisBoundaries[1],yAxisBoundaries[0]:yAxisBoundaries[1]] = betaLabels
                
    return alphaValTestMatrix,betaValTestMatrix

#data,bitMap,kaptonMap,scan_params,weight_params,ut8,voxIndexer,test_print = alphaResults[2],bitMaps[slice],newKaptonMap,scan_params,weightParams,False,False,False

#data,bitMap,kaptonMap,scan_params,weight_params,ut8,voxIndexer,test_print = indexResults[0],bitMaps[slice],newKaptonMap,scan_params,weightParams,False,False,False

#data,bitMap,kaptonMap,scan_params,weight_params,ut8,voxIndexer,test_print = newIndexMaps[slice],bitMaps[slice],newKaptonMap,scan_params,weightParams,False,False,False

def weight_gridder(data,bitMap,kaptonMap,scan_params,weight_params,ut8 = True,voxIndexer = False,test_print = False):
    
    beamdiameter = scan_params[1]/scan_params[2]
    sigma=beamdiameter/(2*np.sqrt(2*np.log(2))) 

    if ut8 == True:    
        ct_slice = data.astype('uint8')
        bin_slice = bitMap.astype('uint8')
    else:
        ct_slice = data
        if voxIndexer == True:
            bin_slice = bitMap
        else:
            bin_slice = np.where(data !=0,1,0)
            
    #ct_zAxis_SAXS_voxels = np.arange(0,kapton_pad.shape[0],(saxs_vox/fibre_vox))
    weight_starts = weight_params[0]
    weight_ends = weight_params[1]
    ct_yAxis_SAXS_voxels = weight_params[2]
        
    indexWeightMatrix = np.zeros((len(ct_yAxis_SAXS_voxels)-1,len(weight_params[3])-1),dtype=np.object_)
    indexCountMatrix = np.zeros((len(ct_yAxis_SAXS_voxels)-1,len(weight_params[3])-1),dtype=np.object_)
    
    indexMatrix = np.zeros((len(ct_yAxis_SAXS_voxels)-1,len(weight_params[3])-1))
    
    index_val = 1
    
    kapton_edge = False
    
    test = []
    
    for i in range(0,len(weight_starts)-1):
        
        weight_start = weight_starts[i]
        weight_end = weight_ends[i]
        if weight_start<0:
            weight_start = 0
        if weight_end>data.shape[0]:
            weight_end = data.shape[0]
                                
        for k in range(0,len(ct_yAxis_SAXS_voxels)-1):
            yAxisBoundaries = np.round([ct_yAxis_SAXS_voxels[k],ct_yAxis_SAXS_voxels[k+1]]).astype(int)
                                      
            bp_sample = np.copy(ct_slice[yAxisBoundaries[0]:yAxisBoundaries[1],weight_start:weight_end]) 
            kapton_sample = np.ceil(np.copy(kaptonMap[yAxisBoundaries[0]:yAxisBoundaries[1],weight_start:weight_end]).astype(float))
            
            
            dbl_pixels = [[bp_sample[j,l] for l in np.arange(0,bp_sample.shape[1],1) if type(bp_sample[j,l]) == list] 
                              for j in np.arange(0,bp_sample.shape[0],1)]
                        
            dbl_pixels = [k for k in dbl_pixels if len(k)>0]
            
            #if len(dbl_pixels)>0 and type(dbl_pixels[0][0]) == list:
                #dbl_pixels = [[k[0] for k in i] for i in dbl_pixels]
            
            if len(dbl_pixels)>0:
                perp_distMatrix = (np.tile(np.arange(1,bp_sample.shape[1]+1,1),(bp_sample.shape[0],1)))
                
                perp_distMatrix = [np.asarray([perp_distMatrix[i,k]-((bp_sample.shape[1]/2)+0.5) if type(bp_sample[i,k])!=list 
                                    else[perp_distMatrix[i,k]-((bp_sample.shape[1]/2)+0.5) for j in np.arange(0,len(bp_sample[i,k]),1)] 
                                    for k in np.arange(0,perp_distMatrix.shape[1],1)]) for i in np.arange(0,perp_distMatrix.shape[0],1)]                               
                
                perp_distMatrix = [np.asarray([perp_distMatrix[i][k] if bp_sample[i,k] !=0 else 0 for
                                               k in np.arange(0,len(perp_distMatrix[i]),1)]) for i in np.arange(0,len(perp_distMatrix),1)]
                
                #if type(perp_distMatrix[0][0]) == list:
                    #perp_distMatrix = [[k[0] if type(k) == list or type(k) == np.ndarray else k for k in i] for i in perp_distMatrix]
                    #perp_distMatrix = [np.asarray(i) for i in perp_distMatrix]
                    
                labelMatrix = [np.asarray([bp_sample[i][k] if bp_sample[i,k] !=0 else 0 for
                                               k in np.arange(0,len(perp_distMatrix[i]),1)]) for i in np.arange(0,len(perp_distMatrix),1)]
                
                perp_distMatrix = [[item for sublist in [[k] if type(k) != list else k for k in i.tolist()] for item in sublist] 
                                   if i.dtype == 'O' else i.tolist() for i in perp_distMatrix]
                                
                labelMatrix = [[item for sublist in [[k] if type(k) != list else k for k in i.tolist()] for item in sublist] 
                                   if i.dtype == 'O' else i.tolist() for i in labelMatrix]
                
                perp_distMatrix = [abs(np.asarray(item)) for sublist in perp_distMatrix for item in sublist if item!=0]
                labelMatrix = [item for sublist in labelMatrix for item in sublist if item!=0]
                
                if len([k for k in perp_distMatrix if type(k) == np.ndarray or type(k) == list])>0:
                    perp_distMatrix = [k[0] if type(k) == np.ndarray or type(k) == list else k for k in perp_distMatrix]
                    labelMatrix = [k[0] if type(k) == np.ndarray or type(k) == list else k for k in labelMatrix]
                    
                int_weightMatrix = (np.exp(-0.5*((perp_distMatrix/sigma)**2)))
                
                index_counts = [[len(np.where(labelMatrix==k)[0]),k] for k in np.unique(labelMatrix)[np.unique(labelMatrix)>0]]
                index_weights = [[np.sum(int_weightMatrix[np.where(labelMatrix == k)]),k] for k in np.unique(labelMatrix)[np.unique(labelMatrix)>0]]
                
                indexWeightMatrix[k,i] = index_weights
                test.append([i,k,index_weights])
                indexCountMatrix[k,i] = index_counts
                
                indexMatrix[k,i] = index_val
                
                index_val = index_val+1
                
            else:
                if bp_sample.shape[0]>0:
                    if np.sum(bp_sample)>0:
                        perp_distMatrix = (np.tile(np.arange(1,bp_sample.shape[1]+1,1),(bp_sample.shape[0],1)))-((bp_sample.shape[1]/2)+0.5)
                        int_weightMatrix = (np.exp(-0.5*((perp_distMatrix/sigma)**2)))*np.where(bp_sample>0,1,0)
                        
                        index_counts = [[len(np.where(bp_sample==k)[0]),k] for k in np.unique(bp_sample)[np.unique(bp_sample)>0]]
                        index_weights = [[np.sum(int_weightMatrix[np.where(bp_sample == k)]),k] for k in np.unique(bp_sample)[np.unique(bp_sample)>0]]  
                        indexWeightMatrix[k,i] = index_weights
                        test.append([i,k,index_weights])
                        indexCountMatrix[k,i] = index_counts
                        
                        indexMatrix[k,i] = index_val
                        
                        index_val = index_val+1
                    else:
                        if np.sum(kapton_sample)>0 and kapton_edge == False:
                            
                            indexWeightMatrix[k,i] = 100000
                            indexCountMatrix[k,i] = 100000
                            
                            indexMatrix[k,i] = 100000
                            
                            kapton_edge = True
                    
                
    return indexWeightMatrix,indexCountMatrix,indexMatrix


def saxs_vox_gridder(voxel):
    labelVals = [item for sublist in voxel for item in sublist]
    labelVals = [[j] if type(j) != list else j for j in labelVals]
    labelVals = np.asarray([item for sublist in labelVals for item in sublist])
    
    return np.asarray(labelVals)

    
#alphaResults = data_gridder(newAlphaMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True) 
#data_gridder(newBetaMaps[slice],alphaResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0]
#data,bitMap,kaptonMap,scan_params,voxel_params, ut8, voxIndexer, test_print = newBetaMaps[slice],alphaResults[2],newKaptonMap,scan_params,voxel_params,False, True, False

#data,bitMap,kaptonMap,scan_params,voxel_params,ut8,voxIndexer = newBetaMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,False,True

#data,bitMap,kaptonMap,scan_params,voxel_params,ut8,voxIndexer =  newIndexMaps[slice],bitMaps[slice],newKaptonMap,scan_params,voxel_params,False,False


#data,bitMap,kaptonMap,scan_params,voxel_params,ut8,voxIndexer,test_print = newIndexMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,False,False,False


#data,bitMap,kaptonMap,scan_params,voxel_params,ut8,voxIndexer,test_print = newThicknessMaps[slice],newThicknessMaps[slice],newKaptonMap,scan_params,voxel_params,False,True,False

def data_gridder(data,bitMap,kaptonMap,scan_params,voxel_params,ut8 = True,voxIndexer = False,test_print = False):
    
    test_print = False
    
    beamdiameter = scan_params[1]
    sigma=beamdiameter/(2*np.sqrt(2*np.log(2))) 

    if ut8 == True:    
        ct_slice = data.astype('uint8')
        bin_slice = bitMap.astype('uint8')
    else:
        ct_slice = data
        if voxIndexer == True:
            bin_slice = bitMap
        else:
            bin_slice = np.where(data !=0,1,0)
            
    #ct_zAxis_SAXS_voxels = np.arange(0,kapton_pad.shape[0],(saxs_vox/fibre_vox))
    ct_xAxis_SAXS_voxels = voxel_params[0]
    ct_xAxis_SAXS_midpoints = voxel_params[1]
    ct_yAxis_SAXS_voxels = voxel_params[2]
    
    #indexRefMatrix = np.zeros((len(voxel_params[0])-1,len(voxel_params[2])-1), dtype=np.object_)
    indexRefMatrix = np.zeros((len(voxel_params[2])-1,len(voxel_params[0])-1))
    voxIndexRefMatrix = np.zeros((len(voxel_params[2])-1,len(voxel_params[0])-1), dtype=np.object_)
    
    indexWeightMatrix = np.zeros((len(voxel_params[2])-1,len(voxel_params[0])-1),dtype=np.object_)
    
    indexMatrix = np.zeros((ct_slice.shape))
    voxIndexMatrix = np.zeros((ct_slice.shape),dtype=np.object_)
    
    valueTestMatrix = np.zeros((ct_slice.shape),dtype = "float32")
    valueMatrix = np.zeros((len(voxel_params[2])-1,len(voxel_params[0])-1), dtype=np.object_)
    countMatrix = np.zeros((len(voxel_params[2])-1,len(voxel_params[0])-1), dtype=np.object_)
    

    voxelIndex = 0
    elementIndex = 0
    
    test = []
    test2 = []
    
    kapton_edge = False
    
    for i in range(0,len(ct_xAxis_SAXS_voxels)-1):
        xAxisBoundaries = np.round([ct_xAxis_SAXS_voxels[i],ct_xAxis_SAXS_voxels[i+1]]).astype(int)
        xAxisMidpoint = int(np.round(ct_xAxis_SAXS_midpoints[i]))
                
        for k in range(0,len(ct_yAxis_SAXS_voxels)-1):
            yAxisBoundaries = np.round([ct_yAxis_SAXS_voxels[k],ct_yAxis_SAXS_voxels[k+1]]).astype(int)
                          
            bp_sample = np.copy(ct_slice[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]])
            bp_index_sample = np.copy(bin_slice[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]])
            
            #bp_sample = np.asarray([[np.mean(k) if type(k)==list or type(k)==np.ndarray else int(k) for k in bp_sample[i]] 
                                    #for i in np.arange(0,len(bp_sample),1)])
            
            #bp_index_sample = np.asarray([[np.mean(k) if type(k)==list else int(k) for k in bp_index_sample[i]] 
                                          #for i in np.arange(0,len(bp_index_sample),1)])
                        
            
            kapton_sample = np.copy(kaptonMap[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]])
            
            voxIndexSample = np.copy(ct_slice[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]]) 
            
            indexSample = (np.where(voxIndexSample!=0,1,0))*voxelIndex
            if type(indexSample) != np.ndarray:
                if type(indexSample) != list:
                    indexSample = np.asarray([indexSample])
                else:
                    indexSample = np.asarray(indexSample)
                
            voxelIndex=voxelIndex+1
            
            if voxIndexer == False:
                dbl_pixels = [[bp_sample[j,l] for l in np.arange(0,bp_sample.shape[1],1) if type(bp_sample[j,l]) == list] 
                              for j in np.arange(0,bp_sample.shape[0],1)]
            else:
                dbl_pixels = [[bp_index_sample[j,l] for l in np.arange(0,bp_index_sample.shape[1],1) if type(bp_index_sample[j,l]) == list] 
                              for j in np.arange(0,bp_sample.shape[0],1)]
            
            dbl_pixels = [k for k in dbl_pixels if len(k)>0]
            
            if voxIndexer == False:
                voxIndexSample = np.asarray([[np.max(k) if type(k)==list or type(k)==np.ndarray else int(k) for k in voxIndexSample[i]] 
                                             for i in np.arange(0,len(voxIndexSample),1)])
            
            #voxIndexSample_vals = [item for sublist in voxIndexSample_vals for item in sublist]
            
            if kapton_sample.shape[0]>0:                           
                if np.sum(kapton_sample) == 0:
                    indexMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = voxelIndex
                    indexRefMatrix[k,i] = voxelIndex
                else:
                    if kapton_edge == False:
                        indexMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = 100000
                        indexRefMatrix[k,i] = 100000  
                    
            if len(dbl_pixels)>0:                
                if voxIndexer == False:
                    labelVals = [item for sublist in voxIndexSample for item in sublist]
                    labelVals = [[j] if type(j) != list else j for j in labelVals]
                    labelVals = np.asarray([item for sublist in labelVals for item in sublist])
                    bp_labels = measure.label(labelVals)
                    labels = bp_labels+elementIndex 
                    matSample = np.copy(voxIndexSample)
                    values = np.unique(labelVals[labelVals>0])
                    alphaCounts = np.unique(labelVals[labelVals>0],return_counts=True)[1]
                    
                    valueMatrix[k,i],countMatrix[k,i] = values,alphaCounts
                    test2.append([k,i])
                    
                    valueTestMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = np.mean(values)

                else:
                    labelVals = saxs_vox_gridder(bp_index_sample)
                    alphaVals = saxs_vox_gridder(voxIndexSample)
                    values = np.asarray([alphaVals[np.where(labelVals == k)] for k in np.unique(labelVals[labelVals!=0])])
                    alphaCounts = [len(k) for k in values]
                    values = [np.mean(k) for k in values]
                    bp_labels = measure.label(labelVals)
                    labels = bp_labels+elementIndex 
                    matSample = np.copy(bp_index_sample)
                    values = np.unique(labelVals[labelVals>0])
                    alphaCounts = np.unique(labelVals[labelVals>0],return_counts=True)[1]
                    
                    valueMatrix[k,i],countMatrix[k,i] = values,alphaCounts
                    test2.append([k,i])
                    
                    valueTestMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = np.mean(values)
                                        
                if len(np.unique(labels))>1 and np.min(indexSample)==0:
                    labels[labels==np.min(labels)] = 0
                elementIndex = np.max(labels) 
                
                labelMat = np.zeros_like(matSample)
                for j in range(0,labelMat.shape[0]):
                    for l in range(0,labelMat.shape[1]):
                        if matSample[j,l] != 0:
                            if type(matSample[j,l]) !=list:
                                labelMat[j,l] = labels[np.where(labelVals==matSample[j,l])][0]
                            else:
                                labelMat[j,l] = [labels[np.where(labelVals==k)][0] for k in matSample[j,l]]
                                
                voxIndexMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = labelMat    
                            
            
            elif len(dbl_pixels)==0 and len(voxIndexSample)>0 and np.sum(voxIndexSample)>0:
                            
                if len(voxIndexSample)>0 and np.max(voxIndexSample)>0 and len(dbl_pixels) ==0:
                    #test.append([i,k])
                    
                    if voxIndexer == False:
                        labels = (measure.label(voxIndexSample))+elementIndex   
                    else:              
                        labels = (measure.label(bp_index_sample))+elementIndex                                
                    if len(np.unique(labels))>1 and np.min(indexSample)==0:
                        labels[labels==np.min(labels)] = 0
                    elementIndex = np.max(labels)   
                    
                    voxIndexMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = labels
                    
                    """
                    Adding section for weighting fibres based on the proximity of their consituent fibre tracing voxels 
                    to the beampath centre
                    """
                    
                    perp_distMatrix = (np.tile(np.arange(1,labels.shape[1]+1,1),(labels.shape[0],1)))-((labels.shape[1]/2)+0.5)
                    int_weightMatrix = (np.exp(-0.5*((perp_distMatrix/sigma)**2)))*np.where(labels>0,1,0)
                    
                    index_weights = [np.sum(int_weightMatrix[np.where(labels == k)]) for k in np.unique(labels)[np.unique(labels)>0]]  
                    indexWeightMatrix[k,i] = index_weights
                                             
                if len(indexSample)>0 and np.max(bp_sample)!=0:
                    #test.append([i,k])                     
                    #if len(np.delete(np.unique(bp_sample), np.where(np.unique(bp_sample) == 0)))>0:  
                    if len(np.delete(np.unique(bp_sample), np.where(np.unique(bp_sample) == 0)))>0:  
                        test.append([i,k])                                   
                        
                        if voxIndexer == False:
                            bp_labels = (measure.label(bp_sample))
                        else:
                            bp_labels = (measure.label(bp_index_sample))
                        
                        labels_for_counts = [bp_labels[np.where(bp_labels == k)] for k in np.unique(bp_labels)[np.unique(bp_labels)>0]]  
                        labels_for_counts = [np.mean(k) for k in labels_for_counts]
                        
                        values = [bp_sample[np.where(bp_labels == k)] for k in np.unique(bp_labels)[np.unique(bp_labels)>0]]                 
                        #values =  np.unique([item for sublist in values for item in sublist if item>0])
                        values = [np.mean(k) for k in values]
                                                              
                        #bp_sample[bp_sample == 0] = np.nan
                        alphaVals = np.unique(bp_labels[bp_labels!=0],return_counts=True,return_index=True)                        
                        alphaCounts = [alphaVals[2][np.where(alphaVals[0]==labels_for_counts[k])[0]][0] for k in np.linspace(0,len(values)-1,len(values)).astype(int)]
                                            
                        #if np.sum(alphaVals[2])>2:
                        valueMatrix[k,i],countMatrix[k,i] = values,alphaCounts
                        test2.append([k,i])
    
                    else:
                        print(i,k)
                        bp_sample[bp_sample==0] = np.nan
                        alphaVals = np.unique(bp_sample[~np.isnan(bp_sample)],return_counts=True,return_index=True)
                        
                        # if np.sum(alphaVals[2])>2:
                        valueMatrix[k,i],countMatrix[k,i] = [alphaVals[0][0]],[alphaVals[2][0]]
                        test2.append([i,k])
                        
                    if test_print == True and len(values) != len(index_weights):
                        print(i,k)
                        
                    valueTestMatrix[yAxisBoundaries[0]:yAxisBoundaries[1],xAxisBoundaries[0]:xAxisBoundaries[1]] = np.mean(alphaVals)
                                        
            else:
                if kapton_sample.shape[0]>0:         
                    if np.sum(kapton_sample)>0 and kapton_edge == False:
                        valueMatrix[k,i],countMatrix[k,i] = 100000,100000
                        kapton_edge = True
                                        
    return valueMatrix,countMatrix,voxIndexMatrix,valueTestMatrix,indexMatrix,indexRefMatrix,indexWeightMatrix



def data_label(data_labels,data):

    pbIdxs = np.unique(data_labels)
    if pbIdxs[0] == 0:
        pbIdxs = pbIdxs[1:len(pbIdxs)]
    
    bpIndexes = [np.where(data_labels == k) for k in pbIdxs]
    alphaValues = [[data[bpIndexes[j][0][k],bpIndexes[j][1][k]] for k in np.arange(0,len(bpIndexes[j][0]),1)] for j in np.arange(0,len(bpIndexes),1)]
    alphaMeans = np.asarray([np.mean(k) for k in alphaValues])
    if np.min(alphaMeans) == 0:
        alphaMeans[alphaMeans==0] = random.uniform(np.min(alphaMeans[alphaMeans>1])-0.1,np.min(alphaMeans[alphaMeans>1])+0.1)
    meanCounts = [(alphaMeans.tolist()).count(k) for k in alphaMeans]
    alphaMeans = [random.uniform(alphaMeans[k]-0.1, alphaMeans[k]+0.1) if meanCounts[k]>1 else alphaMeans[k] for k in np.arange(0,len(alphaMeans),1)]
    pbIdxs = pbIdxs+500#(round(np.max(alphaMeans))+1)
    data_labels[data_labels!=0] = data_labels[data_labels!=0]+500
    
    data_labels = [np.where(np.asarray(data_labels) == pbIdxs[k],alphaMeans[k],0) for k in np.arange(0,len(alphaMeans),1)]
            
    return np.sum(data_labels,0)

def clusters(image,thresh):
    h, w = image.shape

    # reshape to 1D array
    image_2d = image.reshape(h*w,1)

    # set number of colors
    #numcolors = int(round((np.max(np.unique(image[image>0]))-np.min(np.unique(image[image>0])))/thresh))
    numcolors = len(np.unique(np.round(np.unique(image[image>0])/thresh)*thresh))

    # do kmeans processing
    kmeans_cluster = cluster.KMeans(n_clusters=int(numcolors))
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # need to scale result back to range 0-255
    newimage = cluster_centers[cluster_labels].reshape(h, w)
    
    finalImage = np.where(image!=0,1,0)*newimage  

    #imageLabel = measure.label(finalImage*10)
    imageLabel = measure.label(finalImage)
    
    return finalImage,imageLabel


def grouper(iterable, iterRange):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= iterRange:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group
    
    return group


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


def dataFrameSave(data,outputFolder,filename,dataType = "lol",txtSave = "True"):
    
    folder = outputFolder+"/"+filename
        
    if os.path.isdir(folder) == False:
        os.mkdir(folder)
                            
    if len(data.shape) ==3:
        for i in range(0,data.shape[0]):
            dataSeg = data[i,:,:]
            np.save(folder+"/"+str(i), dataSeg)
            if txtSave == "True":
                df = pd.DataFrame(dataSeg)
                with pd.ExcelWriter(folder+"/"+str(i)+'.xlsx', engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                
    else:
        
        for rot in range(0,data.shape[0]):
            rotationFolder = folder+"/rotation "+str(rot) 
            if os.path.isdir(rotationFolder) == False:
                os.mkdir(rotationFolder)
            for slice_no in range(0,data.shape[1]):
                np.save(rotationFolder+"/slice "+str(slice_no), data[rot,slice_no])
                if txtSave == "True":
                    if dataType == "img":
                        dataSeg = data[rot,slice_no]
                        np.savetxt(rotationFolder+"/slice "+str(slice_no)+'.xlsx',dataSeg)
                    else:
                        dataSeg = data[rot,slice_no,:,:]
                        df = pd.DataFrame(dataSeg)
                        with pd.ExcelWriter(rotationFolder+"/slice "+str(slice_no)+'.xlsx', engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                            

#data,og_data,index_data,og_index = alpha_slices_mod[slice],alpha_slices[slice],index_slices_mod[slice],index_slices[slice]
                            
def index_comp(data_1,og_data_1,index_data,og_index,alpha_mod = False):

    list_test = []
    
    data = np.copy(data_1)
    og_data = np.copy(og_data_1)
    
    for j in range(0,index_data.shape[0]):
        for l in range(0,index_data.shape[1]):
            if type(index_data[j,l]) == list:
                if type(data[j,l]) != list:
                    print(j,l)
                    list_test.append([j,l])
    
    if len(list_test)>0:
        
        indexes = [index_data[k[0],k[1]] for k in list_test]
        index_locs = [[np.where(og_index == k) for k in i] for i in indexes]
        
        
        og_vals = [[[og_data[index_locs[i][k][0][l],index_locs[i][k][1][l]] for l in np.arange(0,len(index_locs[i][k][0]))] 
          for k in np.arange(0,len(index_locs[i]),1)] for i in np.arange(0,len(index_locs),1)]
        
        og_idxs = [[[og_index[index_locs[i][k][0][l],index_locs[i][k][1][l]] for l in np.arange(0,len(index_locs[i][k][0]))] 
          for k in np.arange(0,len(index_locs[i]),1)] for i in np.arange(0,len(index_locs),1)]
        
        min_vals =[np.argmin([len(k)for k in i]) for i in og_idxs]
        
        replace_vals = [og_idxs[i][min_vals[i]][0] for i in np.arange(0,len(og_idxs),1)]
        
        #og_vals = [[[og_vals[i][k][0]][0] for k in np.arange(0,len(og_vals[i]),1)] for i in np.arange(0,len(og_vals),1)]
        
        for idx,lt in enumerate(list_test):
            
            #data[lt[0],lt[1]] = replace_vals[idx]
            index_data[lt[0],lt[1]] = replace_vals[idx]
            
    for j in range(0,index_data.shape[0]):
        for l in range(0,index_data.shape[1]):
            if type(index_data[j,l]) != list:
                if type(data[j,l] == list):
                    data[j,l] = np.mean(data[j,l])
                    
    if alpha_mod == True:
        for j in range(0,index_data.shape[0]):
            for l in range(0,index_data.shape[1]):
                if index_data[j,l] != 0:
                    if data[j,l] == 0:
                        data[j,l] = 1
                        
    else:
        for j in range(0,index_data.shape[0]):
            for l in range(0,index_data.shape[1]):
                if index_data[j,l] != 0:
                    if data[j,l] == 0:
                        index_data[j,l] = 0
                        
    
    for j in range(0,index_data.shape[0]):
        for l in range(0,index_data.shape[1]):
            if index_data[j,l] == 0:
                if data[j,l] != 0:
                    data[j,l] = 0
            
    return data,index_data

def kapton_check(data):
    
    #yVals = np.asarray([np.where(data[:,k]>0)[0][0] if len(np.where(data[:,k]>0)[0])>0 else np.where(data[:,k-1]>0)[0][0] for k in np.arange(0,200,1)])
    
    yVals = []
    for k in np.arange(0,200,1):
        if len(np.where(data[:,k]>0)[0])>0:
            yVal = np.where(data[:,k]>0)[0][0]
        else:
            yVal = yVals[-1]
        yVals.append(yVal)
    
    cutoff = yVals[np.where(np.diff(yVals)>1)[0][0]]
    
    cutoff = np.where(np.diff(yVals)>1)[0][0]
    
    cutOffLocs = [[yVals[k],k] for k in np.arange(0,cutoff,1)]
    
    for loc in cutOffLocs:
        data[loc[0]-10:loc[0]+10,loc[1]] = 0
        
    return data

import pickle

"""
Begin script
"""                      
if __name__ == "__main__":
        
    """
    Collect folder and scanning parameters
    """
    with open(input_folder+"/registration_scan_info.pkl", "rb") as f:
        calib_paths = pickle.load(f)
        
    with open(input_folder+"/registration_info.pkl", "rb") as f:
        tomoSAXS_params = pickle.load(f)
    
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
    
    output_folder = calib_paths['-OUTFOLDER-']
    
    output_folder = output_folder.replace("w r","w_r")
    
    scan_angles = np.linspace(scan_start,scan_end,scan_count)
    
    saxsScale = int(calib_paths["-saxsScale-"])
    
    scan_angles = np.linspace(-85,95,9)
    
    print("######\nStarting tomoSAXS registration program for scan\n###### ",calib_paths["-SCANNAME-"])
    
    #tomosaxs_paths = selectFolders()["Subfolders"]
    #with open(input_folder+"/registration_scan_files.pkl", "rb") as f:
    tomosaxs_paths = np.load(input_folder+"/registration_scan_files.npy")
    
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
    ct_path = ct_path.replace("w r","w_r")
    waxs_path = calib_paths["-WAXSFILE-"]
    #kapton_path = calib_paths["-KAPTONFOLDER-"]
    kapton_path = '/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/calibrated/kapton'
    
    thickness_path = '/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/thickness_data/mask/fib_trac'
    
    output_path = calib_paths["-OUTFOLDER-"]
    output_path = output_path.replace("w r","w_r")
    script_path = calib_paths["-SCRIPTFOLDER-"]
    script_path = script_path.replace("w r","w_r")
    os.chdir(script_path+"/")
    #import tomoSAXS_outputs_and_figs as outp
    
    """
    Kapton files created in imageJ may be created at a set number of significant figures.
    So we need to isolate this formatting:
    """
    
    kapton_file = glob.glob(os.path.join(kapton_path, "*.tif"))[0]
    kapton_SF = len(kapton_file.split('/')[-1].split(".")[0])
        
    ct_img = cv2.cvtColor(cv2.imread(ct_path), cv2.COLOR_BGR2GRAY)
                
    top_vert_data = np.load(output_folder+"/vertical_calibration.npy")[0]
    
    top_vert_end_abs,ct_top_vert_end_abs = float(top_vert_data[1]),float(top_vert_data[2])
    ct_top_vert_end = ct_top_vert_end_abs/inv_ct_vox
    
    print("######\nUpper vertebral endpoint found in CT data; vertical registration finished\n######")
    
    """
    Once you have found the top vertebrae end-point, you can estimate the vertical offset between this point 
    and each of the tomoSAXS scan slices.  
    """
    
    print("######\nStarting padding of fibre tracing data and horizontal registration\n######")
    
    """
    Load axis coordinate data for first tomoSAXS scan orientation
    """
    
    SAMPLE_PATH = Path(tomosaxs_paths[0])
    
    with File(SAMPLE_PATH) as sample_file:
        entry = list(sample_file.keys())[0]
        if entry+"/SAXS/data" in sample_file:
            tomoSAXS_frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])
            tomoSAXS_xVals = array(sample_file[entry+"/SAXS_sum/base_x_value"])
            tomoSAXS_yVals = array(sample_file[entry+"/SAXS_sum/base_y_value"])
        sample_file.close()
    
    tomoSAXS_ycoords = tomoSAXS_yVals[:,0]
    
    """
    Find offset between coordinates and Upper vertebral endpoint (in um)
    """
    tomoSAXS_offsets = (tomoSAXS_ycoords - top_vert_end_abs)*1000
    
    saxs_vox = np.round(np.mean(np.diff(tomoSAXS_ycoords)),2)*1000
    
    fibtrac_top_vert_end = ct_top_vert_end/(fibre_vox/inv_ct_vox)
    
    fibtrac_tomoSAXS_slices = np.round(fibtrac_top_vert_end + (np.asarray(tomoSAXS_offsets)/fibre_vox)).astype(int)
    fibtrac_tomoSAXS_slice_starts = np.round(fibtrac_tomoSAXS_slices - ((saxs_vox/fibre_vox)/2)).astype(int)
    fibtrac_tomoSAXS_slice_ends = np.round(fibtrac_tomoSAXS_slices + ((saxs_vox/fibre_vox)/2)).astype(int)
    
    kapton_tomoSAXS_slice_starts = np.round(fibtrac_tomoSAXS_slice_starts*(fibre_vox/kapton_vox)).astype(int)
    kapton_tomoSAXS_slice_ends = np.round(fibtrac_tomoSAXS_slice_ends*(fibre_vox/kapton_vox)).astype(int)    
    
    ct_saxs_slice_img = np. copy(ct_img)
    
    ct_tomoSAXS_slices = ct_top_vert_end + (np.asarray(tomoSAXS_offsets)/inv_ct_vox)
    
    for k in ct_tomoSAXS_slices:
        ct_saxs_slice_img[int(k),:] = np.min(ct_img)/10
             
    figSave(ct_saxs_slice_img,output_path+"/CT_map_with_tomoSAXS_slices.png","tomoSAXS slices in CT data")
                
    """
    Now find corresponding fibre tracing data for the respective tomoSAXS scan
    will be in two seperate folders 
        - one for "theta" (i.e. alpha)
        - one of "phi" (i.e. beta)
    THIS IS WHY ITS IMPORTANT TO MAKE SURE THAT FILE NAMES SYNC FOR EACH OF THE DATA TYPES
    AND WHY THE SCAN NAME SET IN GUI HAS TO MATCH THESE
    """
    pad_path = calib_paths['-PADFILE-']
    pad_path = pad_path.replace("w r","w_r")
    pads =  pd.read_excel(pad_path)    
    
    index_folder = output_path+"/167208_fibreID_5x5 padded/calibrated"
    
    sample_fibtrac_folders = [calib_paths["-BETAFOLDER-"],calib_paths["-ALPHAFOLDER-"],index_folder]
    for idx,folder in enumerate(sample_fibtrac_folders):
        sample_fibtrac_folders[idx] = folder.replace("w r","w_r")
    
    scan_name = "FIVD1_unloaded_new"
    sample = [k for k in pads['sample'] if scan_name in k][0]
        
    if len(sample.split("_")) == 1:
        sample_name_elements = sample.split(" ")
    else:
        sample_name_elements = sample.split("_")
        
    sample_ID = sample_name_elements[0]
    if len(sample_ID) == 1:
        sample_ID = sample_name_elements[0]+sample_name_elements[1]
        
    #sample_fibtrac_folders = [k for k in fibtrac_folders if sample_ID in fibtrac_folders[0].split("_") or sample_ID in fibtrac_folders[0].split(" ")]
    
    if len(sample_fibtrac_folders) == 0:
        print("ABORTING: no fibre tracing data found for sample name")
        print("Please check sample name in fibre tracing padding excel file \n and use this name for registration scan name")
        sys.exit()
        
    else:
        
        """
        When/if these folders are found - use the values in the padding excel file for the respective sample
        to pad the fibre tracing volume so that it matches the original CT scan in ABSOLUTE size (not voxel count)
        and save to seperate folder
        """
        
        alpha_slices,beta_slices = [],[]
        og_alpha_slices,og_beta_slices = [],[]
        index_slices,og_index_slices = [],[]
        alpha_indx,og_alpha_indx = [],[]
        
        thickness_slices = []
        
        for folder in sample_fibtrac_folders:
            
            if "theta" in folder:
                if "calibrated" in folder:
                    print("######\nIsolating alpha fibre tracing data\n######")
                else:
                    print("######\nPadding alpha fibre tracing data to original CT dimensions\n######")
            elif "phi" in folder:
                if "calibrated" in folder:
                    print("######\nIsolating beta fibre tracing data\n######")
                else:
                    print("######\nPadding beta fibre tracing data to original CT dimensions\n######")
            
            fibre_data = []
            if "theta" in folder or "phi" in folder:
                if len(glob.glob(os.path.join(folder+"/", "*.tiff")))>0:
                    for sliceFile in np.sort(glob.glob(os.path.join(folder+"/", "*.tiff"))):
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.cvtColor(cv2.imread(sliceFile), cv2.COLOR_BGR2GRAY))
                else:
                    for sliceFile in np.sort(glob.glob(os.path.join(folder+"/", "*.tif"))):
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.cvtColor(cv2.imread(sliceFile), cv2.COLOR_BGR2GRAY))
            else:
                
                if len(glob.glob(os.path.join(folder+"/", "*.tiff")))>0:
                    for sliceFile in np.sort(glob.glob(os.path.join(folder+"/", "*.tiff"))):
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.imread(sliceFile,-1))
                else:
                    for sliceFile in np.sort(glob.glob(os.path.join(folder+"/", "*.tif"))):
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.imread(sliceFile,-1))
                                                       
            
            fibre_data = []
            
            if "theta" in folder or "phi" in folder:
                if len(glob.glob(os.path.join(folder+"/", "*.tiff")))>0:
                    fibre_data_slices = np.sort([int(k.split("/")[-1].split(".")[0]) for k in 
                                         glob.glob(os.path.join(folder+"/", "*.tiff"))])
                    
                    #for sliceFile in glob.glob(os.path.join(folder+"/", "*.tif")):
                    for data_slice in fibre_data_slices:
                        sliceFile = glob.glob(os.path.join(folder+"/", "*.tiff"))[np.where(np.asarray([int(k.split("/")[-1].split(".")[0]) for k in 
                                             glob.glob(os.path.join(folder+"/", "*.tiff"))]) == data_slice)[0][0]]
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.cvtColor(cv2.imread(sliceFile), cv2.COLOR_BGR2GRAY))
                    
                else:
                    fibre_data_slices = np.sort([int(k.split("/")[-1].split(".")[0]) for k in 
                                         glob.glob(os.path.join(folder+"/", "*.tif"))])
                
                    #for sliceFile in glob.glob(os.path.join(folder+"/", "*.tif")):
                    for data_slice in fibre_data_slices:
                        sliceFile = glob.glob(os.path.join(folder+"/", "*.tif"))[np.where(np.asarray([int(k.split("/")[-1].split(".")[0]) for k in 
                                             glob.glob(os.path.join(folder+"/", "*.tif"))]) == data_slice)[0][0]]
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.cvtColor(cv2.imread(sliceFile), cv2.COLOR_BGR2GRAY))
                        
            else:
                
                if len(glob.glob(os.path.join(folder+"/", "*.tiff")))>0:
                    fibre_data_slices = np.sort([int(k.split("/")[-1].split(".")[0]) for k in 
                                         glob.glob(os.path.join(folder+"/", "*.tiff"))])
                    
                    #for sliceFile in glob.glob(os.path.join(folder+"/", "*.tif")):
                    for data_slice in fibre_data_slices:
                        sliceFile = glob.glob(os.path.join(folder+"/", "*.tiff"))[np.where(np.asarray([int(k.split("/")[-1].split(".")[0]) for k in 
                                             glob.glob(os.path.join(folder+"/", "*.tiff"))]) == data_slice)[0][0]]
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.imread(sliceFile,-1))
                    
                else:
                    fibre_data_slices = np.sort([int(k.split("/")[-1].split(".")[0]) for k in 
                                         glob.glob(os.path.join(folder+"/", "*.tif"))])
                
                    #for sliceFile in glob.glob(os.path.join(folder+"/", "*.tif")):
                    for data_slice in fibre_data_slices:
                        sliceFile = glob.glob(os.path.join(folder+"/", "*.tif"))[np.where(np.asarray([int(k.split("/")[-1].split(".")[0]) for k in 
                                             glob.glob(os.path.join(folder+"/", "*.tif"))]) == data_slice)[0][0]]
                        #fibre_data.append(cv2.imread(sliceFile))                     
                        fibre_data.append(cv2.imread(sliceFile,-1))
                
            
            fibre_data = np.asarray(fibre_data)
            
            #xLength = pads['X axis (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]
            #yLength = pads['Y axis (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]
            #zLength = pads['Z axis (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]
            if "calibrated" not in folder:
                xStartPad = pads['x padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                yStartPad = pads['y padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                zStartPad = pads['z padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                
                og_padding = np.zeros((zStartPad,fibre_data.shape[1],fibre_data.shape[2]))
                og_data = np.concatenate([og_padding,fibre_data],0)
                
                xStartPadding = np.zeros((fibre_data.shape[0],fibre_data.shape[1],xStartPad,))               
                fibre_data = np.concatenate([xStartPadding,fibre_data],2)
                
                yStartPadding = np.zeros((fibre_data.shape[0],yStartPad,fibre_data.shape[2]))
                fibre_data = np.concatenate([yStartPadding,fibre_data],1)
                                
                zStartPadding = np.zeros((zStartPad,fibre_data.shape[1],fibre_data.shape[2]))
                fibre_data = np.concatenate([zStartPadding,fibre_data],0)
                
                xEndPad = pads['X bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                yEndPad = pads['Y bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                zEndPad = pads['Z bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
                
                og_end_Padding = np.zeros((zEndPad,og_data.shape[1],og_data.shape[2]))  
                og_data = np.concatenate([og_data,og_end_Padding],0)
                
                xEndPadding = np.zeros((fibre_data.shape[0],fibre_data.shape[1],xEndPad))               
                fibre_data = np.concatenate([fibre_data,xEndPadding],2)
                
                yEndPadding = np.zeros((fibre_data.shape[0],yEndPad,fibre_data.shape[2]))               
                fibre_data = np.concatenate([fibre_data,yEndPadding],1)
                
                zEndPadding = np.zeros((zEndPad,fibre_data.shape[1],fibre_data.shape[2]))               
                fibre_data = np.concatenate([fibre_data,zEndPadding],0)
            
            
                """
                If the padded data has not been saved in this location before - then save here
                """
                
                if os.path.isdir(folder+" padded/") == False:
                    os.mkdir(folder+" padded/")
                         
                for k in range(0,len(fibre_data)):
                    cv2.imwrite(folder+" padded/"+str(k)+".tiff",fibre_data[k].astype(np.uint8))
             
            """
            Now isolate the fibre tracing slices that represent each of the tomoSAXS slices,
            sum their values, and save 
            """
            
            fibtrac_top_vert_end = ct_top_vert_end/(fibre_vox/inv_ct_vox)
            
            fibtrac_tomoSAXS_slices = np.round(fibtrac_top_vert_end + (np.asarray(tomoSAXS_offsets)/fibre_vox)).astype(int)
            fibtrac_tomoSAXS_slice_starts = np.round(fibtrac_tomoSAXS_slices - ((saxs_vox/fibre_vox)/2)).astype(int)
            fibtrac_tomoSAXS_slice_ends = np.round(fibtrac_tomoSAXS_slices + ((saxs_vox/fibre_vox)/2)).astype(int)
            
            if os.path.isdir(output_path+"/original fibre tracing tomoSAXS slices/") == False:
                os.mkdir(output_path+"/original fibre tracing tomoSAXS slices/")
                os.mkdir(output_path+"/original fibre tracing tomoSAXS slices/alpha data/")
                os.mkdir(output_path+"/original fibre tracing tomoSAXS slices/beta data/")
                os.mkdir(output_path+"/original fibre tracing tomoSAXS slices/index data/")
                
                                                    
            i=0   
            tomoSAXS_slices = []
            kapton_slices = []
            #og_alpha_slices = []
            #og_beta_slices = []
            
            dataset = pd.DataFrame({'fibtrac slice starts': fibtrac_tomoSAXS_slice_starts, 'fibtrac slice ends': fibtrac_tomoSAXS_slice_ends,
                                    'CT slice starts': fibtrac_tomoSAXS_slice_starts*(fibre_vox/ct_vox), 
                                    'CT slice ends': fibtrac_tomoSAXS_slice_ends*(fibre_vox/ct_vox)}, 
                                   columns=['fibtrac slice starts', 'fibtrac slice ends','CT slice starts','CT slice ends'])
            
            dataset.to_excel(output_path+"/fibre tracing slice indexes.xlsx")
            
            #i = 0
            
            """
            POTENTIAL FOR PARALLELISING HERE
            """
            
            """
            TREAT DATA AS POINT CLOUDS THROUGH ROTATION NOT IMAGES - MIGHT AVOID INTERPOLATION ISSUE
            """
            
            for slice_start,slice_end in zip(fibtrac_tomoSAXS_slice_starts,fibtrac_tomoSAXS_slice_ends):
                
                print("######\nSaving fibre tracing data for tomoSAXS scan slice ",str(i),"\n######" )
                 
                if "theta" in folder or "phi" in folder:
                
                    slice_fib_data = np.asarray([[cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2GRAY) for k in glob.glob(os.path.join(folder+"/", "*.tiff")) 
                                     if int(k.split("/")[-1].split(".")[0]) == i][0] for i in np.arange(slice_start,slice_end+1,1)])
                    
                else:
                    
                    slice_fib_data = np.asarray([[cv2.imread(k,-1) for k in glob.glob(os.path.join(folder+"/", "*.tiff")) 
                                     if int(k.split("/")[-1].split(".")[0]) == i][0] for i in np.arange(slice_start,slice_end+1,1)])
                    
                thickness_data = np.asarray([[cv2.imread(k,-1) for k in glob.glob(os.path.join(thickness_path+"/", "*.tif")) 
                                 if int(k.split("/")[-1].split(".")[0]) == i][0] for i in np.arange(slice_start,slice_end+1,1)])
                
                thickness_data = np.sum(thickness_data,0)
                
                
                if folder == sample_fibtrac_folders[0]:
                
                    thickness_slices.append(thickness_data)
                
                sum_fib_data = np.sum(slice_fib_data,0)   
                #sum_og_data =  np.sum(og_data[slice_start:slice_end,:,:],0) 
                sum_og_data = np.copy(sum_fib_data)
                
                
                sum_fib_data = np.where(sum_fib_data!=0,1,0)
                sum_fib_test = np.zeros_like(sum_fib_data)
                
                for idx in range(0,len(slice_fib_data)):
                    if idx == 0:
                        sum_fib_test = sum_fib_test+slice_fib_data[idx]
                    else:
                        sum_fib_test = sum_fib_test+ (np.where(sum_fib_test!=0,0,1)*slice_fib_data[idx])
                
                
                #for j in range(0,sum_fib_data.shape[0]):
                    #for l in range(0,sum_fib_data.shape[1]):
                        #if sum_fib_data[j,l] !=0:
                            #if slice_fib_data[0][j,l] !=0:
                                #sum_fib_test[j,l] = slice_fib_data[0][j,l]
                            #elif slice_fib_data[0][j,l] ==0 and slice_fib_data[-1][j,l] !=0:
                                #sum_fib_test[j,l] = slice_fib_data[-1][j,l]
                                                            
                if "theta" in folder:
                    og_alpha_slices.append(sum_fib_test)
                elif "phi" in folder:
                    og_beta_slices.append(sum_fib_test)
                
                sum_fib_data = np.copy(sum_fib_test)  
                
                if folder == sample_fibtrac_folders[0]:
                                                           
                    og_index = measure.label(sum_fib_data) #index the fibres in the fibre tracing slice 
                    
                    if len(np.unique(og_index))>fib_thresh:#if the number of indexed fibres is above your threshold
                        
                        pruned = False
                        
                        #then keep removing every second fibre until the number is below the threshold
                        while pruned == False:
                            
                            og_index = measure.label(sum_fib_data)
                        
                            for idx in np.arange(0,len(np.unique(og_index)),2):
                                sum_fib_data = sum_fib_data*np.where(og_index==idx,0,1)
                                og_index = og_index*np.where(og_index==idx,0,1)
                                
                            if len(np.unique(og_index))<2000:
                                pruned = True
                                                                            
                    og_index = measure.label(sum_fib_data)
                    beta_index = rot_gridder(og_index,0)
                    
                    sum_fib_data = scatter_indexer(beta_index,measure.label(sum_fib_data),sum_fib_data)
                    
                    if len(sample_fibtrac_folders) <2:
                                                    
                        index_slices.append(beta_index)
                        og_index_slices.append(og_index)
                        
                    else:
                        alpha_indx.append(beta_index)
                        og_alpha_indx.append(og_index)
                
                else:
                    
                    if len(sample_fibtrac_folders)<2:
                    
                        sum_fib_data = scatter_indexer(index_slices[i],og_index_slices[i],sum_fib_data)
                        
                    else:
                        
                        sum_fib_data = scatter_indexer(alpha_indx[i],og_alpha_indx[i],sum_fib_data)
                                                                                                                   
                for j in range(0,sum_fib_data.shape[0],1):
                    for l in range(0,sum_fib_data.shape[1],1):
                        if sum_fib_data[j,l] != 0:
                            if type(sum_fib_data[j,l]) == list:
                                sum_fib_data[j,l] = [x for x in sum_fib_data[j,l] if str(x) != 'nan']
                                if len(sum_fib_data[j,l]) == 0:
                                    sum_fib_data[j,l] = [0]
                                print(sum_fib_data[j,l],np.mean(sum_fib_data[j,l]))
                            #print(sum_fib_data[j,l])
                            if str(sum_fib_data[j,l]) != "nan":
                                sum_fib_data[j,l] = int(np.mean(sum_fib_data[j,l]))
                            else:
                                sum_fib_data[j,l] = 0
                                
                #if len(sample_fibtrac_folders)>1 and folder == sample_fibtrac_folders[2]:
                    
                    #index_slices.append(sum_fib_data)
                    #og_index_slices.append(sum_fib_data)
                                
                
                if os.path.isdir(folder+"/z projections/") == False:
                    os.mkdir(folder+"/z projections/")
                if os.path.isdir(folder+"/z projections/") == False:
                    os.mkdir(folder+"/z projections/orig slices/")
                    
                np.save(folder+"/z projections/"+str(i)+".npy",sum_fib_data)
                if "theta" in folder or "phi" in folder:
                    cv2.imwrite(folder+"/z projections/"+str(i)+".tiff",sum_fib_data.astype(np.uint8))
                    cv2.imwrite(folder+"/z projections/"+str(slice_start)+"_"+str(slice_end)+".tiff",sum_fib_data.astype(np.uint8))
                else:
                    cv2.imwrite(folder+"/z projections/"+str(i)+".tiff",sum_fib_data.astype(np.uint8))
                    cv2.imwrite(folder+"/z projections/"+str(slice_start)+"_"+str(slice_end)+".tiff",sum_fib_data.astype(np.uint8))
                    
                    
                if os.path.isdir(folder+"/z projections/kapton/") == False:
                    os.mkdir(folder+"/z projections/kapton/")
                    
                kapton_data = np.sum([cv2.cvtColor(cv2.imread(kapton_path+"/"+str(k).zfill(kapton_SF)+".tif"), cv2.COLOR_BGR2GRAY) 
                                      for k in np.arange(kapton_tomoSAXS_slice_starts[i],kapton_tomoSAXS_slice_ends[i],1)],0)
                
                kapton_data[:,0:10] = 0
                kapton_data[:,-10:kapton_data.shape[1]] = 0
                
                #kapton_data = rotate(kapton_data,20,reshape=False,mode='nearest',order=0)
                                                
                kapton_slices.append(kapton_data)
                
                cv2.imwrite(folder+" padded/z projections/kapton/"+str(i)+".tiff",kapton_data.astype(np.uint8))
                
                #i=i+1
                
                tomoSAXS_slices.append(sum_fib_data)  
                
                if "theta" in folder:
                    cv2.imwrite(output_path+"/original fibre tracing tomoSAXS slices/alpha data/"+str(i)+".tiff",sum_og_data.astype(np.uint8))                
                    figSave(sum_og_data,output_path+"/Example alpha fibre tracing tomoSAXS slice "+str(i)+".png","Example alpha fibre tracing tomoSAXS slice "+str(i))
                    
                elif "phi" in folder:
                    cv2.imwrite(output_path+"/original fibre tracing tomoSAXS slices/beta data/"+str(i)+".tiff",sum_og_data.astype(np.uint8))                
                    figSave(sum_og_data,output_path+"/Example beta fibre tracing tomoSAXS slice "+str(i)+".png","Example beta fibre tracing tomoSAXS slice "+str(i))
                
                else:
                    cv2.imwrite(output_path+"/original fibre tracing tomoSAXS slices/index data/"+str(i)+".tif",sum_og_data)                
                    figSave(sum_og_data,output_path+"/Example index fibre tracing tomoSAXS slice "+str(i)+".png","Example index fibre tracing tomoSAXS slice "+str(i))
                
                i = i+1
                
            
                                            
            if len(sample_fibtrac_folders)<2:
            
                if "theta" in folder:
                    alpha_slices = np.copy(tomoSAXS_slices)
                                   
                elif "phi" in folder:
                    beta_slices = np.copy(tomoSAXS_slices)
                    
            else:
                
                if "theta" in folder:
                    alpha_slices = np.copy(tomoSAXS_slices)
                                   
                elif "phi" in folder:
                    beta_slices = np.copy(tomoSAXS_slices)
                    
                else:
                    index_slices = np.copy(tomoSAXS_slices)
                    #og_index_slices.append(sum_fib_data)
                
        if scan_angles[0]!=-90:
            
            print("########")
            print("adjusting scan angle")
            print("########")
            
            alpha_slices_mod = []
            beta_slices_mod = []
            index_slices_mod = []            
            thickness_slices_mod = []
            
            for slice in range(0,len(alpha_slices)):
                print("########")
                print("adjusting slice ",str(slice))
                print("########")
                alpha_slices_mod.append(rot_gridder(alpha_slices[slice],scan_angles[0] - (-90)))
                beta_slices_mod.append(rot_gridder(beta_slices[slice],scan_angles[0] - (-90)))
                index_slices_mod.append(rot_gridder(index_slices[slice],scan_angles[0] - (-90)))
                thickness_slices_mod.append(rot_gridder(thickness_slices[slice],scan_angles[0] - (-90)))
                thickness_mod = rot_gridder(thickness_slices[slice],scan_angles[0] - (-90))
                if thickness_mod.shape[0]>alpha_slices_mod[slice].shape[0]:                    
                    thickness_mod = cv2.resize(thickness_mod.astype(np.uint8),(index_slices_mod[slice].shape[0],index_slices_mod[slice].shape[1]))
                thickness_slices_mod.append(thickness_mod)
                
                
            alpha_slices_save = np.copy(alpha_slices_mod)
            beta_slices_save = np.copy(beta_slices_mod)
            index_slices_save = np.copy(index_slices_mod)
            thickness_slices_save = np.copy(thickness_slices_mod)
            
            alpha_test = []
            beta_test = []
            index_test = []
            
            for slice in range(0,len(alpha_slices_mod)):
                
                alpha_results,index_results = index_comp(alpha_slices_mod[slice],alpha_slices[slice],
                                             index_slices_mod[slice],index_slices[slice],alpha_mod = True)
                
                beta_results = index_comp(beta_slices_mod[slice],beta_slices[slice],
                                             index_slices_mod[slice],index_slices[slice])[0]
                
                beta_results,alpha_results = index_comp(beta_results,beta_slices[slice],
                                             alpha_results,alpha_slices[slice])
                
                index_test.append(index_results) 
                
                alpha_test.append(alpha_results)
                
                beta_test.append(beta_results)
                
            alpha_slices = np.copy(alpha_test)
            beta_slices = np.copy(beta_test)
            index_slices = np.copy(index_test)
            thickness_slices = np.copy(thickness_slices_mod)
            
                                            
        ct_zAxis_SAXS_voxels = np.arange(0,fibre_data.shape[0],(saxs_vox/fibre_vox))
        ct_xAxis_SAXS_voxels = np.arange(0,np.min([fibre_data.shape[1],fibre_data.shape[2]]),(saxs_vox/fibre_vox))
        ct_yAxis_SAXS_voxels = np.arange(0,np.min([fibre_data.shape[1],fibre_data.shape[2]]),(saxs_vox/fibre_vox))                                                                                                                
               
        """
        Now we can process each of the fibre tracing slices that correspond to tomoSAXS scan slices:
            specifically - for each tomoSAXS slice:
                a. Segment outer edge of corresponding kapton slice (outer edge has highest scatter in SAXS data).
                b. Rotate corresponding alpha slice, beta slice, and kapton outer edge slice to match each 
                    tomoSAXS rotation.
                c. For each rotation: 
                    i. Use "nsphere_fit" (fits circle to 2D data using coope method) to fit compolete 
                        circle to kapton data.
                    ii. Find position of lhs edge of the kapton orthagonal to the rotation.
                    iii. Load SAXS sum scattering data for corresponding tomoSAXS scan orientation.
                    iv. Find posiotion of lhs kapton edge in sum SAXS data
                    v. This gives horizontal offset between the CT and SAXS data
                    vi. You can then adjust (by padding) the CT data to remove this offset and match the SAXS data
        """

        slice_no = 0
        slice_thicknesses = []
        slice_densities = []
        #for alpha_slice,beta_slice,kapton_slice in zip(alpha_slices,beta_slices,kapton_slices):
                                    
        """
        Find and isolate inner kepton edge
        """
        
        print("######\nEstimating kapton edge offset at start and midpoint of tomoSAXS scan\n######")
        
        bit_slices = np.copy(alpha_slices)
        bit_slices[bit_slices !=0] = 1
        
        alpha_slice = alpha_slices[0]
        beta_slice = beta_slices[0]
        bit_slice = bit_slices[0]
        
        kapton_slice = np.copy(image_resize(kapton_slices[0].astype(np.uint8), width = alpha_slice.shape[0], height = alpha_slice.shape[1], inter = cv2.INTER_AREA))
        kapton_slice[:,0] = 0
        kapton_slice[:,-1] = 0
        kapton_slice[-1,:] = 0
        kapton_slice[0,:] = 0
                    
        edges = cv2.Canny(image=kapton_slice, threshold1=100, threshold2=200)
        
        first_edge = [[k,np.where(edges[k,:]>0)[0][0]] for k in np.arange(0,edges.shape[0],1) if len(np.where(edges[k,:]>0)[0])>0]
        
        for edge in first_edge:
            #print(edge)
            edges[edge[0],edge[1]+1:-1] = 0
        
        kapton_edge_img = edges
        
        if scan_angles[0]!=-90:
            
            kapton_edge_img = rotate(kapton_edge_img,scan_angles[0] - (-90),reshape=False,mode='nearest',order=0)
        
        if os.path.isdir(output_path+"/tomoSAXS kapton segmentations/") == False:
            os.mkdir(output_path+"/tomoSAXS kapton segmentations/")
        
        figSave(kapton_slice,output_path+"/tomoSAXS kapton segmentations"+"/TomoSAXS slice "+str(slice_no)+" kapton segmentation.png","TomoSAXS slice "+str(slice_no)+" kapton segmentation")
        figSave(kapton_edge_img,output_path+"/tomoSAXS kapton segmentations"+"/TomoSAXS slice "+str(slice_no)+" kapton edge.png","TomoSAXS slice "+str(slice_no)+" kapton edge")
        
        kapton_coords = np.where(kapton_edge_img>0)
        data = [([kapton_coords[0][k],kapton_coords[1][k]]) for k in np.arange(0,len(kapton_coords[0]),1)]
        data = np.asarray(data)
        
        r, c = nsphere_fit(data)  
        #r = (kapt_w/saxs_vox)/2
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=True)                    
        t1,t2 = r * np.cos(t) + c[0], r * np.sin(t) + c[1]
        
        kapton_edge_mod = np.copy(kapton_edge_img)
        for tx,ty in zip(t1,t2):
            if np.max([tx,ty])<np.min(kapton_edge_mod.shape):
                kapton_edge_mod[int(tx),int(ty)] = np.max(kapton_edge_img)
                
        figSave(kapton_edge_mod,output_path+"/tomoSAXS kapton segmentations"+"/TomoSAXS slice "+str(slice_no)+" kapton model.png","TomoSAXS slice "+str(slice_no)+" kapton model")
        
        kapton_0_deg = np.copy(kapton_edge_img)

        #if scan_angles[0]!=-90:
            
            #kapton_0_deg = rotate(kapton_0_deg,scan_angles[0] - (-90),reshape=False,mode='nearest',order=0)
            
            #kapton_coords = np.where(kapton_0_deg>0)
            #data = [([kapton_coords[0][k],kapton_coords[1][k]]) for k in np.arange(0,len(kapton_coords[0]),1)]
            #data = np.asarray(data)
                            
        
        if np.min(data[:,0]) < kapton_0_deg.shape[0]/2:
            if np.abs(np.min(data[:,1]) - np.min(t2))<saxs_vox:
                kapton_x_edge = np.max([np.min(t2),np.min(data[:,1])])
            else:
                kapton_x_edge = np.min(t2)
        else:
            kapton_x_edge = np.min(t2)
        if np.min(data[0,:]) < kapton_0_deg.shape[1]/2:
            if np.abs(np.min(data[:,0]) - np.min(t1))<saxs_vox:            
                kapton_y_edge = np.max([np.min(t1),np.min(data[:,0])])
            else:
                kapton_y_edge = np.min(t1)
        else:
            kapton_y_edge = np.min(t1)
            
        if kapton_y_edge == 0:
            kapton_y_edge = 1
        if kapton_x_edge == 0:
            kapton_x_edge = 1    
        
        ct_kapton_x_edge = kapton_x_edge*fibre_vox
        ct_kapton_y_edge = kapton_y_edge*fibre_vox
        
                    
        """
        Load midpoint SAXS scan sum intensity data
        """
        
        SAMPLE_PATH = Path(tomosaxs_paths[int(len(tomosaxs_paths)/2)])
        
        with File(SAMPLE_PATH) as sample_file:
            entry = list(sample_file.keys())[0]
            if entry+"/SAXS/data" in sample_file:
                scan_frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])
                scan_xVals = array(sample_file[entry+"/SAXS_sum/base_x_value"])
                scan_yVals = array(sample_file[entry+"/SAXS_sum/base_y_value"])
            sample_file.close()
        
        """
        Find coordinate location of lhs kapton edge in the first SAXS scan
        """
        x_scan_slice_sums = scan_frames_sum[slice_no,:]
        
        if len(np.where(x_scan_slice_sums>np.abs(x_scan_slice_sums[0])*10)[0])>0:
            first_x_kapton = np.where(x_scan_slice_sums>np.abs(x_scan_slice_sums[0])*10)[0][0]
            if first_x_kapton>len(x_scan_slice_sums)/2:
                first_x_kapton = int(first_x_kapton - (int(calib_paths["-tube-"])/saxs_vox))
                if first_x_kapton<0:
                    first_x_kapton = 0
        else:
            if np.min(x_scan_slice_sums[0:50])>0:
                first_x_kapton = 0
            else:
                first_frame = np.where(x_scan_slice_sums<0)[0][0]
                first_x_kapton = np.where(x_scan_slice_sums[first_frame:-1]>np.abs(x_scan_slice_sums[first_frame])*10)[0][0]
                
        saxs_kapton_x_edge = (scan_xVals[slice_no,first_x_kapton] - scan_xVals[slice_no,0])*1000
        
        
        
        """
        Load first SAXS scan sum intensity data
        """
        
        SAMPLE_PATH = Path(tomosaxs_paths[0])
        
        with File(SAMPLE_PATH) as sample_file:
            entry = list(sample_file.keys())[0]
            if entry+"/SAXS/data" in sample_file:
                scan_frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])
                scan_xVals = array(sample_file[entry+"/SAXS_sum/base_x_value"])
                scan_yVals = array(sample_file[entry+"/SAXS_sum/base_y_value"])
            sample_file.close()
        
        """
        Find coordinate location of lhs kapton edge in the first SAXS scan
        """
        y_scan_slice_sums = scan_frames_sum[slice_no,:]
                
        if len(np.where(y_scan_slice_sums>np.abs(y_scan_slice_sums[0])*10)[0])>0:
            first_y_kapton = np.where(y_scan_slice_sums>np.abs(y_scan_slice_sums[0])*10)[0][0]
            if first_y_kapton>len(y_scan_slice_sums)/2:
                first_y_kapton = int(first_y_kapton - (int(calib_paths["-tube-"])/saxs_vox))
                if first_y_kapton<0:
                    first_y_kapton = 0
                
        else:
            if np.min(y_scan_slice_sums[0:50])>0:
                first_y_kapton = 0
            else:
                first_frame = np.where(y_scan_slice_sums<0)[0][0]
                first_y_kapton = np.where(y_scan_slice_sums[first_frame:-1]>np.abs(y_scan_slice_sums[first_frame])*10)[0][0]
        saxs_kapton_y_edge = (scan_xVals[slice_no,first_y_kapton] - scan_xVals[slice_no,0])*1000
        
        """
        Estimate offset between kapton edge in rotated SAXS vs rotated CT data  
        """
                        
        kapton_x_offset = saxs_kapton_x_edge - ct_kapton_x_edge
        kapton_y_offset = saxs_kapton_y_edge - ct_kapton_y_edge
        
        saxs_ct_kapton_x_edge = first_x_kapton - int(kapton_x_offset/saxs_vox)
        saxs_ct_kapton_y_edge = first_y_kapton - int(kapton_y_offset/saxs_vox)
        
        x_offset_vox = int(np.round(kapton_x_offset/fibre_vox))
        y_offset_vox = int(np.round(kapton_y_offset/fibre_vox))
        
        print("######\nX-axis offset = ",str(kapton_x_offset)," um\n######" )
        print("######\nY-axis offset = ",str(kapton_y_offset)," um\n######" )
        
        if os.path.isdir(output_path+"/tomoSAXS 90 degrees kapton edges/") == False:
            os.mkdir(output_path+"/tomoSAXS 90 degrees kapton edges/")
        
        plt.plot(x_scan_slice_sums)
        plt.scatter(first_x_kapton,x_scan_slice_sums[first_x_kapton],color = "b",label = "TomoSAXS slice "+str(slice_no)+
                    " Detected SAXS outer \n0 degrees lhs edge of kapton tube")
        plt.scatter(saxs_ct_kapton_x_edge,x_scan_slice_sums[first_x_kapton],color = "r",label = "TomoSAXS slice "+str(slice_no)+
                    " Detected CT outer \n0 degrees lhs edge of kapton tube")
        plt.legend()            
        plt.savefig(output_path+"/tomoSAXS 90 degrees kapton edges/"+"0 degrees edge comparison.png")
        plt.show()
        plt.close()
        
        plt.plot(y_scan_slice_sums)
        plt.scatter(first_y_kapton,y_scan_slice_sums[first_y_kapton],color = "b",label = "TomoSAXS slice "+str(slice_no)+
                    " Detected SAXS outer \n-90 degrees lhs edge of kapton tube")
        plt.scatter(saxs_ct_kapton_y_edge,y_scan_slice_sums[first_y_kapton],color = "r",label = "TomoSAXS slice "+str(slice_no)+
                    " Detected CT outer \n-90 degrees lhs edge of kapton tube")
        plt.legend()            
        plt.savefig(output_path+"/tomoSAXS 90 degrees kapton edges/"+"-90 degrees edge comparison.png")
        plt.show()
        plt.close()
                        
        """
        Pad CT data accordingly 
            Find offset along x and y axes of midpoint scan
            pad accordingly
            find new voxels to sample over that match the tomoSAXS beampaths
        """
        
        print("######\nPadding fibre tracing data\n######" )
        
        """
        abs value of offset in fibre tracing voxels
        """
                                            
        SAXS_CT_axis0_start = int(((saxs_kapton_x_edge)/fibre_vox) - kapton_x_edge)      
        SAXS_CT_offset_axis0 = int(np.sqrt(SAXS_CT_axis0_start**2))

        SAXS_CT_axis1_start = int(((saxs_kapton_y_edge)/fibre_vox) - kapton_y_edge)      
        SAXS_CT_offset_axis1 = int(np.sqrt(SAXS_CT_axis1_start**2))

        saxsAxis0Grid = x_scan_slice_sums
        saxsAxis1Grid = y_scan_slice_sums 
        
        """
        Create new map - the size of the tomoSAXS data 
                         (i.e. length of tomoSAXS slice in fibre tracing voxels)
                         plus the abs offset amaounts
                         
        Then - copy original fibre tracing data into new map, starting at the abs offset values 
        """
        
        newKaptonMap = np.zeros((int(len(saxsAxis1Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis1,int(len(saxsAxis0Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis0))        
        if int(SAXS_CT_offset_axis1+kapton_edge_img.shape[0])> newKaptonMap.shape[0] or int(SAXS_CT_offset_axis0+kapton_edge_img.shape[1])>newKaptonMap.shape[1]:
            newKaptonMap = np.zeros((int(SAXS_CT_offset_axis1+kapton_edge_img.shape[0]),int(SAXS_CT_offset_axis0+kapton_edge_img.shape[1])))
        newKaptonMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+kapton_edge_img.shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+kapton_edge_img.shape[1])] = kapton_edge_mod
                    
        newAlphaMaps = [] 
        newBetaMaps = [] 
        bitMaps = []
        newIndexMaps = []
        newThicknessMaps = []

        for k in range(0,len(alpha_slices)):
            newAlphaMap = np.zeros((int(len(saxsAxis1Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis1,int(len(saxsAxis0Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis0))
            
            if int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0])>newAlphaMap.shape[0] or int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])>newAlphaMap.shape[1]:
                   newAlphaMap = np.zeros((int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])))
            
            newBetaMap,newIndexMap,newBitMap = np.copy(newAlphaMap),np.copy(newAlphaMap),np.copy(newAlphaMap)
            newThicknessMap = np.copy(newAlphaMap)
            newAlphaMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])] = alpha_slices[k]
            newBetaMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])] = beta_slices[k]                       
            newBitMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])] = bit_slices[k]
            newIndexMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])] = index_slices[k] 
            
            thickness_test = np.copy(thickness_slices[k])
            for j in range(0,thickness_test.shape[0]):
                for l in range(0,thickness_test.shape[1]):
                    if thickness_test[j,l] !=0:
                        thickness_test[j,l] = 1
            
            newThicknessMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alpha_slices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alpha_slices[k].shape[1])] = thickness_test 
            
            newAlphaMaps.append(newAlphaMap)
            newBetaMaps.append(newBetaMap)
            bitMaps.append(newBitMap)
            newIndexMaps.append(newIndexMap)
            newThicknessMaps.append(newThicknessMap)
               
        """
        Find position of sample in new map
        """
        
        alphaAxis1 = [np.sum(bitMaps[k],axis=1) for k in np.linspace(0,len(alpha_slices)-1,len(alpha_slices)).astype(int)]
        alphaAxis2 = [np.sum(bitMaps[k],axis=0) for k in np.linspace(0,len(alpha_slices)-1,len(alpha_slices)).astype(int)]
           
        ctAxis0 = alphaAxis2[0]
        ctAxis1 = alphaAxis1[0]
        
        ctAxis0Start,ctAxis0End = np.where(ctAxis0>0)[0][0],np.where(ctAxis0>0)[0][-1]

        sampleWidth = np.min([alpha_slice.shape[0],alpha_slice.shape[1]])
        
        """
        Calculate where in the new map you need to sample that corresponds to the tomoSAXS data
        """

        if SAXS_CT_axis0_start<0 and SAXS_CT_offset_axis0 > saxs_vox/fibre_vox:
            
            """
            IF the x-axis kapton edge in CT is to the right of the edge in the 90 degree tomoSAXS data,
            (i.e. SAXS_CT_axis0_start<0), and that difference is more than the size of a tomoSAXS voxel - 
            
                then you need to double the abs value of the offset (in fibre tracing voxels) and start 
                sampling this axis from this point
            """
            
            saxs_axis0_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis0*2)*fibre_vox)/saxs_vox)))).tolist()
            saxs_axis0_test = [saxs_axis0_Addon+saxsAxis0Grid.tolist()][0]
            xAxis0 = np.linspace(0,100*0.05,len(saxs_axis0_test))
            saxsStart = len(saxs_axis0_Addon)*round(saxs_vox/fibre_vox)
            #ctaxis0_SAXS_voxels = np.linspace(saxsStart,saxsStart+((round(saxs_vox/fibre_vox)*(sampleWidth))),sampleWidth+1).astype(int)
            
            ctaxis0_SAXS_voxels = np.round(np.arange(saxsStart,int(saxsStart+((len(saxsAxis0Grid)*saxs_vox)/fibre_vox)),saxs_vox/fibre_vox)).astype(int)
            
            ctaxis0_SAXS_midPoints = ctaxis0_SAXS_voxels + round(round(saxs_vox/fibre_vox)/2)
        elif SAXS_CT_axis0_start>0 and SAXS_CT_offset_axis0 > saxs_vox/fibre_vox:
            
            """
            ELSE IF the x-axis kapton edge in CT is to the LEFT of the edge in the 90 degree tomoSAXS data,
            (i.e. SAXS_CT_axis0_start>0), and that difference is more than the size of a tomoSAXS voxel - 
            
                then you have already adjusted the offset along this axis in the new map so you start sampling at 0
                along this axis
            """
            
            saxs_axis0_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis0)*fibre_vox)/saxs_vox)))).tolist()
            saxs_axis0_test = [saxsAxis0Grid.tolist()+saxs_axis0_Addon][0]
            xAxis0 = np.linspace(0,100*0.05,len(saxs_axis0_test))        
            #ctaxis0_SAXS_voxels = np.linspace(0,(round(saxs_vox/fibre_vox)*(sampleWidth)),sampleWidth+1).astype(int)
            ctaxis0_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
        else:
            #ctaxis0_SAXS_voxels = np.linspace(0,(round(saxs_vox/fibre_vox)*(sampleWidth)),sampleWidth+1).astype(int)
            ctaxis0_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
         
        
        """
        repeat for midpoint tomoSAXS scan axis
        """
        if SAXS_CT_axis1_start<0 and SAXS_CT_offset_axis1 > saxs_vox/fibre_vox:
            saxs_axis1_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis1*2)*fibre_vox)/saxs_vox)))).tolist()
            saxs_axis1_test = [saxs_axis1_Addon+saxsAxis1Grid.tolist()][0]
            xAxis1 = np.linspace(0,100*0.05,len(saxs_axis1_test))        
            saxsStart = len(saxs_axis1_Addon)*round(saxs_vox/fibre_vox)
            #ctaxis1_SAXS_voxels = np.linspace(saxsStart,saxsStart+((round(saxs_vox/fibre_vox)*(sampleWidth))),sampleWidth+1).astype(int) 
            ctaxis1_SAXS_voxels = np.round(np.arange(saxsStart,saxsStart+((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
            
        elif SAXS_CT_axis1_start>0 and SAXS_CT_offset_axis1 > saxs_vox/fibre_vox:
            saxs_axis1_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis1)*fibre_vox)/saxs_vox)))).tolist()
            saxs_axis1_test = [saxsAxis1Grid.tolist()+saxs_axis1_Addon][0]
            xAxis1 = np.linspace(0,100*0.05,len(saxs_axis1_test))
            #ctaxis1_SAXS_voxels = np.linspace(0,(round(saxs_vox/fibre_vox)*(sampleWidth)),sampleWidth+1).astype(int)
            ctaxis1_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
        else:
            #ctaxis1_SAXS_voxels = np.linspace(0,(round(saxs_vox/fibre_vox)*(sampleWidth)),sampleWidth+1).astype(int)
            ctaxis1_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
        
        voxelSize = ctaxis0_SAXS_voxels[1] - ctaxis0_SAXS_voxels[0]
                                
        kaptonData = newKaptonMap[ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1],ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1]]
        if kaptonData.shape[0] > ctaxis1_SAXS_voxels[-1]:
            addOn = np.zeros((kaptonData.shape[0],(kaptonData.shape[1] - ctaxis1_SAXS_voxels[-1])*2))            
            kaptonData = np.concatenate((kaptonData,addOn),axis=1)
            addOn2 = np.zeros(((kaptonData.shape[1] - ctaxis0_SAXS_voxels[-1])*2,kaptonData.shape[1]))            
            kaptonData = np.concatenate((kaptonData,addOn2),axis=0)
        #ctaxis1_SAXS_midPoints = ctaxis1_SAXS_voxels + (round(round(saxs_vox/fibre_vox)/2)*saxsScale)
        
        ctaxis0_SAXS_midPoints = np.round(ctaxis0_SAXS_voxels+((saxs_vox/fibre_vox)/2)*saxsScale).astype(int)
        ctaxis1_SAXS_midPoints = np.round(ctaxis1_SAXS_voxels+((saxs_vox/fibre_vox)/2)*saxsScale).astype(int)
        
        scan_params =[sampleWidth,saxs_vox,fibre_vox,saxsScale]
        voxel_params = [ctaxis1_SAXS_voxels[ctaxis1_SAXS_voxels<newAlphaMaps[0].shape[0]],
                        ctaxis1_SAXS_midPoints[ctaxis1_SAXS_midPoints<newAlphaMaps[0].shape[0]],
                        ctaxis0_SAXS_voxels[ctaxis0_SAXS_voxels<newAlphaMaps[0].shape[1]],
                        ctaxis0_SAXS_midPoints[ctaxis0_SAXS_midPoints<newAlphaMaps[0].shape[1]]]
                                
        """
        Subsample the padded fibre tracing data and register which tomoSAXS voxels individual fibres belong to.
        """
        
        alphaData = []
        betaData = []
        indexData = []
        thicknessData = []
                                
        print("######\nIdentifying per-tomoSAXS voxel fibre information\n######" )
        
        for slice in range(0,len(newAlphaMaps)):
            
            print("######\nIdentifying per-tomoSAXS voxel fibre information for tomoSAXS slice "+str(slice)+"\n######" )
                                
            if len(sample_fibtrac_folders)<2:
            
                abResults = ab_gridder(newAlphaMaps[slice],newBetaMaps[slice],scan_params,voxel_params)
                
                alphaResults = abResults[0]
                betaResults = abResults[1]
                
            else:
                
                alphaResults = newAlphaMaps[slice]
                betaResults = newBetaMaps[slice]
                                            
            alphaData.append(alphaResults[ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1],ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1]])
            betaData.append(betaResults[ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1],ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1]])
            indexData.append(newIndexMaps[slice][ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1],ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1]])
            thicknessData.append(newThicknessMaps[slice][ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1],ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1]])
        
        bitmapData = np.copy(alphaData)
        bitmapData[bitmapData !=0] = 1
                        
        fibreRotMaps = []
        fibreRotMaps.append(alphaData)
        
        print("######\nMidpoint registration completed,\nregistering full tomoSAXS scan\n######" )
        
        
        alphaIndexes = []
        betaIndexes = []
        rotationIndexes = []
        fibreIndexes = []
        fibreCounts = []
        fibreIndexData = []
        weightIndexData = []
        thicknessIndexes = []
        
        sampleThicknesses,sampleDensities = [],[]
        
        rot_alpha_maps,rot_beta_maps,rot_bin_maps,rot_thickness_maps = [],[],[],[]
        rot_index_maps = []
        
        """
        Now we can rotate the registered data to each tomoSAXS scan orientation,
            then caluclated the offset between the lhs outer kaptom edge in CT vs SAXS data,
            pad in the same way as above, 
            sample across the equivalent space in fibre tracing data to tomoSAXS beampaths,
            register fibre tracing data for each tomoSAXS voxel across the beampath
            
            these creates:
                a. "alphaIndex": map of fibre alpha values in tomoSAXS voxels - each voxel containing lists of alpha values for 
                   fibres within the respective voxel
                b. "alphaCount": map of fibre beta values in tomoSAXS voxels - each voxel containing lists of fibre tracing voxels 
                   that individual fibres comprise of within the respective voxel 
                c. "voxIndexes": map of indexed fibres in fibre tracing voxels, with each fibre given a greyscale value equal to its index
                d. "rotIndex": map of indexed fibre tracing voxels that are sumsapled within each tomoSAXS voxel
                e. "fibreIndex": map of fibre index values in tomoSAXS voxels - each voxel containing lists of index values for 
                   fibres within the respective voxel
                f. "voxRefIndex": map of indexed tomoSAXS voxels
                g. "betaIndex": map of fibre alpha values in tomoSAXS voxels - each voxel containing lists of alpha values for 
                   fibres within the respective voxel
        """
        
        if scan_angles[0]!=-90:
            
            angle_adjust = scan_angles[0] - (-90)
            
            
            
        """
        CAN PARALLELISE THIS ROTATION STEP TOO
        """
                                
        for rot in range(0,len(scan_angles)):
        #for rot in range(0,1):
            
            print("######\nRegistering tomoSAXS angle ",str(scan_angles[rot]),"\n######" )
            
            test_rot_save = [rot]
            np.save(input_folder+"/"+str(rot)+".npy",test_rot_save)
            
            rot_angle = map_rot - scan_angles[rot] + angle_adjust
            
            #kapton_rot = rotate(kaptonData,map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0)
            kapton_rot = rotate(kaptonData,rot_angle,reshape=False,mode='nearest',order=0)
            
            if np.argmax(kapton_rot[:,0])>0 and np.argmax(kapton_rot[:,0])<(kapton_rot.shape[0]/3):
                
                kapton_rot = kapton_check(kapton_rot)
                
                if np.argmax(kapton_rot[:,0])>0:
                    
                    kapton_rot = kapton_check(kapton_rot)
            
            alphaRotSlices = []
            betaRotSlices = []
            binRotSlices = []
            indexSlices = []
            rotSlices = []
            thicknessSlices = []
            
            
            """
            Rotate the original registered alpha, beta and bitmap data 
            IF this is the first rotation in the scan:
                use alpha data as fibre index data
            ELIF this is not the first rotation: 
                fibre index data has already been created
                so use the fibre index dataset for the first rotation 
                for every subsequent rotation                
            
            WE ARE LOOSING INDEXES DURING ROTATION - NEED TO FIX
            
            """
            
            for slice in range(0,len(alphaData)):    
                print("######\nRotating slice ",str(slice),"\n######")                
                if rot == 0:
                    #alphaRotSlices.append(rotate(alphaData[slice],map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0))
                    alphaRotSlices.append(rot_gridder(alphaData[slice],rot_angle))
                    #betaRotSlices.append(rotate(betaData[slice],map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0))
                    betaRotSlices.append(rot_gridder(betaData[slice],rot_angle))
                    #binRotSlice = np.copy(betaRotSlices[-1])
                    #binRotSlice[binRotSlice!=0] = 1
                    #binRotSlices.append(rotate(bitmapData[slice],map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0))
                    binRotSlices.append(rot_gridder(bitmapData[slice],rot_angle))
                    #indexSlices.append(rotate(alphaData[slice],map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0))
                    #indexSlices.append(rot_gridder(alphaData[slice],map_rot - scan_angles[rot]))
                    if len(sample_fibtrac_folders)<2:
                        indexSlices.append(alphaRotSlices[slice])
                    else:
                        indexSlices.append(rot_gridder(indexData[slice],rot_angle))
                        
                    #rotSlices.append(rotate(alphaData[slice],map_rot - scan_angles[rot],reshape=False,mode='nearest',order=0))
                    #rotSlices.append(rot_gridder(alphaData[slice],map_rot - scan_angles[rot]))
                    rotSlices.append(alphaRotSlices[slice])
                    
                    thicknessSlices.append(rotate(thicknessData[slice],rot_angle,reshape=False,mode='nearest',order=0))
                    
                elif rot>0:
                    if len(sample_fibtrac_folders)<2:
                        #indexSlices.append(rotate(fibreIndexData[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        indexSlices.append(rot_gridder(fibreIndexData[0][slice],(scan_angles[rot] - scan_angles[0])*-1))
                        rotSlices.append(rotate(rotationIndexes[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        #rotSlices.append(rot_gridder(rotationIndexes[0][slice],(scan_angles[rot] - scan_angles[0])*-1))
                        #alphaRotSlices.append(rotate(rot_alpha_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        #alphaRotSlices.append(rot_gridder(rot_alpha_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1))
                        alphaRotSlices.append(scatter_indexer(indexSlices[slice],fibreIndexData[0][slice],rot_alpha_maps[0][slice]))
                        #betaRotSlices.append(rotate(rot_beta_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        #betaRotSlices.append(rot_gridder(rot_beta_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1))
                        betaRotSlices.append(scatter_indexer(indexSlices[slice],fibreIndexData[0][slice],rot_beta_maps[0][slice]))
                        #binRotSlice = np.copy(betaRotSlices[-1])
                        #binRotSlice[binRotSlice!=0] = 1
                        #binRotSlices.append(rotate(rot_bin_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        #binRotSlices.append(rot_gridder(rot_bin_maps[0][slice],(scan_angles[rot] - scan_angles[0])*-1))
                        binRotSlices.append(scatter_indexer(indexSlices[slice],fibreIndexData[0][slice],rot_bin_maps[0][slice]))
                        thickness_test = np.copy(thicknessData[slice])
                        for j in range(0,thickness_test.shape[0]):
                            for l in range(0,thickness_test.shape[1]):
                                if thickness_test[j,l] !=0:
                                    thickness_test[j,l] = 1
                        
                        thickness_test = rotate(thickness_test.astype(int),rot_angle,reshape=False,mode='nearest',order=0)
                        thickness_test = cv2.resize(thickness_test.astype(np.uint8),(indexSlices[slice].shape[1],indexSlices[slice].shape[0]))
                        thicknessSlices.append(thickness_test)
                        #thicknessSlices.append(rotate(thickness_test.astype(int),(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        #thicknessSlices.append(scatter_indexer(indexSlices[slice],fibreIndexData[0][slice],rot_thickness_maps[0][slice]))
                        
                    else:
                        index_test = rot_gridder(fibreIndexData[0][slice],(scan_angles[rot] - scan_angles[0])*-1)
                        indexSlices.append(scatter_indexer(index_test,fibreIndexData[0][slice],rot_index_maps[0][slice]))
                        rotSlices.append(rotate(rotationIndexes[0][slice],(scan_angles[rot] - scan_angles[0])*-1,reshape=False,mode='nearest',order=0))
                        alphaRotSlices.append(scatter_indexer(index_test,fibreIndexData[0][slice],rot_alpha_maps[0][slice]))                        
                        betaRotSlices.append(scatter_indexer(index_test,fibreIndexData[0][slice],rot_beta_maps[0][slice]))
                        binRotSlices.append(scatter_indexer(index_test,fibreIndexData[0][slice],rot_bin_maps[0][slice]))
                        thickness_test = np.copy(thicknessData[slice])
                        for j in range(0,thickness_test.shape[0]):
                            for l in range(0,thickness_test.shape[1]):
                                if thickness_test[j,l] !=0:
                                    thickness_test[j,l] = 1
                        
                        thickness_test = rotate(thickness_test.astype(int),rot_angle,reshape=False,mode='nearest',order=0)
                        thickness_test = cv2.resize(thickness_test.astype(np.uint8),(indexSlices[slice].shape[1],indexSlices[slice].shape[0]))
                        thicknessSlices.append(thickness_test)
                        
                
            if len(sample_fibtrac_folders)>1 and rot == 0:
            
                alphaRotSlices_save = np.copy(alphaRotSlices)
                betaRotSlices_save = np.copy(betaRotSlices)
                indexRotSlices_save = np.copy(indexSlices)
                
                alpha_test = []
                beta_test = []
                index_test = []
                
                for slice in range(0,len(alpha_slices_mod)):
                    
                    alpha_results,index_results = index_comp(alphaRotSlices[slice],alphaData[slice],
                                                 indexSlices[slice],indexData[slice],alpha_mod = True)
                    
                    beta_results = index_comp(betaRotSlices[slice],betaData[slice],
                                                 indexSlices[slice],indexData[slice])[0]
                    
                    beta_results,alpha_results = index_comp(beta_results,betaData[slice],
                                                 alpha_results,alphaData[slice])
                    
                    index_test.append(index_results) 
                    
                    alpha_test.append(alpha_results)
                    
                    beta_test.append(beta_results)
                    
                    
                alphaRotSlices = np.copy(alpha_test)
                betaRotSlices= np.copy(beta_test)
                indexSlices = np.copy(index_test)
                                
            SAMPLE_PATH = Path(tomosaxs_paths[rot])
            
            with File(SAMPLE_PATH) as sample_file:
                entry = list(sample_file.keys())[0]
                if entry+"/SAXS/data" in sample_file:
                    scan_frames_sum = array(sample_file[entry+"/SAXS_sum/sum"])
                    scan_xVals = array(sample_file[entry+"/SAXS_sum/base_x_value"])
                    scan_yVals = array(sample_file[entry+"/SAXS_sum/base_y_value"])
                sample_file.close()
            
            """
            Find coordinate location of lhs kapton edge in SAXS scan
            """
            x_scan_slice_sums = scan_frames_sum[0,:]
            
            """
            check if the first set of frames is:
                background - typically minus sum WAXS radiation values
                kapton - high sum WAXS rad values followed by positive values
                artefact - high values followed by minus values
            """            
            if len(np.where(x_scan_slice_sums>np.abs(x_scan_slice_sums[0])*10)[0])>0:
                first_x_kapton = np.where(x_scan_slice_sums>np.abs(x_scan_slice_sums[0])*10)[0][0]
                if first_x_kapton>len(x_scan_slice_sums)/2:
                    first_x_kapton = int(first_x_kapton - (int(calib_paths["-tube-"])/saxs_vox))
                    if first_x_kapton<0:
                        first_x_kapton = 0
            else:
                if np.min(x_scan_slice_sums[0:50])>0:
                    first_x_kapton = 0
                else:
                    first_frame = np.where(x_scan_slice_sums<0)[0][0]
                    first_x_kapton = np.where(x_scan_slice_sums[first_frame:-1]>np.abs(x_scan_slice_sums[first_frame])*10)[0][0]
            saxs_kapton_x_edge = (scan_xVals[slice_no,first_x_kapton] - scan_xVals[slice_no,0])*1000
            
            """
            Find coordinate location of lhs kapton edge in CT data and
            calculate offset
            """
            
            kapton_coords = np.where(kapton_rot>0)
            data = [([kapton_coords[0][k],kapton_coords[1][k]]) for k in np.arange(0,len(kapton_coords[0]),1)]
            data = np.asarray(data)
            
            r, c = nsphere_fit(data)  
            #r = (kapt_w/saxs_vox)/2
            t = np.linspace(0, 2 * np.pi, 1000, endpoint=True)                    
            t1,t2 = r * np.cos(t) + c[0], r * np.sin(t) + c[1]
            
            kapton_edge_mod = np.copy(kapton_rot)
            for tx,ty in zip(t1,t2):
                if np.max([tx,ty])<np.min(kapton_edge_mod.shape):
                    kapton_edge_mod[int(tx),int(ty)] = np.max(kapton_rot)
            
            if np.min(data[:,0]) < kapton_rot.shape[0]/2:
               if np.abs(np.min(data[:,1]) - np.min(t2))<saxs_vox:
                   kapton_x_edge = np.max([np.min(t2),np.min(data[:,1])])
               else:
                   kapton_x_edge = np.min(t2)   
            else:
                kapton_x_edge = np.min(t2)            
            
            ct_kapton_x_edge = kapton_x_edge*fibre_vox
            
            kapton_x_offset = saxs_kapton_x_edge - ct_kapton_x_edge
            
            print("######\nTomoSAXS angle "+str(scan_angles[rot])+" X-axis offset = ",str(kapton_x_offset)," um\n######" )
            
            saxs_ct_kapton_x_edge = first_x_kapton - int(kapton_x_offset/saxs_vox)
            
            x_offset_vox = int(np.round(kapton_x_offset/fibre_vox))
            
            if os.path.isdir(output_path+"/tomoSAXS rotations kapton edges/") == False:
                os.mkdir(output_path+"/tomoSAXS rotations kapton edges/")
            
            plt.plot(x_scan_slice_sums)
            plt.scatter(first_x_kapton,x_scan_slice_sums[first_x_kapton],color = "b",label = "TomoSAXS slice "+str(slice_no)+
                        " Detected SAXS outer \n"+str(scan_angles[rot])+" lhs edge of kapton tube")
            plt.scatter(saxs_ct_kapton_x_edge,x_scan_slice_sums[first_x_kapton],color = "r",label = "TomoSAXS slice "+str(slice_no)+
                        " Detected CT outer \n"+str(scan_angles[rot])+" lhs edge of kapton tube")
            plt.legend()            
            plt.savefig(output_path+"/tomoSAXS rotations kapton edges/"+".png")
            plt.show()
            plt.close()
            
            SAXS_CT_axis0_start = int(((saxs_kapton_x_edge)/fibre_vox) - kapton_x_edge)      
            SAXS_CT_offset_axis0 = int(np.sqrt(SAXS_CT_axis0_start**2))

            #SAXS_CT_axis1_start = int(((saxs_kapton_y_edge)/fibre_vox) - kapton_y_edge)      
            SAXS_CT_offset_axis1 = 0

            saxsAxis0Grid = x_scan_slice_sums
            saxsAxis1Grid = y_scan_slice_sums 
            
            """
            pad rotated data by offset
            """
            
            newKaptonMap = np.zeros((alphaRotSlices[0].shape[0]+SAXS_CT_offset_axis1,alphaRotSlices[0].shape[1]+SAXS_CT_offset_axis0))
            newKaptonMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+kapton_edge_mod.shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+kapton_edge_mod.shape[1])] = kapton_edge_mod
                       
            newAlphaMaps = [] 
            newBetaMaps = [] 
            bitMaps = []
            newIndexMaps = []
            newRotMaps = []
            newThicknessMaps = []

            for k in range(0,len(alphaRotSlices)):
                #newAlphaMap = np.zeros((int(len(saxsAxis0Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis1,int(len(saxsAxis0Grid)*saxs_vox/fibre_vox)+SAXS_CT_offset_axis0))
                
                newAlphaMap = np.zeros((alphaRotSlices[0].shape[0]+SAXS_CT_offset_axis1,alphaRotSlices[0].shape[1]+SAXS_CT_offset_axis0),dtype=np.object_)
                
                newBetaMap,newIndexMap,newRotMap,newBinMap = np.copy(newAlphaMap),np.copy(newAlphaMap),np.copy(newAlphaMap),np.copy(newAlphaMap)
                newThicknessMap = np.copy(newAlphaMap)
                
                newAlphaMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alphaRotSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alphaRotSlices[k].shape[1])] = alphaRotSlices[k]
                newBetaMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+betaRotSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+betaRotSlices[k].shape[1])] = betaRotSlices[k]                        
                newBinMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alphaRotSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alphaRotSlices[k].shape[1])] = binRotSlices[k]
                if rot>0 or len(sample_fibtrac_folders)>1:
                    newIndexMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+indexSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+indexSlices[k].shape[1])] = indexSlices[k]   
                    newRotMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+rotSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+rotSlices[k].shape[1])] = rotSlices[k]
                
                thickness_test = np.copy(thicknessSlices[k])
                for j in range(0,thickness_test.shape[0]):
                    for l in range(0,thickness_test.shape[1]):
                        if thickness_test[j,l] !=0:
                            thickness_test[j,l] = 1
                
                newThicknessMap[SAXS_CT_offset_axis1:int(SAXS_CT_offset_axis1+alphaRotSlices[k].shape[0]),SAXS_CT_offset_axis0:int(SAXS_CT_offset_axis0+alphaRotSlices[k].shape[1])] = thickness_test
                newAlphaMaps.append(newAlphaMap)
                newBetaMaps.append(newBetaMap)
                newIndexMaps.append(newIndexMap)
                newRotMaps.append(newRotMap)
                #bitMap = np.copy(newBetaMap)
                #bitMap[bitMap!=0] = 1
                bitMaps.append(newBinMap)
                newThicknessMaps.append(newThicknessMap)
                
            
            rot_alpha_maps.append(newAlphaMaps)
            rot_beta_maps.append(newBetaMaps)
            rot_index_maps.append(newIndexMaps)
            rot_bin_maps.append(bitMaps)
            rot_thickness_maps.append(newThicknessMaps)
            
            """
            Calculate where in the new map you need to sample that corresponds to the tomoSAXS data
            """
                                    
            if SAXS_CT_axis0_start<0 and SAXS_CT_offset_axis0 > saxs_vox/fibre_vox:
                saxs_axis0_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis0*2)*fibre_vox)/saxs_vox)))).tolist()
                saxs_axis0_test = [saxs_axis0_Addon+saxsAxis0Grid.tolist()][0]
                xAxis0 = np.linspace(0,100*0.05,len(saxs_axis0_test))
                saxsStart = len(saxs_axis0_Addon)*round(saxs_vox/fibre_vox)
                #ctaxis0_SAXS_voxels = np.linspace(saxsStart,saxsStart+((round(saxs_vox/fibre_vox)*(sampleWidth))),sampleWidth+1).astype(int) 
                ctaxis0_SAXS_voxels = np.round(np.arange(saxsStart,saxsStart+((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
                ctaxis0_SAXS_midPoints = ctaxis0_SAXS_voxels + round(round(saxs_vox/fibre_vox)/2)
            elif SAXS_CT_axis0_start>0 and SAXS_CT_offset_axis0 > saxs_vox/fibre_vox:
                #saxs_axis0_Addon = np.zeros((int(np.round(((SAXS_CT_offset_axis0)*fibre_vox)/saxs_vox)))).tolist()
                #saxs_axis0_test = [saxsAxis0Grid+saxs_axis0_Addon][0]
                #xAxis0 = np.linspace(0,100*0.05,len(saxs_axis0_test))        
                #ctaxis0_SAXS_voxels = np.linspace(0,(round(saxs_vox/fibre_vox)*(sampleWidth)),sampleWidth+1).astype(int)
                ctaxis0_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
            else:
                #ctaxis0_SAXS_voxels = np.linspace(0,(round((saxs_vox)/fibre_vox))*round(sampleWidth),round(sampleWidth)+1).astype(int)
                ctaxis0_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
                            
            ctaxis1_SAXS_voxels = np.round(np.arange(0,((len(saxsAxis0Grid)*saxs_vox)/fibre_vox),saxs_vox/fibre_vox)).astype(int)
            
            """
            if this is the first tomoSAXS scan rotation:
                create new alpha, beta, and bitmap datasets, starting from the x-axis offset (minimial due to previous
                registration step)                                
            """
            
            if rot == 0 and np.max([ctaxis0_SAXS_voxels[0],ctaxis1_SAXS_voxels[0]])>0:
                alphaData = [np.rot90(newAlphaMaps[k],3)[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]] for k in np.linspace(0,len(newAlphaMaps)-1,len(newAlphaMaps)).astype(int)]
                betaData = [np.rot90(newBetaMaps[k],3)[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]] for k in np.linspace(0,len(newAlphaMaps)-1,len(newAlphaMaps)).astype(int)]
                bitmapData = [np.rot90(bitMaps[k],3)[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]] for k in np.linspace(0,len(newAlphaMaps)-1,len(newAlphaMaps)).astype(int)]                
                kaptonData = kaptonData[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]]
                indexData = [np.rot90(newIndexMaps[k],3)[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]] for k in np.linspace(0,len(newAlphaMaps)-1,len(newAlphaMaps)).astype(int)]
                thicknessData = [np.rot90(newThicknessMaps[k],3)[ctaxis0_SAXS_voxels[0]:ctaxis0_SAXS_voxels[-1],ctaxis1_SAXS_voxels[0]:ctaxis1_SAXS_voxels[-1]] for k in np.linspace(0,len(newAlphaMaps)-1,len(newAlphaMaps)).astype(int)]
            
            ctaxis0_SAXS_midPoints = np.round(ctaxis0_SAXS_voxels+((saxs_vox/fibre_vox)/2)*saxsScale).astype(int)
            ctaxis1_SAXS_midPoints = np.round(ctaxis1_SAXS_voxels+((saxs_vox/fibre_vox)/2)*saxsScale).astype(int)
            
            scan_params =[sampleWidth,saxs_vox,fibre_vox]
            
            beamdiameter = saxs_vox/fibre_vox
            
            sig3 = ((beamdiameter/(2*np.sqrt(2*np.log(2))))*3)/2
            
            voxel_params = [ctaxis1_SAXS_voxels[ctaxis1_SAXS_voxels<newAlphaMaps[0].shape[0]],
                            ctaxis1_SAXS_midPoints[ctaxis1_SAXS_midPoints<newAlphaMaps[0].shape[0]],
                            ctaxis0_SAXS_voxels[ctaxis0_SAXS_voxels<newAlphaMaps[0].shape[1]],
                            ctaxis0_SAXS_midPoints[ctaxis0_SAXS_midPoints<newAlphaMaps[0].shape[1]]]
            
            weightStarts = np.round(ctaxis1_SAXS_midPoints-sig3).astype(int)
            weightEnds = np.round(ctaxis1_SAXS_midPoints+sig3).astype(int)
            
            weightParams = [weightStarts,weightEnds,
                            ctaxis0_SAXS_voxels[ctaxis0_SAXS_voxels<newAlphaMaps[0].shape[1]],
                            ctaxis0_SAXS_midPoints[ctaxis0_SAXS_midPoints<newAlphaMaps[0].shape[1]],
                            ctaxis1_SAXS_voxels[ctaxis1_SAXS_voxels<newAlphaMaps[0].shape[0]]]
            
            alphaIndex = []
            alphaCount = []
            voxIndexes = []
            weightsIndex = []
            
            fibreIndex = []
            
            betaIndex = []
            
            rotIndex = []
            
            voxRefIndex = []
            
            rot_thickness,rot_density = [],[]
            
            thickness_index = []
            
            for slice in range(0,len(newAlphaMaps)):
                
                og_rot_index = []
                og_vox_index = []
                
                print("######\nRegistering per-tomoSAXS voxel fibre information for\nTomoSAXS angle "+str(scan_angles[rot])+" slice "+str(slice)+"\n######")
                if rot == 0:
                    
                    if len(sample_fibtrac_folders)<2:
                    
                        alphaResults =  data_gridder(newAlphaMaps[slice],bitMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False)
                        print(np.sum((np.where(newAlphaMaps[slice]!=0,1,0)) - (np.where(alphaResults[2]!=0,1,0))))
                        alphaIndex.append(alphaResults[0])
                        
                        weightResults =  weight_gridder(alphaResults[2],bitMaps[slice],newKaptonMap,scan_params,weightParams,ut8 = False)
                        
                        alphaCount.append(weightResults[1])
                        weightsIndex.append(weightResults[0])
                        
                        #voxIndexes.append(alphaResults[2][0:alphaData[0].shape[0],0:alphaData[0].shape[1]])
                        voxIndexes.append(alphaResults[2])
                        #rotIndex.append(alphaResults[4])
                        
                        rotIndex.append(alphaResults[4][0:alphaData[0].shape[0],0:alphaData[0].shape[1]])
                        idxResults = data_gridder(alphaResults[2],bitMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False)
                        fibreIndex.append(idxResults[0])
                        voxRefIndex.append(alphaResults[5])
                        
                        #betaIndex.append(data_gridder(newBetaMaps[slice],idxResults[3],scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                        betaIndex.append(data_gridder(newBetaMaps[slice],alphaResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                        
                        thickness_index.append(data_gridder(newThicknessMaps[slice],alphaResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                        
                        alpha_test = alphaResults[0]
                        
                        beta_test = data_gridder(newBetaMaps[slice],alphaResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0]
                        
                        test = [[[i,k] for k in np.arange(0,len(beta_test[i]),1) if type(beta_test[i][k]) == list and type(alpha_test[i][k]) == list and len(beta_test[i][k])!=len(alpha_test[i][k])]
                          for i in np.arange(0,len(alpha_test),1)]
                        
                        test = [k for k in test if len(k)!=0]
                        
                        print(len(test))
                        
                    else:
                        
                        indexResults =  data_gridder(newIndexMaps[slice],bitMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False)
                        print(np.sum((np.where(newIndexMaps[slice]!=0,1,0)) - (np.where(indexResults[2]!=0,1,0))))

                        weightResults =  weight_gridder(indexResults[0],bitMaps[slice],newKaptonMap,scan_params,weightParams,ut8 = False)

                        alphaCount.append(weightResults[1])
                        weightsIndex.append(weightResults[0])

                        #voxIndexes.append(alphaResults[2][0:alphaData[0].shape[0],0:alphaData[0].shape[1]])
                        voxIndexes.append(indexResults[2])

                        rotIndex.append(indexResults[4][0:indexData[0].shape[0],0:indexData[0].shape[1]])
                        #idxResults = data_gridder(alphaResults[2],bitMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False)
                        fibreIndex.append(indexResults[0])
                        voxRefIndex.append(indexResults[5])

                        #betaIndex.append(data_gridder(newBetaMaps[slice],idxResults[3],scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                        betaIndex.append(data_gridder(newBetaMaps[slice],indexResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])

                        alphaIndex.append(data_gridder(newAlphaMaps[slice],indexResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])

                        thickness_index.append(data_gridder(newThicknessMaps[slice],indexResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                        
                        #alpha_test = alphaResults[0]
                        
                        #beta_test = data_gridder(newBetaMaps[slice],indexResults[2],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0]
                        
                        #test = [[[i,k] for k in np.arange(0,len(beta_test[i]),1) if type(beta_test[i][k]) == list and type(alpha_test[i][k]) == list and len(beta_test[i][k])!=len(alpha_test[i][k])]
                          #for i in np.arange(0,len(alpha_test),1)]
                        
                        #test = [k for k in test if len(k)!=0]
                        
                        #print(len(test))
                    
                
                else:
                
                    idxResults = data_gridder(newRotMaps[slice],bitMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False)             
                    rotIndex.append(idxResults[0])
                    fibreIdxResults = data_gridder(newIndexMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)
                    fibreIndex.append(fibreIdxResults[0])
                    
                    alphaResults = data_gridder(newAlphaMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True) 
                    alphaIndex.append(alphaResults[0])
                    weightResults =  weight_gridder(newIndexMaps[slice],bitMaps[slice],newKaptonMap,scan_params,weightParams,ut8 = False)
                    alphaCount.append(weightResults[1])
                    weightsIndex.append(weightResults[0])
                    
                    voxIndexes.append(newIndexMaps[slice])
                    voxRefIndex.append(alphaResults[5])
                    
                    #betaIndex.append(data_gridder(newBetaMaps[slice],alphaResults[2],scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                    betaIndex.append(data_gridder(newBetaMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                    
                    thickness_index.append(data_gridder(newThicknessMaps[slice],newThicknessMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0])
                    
                    
                    #alpha_test = alphaResults[0]
                    
                    #beta_test = data_gridder(newBetaMaps[slice],newIndexMaps[slice],newKaptonMap,scan_params,voxel_params,ut8 = False,voxIndexer = True)[0]
                    
                    #test = [[[i,k] for k in np.arange(0,len(beta_test[i]),1) if type(beta_test[i][k]) == list and type(alpha_test[i][k]) == list and len(beta_test[i][k])!=len(alpha_test[i][k])]
                      #for i in np.arange(0,len(alpha_test),1)]
                    
                    #test = [k for k in test if len(k)!=0]
                    
                    #print(len(test))
                    
                
                """
                Now estimate the thickness (spatial distribution) and density (number of fibres) across every tomoSAXS beampath
                for background correction.
                """
                            
                #if len(sample_fibtrac_folders)<2 and rot==0:
                    #slice_thickness,slice_density = np.zeros((indexResults[0].shape[1])),np.zeros((indexResults[0].shape[1]))
                    #saxs_bps = np.copy(indexResults[0])
                    #bp_range = indexResults[0].shape[1]
                #else:
                    #slice_thickness,slice_density = np.zeros((alphaResults[0].shape[1])),np.zeros((alphaResults[0].shape[1]))
                    #saxs_bps = np.copy(alphaResults[0])
                    #bp_range = alphaResults[0].shape[1]
                slice_thickness,slice_density = np.zeros((thickness_index[slice].shape[1])),np.zeros((thickness_index[slice].shape[1]))
                saxs_bps = np.copy(thickness_index[slice])
                bp_range = thickness_index[slice].shape[1]
                
                for i in range(0,bp_range):
                    saxs_bp = np.copy(saxs_bps[:,i])
                    if len(np.where(np.asarray(saxs_bp)==500)[0])>0:
                        slice_thickness[i] = 10000
                        slice_density[i] = 10000
                    else:
                        saxs_bp[np.where(np.asarray(saxs_bp)==500)] = 0
                        if len(np.where(saxs_bp !=0)[0]):
                            slice_thickness[i] = (np.where(saxs_bp !=0)[0][-1] - np.where(saxs_bp !=0)[0][0])*saxs_vox
                            slice_density[i] = np.sum([len(k) if type(k) == list or type(k) == np.ndarray else 1 for k in saxs_bp[np.where(saxs_bp !=0)]])
                        
                plt.plot(slice_thickness)
                plt.show()
                plt.close()
                
                rot_thickness.append(slice_thickness)
                rot_density.append(slice_density)
                
            fibreRotMaps.append(alphaData)
                                                     
            alphaIndexes.append(alphaIndex)
            betaIndexes.append(betaIndex)
            rotationIndexes.append(rotIndex)
            fibreIndexes.append(fibreIndex)
            fibreCounts.append(alphaCount)
            fibreIndexData.append(voxIndexes)
            weightIndexData.append(weightsIndex)
            
            sampleThicknesses.append(rot_thickness)
            sampleDensities.append(rot_density)
    
    
    print("######\nRegistration now complete for scan ",scan_name,":\nSaving data\n######") 
    
    np.save(output_path+"/first_fibre_index_maps.npy",fibreIndexData[0])
    
    np.save(output_path+"/full_alpha_index_matrices.npy",alphaIndexes)
    np.save(output_path+"/full_beta_index_matrices.npy",betaIndexes)
    np.save(output_path+"/full_rotation_index_matrices.npy",rotationIndexes)
    np.save(output_path+"/full_fibre_index_matrices.npy",fibreIndexes)
    np.save(output_path+"/full_count_index_matrices.npy",fibreCounts)
    np.save(output_path+"/full_fibre_rotation_index_matrices.npy",fibreRotMaps)
    
    fib_index_pad = np.empty(len(fibreIndexData),dtype=object)
    fib_index_pad[:]=fibreIndexData
    
    np.save(output_path+"/full_voxel_index_matrices.npy",fib_index_pad)
    np.save(output_path+"/full_fibre_weight_matrices.npy",weightIndexData)
    
    np.save(output_path+"/full_sample_thicknesses.npy",sampleThicknesses)
    np.save(output_path+"/full_sample_densities.npy",sampleDensities)
    
    dataFrameSave(np.asarray(alphaIndexes),output_path,"alpha index matrices")
    dataFrameSave(np.asarray(betaIndexes),output_path,"beta index matrices")
    dataFrameSave(np.asarray(rotationIndexes),output_path,"rotation index matrices")
    dataFrameSave(np.asarray(fibreIndexes),output_path,"fibre index matrices")
    dataFrameSave(np.asarray(fibreCounts),output_path,"fibre count matrices")
    dataFrameSave(np.asarray(fibreRotMaps),output_path,"fibre rotation maps")
    dataFrameSave(np.asarray(fibreIndexData),output_path,"fibre index data maps") 
    dataFrameSave(np.asarray(sampleThicknesses),output_path,"sample thickness maps")            
    dataFrameSave(np.asarray(sampleThicknesses),output_path,"sample density maps")
    
    print("######\nRegistration now complete for scan ",scan_name,"\n######") 
