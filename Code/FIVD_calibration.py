# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:15:53 2025

@author: Himadri
"""

"""
Scipt for spatially registering low resolution and high 
resolution CT datasets, and fibre tracing data.

Based on previous estimation of vertical offset between 
high/low resolution CT data, and isolation of shared
sample-specific landmarks - SEE "HOW-TO" DOCUMENT:
    https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/Documentation/How-to.rst

Operates 2 GUIs:
    1. select folder containing TomoSAXS scripts.
        this is to load "FIVD_calibration_gui.py" - 
            the data input gui script
    2. Input data for calibration (see "how-to" document)
    
"""

import os,glob
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import sys
import cv2
from sklearn import cluster
from skimage import measure
import pandas as pd
from scipy.ndimage import rotate
from scipy.interpolate import griddata
import pickle

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class DirSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Select Working Directory")
        self.selected_dir = ""

        # Label + entry
        ttk.Label(self, text="Working Directory:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.dir_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.dir_var, width=60)
        entry.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        # Browse button
        ttk.Button(self, text="Browse…", command=self.browse).grid(row=0, column=2, padx=6, pady=6)

        # Confirm button
        ttk.Button(self, text="Confirm", command=self.confirm).grid(row=1, column=0, columnspan=3, pady=10)

        self.columnconfigure(1, weight=1)

        # Fit window to contents
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

    def browse(self):
        path = filedialog.askdirectory(title="Select working directory")
        if path:
            self.dir_var.set(path)

    def confirm(self):
        self.selected_dir = self.dir_var.get()
        if not self.selected_dir:
            messagebox.showwarning("No selection", "Please select a directory.")
            return
        messagebox.showinfo("Confirmed", f"Working directory set to:\n{self.selected_dir}")
        self.destroy()

####
#select working directory
####
app = DirSelector()
app.mainloop()
gui_folder = app.selected_dir

os.chdir(gui_folder)

####
#navigate and load GUI script
####
import FIVD_calibration_gui as calib_gui

####
#Run GUI
####
test = calib_gui.run_gui()

####
#save input to variables
####
start_diff = test.values["start_diff"]

og_shape = test.values["og_shape"]

top_path = test.values["top_path"]

path = test.values["path"]

calibPath = test.values["calibPath"]

lowPath = test.values["lowPath"]

highPath = test.values["highPath"]

beta_path = test.values["beta_path"]

alpha_path = test.values["alpha_path"]

index_path = test.values["index_path"]

pad_file = test.values["pad_file"]

ct_savePath = test.values["ct_savePath"]

fibtrac_savePath = np.copy(path)

scan_name = test.values["scan_name"]

"""
pad and save fibre tracing data
"""
####
#read padding information
####
pads =  pd.read_excel(pad_file)

sample_fibtrac_folders = [beta_path,alpha_path,index_path]

sample = [k for k in pads['sample'] if scan_name in k][0]
    
if len(sample.split("_")) == 1:
    sample_name_elements = sample.split(" ")
else:
    sample_name_elements = sample.split("_")
    
sample_ID = sample_name_elements[0]
if len(sample_ID) == 1:
    sample_ID = sample_name_elements[0]+sample_name_elements[1]

####
#Begin padding fibre tracing data
####
padded_folders = []    
for folder in sample_fibtrac_folders:
    
    if "theta" in folder:
        print("######\nPadding alpha fibre tracing data to original CT dimensions\n######")
    elif "phi" in folder:
        print("######\nPadding beta fibre tracing data to original CT dimensions\n######")
    
    ####
    #load data
    ####
    fibre_data = []
    for sliceFile in np.sort(glob.glob(os.path.join(folder+"/", "*.tif"))):
        #fibre_data.append(cv2.imread(sliceFile))                     
        if "theta" in folder or "phi" in folder:
            fibre_data.append(cv2.cvtColor(cv2.imread(sliceFile), cv2.COLOR_BGR2GRAY))
        else:
            fibre_data.append(cv2.imread(sliceFile,-1))
    
    fibre_data = np.asarray(fibre_data)
    
    ####
    #isolate padding values for each axis
    ####
    xEndPad = pads['X bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
    yEndPad = pads['Y bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
    zEndPad = pads['Z bottom pad (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
            
    xStartPad = pads['x padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
    yStartPad = pads['y padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
    zStartPad = pads['z padding (new voxels)'][np.where(pads['sample'] == sample)[0]].tolist()[0]-1
    
    ####
    #Pad accordingly
    ####
    og_padding = np.zeros((zStartPad,fibre_data.shape[1],fibre_data.shape[2]))
    og_data = np.concatenate([og_padding,fibre_data],0)
    
    xStartPadding = np.zeros((fibre_data.shape[0],fibre_data.shape[1],xStartPad,))               
    fibre_data = np.concatenate([xStartPadding,fibre_data],2)
    
    yStartPadding = np.zeros((fibre_data.shape[0],yStartPad,fibre_data.shape[2]))
    fibre_data = np.concatenate([yStartPadding,fibre_data],1)
                    
    zStartPadding = np.zeros((zStartPad,fibre_data.shape[1],fibre_data.shape[2]))
    fibre_data = np.concatenate([zStartPadding,fibre_data],0)
            
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
             
    if "theta" in folder or "phi" in folder:
        for k in range(0,len(fibre_data)):
            cv2.imwrite(folder+" padded/"+str(k)+".tiff",fibre_data[k].astype(np.uint8))
    else:
        for k in range(0,len(fibre_data)):
            cv2.imwrite(folder+" padded/"+str(k)+".tiff",fibre_data[k])
        
    padded_folders.append(folder+" padded/")


high_res = [k for k in os.listdir(calibPath) if "high" in k][0]
low_res = [k for k in os.listdir(calibPath) if "low" in k][0]

high_res_img = cv2.imread(calibPath+high_res,0)
low_res_img = cv2.imread(calibPath+low_res,0)

ret,high_thresh = cv2.threshold(high_res_img,127,255,0)
 
# calculate moments of binary image
M = cv2.moments(high_thresh)
 
# calculate x,y coordinate of center
high_X = int(M["m10"] / M["m00"])
high_Y = int(M["m01"] / M["m00"])

ret,low_thresh = cv2.threshold(low_res_img,127,255,0)
 
# calculate moments of binary image
M = cv2.moments(low_thresh)
 
# calculate x,y coordinate of center
low_X = int(M["m10"] / M["m00"])
low_Y = int(M["m01"] / M["m00"])

# calculate fibre tracing coordinates of center
fibTest_shape = round(low_res_img.shape[-1]/(high_res_img.shape[0]/fibre_data.shape[-1]))
fibtrac_low_X = round(low_X/(high_res_img.shape[0]/fibre_data.shape[-1]))
fibtrac_low_Y = round(low_Y/(high_res_img.shape[0]/fibre_data.shape[-1]))

fibtrac_high_X = round(high_X/(high_res_img.shape[0]/fibre_data.shape[-1]))
fibtrac_high_Y = round(high_Y/(high_res_img.shape[0]/fibre_data.shape[-1]))


####
#Pad high res data - centring on landmark coordinates
####
if low_X>high_X:
    
    for i in range(0,len(os.listdir(highPath))):
        
        lr_i = i + start_diff
        
        hr_img = cv2.imread(highPath+[k for k in os.listdir(highPath) if k !="rois" and int(k.split(".")[0]) == i][0],0)
        lr_img = cv2.imread(lowPath+[k for k in os.listdir(lowPath) if k !="rois" and int(k.split(".")[0]) == lr_i][0],0)
        
        test_img = np.copy(lr_img)
        
        #test_img[(low_X - high_X):low_X+(high_res_img.shape[1] - high_X),(low_Y - high_Y):low_Y+(high_res_img.shape[0] - high_Y)] = hr_img    
        test_img[(low_Y - high_Y):low_Y+(high_res_img.shape[1] - high_Y),(low_X - high_X):low_X+(high_res_img.shape[0] - high_X)] = hr_img
    
        cv2.imwrite(ct_savePath+str(i)+".tiff",test_img)
        
for folder in padded_folders:
    for  i in range(0,len([k for k in os.listdir(folder) if k !="calibrated" and int(k.split(".")[0])])):           
    
        fibTest_img = np.zeros((fibTest_shape,fibTest_shape))
        
        if "theta" in folder or "phi" in folder:
            fibtrac_img = cv2.imread(folder+[k for k in os.listdir(folder) if k !="calibrated" and int(k.split(".")[0]) == i][0],0)
        else:
            fibtrac_img = cv2.imread(folder+[k for k in os.listdir(folder) if k !="calibrated" and int(k.split(".")[0]) == i][0],-1)
        
        fibTest_img[(fibtrac_low_Y - fibtrac_high_Y):fibtrac_low_Y+(fibtrac_img.shape[0] - fibtrac_high_Y),(fibtrac_low_X - fibtrac_high_X):fibtrac_low_X+(fibtrac_img.shape[1] - fibtrac_high_X)] = fibtrac_img
        #fibTest_img[(fibtrac_low_X - fibtrac_high_X):fibtrac_low_X+(fibtrac_img.shape[0] - fibtrac_high_X),(fibtrac_low_Y - fibtrac_high_Y):fibtrac_low_Y+(fibtrac_img.shape[1] - fibtrac_high_Y)] = fibtrac_img
        
        if os.path.isdir(folder+"/calibrated/") == False:
            os.mkdir(folder+"/calibrated/")
                 
        if "theta" in folder or "phi" in folder:
            cv2.imwrite(folder+"/calibrated/"+str(i)+".tiff",fibTest_img.astype(np.uint8))
        else:
            cv2.imwrite(folder+"/calibrated/"+str(i)+".tiff",fibTest_img)
