#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:53:53 2024

@author: via83767
"""

import PySimpleGUI as sg
import numpy as np
import os,glob

def collectFileParameters():
    
    Font = ("Arial", 11)
    
    sg.theme('Light Blue 2')
            
    """
    First frame: sample input and naming (e.g. "tomoSAXS scan 1")    
    """
    frame_1 = [[sg.Text('Script folder', size=(39, 1)), sg.Input(size=(20,1),default_text = "C:/Users/Himadri/Desktop/Academic work/TomoSAXS/papers/outline scripts"), 
                sg.FolderBrowse(key="-SCRIPT_FOLDER-")],
                [sg.Text('New file folder', size=(39, 1)), sg.Input(size=(20,1),default_text = "C:/Users/Himadri/Desktop/Academic work/TomoSAXS/papers/outline scripts"), 
                sg.FolderBrowse(key="-NEW_FOLDER-")]]
    
    layout = [[sg.Frame('Folder selection', frame_1, pad=(0, 5))],
              [sg.Submit(),sg.Cancel()]]
    
    window = sg.Window('New TomoSAXS recon', layout,font=Font)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Submit':
            window.close()
            return values
            
    window.close()

folderParams = collectFileParameters()
if len(folderParams[0])!= len(folderParams['-SCRIPT_FOLDER-']):
    script_folder = [folderParams[0],folderParams['-SCRIPT_FOLDER-']][np.argmax(len(folderParams[0]),len(folderParams['-SCRIPT_FOLDER-']))]
    new_folder = [folderParams[1],folderParams['-SCRIPT_FOLDER-']][np.argmax(len(folderParams[1]),len(folderParams['-SCRIPT_FOLDER-']))]
else:
    script_folder = folderParams[0]
    new_folder = folderParams[1]


import os
os.chdir(script_folder)

bash_files = [file for file in os.listdir(script_folder) if file.endswith(".sh")]

py_files = [file for file in os.listdir(script_folder) if file.endswith(".py")]

for py_file in py_files:
    
    with open(py_file,'r') as file:
        filedata = file.read()
        
        if len([k for k in filedata if "input_folder =" in filedata])>0: 
            input_line = filedata.split("input_folder")[1]            
            if len(input_line.split('"'))>1:
                filedata = filedata.replace(input_line.split('"')[1],new_folder)
            else:
                filedata = filedata.replace(input_line.split("'")[1],new_folder)
        
    with open(py_file,'w') as file:
        file.write(filedata)
        
for bash_file in bash_files:
    
    with open(bash_file,'r') as file:
        filedata = file.read()
        input_line = filedata.split("cd")[1].split("\n\nmodule")[0]
        input_line = input_line.split(" ")[-1]
        filedata = filedata.replace(input_line,script_folder)
        
    with open(bash_file,'w') as file:
        file.write(filedata)
    
