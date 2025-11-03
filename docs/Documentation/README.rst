TomoSAXS: Multimodal X-ray analysis of collagenous tissues combining micro-CT and volumetric SAXS
=======================

.. _Overview:

Overview
------------
TomoSAXS is a framework for combining micro-CT and tomographic SAXS (small angled X-ray scattering) data, 
to spatially analyse nanoscale structural and strain/pre-strain in soft collagen organs using SAXS, and 
compare with micro-scale structure and strain/pre-strain estimated from micro-CT. All data is originally 
obtained at the Diamond Light Source synchrotron (`DLS <https://www.diamond.ac.uk/Home.html>`_), and the pipeline is currently 
formatted for the data structure applied by DLS.

**HARDWARE REQUIRED - THIS IS A PIPELINE DESIGNED FOR USING CLUSTER COMPUTING TO PARALLELISE BETWEEN SLICES. IT HAS BEEN CARIED OUT AT THE HPC CLUSTER AT DIAMOND LIGHT SOURCE. IF RUNNING WITHOUT THIS ON A STANDARD DEKSTOP, THIS PROCESS MAY TAKE 1-2 WEEKS**


The TomoSAXS pipeline principally operates in the Python (v = 3.10) platform, using cluster computing in the SLURM environment. 
The pipeline is modular, operating along the following modules:

  **1.	Fibre orientation analysis and SAXS/CT data registration.**

  **2.	DVC analysis of CT data.**

  **3.	Background correction using estimates of sample thickness per-rotation (from registration process).**

  **4.	Initial estimation of scattering intensity for individual fibres using single value decomposition (SVD).**

  **5.	Reconstruction of scattering metrics for individual fibres related to nanoscale structure and mechanics.**

  **6.	Spatial mapping of per-fibre metric values and estimation of per-fibre strain through comparison with DVC data.**

This document focusses on how to perform the Python pipeline for TomoSAXS processing and analysis (Modules 1,3,4,5). Details of the algorithms involved in each module are provided in the following pages:

  •	For details on software dependencies and installation - see `here <https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/Documentation/Software.rst>`_ after these packages are installed, please download the necessary scripts as described below to install the tomoSAXS package.

  •	For details on data registration – see `here <https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/coreg.rst>`_

  •	For details on background correction - see `here <https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/bgrcorr.rst>`_

  •	For Reconstruction of per-fibre scattering metrics – see `here <https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon.rst>`_

  •	For details on DVC processing and analysis – see `here <https://github.com/himadri111/tomosaxs/tree/main/Code/CT%20%26%20DVC>`_

To download the necessary scripts, visit the stable build library `here <https://github.com/himadri111/tomosaxs/tree/main/Code>`_
Once downloaded, copy to the desired working folder for your analysis. The “folder_swap.py” script is a useful tool for changing the input folder in downloaded scripts.

**Example data** to run reconstruction on a single slice can be found on the Figshare repository `here <https://figshare.com/s/ad8a7f8cef880b62f28d>`_


.. _Module 1:

Module 1: SAXS/CT data registraion
------------------------------------
This module operates over several discrete processing steps:

 **a.	Processing of CT data.**

 **b.	Processing of fibre orientation data.**

 **c.	Registration of fibre orientation and SAXS data.**

**a.	Processing of CT data**
------------------------------------

For imaging large samples such as complete intervertebral discs, two CT scans are used; 
 
 **a) a high resolution scan including the entire sample;**

 **b) a low resolution scan including the sample and at least one half of the kapton tube sample holder.** 
This is to allow the low resolution scan to calibrate the distance of the sample from the kapton tube in the high resolution scan (used for fibre orientation analysis and DVC), which is vital for the registration process. The scans are then combined by scaling each to that they are the same size, calculating the vertical offset between them using a suitable fiducial marker in the sample, and using the spatial offset between the marker in each scan to centre the high resolution data onto the low resolution data.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/resolution_comp.png
**FIG. 1. low versus high resolution SRCT reconstructions.** 


This process follows the following steps:

*Scaling CT data.* The high resolution and low resolution CT reconstructions are opened in imageJ/Fiji (herein referred to as `Fiji <https://imagej.net/>`_), by locating their folder and dragging the folder icon into the Fiji taskbar. Once loaded, reduce the bit-rate to 8bit by selecting **Image>Type>8bit**. Then save the 8bit version to your working directory by selecting **File>Save As>Image Sequence**. In the proceeding “Save Image Sequence” window, select the “Browse” button. Navigate to your working directory and create a new subfolder, naming it “CT data”. Within “CT data” create a further subfolder called “low res” if you are saving the low resolution scan, or “high res” if you are saving the high resolution scan. Finally create a final subfolder called “8bit original”. Navigate inside this new folder and hit the “select” button. Then in the “Save Image Sequence” window, delete the information in the sub-window next to “Name” and hit the “OK” button. Repeat this for each dataset.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/8bit_img.png
**FIG. 2. downsampling to 8bit in Fiji.** 


For registration, both datasets must be modified so that they are the same absolute voxel size. The current default high resolution voxel size is 1.625 μm\ :sup:`3`, and low resolution voxel size is 2.6 μm\ :sup:`3`. Both datasets are modified to produce voxels sizes of 6.5 μm\ :sup:`3`.  :sup:`2`.

 •	Modify the **high resolution** 8bit data by selecting **Image>Adjust>Size** in Fiji and changing the **width and height to 640 and the depth to 540**. 

 •	Modify the **low resolution** 8bit data by changing the **width and height to 1024 and depth to 864**. 

 •	Save each modified dataset as an image sequence in a new subfolder within their respective “high res” or “low res” folder called “inverse scaled”.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/resize_img.png
**FIG. 3. Resizing data in Fiji.**


High resolution data only included a subsection of the low resolution data (smaller field of view), so the vertical offset between the two scaled datasets must be calculated. Open both scaled datasets in Fiji and isolate a slice in the low resolution dataset that includes a diagnostic element of the sample. This can be a portion of sample with a definitive and unique 2D shape or size. Once selected, find the same portion in the high resolution scaled data and log the offset in the slice number between both datasets. Duplicate the slice in both datasets by right clicking inside the slice and selecting “Duplicate” in the proceeding window. 

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/threshold_img.png
**FIG. 4. Thresholding data in duplicated slices representing the same region-of-interest in rescaled low resoluton and high resoluton datasets in Fiji.**


Now the spatial registration between the low resolution and high resolution datasets can be calibrated. To do this: 
 •	first select **Process>Binary>Options** in the Fiji taskbar and in the proceeding “Binary Options” window, tick “Black background” before hitting “ok”. 
 •	You can now use thresholding to isolate the selected feature in both duplicates by clicking on the duplicate and selecting **Image>Adjust>Threshold** in the Fiji taskbar. 
 •	In the “Threshold” window, adjust the lower bound of the threshold so that the feature is kept as red but the surrounding background is not. 
 •	Once this has been optimized, make sure “Dark background” has been ticked in the “Threshold” window and then hit the “Apply” button. 
 •	This converts the duplicate into a binary image consisting of greyscale values of 255 for all regions marked with red in the threshold and 0 for all other regions. 
 •	You can now further isolate the chosen feature by using the Polygon selection tool in the Fiji taskbar to select around the feature, before selecting **Edit>Clear outside** to remove any other material. 
 •	Once only the feature is left in the duplicate, save using **File>save as>tiff** and create a new subfolder in “CT data” called “calibration”, then saving within that folder by naming the image after the slice that it originates from (e.g.**“low_res_168.tiff”** or **“high res_450.tiff”** respectively).

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/region_select.png
**FIG. 4. Selection of diagnostic sample element and isolation using the Polygon selection tool in Fiji.**

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/region_isolate.png
**FIG. 5. Isolated sample element after using "Clear outside" tool in Fiji.**


The data is now ready to be calibrated alongside the orientation data.


**b) Processing of fibre orientation data**
--------------------------------------------
Fibre orientation data is provided in the form of downsampled and subsampled stacks of tiffs, saved as single 3D tiff files, each with its own .png file highlighting the parameters used for creating the data:
•	 Fibre theta angle (azimuthal angle) - 8bit tiffs
•	 Fibre theta angle.png
•	 Fibre phi angle (lateral angle) – 8bit tiffs
•	 Fibre phi angle.png
•	 Fibre index (index value for each fibre) – 16bit tiffs
•	 Fibre index.png

These files must be exported to stacks of single tiff files. Then, for registration, processed by padding to the same absolute size as the original CT data.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/theta_stack.png
**FIG. 6. Per-fibre Azimuthal orientation data.**


*Exporting data.* Data is exported to single tiffs in Fiji. Open Fiji and drag each of the above 3D tiffs into the tool bar, which will load them into Fiji. For each stack, select **File>Save As>Image Sequence** and in the proceeding “Save Image Sequence” window, select the “Browse” button. Navigate to your working folder and create a new subfolder, naming it the same name as the original file. Navigate inside this new folder and hit the “select” button. Then in the “Save Image Sequence” window, delete the information in the subwindow next to “Name” and hit the “OK” button. Repeat this for each dataset.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/pad_settings.png
**FIG. 7. Padding settings from associated .png file for fibre orientation data.**


*Inputting calibration details.* This uses the associated .png file for the fibre orientation data. Only one needs to be used as the parameters are the same for each. The process also uses the **“voxel processing.xlsx”** and **“vox_padding.xlsx”** files (LINK TO FILES HERE). 

•	 Open one of the .png files, and both .xlsx files. 
•	 In the “voxel_processing.xlsx” file, in the “sample” column provide the desired name of the sample. **This is very important** as the exact name will be used for all further scripts. Do not include spaces (use underscores _ instead). 
•	 In the three columns to the right (X axis length (new voxels); Y axis length (new voxels); Z axis length (new voxels)), input the “Lattice info” in the .png file. 
•	 Then in “X axis length (old voxels); Y axis length (old voxels); Z axis length (old voxels)”, input the “physical size” in the .png file. 
•	 Finally, in the “x start point old voxels; y start point old voxels; z start point old voxels” columns, input the starts points of the “physical size” data in the .png file (numbers starting after “from”). The rest of the columns should automatically generate.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/voxel_padding_entry_1.png
**FIG. 8. voxel_processing.xlsx dataset.**

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/voxel_padding_entry_2.png
**FIG. 9. voxel_processing.xlsx dataset continued.**


Once this data has been inputted: 

•	 Copy the data from columns “X start padding (new voxels); Y start padding (new voxels); Z start padding (new voxels)” in “voxel_processing.xlsx” into the “x padding (new voxels); y padding (new voxels); z padding (new voxels)” of the “vox_padding.xlsx” file. Round these values up to the nearest integer. 
•	 Then copy the data from “X end padding; Y end padding; Z end padding” into the “X bottom pad (new voxels); Y bottom pad (new voxels); Z bottom pad (new voxels)” of the “vox_padding.xlsx” file. 
•	 Finally, add the same sample name to the “sample” column of “vox_padding.xlsx” and save both files in the working directory. These files will be used in the next step.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/vox_padding.png
**FIG. 10. vox_padding.xlsx dataset.**


**Calibrating to CT data.** This process uses the `folder_swap.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/folder_swap.py>`_ and `FIVD_calbration.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/FIVD_calbration.py>`_ scripts. Both are ran within the **“Spyder” (V.5+)** Interactive Developer Environment (`IDE <https://www.spyder-ide.org/>`_). Open “folder_swap.py” in Spyder and hit run. You will be greeted by a Graphical User Interface (GUI):

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/folder_swap.png
**FIG. 11. Folder Swap GUI.**


Hit the Browse button for “Script Folder” and navigate to your script folder, then press Select. Then hit the Browse button for “New file folder” and navigate to your working directory and press Select. Finally hit the “Submit” button in the main GUI. This will change the input folder in all python and bash scripts to the working directory. 

Now run the “FIVD_calibration.py” script in spyder. This will create a new dataset and folder in the CT Data subfolder called “calibrated”. This consists of copies of the scaled low resolution dataset for slices representing the same region of interest as the high resolution dataset, with the scaled high resolution slices copied onto them according to the spatial offset between the low resolution and high resolution representations of the isolated features characterized in each dataset. The script also pads the fibre orientation data to the same absolute size as the scaled low resolution data, within the “[orientation data] padded” subfolder for each orientation dataset, created in the working directory. Within this folder, another subfolder is created called “calibrated” which consists of the fibre orientation data padded to the same absolute sixe as the calibrated CT data.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/calibration_data.png
**FIG. 12. Padded and calibrated fibre oirentation and SRCT data.**


*Creating inverted reslice for vertical registration.* The registration process (see below) uses a 2D map of summed Wide Angle X-ray Scatter (WAXS; collected alongside SAXS data at I22) intensity for the user to select a distinct region and compare it to a comparable map of the CT data. While the WAXS map is created in the registration script, the CT map must be created by the user. 

•	 Load the calibrated CT data into Fiji and select **Image>Stacks>Reslice**. 
•	 In the proceeding “Reslice” window, ensure the “Start At” position is set to “top” and hit the OK button. 
•	 This will create a “resliced” dataset oriented the same way as the WAXD map. 
•	 To create a single map image, select the resliced dataset and go to **image>stacks>z-project...**, and in the proceeding “Z-projection” window select “sum slices” before hitting OK. 
•	 This creates a single image, with grey vales the sum of all slices for the respective voxel. Finally, select edit>invert to invert these values (mirroring the WAXS map) and save this image as a tiff in the “calibration” subfolder of CT data.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/reslice_data.png
**FIG. 13. Reslice window in Fiji**

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/inverted_ct.png
**FIG. 14. Inverted CT map for registration**


*Processing kapton data.* The kapton tube is used for the spatial registration between the fibre orientation and SAXS data. To do this, the tube must be isolated in the calibrated CT data. 

•	 Load the calibrated data into Fiji, and either select **Image>Adjust> Brightness/contrast** or use the keyboard shortcut **ctrl+shift+c** to open the brightness/contrast window. 
•	 This window will show two distinct peaks in the greyscale histogram. Drag the “Minimum” bar to change the minimum dynamic range until only the kapton tube (and probably bone) is visible, and the “maximum” bar to the right-hand limit of the right-hand peak and hit the “apply” button. 
•	 This will change the dynamic range of the dataset so that the kapton window is clear. 
•	 Now go to **Analyse>tools>ROI manager** in the Fiji taskbar, which will open the ROI manager subwindow. 
•	 In the dataset, navigate to the first slice that shows a complete kapton tube (small sections may be lost from the overlapping by the high resolution data). Choose the “polygon selection” tool in the Fiji taskbar and draw a polygon around the inner surface of the kapton tube, then hit “add” in the ROI manager window. 
•	 Navigate to the slice **50 slices higher** than the current slice in the dataset and repeat the polygon selection and add to the ROI manager. 
•	 Repeat this for the rest of the dataset. 
•	 Once finished, in the ROI manager window, select every ROI (hold shift and select the first and last ROI), then hit the “more” button. Within the proceeding popup window, select “interpolate ROIs”. 
•	 This will interpolate for each slice between the created ROIs. 
•	 Then hit **ctrl+shift+n** to open the macro editor window and **File>open** within this window to open the **“ROI_manager.ijm”** macro. Hit **“run”** in this window to clear the inside of every ROI, removing the sample from the image and leaving only the kapton tube. 
•	 Save this dataset as an image sequence in a new subfolder within the “calibrated” folder called “kapton”. 

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/kapton_segment.png
**FIG. 15. Segmnetation of kapton data in Fiji**


**c.	Registration of fibre orientation and SAXS data**
--------------------------------------------------------

This process registers the padded fibre orientation and index data with tomographic SAXS data. It uses two python scripts:
 
 1.	`Registration_user_input.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/registration_user_input.py>`_


 2.	`Fivd_registration_cluster.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/fivd_registration_cluster.py>`_

*Inputting registration information.* This process is user operated, providing all of the necessary information for the main script, “FIVD_registration_cluster.py”. Registration_user_input.py is operated locally in the Spyder IDE. Load the script in Spyder and hit “run”. The script is GUI based, first providing a GUI window for the user to input folder locations and scan information:

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/reg_gui_1.png
**FIG. 16. Registration GUI 1.**

•	“Scan name” – this must be the same name inputted in the “vox_padding.xlsx” file.
•	“original CT data” – Hit the browse button and navigate to the CT data folder and press Select.
•	“inverted resliced CT Map” – Hit the browse button and select the inverted resliced z projection in “calibration”, and press Select.
•	“kapton CT dataset” – navigate to the “kapton” folder of the “calibrated” subfolder in CT data, then press Select.
•	“Beta/phi fibre tracing data” – hit the browse button and navigate to the working directory. Naviagate to the “calibrated” subfolder of the padded folder for the phi data and press Select.
•	“alpha/theta fibre tracing data” – hit the browse button and navigate to the working directory. Naviagate to the “calibrated” subfolder of the padded folder for the theta data and press Select.
•	“WAXS map data” – hit the browse button and navigate to the folder storing the SAXS data for the respective sample. Select the .nxs file for the coarse mapping scan (performed before each tomography to locate the sample within the sample holder, then press Select.
•	“Output folder” – hit Browse and navigate to the working directory, then press select.
•	“Script folder” – hit Browse and navigate to the folder including all downloaded TomoSAXS scripts, then press Select.
•	“Fibre tracing padding file” – navigate to the working directory and select the “vox_padding.xlsx” file, then press Select.
•	“Original CT voxel size (um)” – input the voxel size of the original CT data (default set to 1.625 μm).
•	“Inverted CT voxel sixe” – input the voxel size of the inverted CT map (default set to 6.5 μm).
•	“kapton data voxel size) – input the voxel size of the isolated kapton dataset (default set to 6.5 μm).
•	“Fibre tracing voxel size” – input the voxel size of the fibre orientation and fibre index data (default set to 5 μm).
•	“kapton tube diameter (um)” – input the diameter of the kapton tube used for the sample (default is 4000 μm but set to 6000 μm if using full IVD as this was the diameter used for these samples). 
•	“SAXS rotational direction” – set to the direction of rotation for the respective SAXS tomography (set to clockwise if using development scans).
•	“TomoSAXS binning” – binning of SAXS tomography data (default set to 1 – i.e. no binning).

Input all of the above data then hit the “submit” button.

This will open up a new GUI titled “3D registration: TomoSAXS parameters” for the user to input the parameters of the SAXS tomography:

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/reg_gui_2.png
**FIG. 17. Registration GUI 2.**

•	“Number of rotational angles in TomoSAXS scan” – default set to 9.
•	“Start angle” – default set to -90\ :sup:`o`.
•	“end angle” – default set to 90\ :sup:`o`.
•	“Angle of WAXS map” – default set to 00.

Input the parameters of the respective scan, then hit “submit”.

This will now open a third GUI, titled “Select files in TomoSAXS scan”. Hit the Browse button and navigate to the folder containing your SAXS data. Select all of the .nxs files in the respective scan (hold ctrl while selecting to highlight all scans, then press Select. Hit “ok” to submit.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/reg_gui_3.png
**FIG. 18. Registration GUI 3.**

This will now open up a pop-up, displaying the WAXS map. Target a characteristic element of the sample in this map (zoom using middle mouse button) and click to place a cross-hairs at this position. I usually use the highest point of the lower vertebral endplate. Once you are happy with the placement of the cross-hair. **Hit esc twice**.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/waxs_map.png
**FIG. 18. Map of WAXS intensity across sample.**

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/waxs_map_zoom.png
**FIG. 19. Selection of diagnostic sample region in WAXS data.**


A new pop-up will then appear displaying the cross-hair to double check that you are satisfied with the placement. If you hit “Cancel” you can reapply the cross-hair and repeat until you are happy. Once you are satisfied, hit the “yes” then “submit” button.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/waxs_selection.png
**FIG. 20. Confirming selection in GUI.**


This will trigger the inverted CT map to pop-up. Find the same point in this map and apply the cross-hair, repeating the above steps until you are satisfied that the cross-hairs are at the same position (vertically) in both maps. Hit “yes” and the “submit” button.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/ct_invert_map.png
**FIG. 21. Select same sample element in inverted SRCT map.**

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/ct_invert_map.png
**FIG. 22. Confirming selection in GUI.**


This operation will create the “registration_scan_info.pkl”, “bgcorr_info.pkl”, “registration_info.pkl”, and “registration_scan_files.npy” files in the folder selected as the output folder in the original GUI. These are used for the main registration script “FIVD_registration_cluster.py”.


*Registering data.* This operation is all performed in the “FIVD_registration_cluster.py” script. This can either be operated locally, or using SLURM on a computer cluster. If used locally, load “FIVD_registration_cluster.py” into Spyder and hit “Run”.

If using a cluster, navigate to the operations node. 
•	If using the DLS cluser, open a terminal and enter “ssh Wilson” – you may then be prompted to input your fedID password). 
•	Navigate to your script folder using **“cd /path/to/your/script_folder”**. 
•	Then enter **“sbatch --partition=#partion_you_want_to_use# FIVD_full_reg_bash.sh”**. for dls an example would be **“sbatch --partition=cs04r FIVD_full_reg_bash.sh”**. 
•	This should create an output similar to **“Submitted batch job 9999”**.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/slurm_terminal.png
**FIG. 23. Operating in cluster terminal.**

You can monitor the progress of the job using **“squeue –u YOUR_FEDID”** (swap YOUR_FEDID for your federal ID).


.. _Module 2:

**Module 2. Background correction.**
---------------------------------------

This process uses two scripts – `bgcr_user_input.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/bgr_user_input.py>`_ and `bgcr_cluster.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/bgr_cluster.py>`_. “bgcr_user_input.py” is run locally within Spyder. “bgcr_cluster.py” is most efficiently run using cluster computing.

“bgcr_user_input.py”. Load the script into Spyder and hit Run. You will be greeted by the following GUI, titled “TomoSAXS background correction”:

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/bg_corr_gui_1.png
**FIG. 24. background correction GUI.**

•	“SAXS data folder” – hit Browse and navigate to the folder containing the SAXS data for the respective scan.
•	“Mask file” – hit Browse and navigate to the folder containing the .nxs files for the mask created during the SAXS experiment (usually within the “processing” folder), select the mask.nxs file, and hit Select.
•	“Calibration file” – hit Browse and select the calibration.nxs file created during the SAXS experiment (usually in the same 2processing folder as the mask file”), select the file and hit Select.
•	“Scan name” - this must be the same name inputted in the “vox_padding.xlsx” file.
•	“Background file” – hit Browse and navigate to the SAXS data folder, select the .nxs file representing the empty kapton tube background scan and hit Select.
•	“Dispersant used?” – tickbox. If you used a hydrating fluid (e.g. PBS) during the scan, tick this box. This creates a new input called “Dispersant file” (see below).
•	“sample thickness file?” – tickbox. If you have complete the registration process, it will have generated a sample thickness file (see bloew), so tick and the “sample width file” input will appear.
•	“dispersant file” – hit Browse and navigate to your SAXS data folder. Select the .nxs file representing the background scan collected of the kapton tube filled with your hydrating fluid, then hit Select.
•	“sample width file” – hit Browse and navigate to the output folder selected for the registration process (usually your working directory). Select the file name “full_sample_thickness.npy” then hit Select. 
•	“scan info”: 
 •	For a tomoSAXS scan, select “TomoSAXS”.
 •	If a hydrating fluid was used, select “Correct background and dispersant”, if not then select “Corrct just background”.
 •	If your background scans were performed using a line scan (default for TomoSAXS), select “line-scan background”, if not select the most appropriate between “single background” (one background frame collected for empty sample holder, and sample holder filled with fluid, respectively) or “sample background” (sample of backgrounds taken but not using a line-scan).
 •	Select the appropriate sample holder from “kapton tube”, “kapton cuboid”, or “no chamber”.
 •	If “kapton tube” selected: 
   •	Input the sample width – or if a biological sample of no fixed width used, input “NA”.
   •	Input the sample holder width in mm (6 for full IVD).
   •	Input with the width of the kapton material in mm (0.125 for TomoSAXS experiments).


Hitting “submit” will open a second GUI, titled “Select files in TomoSAXS scan”. Hit the Browse button and navigate to the folder containing your SAXS data. Select all of the .nxs files in the respective scan (hold ctrl while selecting to highlight all scans, then press Select. Hit “ok” to submit.

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/reg_gui_3.png
**FIG. 25. tomoSAXS file selection GUI.**


This process creates the **“bgcorr_info.pkl”** and **“registration_scan_files.npy”** files in the directory set as the output folder. 


*“bgcr_cluster.py”.* This is most efficiently ran using cluster computing. Before using this, open the **“folder_swap.py”** script in Spyder, hit run and for “script folder”, browse to the “bcgr_corr” sub folder in your scripts folder and press Select. Then for “new file folder”, navigate to the working directory and press Select. Then hit Submit. This will convert the input folder of all the background correction scripts to the working directory.

If using a clyster, navigate to the operations node. if using the DLS cluser, open a terminal and enter “ssh Wilson” – you may then be prompted to input your fedID password). Navigate to your background correction script folder folder using **“cd /path/to/your/script_folder/bgcr_corr”**. Then submit the bash scripts for each TomoSAXS slice using: 

•	**“sbatch --partition=#partion_you_want_to_use# FIVD_bgcorr_0_bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# FIVD_bgcorr_1_bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# FIVD_bgcorr_2_bash.sh”**
•	…
•	**“sbatch --partition=#partion_you_want_to_use# FIVD_bgcorr_10_bash.sh”**


.. _Module 3:

**Module 3. Initial estimation of scattering intensity for individual fibres using single value decomposition (SVD).**
---------------------------------------

This process uses two scripts, most efficiently ran using cluster computing. The first script is `chi_exp_multiproc.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/1_chi_exp_multiproc.py>`_ , which builds a python library consisting of measured intensity values across the χ axis for every beampath in the SAXS tomography. Run this on the cluster. Before using this, open the “folder_swap.py” script in Spyder, hit run and for “script folder”, browse to the “chiExp_multiproc” sub folder in your scripts folder and press Select. Then for “new file folder”, navigate to the working directory and press Select. Then hit Submit. This will convert the input folder of all the background correction scripts to the working directory.

If using the DLS cluser, open a terminal and enter “ssh Wilson” – you may then be prompted to input your fedID password). Navigate to your background correction script folder folder using **“cd /path/to/your/script_folder/chiExp_multiproc”**. Then submit the bash scripts for each TomoSAXS slice using: 

•	**“sbatch --partition=#partion_you_want_to_use# chiExp_multiproc_0_bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# chiExp_multiproc_1_bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# chiExp_multiproc_2_bash.sh”**
•	…
•	**“sbatch --partition=#partion_you_want_to_use# chiExp_multiproc_10_bash.sh”**

Once these scripts have finished running, you can run the `svd_module.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/1_svd_module.py>`_ script. This uses the index and orientation data of fibres in each beampath to simulate scattering across the beampath and compare it to measured data held in the above library. Comparisons are used to estimate the amplitude of scattering intensity required for simulations to match measured intensity for every possible fibre using single value decomposition (SVD) (see LINK TO SVD PAGE). 

Run this on the cluster. Before using this, open the “folder_swap.py” script in Spyder, hit run and for “script folder”, browse to the “svd” sub folder in your scripts folder and press Select. Then for “new file folder”, navigate to the working directory and press Select. Then hit Submit. This will convert the input folder of all the background correction scripts to the working directory.

If using the DLS cluser, open a terminal and enter “ssh Wilson” – you may then be prompted to input your fedID password). Navigate to your background correction script folder folder using **“cd /path/to/your/script_folder/svd”**. Then submit the bash scripts for each TomoSAXS slice using: 

•	**“sbatch --partition=#partion_you_want_to_use# _0_svd_Bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# _1_svd_Bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# _2_svd_Bash.sh”**
•	…
•	**“sbatch --partition=#partion_you_want_to_use# _10_svd_Bash.sh”**


.. _Module 3:

**Module 4. Reconstruction of scattering metrics for individual fibres related to nanoscale structure and mechanics.**
------------------------------------------------------------------------------------------------------------------------
Once the SVD scripts have finished, you can now start the reconstruction process. This is performed using the **“recon_module.py”** script.

`recon_module.py <https://github.com/himadri111/saxs-docs-tutorial/blob/main/Code/1_recon_module.py>`_

This script performs simulations of each beampath using its fibre orientation and index data. If either single fibres or neighbouring fibres with sufficient overlap between each other and indepdendence from neighbouring fibres along χ are found to provide a proportion of total scattering above a certain threshold (see `here <https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon.rst>`_).

Run this on the cluster. Before using this, open the “folder_swap.py” script in Spyder, hit run and for “script folder”, browse to the “recon” sub folder in your scripts folder and press Select. Then for “new file folder”, navigate to the working directory and press Select. Then hit Submit. This will convert the input folder of all the background correction scripts to the working directory.

If using the DLS cluser, open a terminal and enter “ssh Wilson” – you may then be prompted to input your fedID password). Navigate to your background correction script folder folder using **“cd /path/to/your/script_folder/svd”**. Then submit the bash scripts for each TomoSAXS slice using: 

•	**“sbatch --partition=#partion_you_want_to_use# _0_recon_Bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# _1_ recon_Bash.sh”**
•	**“sbatch --partition=#partion_you_want_to_use# _2_ recon_Bash.sh”**
•	…
•	**“sbatch --partition=#partion_you_want_to_use# _10_ recon_Bash.sh”**
