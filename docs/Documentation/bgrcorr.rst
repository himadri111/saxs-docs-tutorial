Background Corrections
=======================

.. _Overview:

Overview
------------
Describes the steps in normalising and subtracting the background for a sample in cylindrical Kapton tube. Background correction is based on the `Adcorr <https://github.com/DiamondLightSource/adcorr>`_ library developed for background correction of SAXS data collected at the I22 beamline of the Diamond Light Source synchrotron.

TomoSAXS samples have background SAXS signals from two main sources:

1. Cylindrical kapton tube ("Background")

2. Phosphate buffer solution (PBS) ("Dispersant")

.. _frame_intro-label:
.. figure:: frame_comparison_clipped.png

.. image:: tube_comp.png

Background correction for tomoSAXS samples is complicated by several factors:

1. The cylindrical shape of kapton tubes changes the proportion of scattering background (amount of PBS solution per-beampath

2. The non-uniform shape of samples changes the proportion of PBS displaced by the sample both between beampaths and between rotations.

These factors must be accounted for when applying the Adcorr background correction procedure on tomoSAXS data. 

This technique accounts for these factors by:

1. Using the known diameter of the kapton tube to estimate the thickness of the tube for every beampath and thus the amount of undisplaced PBS.

2. Estimating the thickness of the sample for every beampath of the tomoSAXS scan during the co-registration `Co-registration <https://himadri111-saxs-docs-tutorial.readthedocs.io/en/latest/coreg.html>`_ process.

These values allow estimation of the displaced volume fraction for every frame of the tomoSAXS scan.


Prerequisite data:
^^^^^^^^^^^^^^^^^^

1. TomoSAXS dataset: series of individual SAXS raster map files and associated data (accessed using .nxs file) representing each of the tomoSAXS orientations.

2. Mask file: file ( Figure :ref:`mask-label` ) created during calibration of experiment at I22 beamline identifying regions of 2D SAXS frames to be masked from analyses.

3. Calibration file: file ( Figure :ref:`calib-label` ) created during calibration of experiment at I22 beamline containing calibration data for analysis of 2D SAXS frames.

4. Background file: raster map :ref:`frame_intro-label`  acquired using the same parameters as tomoSAXS data taken of an empty kapton tube (diameter the same width as that used in the respective tomoSAXS scan).

5. Dispersant file: raster map :ref:`frame_intro-label` acquired using the same parameters as tomoSAXS data taken of a kapton tube (diameter the same width as that used in the respective tomoSAXS scan) filled with hydrating fluid used in tomoSAXS scan (e.g. PBS/phosphate buffer saline solution).

6. Sample thickness file: file containing data for the estimate thickness of the sample across every beampath in the tomoSAXS scan. Can be either a .txt file or .npy file (.npy preferred). Generated as part of the `Co-registration <https://himadri111-saxs-docs-tutorial.readthedocs.io/en/latest/coreg.html>`_ process. npy file is saved as a 3D arary; first dimension for tomoSAXS orientation; second dimension for tomoSAXS slice; third dimension for estimated sample thickness for each frame. 

.. image:: sample_thickness_structure.png

7. "adcorr_multiFuncs": library python file for multiprocessing of background correction data.

8. `Beamline-specific information <https://diamondlightsource.github.io/adcorr/main/tutorials/i22_corrections.html>`_ for the Adcorr procedure:

.. code-block:: python

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


Steps:
^^^^^^^

1. User input folder and scan  parameters (tomoSAXS files, output directories, type of scan, kapton width etc).

2. Load background and dispersant data and estimate position of kapton edges.

3. Load sample thickness data.

4. for each scan: 

  a. Load SAXS data slice by slice and estimate position of kapton edges.

  b. Use these estimates to estimate the thickness of the kapton tube at each beampath.

  c. Use the estimated sample thickness for the same beampath to in-turn estimate the amount of PBS displaced by the sample for the beampath.

  D. Input these values into the `pauw_dispersed_sample_sequence() <https://github.com/DiamondLightSource/adcorr/blob/main/src/adcorr/sequences/pauw.py>`_ function for this frame to perform the background correction.

5. Save corrected data in new hdf5 file.


.. _gui:
1. User input
--------------

User input is performed using a series of graphical user interfaces (GUIs), where the user inputs the correct folders, files, and scan parameters for the background correction process.

The first:

.. image:: adcorr_gui_1.png

Reads in:

a. "SAXS data folder" - the folder containing the tomoSAXS data.

b. "Mask file" - the "SAXS_mask.nxs" file for the respective experiment.

c. "Calibration file" - the "SAXS_calibration.nxs" file for the respective experiment.

d. "Scan name" - name for scan to use for saving backgroudn corrected files (if left empty then saves the same names as the individual tomoSAXS scan names).

d. "Background file" - the .nxs file for the empty kapton tube SAXS raster map.

e. "Dispersant used?" - tick box to state that a dispersant file should be used in the background correction. Creates new file selection box to select the .nxs file for the pbs-filled kapton tube raster map.

f. "Sample thickness folder?" - tick box to state that a sample thickness file should be used in the background correction. Creates new file selection box to select the file.

g. "Script folder" - the folder containing the python file "tomoSAXS_disp_multiproc" python file.

h. "Output folder" - the folder chosen for outputting background corrected data for.

"Scan info" - three check boxes for the nature of the scan. for tomoSAXS, select:

  "tomoSAXS"

  "Line-scan background"

  "kapton tube"

Then input the respective values for the kapton tube and gross sample thickness.


The second:

.. image:: adcorr_gui_2.png

Reads in the nexus files for each of the individual raster maps that make up the respective tomoSAXS scan.


.. load_data:
2. Loading data
----------------

Data types loaded for all datasets (background, dispersant, and sample) are:
  i.   Frames (2D SAXS detector frames)
  ii.  Count times (exposure time in seconds for each frame)
  iii. Incident flux (I0 data for each frame)
  iv.  Transmitted flux (bs diodes data for each frame)

.. code-block:: python

   """
   Example for loading dispersant data
   """
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



a. The script starts by loading the data for the the background (empty kapton tube) and dispersant (filled kapton tube) data. The outputs (not shown during the script) are: 

.. image:: bg_and_disp.png

.. image:: bg_disp_sum_comp_clip.png

b. Then finds the edges of the kapton tube for both datasets:

.. code-block:: python
  
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


.. image:: Background_kapton_edges.png

.. image:: Dispersant_kapton_edges.png

c. The script then loads the Sample thickness data:

.. image:: sample_thickness_plot.png

.. image:: sample_thickness_img_clip.png
  :width: 400

and corrects for inconsistencies (from low density regions of fibre tracing data :ref:`.. padding:` ) by fitting a 3rd order polynomial to the peaks in the thickness dataset:

.. code-block:: python

  from scipy.signal import find_peaks
  
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
  

.. image:: sample_thickness_comp_3deg.png

.. image:: corrected_frame_thickness.png


d. The script then loads the mask:

.. _mask-label:
.. figure:: Mask.png


and calibration data for the tomoSAXS scan.

.. _calib-label:
.. figure:: calib.png


.. bg_corr:
3. Background correction
-------------------------

Background correction is performed on a per-scan basis for tomoSAXS (i.e. each individual raster map representing a sequential sample orientation is loaded individually and backgroundcorrected). For each scan, an empty hdf5 file is created for populating with corrected frames.

Background correction is then applied on a per-slice basis. Each vertical slice is loaded sequentially, and for each slice:

a. A new row is created in the hdf5 file for the respective tomoSAXS orientation.

b. the kapton tube edges are found

.. image:: sample_kapton_edges.png

c. The data for the sample:
  a. SAXS frames
  b. Count times
  c. Incident flux (I0 data)
  d. transmitted flux (bs diodes data)
  e. Sample thickness data for this slice
are then subsampled to just those frames within the kapton edges

d. The X axis positions are found for each of these frames, and the difference between these positions and the lefthand-side (lhs) kapton edge are used to subsample the frames, count times, incident flux values, and transmittedflux values  from the equivalent position of the kapton tube width for the background and dispersant data.

e. The width of the kapton tube can then be estimated for each frame by estimating the chord length of the frame from its distance from the centre point of the tube:

.. code-block:: python

  disp_sample_range = sample_axis[-1]-sample_axis                
  disp_dist_frm_ctr = np.sqrt((disp_sample_range-(disp_sample_range[0]/2))**2)                
  choord_len = [((disp_dist_frm_ctr[0]**2)-(disp_dist_frm_ctr[k]**2))*1000 for k in np.arange(0,len(disp_dist_frm_ctr),1)]
  choord_len = np.asarray(choord_len)*1e-3

f. We can then input the subsampled data (frames, count times, incident flux, transmitted flux), as well as the estimated kapton tube width, and estimated sample width, and original index (position within the scan) for every subsampled frame into the "tomSAXS_disp_mutliproc()" multiprocessing function. This function uses multiprocessing to apply the `pauw_dispersed_sample_sequence() <https://github.com/DiamondLightSource/adcorr/blob/main/src/adcorr/sequences/pauw.py>`_ function to background correct each subsampled frame, using the ratio between the sample thickness and kapton tube width as a metric for the displaced volume fraction.

g. For each frame, this function outputs a background corrected frame, and its original index:

.. image:: orig_vs_corr_clipped.png

.. image:: full_iq_comp_plot.png

.. image:: full_iq_log_comp_plot.png

.. image:: 3rd_peak_iq_comp_plot.png

.. image:: 3rd_peak_iq_log_comp_plot.png

.. image:: 3rd_peak_log_just_iq_comp_plot.png


h. The original frames for the entire tomoSAXS slice are then copied, and copies are replaced by the corrected frame for the respective index.

i. The new slice containing corrected frames is then saved into the hdf5 file for corrected data. 


.. _principles:
Principles of Background Corrections
--------------
Summarize relations in A. Smith et al J. App. Cryst. (2017)
(Primary responsibility: EN/HG)
Summarize equations
