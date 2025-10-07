Reconstruction of fibre scattering parameters
==================================

.. _Overview:

Overview
------------
Describes the process for the reconstruction of scattering metrics for individual fibrils in a TomoSAXS slice.

This process uses 3D diffraction modelling to estimate four properties related to nanoscale structure and strain in sampled collagen fibrils:
  
1.	D-period. The mean gauge length of the gap/overlap region of constituent collagen fibrils, measured as the peak position of meridional peaks along the Q axis of SAXS detectors.
  
2.	Wa. The Variation in D-period of constituent collagen fibrils, measured as the width of meridional peaks along the Q axis.
  
3.	WMu. The variation in 3D orientation of constituent collagen fibrils, measured asa the width of meridional peaks along the χ axis.
  
4.	Delta. Related to the width of the fibre, measured as the relative degree of ellipticality of the meridional peak (0 = straight peak; 1 = elliptical peak). 


.. _frame_intro-label:

.. image:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_1.png
**FIG. 1. Scattering properties.** (a) SAXS frame with axes of interest highlighted. Orange dashed line running through meridional collagen peak represents the azimuthal q axis  (measured in nm :sup:`-1`); dashed purple lines on either side of peak represent the radial χ axis (measured in degrees :sup:`o`). The relationship between Delta values and peak shape is highlighted by simulations inset on the lower RHS. (b) I(χ) integration between 0-180 :sup:`o`, displaying wMu as the region of χ represented by the scatter of the respective peak. (c) I(q) integration across the peak with q0 highlighted as the peak position of modelled function in q, and wa as the width of the model in q (nm :sup:`-1`).




These properties are measured for scattering instances where a user defined region (∆χ ; default = >10o) of the scattering signal across the χ axis is represented by the independent scattering of either 1(∆χs) or 2 (∆χo) collagen fibres.
These steps are repeated through multiple scans on the same dataset to first solve for all single solvable fibres, and then to include double (2) overlapping fibres in the optimisation process. Future optimisations may solve for single and double fibres concurrently, or using other refinements.


.. figure:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_2.png

**FIG. 2. Fibre fitting scenarios.** (a) Frame consisting of scattering from five fibrils, whose patterns are estimated along χ in (b). fibre 32 is predicted to provide >90% of scattering intensity for the portion of χ marked with blue vertical lines.
(c) Frame consisting of scattering from seven fibrils, whose patterns are estimated along χ in (d). The combined scattering of fibres 35 and 39 is predicted to provide >90% of scattering intensity for the portion of χ marked with red/green vertical lines.


Independent scattering instances are estimated through the simulation of the respective tomoSAXS scan. Here, the angular orientation values for each indexed fibre (α,β), registered to the voxel coordinates of the tomoSAXS scan (see `registration <https://himadri111-saxs-docs-tutorial.readthedocs.io/en/latest/coreg.html>`_), are used to simulate scattering in each beampath of the TomoSAXS scan. Simulated scattering is integrated across χ, and fibres that produce simulated scattering intensities that provide a percentage of total scatter above a user-defined threshold (rs; default is 90%) are isolated as “solvable”.


Isolated scattering signals are then sampled in the real data for the respective simulated beampath, using 1D azimuthal integration over a user-defined number of angular sectors (“cakes”) (nχc) of user-defined width along χ (∆χc). These samples are then fitted using nonlinear optimisation to the 3D model of diffraction to obtain the fibril parameters (q0, wa, wMu, delta) (see `fibre_model <https://himadri111-saxs-docs-tutorial.readthedocs.io/en/latest/fibremodel.html>`_). The default model for diffraction is set to Nelder-Mead, but can be reset. 


.. figure:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_3.png 

**FIG. 3. Fibril fitting.** (a) SAXS frame with scattering dominated by single fibril. Scattering region of fibril highlighted with red arrows. (b) I(chi) simulation of scattering in frame, with red arrows corresponding to those in (a). Blue, green, and yellow arrows indicate "cakes" of χ sampled of I(q) integration. (c) Comparison between measured I(q) integration and modelled I(q) using default values for scattering parameters. (d) Comparison between measured I(q) integration and modelled I(q) using optimised values for scattering parameters. 

Fitting produces optimised values for the 4 scattering metrics of the respective fibril. Fitted fibrils are in-turn used to aid deconvolution of neighbouring fibrils (along χ) for all of their scattering instances. This system of fibre fitting deconvolution continues across repeats of the TomoSAXS scan until every possible fibre is reconstructed. E.g. from a tomoSAXS slice with 700 indexed fibrils, 400 should be reconstructed.


.. figure:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_5.png

**FIG. 4. Fibril fitting.** (a) I(χ) integration of SAXS frame with fibril 26 isolated for single fibril fitting. (b) Comparison between measured I(q) integration and modelled I(q) using optimised values for scattering parameters of fibril 26. (c) I(χ) integration of SAXS frame with reconstructed fibril 26 highlighted with dash-dot, showing that solving of fibril 26 allows fibril 20 to be reconstructed. (d) Comparison between measured I(q) integration and modelled I(q) using optimised values for scattering parameters of fibril 20 and including fibril 26. Differing contributions of fibrils 20 and 26 highlighted in inset plot.


.. figure:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_4.png


This operation is performed independently for each vertical “slice” in a TomoSAXS scan, the details provided herein are with-respect to a single “slice”.

Prerequisite data:

1.	Spatially registered datasets of per TomoSAXS voxel:

  (a)	Alpha values (herein α): vertical orientation angle.

  (b)	Beta values (herein β): horizontal orientation angle.

  (c)	Fibre indexes.

  (d)	Fibre counts: counts of fibre tracing voxels that comprise each indexed fibre.

  (e)	Fibre weights: weighting value for the distance between fibre voxels and the SAXS beam centre.

2.	Background-corrected SAXS data for each orientation of the TomoSAXS scan (see `bgrcorr <https://himadri111-saxs-docs-tutorial.readthedocs.io/en/latest/bgrcorr.html>`_).

3.	Calibration file for the TomoSAXS scan

4.	Mask file for the TomoSAXS scan

5.	TomoSAXS scan information:
  a.	Start/end orientation (in degrees)
  b.	Rotation direction (clockwise vs anticlockwise)
  c.	Number of angular orientations

6.	“fibrilParam” dictionary comprising entries for each indexed fibre in the respective scan. Each entry contains the index, α, β, estimated amplitude, and !! for each fibre (created during the amplitude estimation process – LINK TO PAGE).
  Example entry:

    {'indx': 9727.0,
  
    'x': -0.5258687258687258,
  
    'y': 0.1,
  
    'alpha': -9457.0,
  
    'beta': 9257.0,
  
    'weight': 100.0,
  
    'count': 100.0,
  
    'solved': False,
  
    'intersected': False,
  
    'simu': {'q0': 0.2815943369754681,
  
     'wa': 0.002062910203719436,
  
     'wMu': 0.19952151910114227,
  
     'amp': 0.9809155838804501,
  
     'delta': 0.5005588922468329},
  
    'fit': {},
  
    'amplitude': 1,
  
    'number': 1801,
  
    'x_full': 16,
  
    'z': -0.020849420849420874,
  
    'z_full': 125,
  
    'initial_amp_est': 14444.973762829211}

7.	“rotated_beampaths” dictionary, containing dictionaries of indexed fibres for each orientation of the TomoSAXS scan, with corrected β values (change with orientation of beampaths relative to fibres), and added details for the weighting factor and voxel count of each fibre for the beampaths that they encounter for the respective orientation.

8.	“cake_params” dictionary, containing information for subsampling χ and q axes for 1D integrations (created during the amplitude estimation process – LINK TO PAGE).

9.	“initStruct” dictionary, containing initial estimates and estimated maximum variation of for scattering metric values (created during the amplitude estimation process – LINK TO PAGE).


.. _Glossary:

Glossary
------------


•	 “χ”: Radial axis across a 2D SAXS detector.
•	 “q”: Azimuthal axis from the beam centre in a 2D SAXS detector
•	I(χ): one-dimensional integration of scattering intensity across χ, sampled over a user-determined range in q.
•	I(χ): one-dimensional integration of scattering intensity across q, sampled over a user-determined range in χ.
•	“Scattering instance”: single interaction of a fibre and a SAXS X-ray beampath. Individual fibres can have multiple independent scattering instances, as they interact with different X-ray beampaths at different sample orientations.
•	“orientation”: here refers to the angular orientation of a single SAXS “map” (2D raster map) in the respective TomoSAXS scan. TomoSAXS scans are performed by taking multiple maps of a sample, each at a differing (user-defined) orientation relative to the sample.
•	 “Single fibre fitting”: Isolation and fitting of a scattering instance where a single fibre provides scattering intensity that is estimated to be above a user-defined threshold for minimum absolute scattering intensity, and minimum relative intensity with respect to total scattering intensity for a beampath, across a range in χ of user-defined length.
•	“double fibre fitting”: Isolation and fitting of two overlapping scattering instances, whose combined scattering intensity is estimated to be above a user-defined threshold for minimum absolute scattering intensity, and minimum relative intensity with respect to total scattering intensity for a beampath across a range in χ of user-defined length, and their peaks in scatter along χ are within 3o with respect to each other.   
•	“Fitting”: Estimation of scattering metrics for a fibre by optimising the fit between modelled I(q) projections and sampled I(q) projections of independent scatter of one (single fibre fitting) or two (double fibre fitting) fibre(s) through changing the values of these metrics.
•	∆χ: Minimum region of independent scatter for fitting scattering instance.
•	∆χs: Independent scattering region for a single fibril.
•	∆χo: Independent scattering region for two overlapping fibrils.
•	rs: Minimum percentage of total scatter for a scattering instance (single or double) to be considered independent. 
•	nχc: Number of angular cakes to sample independent scatter across q.
•	∆χc: Size of cakes in χ (degrees).


.. _Steps:

Steps
------------

Steps are:

1.	Perform TomoSAXS simulation with no fitting, recording all instances of potential single fibre fitting (where single fibres provide independent scattering), alongside the maximum estimated scattering intensity of the fibre in this instance.

2.	For fibres that have multiple independent scattering instances, isolate the instance with the highest simulated scattering intensity.

3.	Perform Single fibre fitting for these scattering instances. If the fit quality exceeds a pre-determined threshold (standard error < 30%), add fit data to the fibre index library.

4.	For all fibres with multiple independent scattering instances, whose most intense instance did not provide an accurate model fit to real data, repeat fitting for second-most intense scattering instance.

5.	Log all instances with inaccurate model fits, and repeat steps 1-4 (removing instances determined to be inaccurate) until no remaining single fibre fitting instances are detected. The deconvolution of single fibres should allow for the deconvolution of an increasing number of neighbouring fibres.

6.	Repeat the TomoSAXS simulation by a user-defined number of times (default = 10), allowing both single fibre fitting and double fibre fitting to be attempted for each respective independent scattering instance.

7.	If any fibres remain un-reconstructed, reduce the minimum absolute intensity threshold for remaining fibres and repeat the simulation by a user-defined number of times (default = 10).


.. _Methodology:

Methodology
------------
TomoSAXS simulation:

•	For each orientation of the TomoSAXS scan, loads the respective fibre dictionary from “rotated_beampaths”.
.. code-block:: python

  sys.path.append(r'path\to\scripts')
  import threeDXRD_080923 as t3d
  import recon_library as rec_lib
  
  """
  input varables and data
  """
  
  graph_display = False
     
  # input rotation range
  rot_range = [-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]

  #input number of cakes for sampling and reconstruction
  nslices = 3

  # input wavelength
  wavelen = 0.08856
  
  #load prerequisite data
  with open(fibrilParams_file, 'rb') as f:
      fibrilParams = pickle.load(f)
      
  with open(beampath_file, 'rb') as f:
      rotated_beampaths = pickle.load(f)
      
  with open(ichi_dict_file, 'rb') as f:
      ichiExp = pickle.load(f)
      
  with open(cake_params_file, 'rb') as f:
      cake_params_chi = pickle.load(f)
      
  with open(init_struct_file, 'rb') as f:
      initStruct = pickle.load(f)
  
  #list of fibril indexes    
  fibril_idxs = [k["indx"] for k in fibrilParams]
  
  """
  populate list of default scattering parameter values 
  """
  q0_m,wMu_m = initStruct['q0']["m"],initStruct['wMu']["m"]
  wa_m,delta_m = initStruct['wa']["m"],initStruct['delta']["m"]
  amp_m = 1e6
  
  sim_vals = [q0_m,wa_m,wMu_m,amp_m,delta_m]
  
  """
  generate sampling limits
  """
  q0m_low,q0m_high,nq0,wavelen = cake_params_chi["q1"],cake_params_chi["q2"],cake_params_chi["nq"],cake_params_chi["wavelen"]    
  
  chi1, chi2, nchi = 0, 180, 180
  chirange = np.linspace(chi1, chi2, nchi)
  gamma0, deltaGamma0 = np.radians(0.0), 0.01
  mu = np.radians(0.0)
  
  q_fitIq1, q_fitIq2, nq_fitIq = q0_m-0.02, q0_m+0.03, 120
  #q_fitIq1, q_fitIq2, nq_fitIq = q0_m-0.02, q0_m+0.03, 60
  dq_fitIq = (q_fitIq2-q_fitIq1)/nq_fitIq
  q_fitIqr = np.arange(q_fitIq1, q_fitIq2, dq_fitIq)
  
  qx, qy, qz, qxD, qyD, qxD_offset = t3d.calc_ewald_surface(0.08856,1.0, 1.0, 0.011)
  ewald = (qx, qy, qz)
  
  """
  Set thresholds for scattering independence and detection
  """
  
  threshold_detection = 0.5e7
  chiRefWindow=10
  threshold_interference = 10
  
  #background corrected SAXS file
  slice_saxs_file = glob.glob(os.path.join(data_path, "*.h5"))[0]
  
  #isolate vertical slice being reconstructed
  vert_slice = slice_saxs_file.split("\\")[-1].split(".")[0]
  vert_slice = int(re.findall(r'\d+', vert_slice)[0])
  
  #load indexed fibre data for respective slice
  slice_index_index = np.copy(np.asarray([k[vert_slice] for k in index_data]))
  
  """
  load mask and identify points in chi that are masked within the sampling region of q and chi    
  """
   
  Nx, Ny = slice_index_index[0].shape[0], 1
  Nz = slice_index_index[0].shape[1]
  samplex1, samplex2, Nxf = -.6,.6,Nx
  sampley1, sampley2, Nyf = .1,.1,Ny
  samplez1, samplez2, Nzf = -.6,.6,Nz
  dxs, dys, dzs = (samplex2-samplex1)/Nx,(sampley2-sampley1)/Ny,(samplez2-samplez1)/Nz
  
  recon_mask_chi = rec_lib.ichi_sample(0,Mask,0,0,recon_cake_params,a1,0,slice_saxs_file,[-180,0],
                        fibre_chi = False,fit_chi=False,iq_plot = True)
  
  recon_mask_chis = np.flip(recon_mask_chi[0][1])
  recon_mask_pts = np.where(recon_mask_chis==0,True,False)
  
  """
  Create sampling_params object - a list of sampling parameters
  """
  sampling_params = [chi1,chi2,chirange,nchi,qx,qy,qz,dxs,q0_m,q0m_low,q0m_high,nq0,
                     threshold_interference,threshold_detection,q_fitIqr,binning]
  
  """
  Create list of single solve instances
  """

  single_solves = []

  """
  let r be the rotation we are currently simulating
  """
  
  r = 0
  
  rot_beampath = np.asarray(rotated_beampaths[r])



•	This dictionary object is split into entries that represent every beampath in the mapping of the respective SAXS orientation (for this vertical “slice”).
.. code-block:: python
  
  """
  let i be the beampath index for this rotation, and path_dict the dictionary entry for the beampath
  """
  
  for i, path_dict in enumerate(rot_beampath):
      
      if path_dict != None: 
          """
          If the dictionary entry contains information (beampaths not predicted 
          to encounter any fibres have empty entries) 
          """
                      
          if path_dict["voxels"] != None:
              
              """
              If the entry contains listed and indexed voxels
              """
              
              if i in ichiExp[r] and type(ichiExp[r][i]["kapton"]) == bool:
                  
                  """
                  If the beampath did not encounter the kapton edge at this orientation
                  """

•	Each beampath entry is inspected, and those that interact with indexed fibres are investigated. 
.. code-block:: python

  Ichi1D = np.arange(chi1,chi2,nchi)                
                  
  Ichi1D_unsolved = np.zeros_like(Ichi1D)
  Ichi1D_solved = np.zeros_like(Ichi1D)
  max_chis,slv_max_chis = [],[]
  chi_indxs,slv_indxs = [],[]
  ichis,slv_ichis = [],[]
  x_locs,y_locs,z_locs = [],[],[]
  for idx, voxel in enumerate(path_dict["voxels"]):
      vox_indx = voxel["fibril_param"]["indx"]
      indx = np.where(fibril_idxs == vox_indx)[0][0]
      fibrilParams[indx]["intersected"]=True
      #function for simulating I(chi) integration - see recon functions library github page
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


•	The total combined scattering intensity of all fibres encountered along the beampath that have yet to be reconstructed is simulated across a user-defined (default 0-180 :sup:`o`) χ-range, using their α and β values and estimated amplitude.

.. figure:: https://github.com/himadri111/saxs-docs-tutorial/blob/main/docs/source/recon_Fig_6.png

•	Single fibre fitting: The estimated scattering signal for each individual unreconstructed fibre through this region is then compared to the total scattering intensity. Fibres that produce a percentage of total scatter above a user-defined threshold (default = 90%) for a subregion of this χ-range wider than a user-defined threshold (default = 10o) are denoted as available for single fibre fitting, with their details added to a log list of potential fibres for single fibre fitting.

•	Double fibre fitting: Simulated scattering signals for fibres that do not meet either of the above thresholds are combined with those of fibres with scattering peaks within 3o of the peak of the fibre along χ. These combined signals are then compared to the total scattering across the χ-range. If this passes the above thresholds, both fibres are denoted as available for double fibre fitting, with their details added to a log list of potential fibres for double fibre fitting.



