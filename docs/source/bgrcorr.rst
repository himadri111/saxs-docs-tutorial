Background Corrections
=======================

.. _Overview:

Overview
------------
Describes the steps in normalising and subtracting the background for a sample in cylindrical Kapton tube.

.. _load:
Loading Data

Summarize folder structure, needed files (including calibration), sample file, water bgr, empty kapton bgr, empty air bgr, input file locations, ...

.. _principles:
Principles of Background Corrections
--------------
Summarize relations in A. Smith et al J. App. Cryst. (2017)
(Primary responsibility: Elis Newham)
Summarize equations

Load SAXS scans along with background files and apply adcorr correction
  a. Theory behind absorption corrections in variable geometries
  b. Example usage with experimental data
  c. Example usage with simulated data
    i. Using `pyFAI <https://pyfai.readthedocs.io/>`_ to generate synthetic data

.. _variablethickness:
Accounting for variable thickness with CT image
------------------
(Primary responsibility: Elis Newham with input from Alissa Parmenter and Jishizhan Chen). Code examples

.. _examplesim:
Simulated data examples
--------------------
Linescan of tissue plane with small voxel size in saline/PBS

.. _exampleexp:
Experimental data examples
--------------------
Linescan of tissue plane from experimental data in saline/PBS
