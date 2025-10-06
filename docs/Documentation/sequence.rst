Sequence
=====

.. _Overview:

Overview
------------
Check out the :doc:`usage` section for further information.

Sequence of steps
------------------

.. _bgrcorr:
Bgr Correction
--------------
Described in :doc:`bgrcorr` section.
(Primary responsibility: EN/HG)
Load SAXS scans along with background files and apply adcorr correction
  a. Theory behind absorption corrections in variable geometries
  b. Example usage with experimental data
  c. Example usage with simulated data
    i. Using `pyFAI <https://pyfai.readthedocs.io/>`_ to generate synthetic data

.. _ctcoreg:
CT coregistration
------------------
(Primary responsibility: EN, input from AP/JC). Load CT image and co-register
Described in :doc:`coreg` section.

.. _datapreproc:
Data preprocessing
--------------------
Described in :doc:`imgtreatment` section.

  a. Diffuse scatter/background subtraction methods
  b. Mask effects and data augmentation

.. _3drecon:
3D reconstruction
------------------
Described in :doc:`saxsrecon` section.

  a. Initial estimation of scattering amplitudes
  b. Demonstration on:
    i. Using simulated organ-like geometries
    ii. Using real CT/SAXS scan data
