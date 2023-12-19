Sequence
========

.. _Overview:
Overview
------------
As described :doc:`usage`, the following is a hyperlinked series of data reduction steps reducing SAXS data, aligning with CT images, applying the reconstruction pipeline and displaying the output 2D and 3D parameter fields. The 

Sequence of steps
-----------------

.. _bgrcorr:
Bgr Correction
--------------

Load SAXS scans along with background files and apply adcorr correction
  a. Theory behind absorption corrections in variable geometries
  b. Example usage with experimental data
  c. Example usage with simulated data
    i. Using pyFAI <https://pyfai.readthedocs.io/>`_ to generate synthetic data

.. _ctcoreg:
CT coregistration
-----------------

Load CT image and co-register 
