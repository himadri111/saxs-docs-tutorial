Artefact corr, diffuse bgr subtraction, denoising etc.
=======================================================

.. _Overview:
Overview
------------
Background corrected SAXS frames need to be processed to:
1. extract the ROI of interest (a sector around 3rd meridional order) using pyFAI based cake remapping
2. augment ROI data using chi->chi+180 symmetry
3. subtract the diffuse SAXS signal
4. apply denoising filter if needed
5. correct for streak artefacts if needed

.. _roi:
Extract ROI of interest
-------------------------
Using synthetic SAXS data (pyFAI generated)
1. code showing remapping with images

.. _augment:
Augment using SAXS symmetry
---------------------------
Using above data set
1. code showing augmentation of image with image

.. _diffuse bgr:
Subtract diffuse bgr
-------------------------
Using synthetic SAXS data (pyFAI generated) with a meridional and diffuse ellipsoidal term
1. Display image with and without diffuse term

.. code-block:: python

  Some ruby code

.. image:: testerpillar1.jpg
  :width: 400
  :alt: Alternative text

2. Display I(q) profile with and without diffuse term
3. Fit different background terms (cubic spline, exponential background, power law) and test fit quality
4. Repeat with noise added
5. Repeat for different levels of peak height and diffuse bgr, plotting original and final meridional component

.. _denoising:
Apply denoising filter
-------------------------
noisy-to-noisy filter: details to be worked out

.. _streak:
Correct streak artefacts
-------------------------
correct streak artefacts: details to be worked out
