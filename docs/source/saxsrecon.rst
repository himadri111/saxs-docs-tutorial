.. role:: python(code)
  :language: python
  :class: highlight

SAXS-Recon
============

.. _Overview:

Overview
------------
Details of the reconstruction process, giving examples of single-, double- and higher-overlap cases.

.. _initial:

Initialisation
---------------
The fibre/voxel index is :math:`i` and the scan index is :math:`k`. Since the amplitudes of the fibres in each voxel :math:`a_{i}` is not known, the following estimation procedure is used. Approximate the fibres as having the mean fibril-parameter values of :math:`q_{i,0},w_{i,a},w_{i,\mu}` (which will be estimated from :math:`I^{k}(q)` for all scan points k). 

Calculate :math:`I^{k}_{r}(\chi)` for all scan points k and all rotation angles :math:`r` (with diffuse scattering subtracted as in the :: previous section ::. Define the observable scattering from each voxel as :math:`w_{i,r}^{k} \times V_{i}(q,\chi;{\bf{f}};(\alpha,\beta);r)` where :math:`\bf{f}` is a shorthand for "fibril characteristics" :math:`q_{i,0},w_{i,a},w_{i,\mu}`, and :math:`w_{i,r}^{k}` is a weighting factor which is proportional to the intersection of the X-ray microfocus beam with the voxel :math:`i` at a rotation :math:`r`. Since the X-ray beam is a pencil beam, each scattering path will contain only a small subset of voxels; i.e. most of the :math:`w_{i,r}^{k}` will be zero (across :math:`i`) for each :math:`k`. We denote :math:`\bf{\alpha}\equiv(\alpha,\beta).`

Then we can equate 

.. math::
   :nowrap:

   \begin{eqnarray}
      I^{k1}_{r1}(\chi)    & = & w_{1,r1}^{k1} \times a_{1} V_{1}(\chi;{\bf{f_{1},\alpha_{1}}};r1) + \ldots + w_{M,r1}^{k1} \times a_{M} V_{M}(\chi;{\bf{f_{M},\alpha_{M}}};r1)\\
      I^{k2}_{r1}(\chi)    & = & w_{1,r1}^{k2} \times a_{1} V_{1}(\chi;{\bf{f_{1},\alpha_{1}}};r1) + \ldots + w_{M,r1}^{k2} \times a_{M} V_{M}(\chi;{\bf{f_{M},\alpha_{M}}};r1)\\
      I^{k3}_{r1}(\chi)    & = & \ldots \\
      I^{kP}_{rN}(\chi)    & = & w_{1,rN}^{kP} \times a_{1} V_{1}(\chi;{\bf{f_{1},\alpha_{1}}};rN) + \ldots + w_{M,rN}^{kP} \times a_{M} V_{M}(\chi;{\bf{f_{M},\alpha_{M}}};rN)
   \end{eqnarray}

where :math:`k1,\ldots,kP` represents the scan points, :math:`r1,\ldots,rN` the number of rotations and :math:`1,\ldots,M` the number of voxels in the sample volume. 

Since the mean fibril characteristics are known (:math:`\bf{f}`) the above set of equations can be evaluated at multiple angular (:math:`\chi`) points from 0 to :math:`\pi`, leading to a set of linear equations in :math:`a_{i}`. As the number of angular points can in principle be arbitrarily increased (:math:`Q_{\chi}`) we can adjust the parameters such that :math:`N \times Q_{\chi} > M`, leading to an overdetermined system of linear equations. These can be solved using the numpy library :python:`np.linalg.lstsq` 

.. code-block:: python
   :linenos:

   import numpy as np
   """
   the equation works like
   1                2       3   ....   Nfibrils = I(chi1)
   a1*model(chi1) + a2*model(chi1) ... aN*model(chi1) = I(chi1)
   a1*model(chi2) + a2*model(chi2) ... aN*model(chi2) = I(chi2)
   .
   .
   .
   M = Nx*r*n_chi_svd
   a(M)*model(chiM) + a2*model(chiM) ... aN*model(chiM) = I(chiM)
   """
   """
   for chi_s_svd points (j: 0 to n_chi_svd-1)
   for each voxel in pdict, evaluate I(chi) and model weight at chi_s_svd points
   use mean values of q0, wa, wMu
   if the value is > threshold (e.g. 1% of max val) then 
   in matrix A, add model weight to "indx" column; "nxscan*n_chi_svd + j" row
   add it to the Ichi value at that point chi_s_svd; add to b_svd
   """
   ampval2=np.linalg.lstsq(a_svd_arr, b_svd_arr, rcond=None)[0]

.. _validationinitial:

Validation of initialisation
--------------------------------

.. _angular:

Simulating the tomoSAXS scan and rotation
-----------------------------------------------
Using these estimated amplitude and fibril characteristics, the 2D- and 1D- SAXS pattern can be simulated for each scan-point and rotation angle, as shown in the schematics below.

SHOW EXAMPLE PLOT

The color scheme for the voxels represents their status - green (unsolved), dark pink (already solved), and bright pink (solved at the current scan point). The procedure for solving the voxels is described next. 

.. _sv:

Identifying voxel-specific diffracting sectors
-----------------------------------------------
Each fibre :math:`i` contributes significantly (above a noise threshold) only at specific rotation angles :math:`i` and angular sectors :math:`\delta \chi_q`. To calculate this, using the estimated :math:`\{a_{i},\bf{f}_{i},,\bf{\alpha}_{i}\}_{M}` parameters, the total measured angular SAXS intensity :math:`I^{k}_{r}(\chi)` for each rotation angle :math:`r_{j}`, and the individual components :math:`w_{i,r}^{k} \times a_{i} V_{i}(\chi;{\bf{f_{i},\alpha_{i}}};r)` are calculated. 

SHOW EXAMPLE PLOT

As can be seen, some fibres are the predominant contributors to the SAXS signal in certain angular sectors :math:`\delta \chi_q` (shown as shaded), while other fibres are overlapping. For the first category, the fibre characteristics can be extracted by fitting the radial intensity profiles along these angular sectors to the model scattering functions. The angular sector where the fibre :math:`i` is the predominant contributor is estimated by taking the ratio of the simulated :math:`w_{i,r}^{k} \times a_{i} V_{i}(\chi;{\bf{f_{i},\alpha_{i}}};r)` to :math:`I^{k}_{r}(\chi)` over the full :math:`\chi` range, and finding if there exists any :math:`\delta \chi_q` where the ratio is :math:`>t_{s}` where :math:`t_{s}` is a single-voxel dimensionless ratio e.g. = 0.95 (i.e. the fibre :math:`i` contributes at least :math:`t_{s}` of the intensity over :math:`\delta \chi_q`. This step is called Single-Voxel Estimation, and an example of the :math:`I(q)` fits is shown below. 

SHOW EXAMPLE PLOT

For the next level of complexity, 
 
