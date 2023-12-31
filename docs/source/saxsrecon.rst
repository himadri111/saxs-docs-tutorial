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

Since the mean fibril characteristics are known (:math:`\bf{f}`) the above set of equations can be evaluated at multiple angular (:math:`\chi`) points from 0 to :math:`\pi`, leading to a set of linear equations in :math:`a_{i}`. As the number of angular points can in principle be arbitrarily increased (:math:`Q_{\chi}`) we can adjust the parameters such that :math:`N \times Q_{\chi} > M`, leading to an overdetermined system of linear equations. These can be solved using the numpy library np.lstsq. A simple text to :meth:`dict.update`

.. code-block:: python
   :linenos:

   import antigravity

   def main():
       antigravity.fly()
