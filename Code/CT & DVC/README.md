# CT & DVC Analysis Code

This repository contains the code for performing synchrotron computed tomography (sCT) image analysis, fibre tracing, and digital volume correlation (DVC) analysis as described in:

**"Multimodal X-ray imaging reveals hierarchical fibre mechanics"**  
Newham et al., 2025

## Overview

This code performs microscale structural analysis and strain mapping on intervertebral disc (IVD) samples using synchrotron CT imaging and DVC. The analysis traces collagen fibres in the annulus fibrosus and measures their mechanical response under compressive loading.

### Key Capabilities

- Fibre tracing from high-resolution synchrotron CT volumes
- Digital volume correlation (DVC) for 3D strain mapping
- Fibre-based orientation analysis (theta and phi angles)
- Calculation of fibre-level mechanical metrics including:
  - Fibre strain and curvature
  - Principal strain components
  - Radial and circumferential tissue strains
- Integration with TomoSAXS reconstruction for nano-to-microscale mechanics

## Requirements

### Software Dependencies

- **Avizo** - for 3D fibre tracing and spatial graph generation
- **MATLAB** - for fibre orientation analysis
- **Python 3.x** - for additional processing scripts
- **iDVC** - for DVC strain calculation
- **ImageJ/Fiji** - for image preprocessing

### Python Libraries

```python
numpy
pandas
scipy
matplotlib
```

## Data Requirements

All input data files are available on Figshare (currently embargoed). The dataset includes:

### CT Image Data

- `167208_2560xy_2160z_pag100_us2p5.raw` - Synchrotron CT of IVD sample at 1N preload (reference)
- `167209_2560xy_2160z_pag100_us2p5.raw` - Synchrotron CT after 50 µm compression (deformed)
  - Dimensions: 2560 × 2560 × 2160 voxels
  - Format: 16-bit unsigned, big endian
  - Voxel size: 1.625 µm
  - Reconstruction: Paganin phase retrieval (δ/β = 100), unsharp mask applied

### Orientation Data for TomoSAXS

- `167208_fibreID_5x5.tif` - Fibre identification map (8.125 µm voxel size)
- `167208_theta_5x5.tif` - Theta orientation (SAXS α angle)
- `167208_phi_5x5.tif` - Phi orientation (SAXS β angle)

### Intermediate Processing Files

The Figshare dataset also includes intermediate files from the analysis pipeline (see Data Structure section below).

## Workflow

### 1. Image Preprocessing

CT images are reconstructed using filtered back projection with Paganin phase retrieval and enhanced with an unsharp mask in ImageJ:

- Radius: 2.5 pixels
- Weight: 0.9

### 2. Fibre Tracing (Avizo)

**Input:** `167208_2560xy_2160z_pag100_us2p5.raw`

**Output:** 
- `167208_fibres.xml` - Traced fibre data
- `167208_fibres_spatialgraph.am` - Avizo spatial graph
- `167208_fibres.xlsx` - Excel conversion for Python

Trace collagen fibres through the annulus fibrosus using Avizo's fibre tracking module. This generates a 3D representation of the fibre network.

### 3. Initial Orientation Analysis (MATLAB)

**Script:** `TomoSAXS_initial_orientation.m`

**Inputs:**
- `167208_fibres.xml`
- `167208_theta_5x5.tif`
- `167208_phi_5x5.tif`

**Outputs:**
- `167208_orientation.csv` - Initial fibre orientation (x, y, z, fibre ID, theta, phi)
- `167208_pointcloud_spacecurve.mat` - Fibre fitting data for loaded analysis

Calculates the initial orientation of each traced fibre in terms of theta (in-plane angle) and phi (out-of-plane angle) from the CT data.

### 4. Digital Volume Correlation (DVC)

**Software:** iDVC (https://tomographicimaging.github.io/iDVC/executable.html)

**Input Configuration:** `dvc_in_167208.txt`

**Inputs:**
- Reference volume: `167208_2560xy_2160z_pag100_us2p5.raw`
- Deformed volume: `167209_2560xy_2160z_pag100_us2p5.raw`
- Fibre point cloud: `167208_dvc_pointcloud.txt`

**Outputs:**
- `167208_displacement.disp` - 3D displacement field
- `167208_tissuestrain-sw75.Lstr.csv` - Tissue-level strain tensor

Run DVC analysis to calculate 3D displacement and strain fields between the reference and loaded configurations.

### 5. Loaded Orientation Analysis (MATLAB)

**Script:** `TomoSAXS_loaded_orientation.m`

**Inputs:**
- `167208_pointcloud_spacecurve.mat`
- `167208_displacement.disp`

**Outputs:**
- `167209_orientation.csv` - Loaded fibre orientations
- `167208_fibre_metrics.csv` - Per-point metrics (x, y, z, theta, phi, delta theta, delta phi, curvature k, delta k, etc.)
- `167209_pointcloud_mask.csv` - Points for creating loaded orientation masks

Propagates the fibre network through the deformation field and calculates changes in orientation and curvature.

### 6. Strain Analysis

#### Radial and Circumferential Strain

**Script:** `contourlines_strain.py`

**Inputs:**
- `167208_AFstrain_labels.tif` - Annulus fibrosus segmentation
- `167208_tissuestrain-sw75.Lstr.csv`

**Output:**
- `167208_radcirc_strains.csv` - Radial and circumferential strain components

#### Strain-Fibre Direction Analysis

**Script:** `angleofeigenvectors.m` or `eigenvectoranglewithfibre.py`

**Inputs:**
- `167208_tissuestrain-sw75.Lstr.csv`
- `167208_orientation.csv`

**Output:**
- `167208_strain_direction.csv` - Angles between principal strain axes and fibre directions (ep1_alpha, ep3_alpha)

### 7. Fibre-Based Metrics Compilation

**Output:** `167208_fibre_means.csv`

Final compilation of per-fibre mean values including:
- Fibre orientation (theta, phi) and changes (delta theta, delta phi)
- Curvature and curvature changes
- Fibre strain (with and without polynomial smoothing)
- Principal strains (1st and 3rd)
- Angles between principal strain vectors and fibre direction
- Radial and circumferential strains

## File Naming Convention

Files follow the convention: `[scan_number]_[description].[extension]`

- **167208** - Reference scan (preload)
- **167209** - Loaded scan (50 µm compression)

## Data Structure

### Input Files
```
├── Raw CT data
│   ├── 167208_2560xy_2160z_pag100_us2p5.raw
│   └── 167209_2560xy_2160z_pag100_us2p5.raw
│
├── Orientation data (for TomoSAXS)
│   ├── 167208_fibreID_5x5.tif
│   ├── 167208_theta_5x5.tif
│   └── 167208_phi_5x5.tif
│
└── Segmentation
    └── 167208_AFstrain_labels.tif
```

### Output Files
```
├── Fibre tracing outputs
│   ├── 167208_fibres.xml
│   ├── 167208_fibres.xlsx
│   ├── 167208_fibres_spatialgraph.am
│   └── 167208_orientation.csv
│
├── DVC outputs
│   ├── 167208_dvc_pointcloud.txt
│   ├── 167208_displacement.disp
│   └── 167208_tissuestrain-sw75.Lstr.csv
│
├── Loaded analysis outputs
│   ├── 167209_orientation.csv
│   ├── 167208_fibre_metrics.csv
│   └── 167209_pointcloud_mask.csv
│
├── Strain analysis outputs
│   ├── 167208_radcirc_strains.csv
│   ├── 167208_strain_direction.csv
│   └── 167208_fibre_means.csv (final summary)
│
└── Intermediate files
    └── 167208_pointcloud_spacecurve.mat
```

## Key Processing Scripts

### MATLAB Scripts

- `TomoSAXS_initial_orientation.m` - Initial fibre orientation from CT
- `TomoSAXS_loaded_orientation.m` - Propagate fibres through deformation
- `angleofeigenvectors.m` - Calculate angles between strain and fibre directions

### Python Scripts

- `contourlines_strain.py` - Radial/circumferential strain calculation
- `eigenvectoranglewithfibre.py` - Alternative strain-fibre angle calculation
- `TomoSAXS_initial_orientation.py` - Python version of initial orientation (if used)

### Configuration Files

- `dvc_in_167208.txt` - iDVC configuration parameters

## Analysis Parameters

### CT Reconstruction
- **Paganin phase retrieval:** δ/β = 100
- **Unsharp mask:** radius = 2.5 pixels, weight = 0.9

### DVC Settings
- **Subvolume size:** sw75 (75 voxel subset size)
- **Strain calculation:** Lagrangian strain tensor

### Voxel Sizes
- **CT data:** 1.625 µm
- **Orientation maps:** 8.125 µm (5× CT voxel size)

## Integration with TomoSAXS

The outputs from this CT/DVC analysis serve as critical inputs for TomoSAXS reconstruction:

1. **Fibre orientation data** (`167208_theta_5x5.tif`, `167208_phi_5x5.tif`) - Guides SAXS orientation sampling
2. **Fibre ID mapping** (`167208_fibreID_5x5.tif`) - Links SAXS voxels to specific fibres
3. **Loaded orientation** (`167209_pointcloud_mask.csv`) - Creates masks for loaded state reconstruction
4. **Strain data** - Calibrates nano-to-microscale mechanical relationships

## Citation

If you use this code, please cite:

```
Newham, E., Parmenter, A.L., et al. (2025). 
Multimodal X-ray imaging reveals hierarchical fibre mechanics. 
bioRxiv. doi: 10.1101/2025.09.19.677294
```

## Data Availability

All input data and example outputs are available on Figshare (currently embargoed):
[Link to be updated when embargo is lifted]

## Support

For questions about the code or methodology, please refer to the paper's methods and supplementary materials, or contact the corresponding authors.

## License

[To be specified]

## Acknowledgments

This work was performed using synchrotron imaging at Diamond Light Source, UK. The TomoSAXS method was developed to enable full-field 3D mapping of fibril-to-fibre mechanics across intact tissues.

Funding: Engineering and Physical Sciences Research Council (EPSRC) grants EP/V011235/1, EP/V011006/1, EP/V011383/1, EP/V011065/1; Medical Research Council (MRC) grants MR/R025673/1, MR/V033506/1.
