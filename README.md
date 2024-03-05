# Integration of PyTorch and 3D Slicer

This repository contains the code for two Slicer modules that can be used to segment brain structures on T1-weighted MRIs.

Segmentations are performed using convolutional neural networks (CNNs), i.e., deep learning models. They take less than a minute on a graphics processing unit (GPU).

This is a project for the [35th NA-MIC Project Week](https://github.com/NA-MIC/ProjectWeek/tree/master/PW35_2021_Virtual/Projects/PyTorchIntegration).

## Installation

### Download module

#### Option 1: clone repository

```shell
git clone https://github.com/fepegar/SlicerParcellation.git
```

#### Option 2: download zipped repository

[Download the zipped directory](https://github.com/fepegar/SlicerParcellation/archive/refs/heads/master.zip) and unzip it.

### Add directory in Slicer

In Slicer, go to `Edit -> Application Settings -> Modules` and add the cloned/downloaded folder to the `Additional module paths`. When prompted, restart Slicer.

## Modules

## Brain Parcellation

![Brain Parcellation module](./screenshots/parcellation.png)

Based on [Li et al., 2017, *On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task*](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28).

It splits the brain in 160 different structures, similar to the geodesic information flows (GIF) algorithm by [Cardoso et al. 2015](https://pubmed.ncbi.nlm.nih.gov/25879909/).

The PyTorch model was ported from NiftyNet at the MICCAI Educational Challenge 2019: [Combining the power of PyTorch and NiftyNet](https://github.com/fepegar/miccai-educational-challenge-2019/).

The [`highresnet`](https://github.com/fepegar/highresnet) Python package can be installed running `pip install highresnet` to parcellate images outside 3D Slicer.

## Brain Resection Cavity Segmentation

![Brain Resection Cavity Segmentation module](./screenshots/cavity.gif)

Based on [Pérez-García et al., 2021, *A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections*](https://link.springer.com/article/10.1007/s11548-021-02420-2).

The segmentation works best if the input images are in the [MNI space](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009). The model was trained on T1-weighted MRIs with simulated resections, but it seems to work well with T1 images with gadolinium as well.

Sample images from the [EPISURG dataset](https://doi.org/10.5522/04/9996158.v1) can be used to try this module.

The [`resseg`](https://github.com/fepegar/resseg) Python package can be installed to segment cavities outside 3D Slicer.
