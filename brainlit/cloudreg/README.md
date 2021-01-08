# CloudReg

CloudReg is a cloud-enabled pipeline for scalable registration, originally built for mouse light-sheet microscopy, but now with functionality to support many different volumetric registration challenges in the cloud.

- [Data management](#data)
- [Preprocessing](#peprocessing)
- [Registration](#registration)
- [Visualization](#visualization)

## Data management

Portal for managing data and initiating pipeline

## Peprocessing

Pipeline to run

1. Artifact correction
1. Tile stitching
1. Neuroglancer volume creation
1. Downsample

## Registration

GPU-accelerated registration pipeline using [LDDMM](https://link.springer.com/article/10.1023/B:VISI.0000043755.93987.aa) implemented by [ardent](https://github.com/neurodata/ardent) providing

1. Registration of volumes to
   1. [Allen CCFv3 2017](http://atlas.brain-map.org)
   1. [Allen Mouse Brain Reference Atlas](http://atlas.brain-map.org) (ARA)
   1. Other custom supplied targets
1. Multiscale registration
1. Contrast transforms
1. Invertible transformation matrix smoothly upscaled to higher resolution

## Visualization

Portal running [Neuroglancer](https://github.com/neurodata/neuroglancer) to

1. View data in native or [Allen Mouse Common Coordinate Framework](http://help.brain-map.org/download/attachments/2818169/MouseCCF.pdf) (CCF) space
1. View data overlaid with segmentations and lables from different atlases or other data/channels co-registered to CCF
1. View data overlaid with meshes of atlas segmentations
