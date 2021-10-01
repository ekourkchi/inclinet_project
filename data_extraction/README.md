# Data Extraction

1. [Preparation of Labels](#PreparationofLabels)
2. [Preparation of Galaxy Images](#PreparationofGalaxyImages)
3. [Image formats](#Imageformats)
4. [Data Augmentation](#DataAugmentation)


##  1. <a name='PreparationofLabels'></a>Preparation of Labels

[This notebook](https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/incNET_dataClean.ipynb) discuss how we have manually labelled spiral galaxies using an [online GUI](http://edd.ifa.hawaii.edu/inclination/) in a collaborative project. Users of the interface are asked to situate a target galaxy within a lattice of galaxies with established inclinations. In this graphical interface, we use the colorful images provided [SDSS](https://www.sdss.org/) as well as the g, r and i band images generated for our photometry program. These latter are presented in black-and-white after re-scaling by the `asinh` function to differentiate more clearly the internal structures of galaxies. The inclination of standard galaxies were initially measured based on their I-band axial ratios.

![Fig2](https://user-images.githubusercontent.com/13570487/135530645-28f40a5f-79e0-4d01-8307-2ce91d2d1e0c.png)
 
*Figure: The distribution of the labels across the sample*

Here, we show how to load galaxy images form the *SDSS* image server by providing the galaxy coordinates, pixel scale size, and field of view.
https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/SDSS_load_image.ipynb


##  2. <a name='PreparationofGalaxyImages'></a>Preparation of Galaxy Images

Galaxy images are originally in 512x512 format. These images have been used in the Galaxy Inclination Zoo for the purpose of manual investigations. One of the data preparation process involve bundling all transformed images (all are downsampled to 128x128 pixels) and their inclination labels and compressing them for the use in the augmentation and training process. 

- Grayscale Image (`g`, `r`, and `i` bands): https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/Data_Preparation_gri.ipynb

- Colorful Images (`RGB`): https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/Data_Preparation_RGB.ipynb


##  3. <a name='Imageformats'></a>Image formats

All images have been preprocessed into 4 separate batches and stored in `numpy` compressed format, i.e. `npz`.

The batch names follow this naming format `data_128x128_<filter>_originals.npz`, where `<filter>` is to be replaced with either of `g`, `r`, and `i` or `RGB`. `gri` images are presented in grayscale, in three channels all of which are equivalent. RGB images are colorful and have been processed separately.


![image](https://user-images.githubusercontent.com/13570487/135597713-6646f3a3-0336-4b5a-86f2-17851576b286.png)


##  4. <a name='DataAugmentation'></a>Data Augmentation

The inclination of each galaxy is independent of its position angle on the projected image as well as the image quality. Therefore, we augment our image sample by running them through a combination of transformations such as rotation, translation, mirroring, additive Gaussian noise, altering contrast, blurring, etc. All augmentation transformations keep the aspect ratio of images intact to ensure that the elliptical shape of galaxies and their inclinations are preserved.

In this notebook we present how we prepared training/testing sub-samples as well as the augmentation process.
https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/incNET_model_augmentation.ipynb

![image](https://user-images.githubusercontent.com/13570487/135596539-3cebea47-583c-48fa-afac-0421021b24a6.png)
