# IncliNET

## Introduction

The inclination of spiral galaxies plays an important role in measurements of their distances and other astronomical analyses. Each galaxy has its own unique morphology, luminosity, and surface brightness profiles. In addition, galaxy images are covered by foreground stars of the Milky Way galaxy. Therefore, it is challenging to design an algorithm that automatically determines the 3D inclination of spiral galaxies. The inclinations of spiral galaxies can be coarsely derived from the ellipticity of apertures used for photometry, assuming that the image of a spiral galaxy is the projection of a disk with the shape of an oblate spheroid. For ~1/3 of spirals, the approximation of axial ratios provides inclination estimates good to better than 5 degrees, with degradation to ~5 degrees for another 1/3. However in ~1/3 of cases, ellipticity-derived inclinations are problematic for a variety of reasons. Prominent bulges can dominate the axial ratio measurement. Some galaxies may not be axially symmetric due to tidal effects. High surface brightness bars within much lower surface brightness disks can lead to large errors. Simply the orientation of strong spiral features with respect to the tilt axis can be confusing. The statistical derivation of inclinations for large samples has been unsatisfactory.

The task of manual evaluation of galaxy inclinations is tedious and time consuming. In the future, with the large astronomical survey telescopes coming online, this task could not be efficiently done manually for thousands of galaxies. The objective of this project is to automatically determine the inclination of spiral galaxies with the human level accuracy providing their images, ideally in both colorful and black-and-white formats.

## Data Preparation

### Image Extraction

To obtain the cutout image of each galaxy at g, r, and i bands, we download all corresponding calibrated single exposures from the [SDSS DR12](https://www.sdss.org/dr12/) database. We use [MONTAGE](http://montage.ipac.caltech.edu/), a toolkit for assembling astronomical images, to drizzle all frames and construct galaxy images. The angular scale of the output images is 0.4'' pixel-1. Out data acquisition pipeline is available [online](https://github.com/ekourkchi/SDSS\_get). For the task of manual labeling we presented images to users in 512x512 resolution. For the task of generating and testin multiple ML models, we degrade the resolution of images to 128x128 to make the project feasible given the available resources.


### Image Augmentation

We convert all input images to 128x128. To avoid over-fitting, we increase the sample size by leveraging the augmentation methods. The inclination of each galaxy is independent of its position angle on the projected image as well as the image quality. Therefore, we augment our image sample by running them through a combination of transformations such as rotation, translation, mirroring, additive Gaussian noise, altering contrast, blurring, etc. 
All augmentation transformations keep the aspect ratio of images intact to ensure that the elliptical shape of galaxies and their inclinations are preserved. Other augmentation criteria are as following
The numbers of colorful (RGB) and grayscale (g, r, i) images are equal
The numbers of black-on-white and white-on-black images are the same
g, r, i images have equal chance of appearance in each batch



![fig1](https://user-images.githubusercontent.com/13570487/135529446-134617dc-9ba6-4834-a11d-487c7f5a7025.png)
*Fig. 1: Left: Distribution of the inclinations of the original sample galaxies. Right: Distribution of the augmented sample inclinations*

![fig2](https://user-images.githubusercontent.com/13570487/135529776-cbe10cf2-22dc-4a87-b447-8d9b94842d20.png)
*Fig. 2 : Examples of augmented images. In each panel, the galaxy ID is in cyan. Red is the inclination and magenta is the image pass-band, i.e. g, r, i and c, where c stands for RGB.*

**Notebooks:**

1. Here, data is processed for models to determine the inclinations of spiral galaxies by leveraging the regression methodologies. Fig.1 illustrates the distribution of inclinations in both original and augmented samples. Fig. 2 displays a set of augmented images as example. 
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/incNET_model_augmentation.ipynb

2. Here, data is processed for models to determine the usefulness of spiral galaxies for the task of inclination measurements by leveraging the classification methodologies. We are dealing with binary labels, where 0 denotes accepted galaxies, and 1 represents rejected images due to various anomalies, and ambiguities, or having poor quality. A few face-on galaxies (those with inclination lower than 45 degrees from face-on) have also been rejected. Fig. 3 illustrates the distribution of labels in the original and augmented samples. Fig. 4 shows a subset of augmented galaxies with their labels.  
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/incNET_model_augmentation-binary.ipynb


![Fig3](https://user-images.githubusercontent.com/13570487/135529972-eb037bbd-7531-4da2-88d2-85a64ac7a77b.png)
*Fig. 3: Left: Distribution of the labels of the original sample galaxies. Right: Distribution of the sample labels after augmentation. As expected, each batch covers both labels uniformly. This reduces any biases that originate from the data imbalance.*


![Fig4](https://user-images.githubusercontent.com/13570487/135530054-987a33b4-8207-4e4a-8f3b-8e98a7525f85.png)
*Fig. 4: Examples of augmented images. IIn each panel, the PGC ID of the galaxy is in cyan. Red is the classification label and magenta is the image pass-band, i.e. g, r, i and c, where c stands for RGB.*


![GIZ_demo](https://user-images.githubusercontent.com/13570487/85185022-6c752b00-b24f-11ea-9f9a-9d1d007f4fb7.png)

## Table of contents <a name="data_folders"></a>

- [incNET (data extraction)](#incNET)
   * [Imaging Data Folders](#data_folders)
   * [Tabular Data](#tabular_data)
   * [Data Preparation](#data_prep)
   * [About the Project](#about)
      * [Problem](#problem)
      * [Goal](#goal)
      * [Inclination Labels](#labels)
      * [Demo Version](#demo)  
- [Other similar ML projct with similar aspects](#otherML)
- [Disclaimer](#Disclaimer)

# incNET (data extraction) <a name="incNET"></a>

   - Basic codes to extract and organize the data that includes images of more than 15,000 spiral galaxies taken from the SDSS DR12. To access the code to download the SDSS images please refer to [https://github.com/ekourkchi/SDSS_get](https://github.com/ekourkchi/SDSS_get).
   - Demo codes for training a CNN
   
    

## Imaging Data Folders <a name="data_folders"></a>

The images are stored in these folders. All images are rotated so that the semi-major axis of all spirals are aligned horizontally. Images are provided at the resolutions of 128x128 and 64x64 pixel^2 in the following folders. The naming convention of images follows the *pgcxxxx_NxN_yy.jpg*, where "xxxx" is the ID number of the galaxy in the Principal Galaxies Catalogue, "N" is the number of pixels along each side, and "yy" is the spatial inclination of galaxies in degree, that are measured in a manual inspection procedure via the **Galaxy Inclination Zoo** online GUI: [http://edd.ifa.hawaii.edu/inclination](http://edd.ifa.hawaii.edu/inclination/). The inclination of our sample galaxies from face-on ranges between 45 and 90 degrees, where at 90 degrees the disk of the spiral galaxy is totally face-on. For galaxies more face-on than 45 degrees "yy" is replaced by "F" and the anomalous galaxies are denoted by setting "yy" to 0.

Both of the following 128x128 and 64x64 folders contain ~15,000 spiral images.

   - galaxie: just includes a sample of preprocessed images for the use in [GIZ](http://edd.ifa.hawaii.edu/inclination/).
   Image names are formatted as follows:
       * `pgc####_d25x2_rot_band.png`
       * `pgc####_d25x2_rot_gri.jpg`
       * `pgc####_d25x2_rot_gri.sdss.jpg`
       
   - "####" is the PGC ID of galaxies. 'band' is replaced with 'g', 'r', or 'i'. Files with "gri" in their names are composite colorful images. If they include 'sdss', it means that they are directly taken from the SDSS database, otherwise, they are originated from the [Pan-STARRS](https://ps1images.stsci.edu/cgi-bin/ps1cutouts) image cutout database. For many cases both versions are available.
    
   - 128x128: tracked via Git Large File Storage
   - 64x64: tracked via Git Large File Storage
   
## Tabular Data <a name="tabular_data"></a>

   - [EDD_distance_cf4_v27.csv](https://raw.githubusercontent.com/ekourkchi/incNET-data/master/EDD_distance_cf4_v27.csv): The list of studied galaxy candidates for the Cosmicflows program. [Click here](http://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4cand) for the description of the columns. In our inclination program, the applicable columns are *inc*, *inc_e*, *inc_flag*, *inc_n* and *Note_inc* that holds the notes entered by the [GIZ](http://edd.ifa.hawaii.edu/inclination/) users.


## Data Preparation <a name="data_prep"></a>

The images of our galaxies are extracted from [SDSS DR12](https://www.sdss.org/dr12/) data collection.
For each galaxy with available *SDSS* data, we download all the single exposure cutouts at *u, g, r, i* and *z* bands. Our data acquisition pipeline is available [here](https://github.com/ekourkchi/SDSS\_get), which are drizzled and combined using [MONTAGE](http://montage.ipac.caltech.edu/docs/mProject.html), an astronomical application to assemble images. Our pipeline provides galaxy cutouts at all *ugriz* passbands with the spatial resolution of 0.4'' /pixel.

The constructed images are in *Flexible Image Transport System* (FITS) format commonly used by astronomers. Therefore, we convert and rotate these images for the manual task of evaluating galaxy inclinations. The associated codes are stored in the `imRotate` folder. To align the semi-major axis of spirals along the horizontal axis, the primary position angles of galaxies are either taken from the [HyperLEDA](http://leda.univ-lyon1.fr/) catalog or the outputs of our photometry program that are stored in `EDD_distance_cf4_v27.csv`. For the details of our photometry procedure and the codes, please refer to this repository: [https://github.com/ekourkchi/IDL_photometry](https://github.com/ekourkchi/IDL_photometry). The galaxy images that are used in the *GIZ* are stored in the *galaxy* folder.

The `Data_Preparation.ipynb` and `im_scale_batch.ipynb` notebooks provide code to resize and rescale the SDSS galaxy images and to store them in the numpy `npz` format.
Galaxy images are taken from the `galaxy` folder and reprocessed in various forms. The data augmentation has been done by flipping images vertically and horizontally.
   
## About the Project <a name="about"></a>

![Screenshot from 2020-06-19 20-10-40](https://user-images.githubusercontent.com/13570487/85189112-fda4cb80-b268-11ea-83d5-172bca0fc78f.png)

### Problem <a name="problem"></a>
The inclination of spiral galaxies plays an important role in measurements of their distances using the Tully-Fisher relationship. Each galaxy has its own unique morphology, luminosity, and surface brightness profiles. In addition, galaxy images are covered by foreground stars of the Milky Way galaxy. Therefore, it is challenging to design an algorithm that automatically determines the 3D inclination of spiral galaxies.  The inclinations of spiral galaxies can be coarsely derived from the ellipticity of apertures used for photometry, assuming that the image of a spiral galaxy is the projection of a disk with the shape of an oblate spheroid. For ~1/3 of spirals, the approximation of axial ratios provides inclination estimates good to better than 5 degrees, with degradation to ~5 degrees for another 1/3.  However in ~1/3 of cases, ellipticity-derived inclinations are problematic for a variety of reasons.  Prominent bulges can dominate the axial ratio measurement, with the Sombrero galaxy (above picture: second panel from the right) providing an extreme example.  Some galaxies may not be axially symmetric due to tidal effects.  High surface brightness bars within much lower surface brightness disks can lead to large errors.  Simply the orientation of strong spiral features with respect to the tilt axis can be confusing.  The statistical derivation of inclinations for large samples has been unsatisfactory.

### Goal <a name="goal"></a>

The task of manual evaluation of spirals inclinations is tedious and time consuming. In the future, with the large astronomical survey telescopes coming online, this task could not be efficiently done manually for thousands of galaxies. The objective of this project is to automatically determine the inclination of spiral galaxies with the human level accuracy providing their images (ideally in both colorful and black-and-white formats).

### Inclination Labels <a name="labels"></a>

We have investigated the capability of the human eye to evaluate galaxy inclinations. We begin with the advantage that a substantial fraction of spiral inclinations are well defined by axial ratios.  These good cases give us a grid of standards of wide morphological types over the inclination range that particularly interests us of 45-90 degrees. The challenge is to fit random target galaxies into the grid, thus providing estimates of their inclinations.

To achieve our goal, we designed an online graphical tool, Galaxy Inclination Zoo ([GIZ](http://edd.ifa.hawaii.edu/inclination)), to measure the inclination of spiral galaxies in our sample. In this graphical interface, we use the colorful images provided by SDSS as well as the g, r, i  band images generated for our photometry program. These latter are presented in black-and-white after re-scaling by the asinh function to differentiate more clearly the internal structures of galaxies. The inclination of standard galaxies were initially measured based on their axial ratios.
Each galaxy is compared with the standard galaxies in two steps. First,  the user locates the galaxy among nine standard galaxies sorted by their inclinations ranging between 45 degrees and 90 degrees in increments of 5 degrees. In step two, the same galaxy is compared with nine other standard galaxies whose inclinations are one degree apart and cover the 5 degree interval found in the first step. At the end, the inclination is calculated by averaging the inclinations of the standard galaxies on the left/right-side of the target galaxy. In the first step, if a galaxy is classified to be more face-on than 45\dg, it is flagged and step two is skipped.

The program is open to the public and hence it benefits from the participation of citizen scientists and tens of amateur astronomers. Ultimately, for each spiral we use all the measured inclinations by all users who worked on that galaxy.

We have taken the following precautions to minimize user dependent and independent biases. 
   - We round the resulting inclinations to the next highest or smallest integer values chosen randomly. 
   - At each step, standard galaxies are randomly drawn with an option for users to change them randomly to verify their work or to compare galaxies with similar structures. 
   - To increase the accuracy of the results, we catalog the median of at least three different measurements performed by different users. 
   - Users may reject galaxies for various reasons and leave comments with the aim of avoiding dubious cases.



### Demo Version <a name="demo"></a>

We used the GIZ interface  for the tedious task of manually sorting and classifying ~20,000 galaxies. In the future, with the large astronomical survey telescopes coming online, this task could not be efficiently done manually for thousands of galaxies. The plan is to use the evaluated inclinations by the GIZ online tool to orchestrate a machine-learning algorithm for classifying spiral galaxies in terms of their inclinations. The large sample of human labeled galaxies will be used for training a convolutional neural network. This algorithm will save a lot of time and also will have many other applications in scientific and industrial fields. The goal is to design a deep learning algorithm similar to those used in image processing and face/object recognition techniques to extend the realm of our current research to a more distant universe.

To demonstrate the concept, a simple convolutional neural network was constructed using `TensorFlow` and we trained it using the outputs of the GIZ. Only the JPG colorful images are taken for this demo version. 1,500 images are randomly chosen to test the reliability of the resulting algorithm. The control sample is not used in the training process. The training process generates a model that outputs inclinations with an RMS of 3 degrees. In this collection, the Jupyter notebooks whose names start with *CCN* consist of the codes that were used to train and test our networks.

The following figure plots the difference between the predicted values, `i_p`, and the measured values by GIZ, `i_m`, for 1,500 control spirals.
Each black point represents a galaxy in the control sample. Both axes are in degree. The RMS of deviations is 3 degrees.

![basic_evaluation](https://user-images.githubusercontent.com/13570487/85190851-9ba09200-b279-11ea-9165-751741d5502a.png)


This model is now available to test on any arbitrary spiral galaxy. Please follow this link and open the online application at [incNET](http://edd.ifa.hawaii.edu/incNET). The following figure shows a screenshot of the page.

![incNET_skin](https://user-images.githubusercontent.com/13570487/85190883-e5897800-b279-11ea-9cd1-135e46b36b29.png)

On the left side of this tool, users have different options to find and load a galaxy image. The PGC-based query relies on the information provided by the [HyperLEDA](http://leda.univ-lyon1.fr/) catalog. Each image is rotated and resized based on the LEDA entries for *logd25* and position angle, which are reasonable in most cases. Further manual alignment features are provided. Clicking on the orange button, users can *Evaluate* the inclination. This step feeds the image to a pre-trained neural network and outputs the inclination value. In addition, there is another network that parallelly predicts the rejection probability of the galaxy by human users. For computational matters, the resolution of the arbitrary square images are degraded to 64x64 pixels prior to the evaluation process.

# Other ML projct with similar aspects <a name="otherML"></a>

   * Cloud detection (Understanding Clouds From Satellite Images): [description and data](https://www.kaggle.com/c/understanding_cloud_organization/overview)
   * 17 Category Flower Dataset for flower classification ([Click here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html))
   * LAG Database, containing 11,760 fundus images corresponding to 4,878 suspecious and 6,882 negative glaucoma samples ([Click here](https://github.com/smilell/AG-CNN)). 
   * Skin Cancer: Malignant vs Benign, a balanced [dataset](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign) of images of benign skin moles and malignant skin moles


## Disclaimer <a name="Disclaimer"></a>

 * All rights reserved. The material may not be used, reproduced or distributed, in whole or in part, without the prior agreement. 
 * Contact: *Ehsan Kourkchi* <ekourkchi@gmail.com>



