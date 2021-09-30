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

## Inclination Labels

We have investigated the capability of the human eye to evaluate galaxy inclinations. We begin with the advantage that a substantial fraction of spiral inclinations are well defined by axial ratios. These good cases give us a grid of standards of wide morphological types over the inclination range that particularly interests us of 45-90 degrees. The challenge is to fit random target galaxies into the grid, thus providing estimates of their inclinations.


![Fig5](https://user-images.githubusercontent.com/13570487/135530392-123f7689-8e11-4af3-9e9e-c64810351af8.png)
*Fig. 5: Users compare each target galaxy with a set of standard galaxies to evaluate their inclinations.*


To obtain a large enough sample for an ML project, we have manually labelled ~20,000 spiral galaxies using an online GUI, [Galaxy Inclination Zoo(http://edd.ifa.hawaii.edu/inclination/)], in collaboration with amateur astronomers across the world. In this graphical interface, we use the colorful images provided by SDSS as well as the g, r, i band images generated for our photometry program. These latter are presented in black-and-white after re-scaling by the asinh function to differentiate more clearly the internal structures of galaxies. The inclination of standard galaxies were initially measured based on their axial ratios. Each galaxy is compared with the standard galaxies in two steps (see Fig. 5). First, the user locates the galaxy among nine standard galaxies sorted by their inclinations ranging between 45 degrees and 90 degrees in increments of 5 degrees. In step two, the same galaxy is compared with nine other standard galaxies whose inclinations are one degree apart and cover the 5 degree interval found in the first step. At the end, the inclination is calculated by averaging the inclinations of the standard galaxies on the left/right-side of the target galaxy. In the first step, if a galaxy is classified to be more face-on than 45, it is flagged and step two is skipped. We take the following precautions to minimize user dependent and independent biases:

- We round the resulting inclinations to the next highest or smallest integer values chosen randomly.
- At each step, standard galaxies are randomly drawn with an option for users to change them randomly to verify their work or to compare galaxies with similar structures.
- To increase the accuracy of the results, we catalog the median of at least three different measurements performed by different users.
- Users may reject galaxies for various reasons and leave comments with the aim of avoiding dubious cases.

The uncertainties on the measured inclinations are estimated based on the statistical scatter in the reported values by different users. Fig. 6 illustrates the distribution of labels, where `J` and `F` labels reject and face-on galaxies, respectively. Numbers indicate the inclination angle of galaxies from face-on in degrees. As seen, out of 19,907, ~22% are rejected for various astronomical reasons and ~8% are face-on thus not acceptable for our original research purpose.


![Fig6](https://user-images.githubusercontent.com/13570487/135530645-28f40a5f-79e0-4d01-8307-2ce91d2d1e0c.png)
*Fig. 6: The distribution of the labels across the sample*


### Comparin users' adjusted measurements against each other

This comparison is very similar to A/B testing. The only difference here is that we divide users into two different groups, A and B. Then, we build the median tables for these groups and compare the inclination of the common galaxies of both tables. This way, we are able to evaluate the statistical uncertainties on the measurements.
Fig. 7 compares the results of two different groups of users. Evidently, on average, the agreement between two different people about their inclination measurements using our methodology is ~3 degrees. Any machine learning tool that has almost the similar performance is acceptable for our purpose.
As expected, we see smaller scatter at larger inclination values towards the edge-on galaxies. Practically, it is much easier for users to recognize and evaluate edge-on galaxies, which fortunately is in the favor of our astronomy research, because our sample mainly consists of edge-on galaxies. On the other hand, more scatter about less inclined galaxies (that indicate larger uncertainty on the measured values) and having much smaller number of evaluated galaxies in that region makes it hard for machine learning algorithms to learn from data and have accurate predictions when galaxies tend to be more face-on (smaller inclination values).

![Fig7](https://user-images.githubusercontent.com/13570487/135530717-544e2580-b940-494e-8063-638b09e70f4c.png)
*Fig. 7: Median of the evaluated inclinations by two different groups of users for ∼2000 galaxies.*

**Notebook**

For more detail on how we have processed the ground truth results in order to avoid any human related mistakes or biases please refer to this notebook: https://github.com/ekourkchi/inclinet_project/blob/main/data_extraction/incNET_dataClean.ipynb 

# Models

Models have been define in this auxiliary code: https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/TFmodels.py

Our main objectives are to evaluate the inclination of a spiral galaxy from its images, whether it is presented in grayscale or colorful formats. Moreover, we need to know what is the possibility of the rejection of the given image by a human user. Images are rejected in cases they have poor quality or contain bright objects that influence the process of measuring the inclination of the target galaxy. Very face-on galaxies are also rejected.

We tackle the problem from two different angles.

1. To determine the inclination of galaxies, we investigate 3 models, each of which is constructed based on the VGG network, where convolutional filters of size 3x3 are used. The last layer benefits from the tanh activation function to generate numbers in a finite range, because the spatial inclination of our spirals sample lies in a range of 45 to 90 degrees.

2. To determine the accept/reject labels, we build three other networks which are very similar to the regression networks except the very last layer which is devoted to binary classification.

Models have different complexity levels and are labeled as model4, model5 and model6, with model4 being the simplest one. Here, we briefly introduce different models that we consider in this study.

For clarity, we label our models as `model<n>(m)`, where`<n>` is the model label which can be 4, 5, and 6. `<m>` denotes the “flavor of the model”, where m=0 represents a model that is trained using the entire training sample. m0  stands for models that have been trained using 67% of data.

## Model 4

This is the simplest model in our series of models. The total weight number of this model is ~1.600,000. It has two sets of double convolutional layers. Fig. 8 visualizes the convolutional neural network of Mode 4.

![Model4](https://user-images.githubusercontent.com/13570487/135531201-cdcf96eb-b793-4bbb-8a20-bcbefa2b3251.png)

![Fig8](https://user-images.githubusercontent.com/13570487/132303628-6657d08f-7ae3-4fe9-a96d-335569b5b150.png)

## Model 5

This model is the most complex model in our study. It has ~2,500,000 free parameters and three sets of double convolutional layers.

![model5_table](https://user-images.githubusercontent.com/13570487/132303862-d7901455-d591-45c5-9616-beaa6cb54eb4.png)

## Model 6

This model is comparable to Model4, in terms of complexity, although the number of convolutional units is larger in this model.

![model6_table](https://user-images.githubusercontent.com/13570487/132305223-fd946618-d7aa-40da-b21b-096345804366.png)


# Training

Here, we train a VGG model using the augmented data. The data augmentation process has been explained above. The output augmented data has been stored on disk for the purpose of the following analysis.

## Regression, determining inclinations

128x128 images are used for this analysis, which are in grayscale (g,r,i filter) or colorful (RGB). All images are presented in 3 channels. For grayscale images, all three channels are the same. Inclinations range from 45 to 90 degrees. Half of the grayscale images have black background, i.e. galaxies and stars are in white color. In half of the grayscale cases, objects are in black on white background. Half of the sample is in grayscale, and the other half is in grayscale. The augmentation process has been forced to generate images that cover the entire inclinations uniformly. We adopt the “Adam” optimizer, and the “Mean Square Error” for the loss function, and we keep track of the MSE and MAE metric during the training process.

### Notebooks

- Model 4: https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_04.ipynb
- Model 5: 
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_05.ipynb
- Model 6:
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_06.ipynb

### Challenges

With 128x128 images, and adding augmentation, the size of the required memory to open up the entire augmented sample is out of the capability of the available machines. Thus, we resolved the problem by saving the training sample in 50 separate batches that are already randomized in any way. Each step of the training process starts with loading the corresponding batch, reconstructing the CNN as it was generated in the previous iteration, and advancing the training process for one more step. At the end of that repetition, we store a snapshot of the network weights for the next training step.

### Notes

- Since we are dealing with a large training sample, we need to repeat updating the network weights for many steps to cover all training batches several times
- Over-fitting sometimes helps to minimize the prediction-measurement bias

### Pros

- We can generate as many training galaxies as required without being worry about the memory size
- We can stop the training process at any point and continue the process from where it is left. This helps is the process crashes due to the lack of enough memory that happens if other irrelevant processes clutter the system
- We are able to constantly monitor the training process and make decisions as it goes on

### Cons

- This process is slow. The bottleneck is the i/o process, not training and updating the weight numbers of the network.

### Plotting the evaluation metrics

As illustrated in Fig. 9, the training process can be stopped about the iteration #1000, however a little bit of over training helps to remove the prediction-measurement bias and reduce the size of fluctuations in the metrics.


![Fig9](https://user-images.githubusercontent.com/13570487/135531689-799b0abd-0ba5-4da6-8122-10cec42aab48.png)
*Fig. 9: Regression evaluations metrics vs. training epochs*

## Classification, determining good/bad galaxy images

Here, we train a VGG model using the augmented data. Data augmentation has been done separately in another code and the output data has been stored on disk for the purpose of the following analysis.

128x128 images are used for this analysis, which are in grayscale (g,r,i filter) or colorful (RGB). All images are presented in 3 channels. For grayscale images, all three channels are the same. Labels are either 0 or 1.


- 0: galaxy images with well defined and measured inclinations, that are used for the distance analysis in a separate research
- 1: galaxies that are flagged to be either face-on (inclinations less than 45 degrees from face-on), or to have poor image quality. Deformed galaxies, non-spiral galaxies, confused images, multiple galaxies in a field, galaxy images that are contaminated with bright foreground stars have been also flagged and have label 1.

We adopt the same network trained to determine inclinations. Here for binary classification, the last layer activation function has been changed to Softmax with sparse categorical entropy as the loss function. We keep track of the accuracy metric during the training process.

**Objective:** The goal is to train a network for automatic rejection of unaccepted galaxies. These galaxies can be later manually studied or investigated by human users.

### Notebooks:

- Model 4: https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_04-binary.ipynb
- Model 5:
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_05-binary.ipynb
- Model 6:
https://github.com/ekourkchi/inclinet_project/blob/main/VGG_models/128x128_Trainer_06-binary.ipynb


### Plotting the evaluation metrics
According to Fig. 10, after about 200 iterations, the loss function of the training sample decreases while the performance gets worse on the test sample. This means that over-training the network after this point doesn't improve the outcome.

![Fig10](https://user-images.githubusercontent.com/13570487/135531763-7dc87f62-243a-4c87-a3df-9992344e1518.png)
*Fig. 10: classification evaluations metrics vs. training epochs*


# Testing

## Comparing the Regression Models

**Predictions vs. Actual measurements**

We cross compare the evaluated inclinations versus the actual labels.

To get better understanding of the model performance, we plot the difference between the evaluated inclinations and the measured values by users ($\Deltai = i_m-i_p$). In Fig. 11, the horizontal axis shows the measured inclinations. Each point represents a galaxy in the test sample. 

- **Left Panel:** Predicted values $i_p$, are directly generated by applying the trained network on the test sample. Red solid line displays the results of a least square linear fit on the blue points. Evidently, there is an inclination dependent bias that is inclination dependent. This bias has been linearly modeled by the red line, which is utilized to adjust the predicted values. The slope and intercept of the fitted line are encoded in the `m` and `b` parameters.
- **Right Panel:** Same as the left panel, with adjusted predictions, $i_{pc}$, that is calculated using $i_{pc}=(i_p+b)/(1-m)$.

![Fig11](https://user-images.githubusercontent.com/13570487/135532545-021d74c6-4cfb-49bf-886f-227d096e3bb7.png)

*Fig. 11: Discrepancy between predictions and actual values for mode 4 with the test sample*

**Note:** The root mean square of the prediction-measurements differences is ~4o. The similar metric is ~2.6o when we compare the measured values of two groups of the human users. This means our model performs slightly worse than human, and most of that poor performance is attributed to the outliers and features (like data noise, point sources, stellar spikes, poor images, etc.) with not enough data coverage.

In a similar way Fig. 12 shows the performance of all models. As seen, in almost all cases the prediction bias is at minimum and not that significant. Each panel displays the results of a model, and is labeled with the name of the corresponding model. RMS and MAE denote “Root Mean Square” and “Mean Absolute Error” of the deviations of i about zero. At first glance, Model #5 seems to have the best performance, which does not come as a surprise, because it is the most complicated model that we have considered. In general, the differences in the performance of different models is not that significant.

![Fig12](https://user-images.githubusercontent.com/13570487/135532635-2afc04c5-d7b5-43e0-9c6f-25d1c5e62454.png)

*Fig. 12: The performance of all studied models using the test sample. Each panel illustrates the results of a model labeled as `model<n>(m)`, with `<n>` being the model number which can be 4, 5, and 6.  `<m>` denotes the “flavor of the model”.*

## Visualizing the outliers

Fig. 13 displays the first 49 galaxy images in the test sample, where the predicted value is far from the actual measurements, i.e. $\Delta i > 10^o$).  In each panel, cyan label is the galaxy ID in the Principal Galaxy Catalog (PGC), and green and red labels represent the measured and the predicted inclinations. Magenta labels denote the panel numbers.

Some cases are interesting:

- In panel #42, the galaxy image has been masked out because the bright center of the galaxy has saturated the center of image.
- Case #26 is a typical case, where the galaxy image has been projected next to a bright star.
- Cases like #2, #14, #23, #29, #43 have poor quality images.
- Case #37 has been ruined in the data reduction process, when the telescope data has been preprocessed.

![Fig13](https://user-images.githubusercontent.com/13570487/135532840-bcfb91f7-8f99-4ed9-ac52-1137e2e68c4c.png)

*Fig. 13: Example of outliers for Model 4*


### Averaging models
We take two average types to combine the results of various models and possibly obtain better results, mean and median.  We generate 4 sets of averages

- averaging the results of all model4 flavors
- averaging the results of all model5 flavors
- averaging the results of all model6 flavors
- averaging the results of all models with various flavors

We don't see any significant differences between mean and median, so there is no way we prefer one of them. However, we recommend using the median just to ignore very severe outliers.

### The power of bagging

As inferred from Figures 14 and 15, when the results of all models are averaged out, we get the best performance. RMS and MAE of deviations of the average of all models from the measured values are 3.09 and 2.12 [deg], respectively, which is comparable with the performance of humans.

![Fig14](https://user-images.githubusercontent.com/13570487/132321203-7362280a-8213-4cd8-80a5-efa53188e2e3.png)

*Fig. 14: Median of all predictions*

![Fig15](https://user-images.githubusercontent.com/13570487/132321248-75d78df5-0bda-48b8-b75e-884851007a59.png)

*Fig. 15: Mean of all predictions*

## Evaluation of the Binary Models
Three different metrics have been considered to evaluate the classification models, namely “precision”, “recall” and “accuracy”. These metrics are defined as 

- precision = TP/(TP+FP)
- recall = TP/(TP+FN)
- accuracy = (TP+TN)/(TP+FP+FN+TN)

where `TP` and `FP` are true and false positives, respectively. In a similar way, `TN` and `FN` are true and false negatives, respectively. Precision measures how many of the positive predictions are actually positive. Recall indicates how many of the actuala positive cases have been detected correctly. Accuracy shows the fractions of correct predictions in the entire sample. Fig. 16 visualizes these metrics for different classification models that we considered in this project.

![Fig16](https://user-images.githubusercontent.com/13570487/132331083-961596a8-99a0-442e-bd57-7cda7bbcae6c.png)

*Fig. 16: The performance metrics of the various classification models in this study*

Evidently, model #5 has a better overall performance compared to the other two models, which is expected knowing that model #5 is the most complicated one.
Averaging out the evaluated labels does make significant improvements, however model #5 seems to perform slightly better than the average.









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



