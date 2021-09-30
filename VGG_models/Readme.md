# Training a CNN model based on VGG

![model4](https://user-images.githubusercontent.com/13570487/132303628-6657d08f-7ae3-4fe9-a96d-335569b5b150.png)

## Relevant Notebooks:

### Data Preparation/Augmentation

To prepare data for the full analysis, we increase the resolution of the input images to 128x128. Each images can be rotated arbitrarily, with some additional noise and other tweaks.

**Note** that the aspect ratio of images should not be changed to preserve the elliptical shape of the projected galaxies, and thus their inclinations.

- `incNET_model_augmentation.ipynb` [Click Here](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/incNET_model_augmentation.ipynb)
- `incNET_model_augmentation-binary.ipynb` [Click Here](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/incNET_model_augmentation-binary.ipynb)

## Objectives

Our main objectives are to evaluate the inclination of a spiral galaxy from its images, whether it is presented in grayscale or colorful formats.
Moreover, we need to know what is the possibility of the rejection of the given image by a human user. Images are rejected in cases they have poor quality or contain 
bright objects that influence the process of measuring the inclination of the target galaxy. Very face-on galaxies are also rejected.

We tackle the problem from two different angles.

1. To determine the inclination of galaxies, we investigate 3 models, each of which is constructed based on the VGG network, where convolutional filters of size 3x3 are used. The last layer benefits from the `tanh` activation function to generate number in a finite range, because the spatial inclination of our spirals sample lie in range of 45 to 90 degrees.
2. To determine the accept/reject labels, we build three other networks which are very similar to the regression networks except the very last layer which is devoted to binary classification.

Models have different complexity levels and labeled as `model4`, `model5` and `model6`, with `model4` being the simplest one. 


## Training the CNN models

- `128x128_Trainer_04.ipynb` [Click Here](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_04.ipynb) 
    - In this notebook, we train a VGG model using the augmented data. Data augmentation has been done separately in [another code](https://github.com/ekourkchi/incNET-data/blob/master/incNET_dataPrep/Data_Preparation_gri.ipynb) and the output data has been stored on disk for the purpose of the following analysis. See also [here](https://github.com/ekourkchi/incNET-data/blob/master/incNET_dataPrep/Data_Preparation_RGB.ipynb).

    - 128x128 images are used for this analysis, which are in grayscale (g,r,i filter) or colorful (RGB). All images are presented in 3 channels. For grayscale images, all three channels are the same.

    - Inclinations range from 45 to 90 degrees.

- `128x128_Trainer_04-binary.ipynb` [Click Here](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_04-binary.ipynb)

    - In this notebook, we train a VGG model using the augmented data. Data augmentation has been done separately in another code and the output data has been stored on disk for the purpose of the following analysis.

    - 128x128 images are used for this analysis, which are in grayscale (g,r,i filter) or colorful (RGB). All images are presented in 3 channels. For grayscale images, all three channels are the same.

    - Labels are either 0 or 1.

        - 0: galaxy images with well defined and measured inclinations, that are used for the distance analysis in a separate research
        - 1: galaxies that are flagged to be either face-on (inclinations less than 45 degrees from face-on), or to have poor image quality. Deformed galaxies, non-spiral galaxies, confused images, multiple galaxies in a field, galaxy images that are contaminated with bright foreground stars have been also flagged and have label 1.

    - We adopt the same network structure we used to determine inclinations. Here for binary classification, the activation functions of the last layer has been changed to `Softmax` with sparse categorical entropy as the loss function.

- Other similar notebooks to train similar models are as follows. All comments are similar to those in [128x128_Trainer_04.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_04.ipynb) and [128x128_Trainer_04-binary.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_04-binary.ipynb).
    - [128x128_Trainer_05.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_05.ipynb)
    - [128x128_Trainer_06.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_06.ipynb)
    - [128x128_Trainer_05-binary.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_05-binary.ipynb)
    - [128x128_Trainer_06-binary.ipynb](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/128x128_Trainer_06-binary.ipynb)


## Comparing the results of various models

Refer to [this notebook](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/model_evaluations_plots.ipynb) for further details.

Plotting prediction-measurement vs. measurements. In the following figure, the horizontal axis shows the measured inclinations. Each point represents a galaxy in the test sample. 

In almost all cases the prediction bias is at minimum and not that significant. Each panel displays the results of a model, and is labeled with the name of the corresponding model.  

At first glance, Model #5 has the best performance, which does not come as surprise since it is the most complicated model that we considered. In general, the difference in the performance of the model is not that significant. 

![image](https://user-images.githubusercontent.com/13570487/132320948-a4d2dd02-b81f-4ca9-9565-5dc06b021c8d.png)


## Average models

We take two average types to combine the results of various models and possibly obtain better results

- median
- mean

We generate 4 sets of averages

- averaging the results of all model4 flavors
- averaging the results of all model5 flavors
- averaging the results of all model6 flavors
- averaging the results of all models with various flavors

We don't see any significant differences between `mean` and `median`, so there is not way we prefer one of them. However, we recommend using the median just to ignore very severe outliers.

### The power of bagging

Clearly, when the results of all models are averaged out, we get the best performance. RMS and MAE of deviations of the all models average from the measured values are 3.09 and 2.12 [deg], respectively.

![image](https://user-images.githubusercontent.com/13570487/132321203-7362280a-8213-4cd8-80a5-efa53188e2e3.png)

![image](https://user-images.githubusercontent.com/13570487/132321248-75d78df5-0bda-48b8-b75e-884851007a59.png)



## Models

Here, we brielfy introduce different models that we consider in this study.

### Model4

This is the simplest model in our series of models. The total number of weight number of this model is ~1.600,000. It has two sets of double convolutional layers.


![model4_table](https://user-images.githubusercontent.com/13570487/132303705-b84cea19-a492-4832-9cd4-57bd9535599b.png)


### Model5

This model is the most complex model in our study. It has ~2,500,000 free parameters and three sets of double convolutional layers.

![model5_table](https://user-images.githubusercontent.com/13570487/132303862-d7901455-d591-45c5-9616-beaa6cb54eb4.png)


### Model6

This model is comparable to Model4, in terms of complexity, although the number of convolutional units is larger in this model.

![model6_table](https://user-images.githubusercontent.com/13570487/132305223-fd946618-d7aa-40da-b21b-096345804366.png)



# Evaluation of the binary classification models

Refer to [this notebook](https://github.com/ekourkchi/incNET-data/blob/master/incNET_VGGcnn_withAugmentation/model_evaluations_plots.ipynb) for further details.

- Evidently, model #5 has a better overall performance compared to the other two models, which does not come as surprise knowing that model #5 is the most complicated one.

- Averaging out the evaluated labels does make some improvements, however model #5 seems to perform slightly better than the average.

![image](https://user-images.githubusercontent.com/13570487/132331083-961596a8-99a0-442e-bd57-7cda7bbcae6c.png)




