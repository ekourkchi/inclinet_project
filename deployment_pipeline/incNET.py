#!/usr/bin/python
__author__ = "Ehsan Kourkchi"
__copyright__ = "Copyright 2021"
__status__ = "Production"

import sys
import os
import subprocess
from math import *
import numpy as np
from datetime import *
from pylab import *
import scipy.ndimage
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

###############################################################
def converIMAGE(img_arr, angle=0., scale=1., size=64):
    """Converting an image array
    ``anlgle``: how much to rotate the image
    ``scale``: scale of the output image
        zoomin out if ``sale<1`` and zooming in if ``scale>1``
    ``size``: the number of pixels on each side of the output image

    :param img_arr: input image
    :type img_arr: numpy ``ndarray``
    :param angle: image rotation in ``degrees``, defaults to 0.
    :type angle: ``int``, optional
    :param scale: the scale of the output image relative to the input image, defaults to 1.
    :type scale: ``int``, optional
    :param size: resolution of the output image in pixel, defaults to 64
    :type size: ``int``, optional
    :return: output image 
    :rtype: ``PIL`` image object
    """

    img_arr = scipy.ndimage.rotate(img_arr, -angle)

    d1, d2, ch = img_arr.shape

    if scale<1:
        dmax = int(np.max([d1,d2])/scale)
        new_arr = np.zeros((dmax,dmax,ch)).astype(np.uint8)+255
        x1 = int((dmax-d1)/2)
        x2 = x1+d1
        y1 = int((dmax-d2)/2)
        y2 = y1+d2
        new_arr[x1:x2,y1:y2,:] = img_arr
        img_arr = new_arr
    else:
        N = img_arr.shape
        d = N[0]
        p = int(d / scale)
        d1 = int(d / 2 - p / 2)
        d2 = int(d1 + p)
        img_arr = img_arr[d1:d2, d1:d2, :]


    img = Image.fromarray(img_arr, 'RGB').resize((size, size))

    return img

###############################################################
def openImage(params):
    """importing the image and preparing it for the analysis

    :param params: a set of parameters including the name of the file that contains the image
    :type params: python ``dictionary``
    :return: ``(image_array, scale, angle)``
    :rtype: tuple 

    ``image_array``: numpy ndarray holding the image
    ``angle``: how much to rotate the image
    ``scale``: scale of the output image   

    """

    if not 'scale' in params:
        scale = 1
    else:
        scale = params["scale"]
    if not 'angle' in params:
        angle = 0 
    else:
        angle = params["angle"]
    try:
        fileName = params['fileName']
        img = Image.open(fileName)
        img_arr = np.asarray(img)

        if len(img_arr.shape) == 2:
            img_arr = np.stack((img_arr,) * 3, axis=-1)
        elif len(img_arr.shape) == 1:
            return(None, scale, angle)

        if len(img_arr.shape) == 3 and img_arr.shape[2] > 3:
            img_arr = img_arr[:, :, 0:3]
    except:
        return(None, scale, angle)

    
    return img_arr, scale, angle

###############################################################
def predictor_binary(model, All_images):
    """The prediction of the classification model
    This model returns a number between 0 and 1, which corresponds to the
    likelihood of the users rejecting this image, because of poor quality, bright star in the field, etc.
    
    ``All_images`` hold the images of the same galaxy projected at 4 different position angles, 90 degrees apart.

    :param model: classification model
    :type model: A convolutional neural network in ``TensorFlow``
    :param All_images: ``N`` images to be evaluated 
    :type All_images: numpy ``ndarray`` with the shape ``(N, size, size, n_channels=3)``
    :return: the median of all evaluated rejection likelihoods
    :rtype: ``float``
    """
    
    prediction = model.predict(All_images)

    ## return the likelihood of rejection in %
    return np.median(prediction[:,1])*100

###############################################################
def predictor(model, All_images):
    """The prediction of the regression model
    This model returns the galaxy inclinations of the input galaxy images.
    
    ``All_images`` usually hold the images of the same galaxy projected at 4 different position angles, 90 degrees apart.

    :param model: regression model
    :type model: A convolutional neural network in ``TensorFlow``
    :param All_images: ``N`` images to be evaluated 
    :type All_images: numpy ``ndarray`` with the shape ``(N, size, size, n_channels=3)``
    :return: the median of all evaluated inclinations
    :rtype: ``float``
    """
    prediction = model.predict(All_images)
    prediction = np.median(prediction)
    prediction = 0.5*(prediction+1.)*45.+45.
    prediction = np.round(prediction)
    if prediction > 90:
        prediction = 90.
    
    return prediction

###############################################################
def scaleFileName(params):
    """generating the name of the scaled image
    based on the name of the input image and the time stamp of the analysis

    :param params: a set of parameters including the name of the desired image to be rescaled
    :type params: python ``dictionary``
    :return: the name of the scaled image
    :rtype: ``string``
    """

    if 'fileName' in params:
        scaledImage = '128x128_' + params['fileName'].split('.')[-1] + 'jpg'
    else:
        scaledImage = '128x128_' + str(time.time()).replace('.','_') + '.jpg'

    return scaledImage

###############################################################

def model(params, myModels, saveRescaled=True, scaledImage=None):
    """providing the input parameters and a set of TensorFlow models, this function
    iterates through all models and parses the predictions in separate python dictionaries 

    :param params: input parameters
    :type params: python ``dictionary``
    :param myModels: ML predictors, a dictionary of models
    :type myModels: ``TensorFlow`` models organized in a python ``dictionary``
    :param saveRescaled: save the rescaled image?, defaults to ``True``
    :type saveRescaled: ``bool``, optional
    :param scaledImage: name of the rescaled image, defaults to ``None``
    :type scaledImage: ``str``, optional
    :return: 
        ``evalINCs``: python dictionary containing all evaluated inclinations
        ``evalRej``: python dictionary containing all rejection likelihoods
    :rtype: tuple ``(evalINCs, evalRej)``
    """

    evalINCs = {}

    try: 
        if len(params) == 6:
            RA    = params['alfa']
            Dec   = params['delta']
            npix  = params['npix']
            scale = params['scale']
            angle = params['angle']
            pix   = params['pix']

            url = "http://skyserver.sdss.org/dr12/SkyserverWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=" + \
                str(RA) + "&dec=" + str(Dec) + "&scale=" + str(pix) + "&width=" + str(npix) + "&height=" + str(npix)
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_arr = np.asarray(img)
        else:
            img_arr, scale, angle = openImage(params)
    except:
        return(evalINCs)


    if img_arr is None:
        return(evalINCs)
  
    ###############################
    img = converIMAGE(img_arr, angle=angle, scale=scale, size=128)

    if scaledImage is None:
        scaledImage = scaleFileName(params)

    if saveRescaled:
        img.save('./static/tempImages/resized/'+scaledImage, "JPEG")

    img_arr_ = np.asarray(img)
    nx, ny, channels = img_arr_.shape
    All_images = np.zeros((2, nx, ny, channels), dtype=np.dtype('>i4'))
    All_images[0] = img_arr_.reshape(1, nx, ny, channels)

    i = 1
    for delAngle in range(180,360,180):
        img = converIMAGE(img_arr, angle=angle+delAngle, scale=scale, size=128)
        img_arr_ = np.asarray(img)
        All_images[i] = img_arr_.reshape(1, nx, ny, channels)
        i+=1

    All_images = tf.cast(All_images, tf.float32)
    ###############################

    evalRej = {}
    evalRej_list = []
    for Model in myModels[0]:
        evalINC = {}
        modelName = Model[0]
        rej = predictor_binary(Model[1], All_images)
        evalRej[modelName] = rej
        evalRej_list.append(rej)

    evalINCs_list = []
    for i, modelGroup in enumerate(myModels[1:]):

        evalINC = {}
        
        for Model in modelGroup:
            modelName = Model[0]
            inc = predictor(Model[1], All_images)
            evalINC[modelName] = inc
            evalINCs_list.append(inc)

        gKey = "Group_"+str(i)
        evalINCs[gKey] = evalINC

    evalINCs["summary"] = {}
    evalINCs["summary"]["mean"] = np.mean(evalINCs_list)
    evalINCs["summary"]["median"] = np.median(evalINCs_list)
    evalINCs["summary"]["stdev"] = np.std(evalINCs_list)


    evalRej["summary"] = {}
    evalRej["summary"]["mean"] = np.mean(evalRej_list)
    evalRej["summary"]["median"] = np.median(evalRej_list)
    evalRej["summary"]["stdev"] = np.std(evalRej_list)

    return evalINCs, evalRej
            
            
###############################################################
def model2html(params, myModels):
    """parsing the output results in html format
    Given a set of parameters, all evaluations are carried out and returned in ``html`` format
    for the use in the online ``GUI``.

    :param params: input parameter set
    :type params: python ``dictionary``
    :param myModels: ``TensorFlow`` models organized in a python ``dictionary``
    :type myModels: python ``dictionary``
    :return: a ``summary`` of all evaluations
    :rtype: ``html`` text
    """   

    scaledImage = scaleFileName(params)
    evalINCs, evalRej  = model(params, myModels, scaledImage = scaledImage)

    results = """
    <div style="border: 1px solid black;margin: 15px;padding:15px">

                    <table margin="15px">
                    <tr><td>


                        <p>Input image: 128x128 pixels</p>

                        <div id="incImage">
                        <img src="/static/tempImages/resized/" """ + scaledImage + """"  width="256px" height="256px">
                        </div>


                    </td></tr>
                    <tr><td valign="top">

                        <div id="incResults">

                        <p>Neural Network Evaluation ....</p>

                        
    """ 
    allincs = []
    for key, evalGroup in evalINCs.items():
        if key != 'summary':
            for modelName, prediction in evalGroup.items():
                allincs.append(prediction)

    
    results += "<hr>"
    results += '<p style="background-color:yellow;"><b>Evaluated Inclinations</b></p>'
    allincs = np.asarray(allincs)
    med = np.median(allincs)
    mean = np.mean(allincs)
    stdev = np.std(allincs)
    results += "<p style=\"color:blue;\"><b>Median: </b>" + str('%d' % med) + "&nbsp;[deg]</p>"
    results += "<p style=\"color:blue;\"><b>Mean: </b>" + str('%d' % mean) + "&nbsp;[deg]</p>"
    results += "<p style=\"color:blue;\"><b>Stdev: </b>" + str('%d' % stdev) + "&nbsp;[deg]</p>"


    results += "<hr>"
    results += '<p style="background-color:yellow;"><b>Rejection Likelihood</b></p>'

    rejs = []
    for rejmodel, rejlike in evalRej.items():
        if rejmodel != 'summary':
            rejs.append(rejlike)
    
    med = np.median(rejs)
    mean = np.mean(rejs)
    stdev = np.std(rejs)

    color = 'green'
    if med > 50:
        color = 'red'
    results += "<p style=\"color:"+color+";\"><b>Median: </b>" + '%.0f'%(med) + ' %</p>'
    results += "<p style=\"color:"+color+";\"><b>Mean: </b>" + str('%d' % mean) + ' %</p>'
    results += "<p style=\"color:"+color+";\"><b>Stdev: </b>" + str('%d' % stdev) + ' %</p>'


    results += """
                        </div>


                    </td></tr>
                    </table>
    </div>
    """

    return results, scaledImage
