#!/usr/bin/python
__author__ = "Ehsan Kourkchi"
__copyright__ = "Copyright 02-11-2020"
__version__ = "v1.0"
__status__ = "Production"

import sys
import os
import subprocess
from math import *
import numpy as np
from datetime import *
from pylab import *
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import scipy.ndimage
import random
import requests
from io import BytesIO
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
###############################


def converIMAGE(img_arr, angle=0., scale=1., size=64):

    if scale < 1.:
        scale = 1

    img_rot = scipy.ndimage.rotate(img_arr, -angle)

    N = img_rot.shape
    d = N[0]
    p = int(d / scale)
    d1 = int(d / 2 - p / 2)
    d2 = int(d1 + p)

    imgut = img_rot[d1:d2, d1:d2, :]

    img = Image.fromarray(imgut, 'RGB').resize((size, size))

    return img

###############################
if len(sys.argv) == 7:
    RA = sys.argv[1]
    Dec = sys.argv[2]
    npix = sys.argv[3]
    scale = float(sys.argv[4])
    angle = float(sys.argv[5])
    pix = sys.argv[6]

    url = "http://skyserver.sdss.org/dr12/SkyserverWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra=" + \
        RA + "&dec=" + Dec + "&scale=" + pix + "&width=" + npix + "&height=" + npix
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_arr = np.asarray(img)
elif len(sys.argv) == 4:
    if sys.argv[1] == 'local':
        scale = float(sys.argv[2])
        angle = float(sys.argv[3])
        url = 'http://edd.ifa.hawaii.edu/incNET/tmp.jpg'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img_arr = np.asarray(img)

        if len(img_arr.shape) == 2:
            img_arr = np.stack((img_arr,) * 3, axis=-1)
        elif len(img_arr.shape) == 1:
            print("error")
            sys.exit()

        if len(img_arr.shape) == 3 and img_arr.shape[2] > 3:
            img_arr = img_arr[:, :, 0:3]
    else:
        print("error")
        sys.exit()
else:
    print("error")
    sys.exit()

nx, ny, _ = img_arr.shape
if nx != ny:
    print("size")
    sys.exit()
###############################
img = converIMAGE(img_arr, angle=angle, scale=scale, size=64)
img.save('./demo.jpg', "JPEG")


img_arr = np.asarray(img)
nx, ny, channels = img_arr.shape

img_arr = img_arr.reshape(1, nx, ny, channels)

###############################
regression = load_model(
    "../../secure/incNET/evaluate/CNN_inc_VGG6_regr_seed100.h5")
#classify = load_model("../../secure/incNET/evaluate/CNN_inc_VGG6_classify_seed100.h5")
binary = load_model("../../secure/incNET/evaluate/CNN_inc_VGG6_binary.h5")

img_arr = tf.cast(img_arr, tf.float32)

inc_pr = regression.predict(img_arr)
#inc_pc = classify.predict(img_arr)
#inc_pc = np.argmax(inc_pc, axis=1) + 51
flag = binary.predict(img_arr)

inc_pr = np.round(inc_pr[0][0])
#inc_pc = inc_pc[0]
rejection = np.round(flag[0][0] * 100)

#print(inc_pr, inc_pc, rejection)


if inc_pr > 90:
    inc_pr = 90.
###############################

results = """
<div style="border: 1px solid black;margin: 15px;padding:15px">

                <table margin="15px">
                <tr><td>


                    <p>Input image: 64x64 pixels</p>

                    <div id="incImage">
                    <img src="https://edd.ifa.hawaii.edu/incNET/evaluate/demo.jpg"  width="256px" height="256px">
                    </div>


                </td></tr>
                <tr><td valign="top">

                    <div id="incResults">

                    <p>Neural Network Evaulation ....</p>
                    <p><b>Inclination 1: </b>""" + str('%d' % inc_pr) + """&nbsp;[deg]

                    <p><b>Rejection Likelihood: </b>""" + str('%d' % rejection) + """%
                    </div>


                </td></tr>
                </table>
</div>
"""

print(results)
