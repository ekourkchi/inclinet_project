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
import scipy.misc as scimisc
import random
from datetime import datetime
import json
from optparse import OptionParser
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from sklearn.model_selection import train_test_split
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import copy 

######################################
## This function allows to execute the OS commands
def xcmd(cmd, verbose):

    if verbose: print('\n'+cmd)

    tmp=os.popen(cmd)
    output=''
    for x in tmp: output+=x
    if 'abort' in output:
        failure=True
    else:
        failure=tmp.close()
    if False:
        print('execution of %s failed' % cmd)
        print('error is as follows', output)
        sys.exit()
    else:
        return output

######################################

#################################################################
def esn_shuffle(array, seed=None):
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(array)
        return array

#################################################################

ia.seed(100)

seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),               # bluring the image 50% of times
    iaa.GammaContrast(gamma=(0.97,1.03)),                               # altering the contrast     
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3), per_channel=0.5),  # adding Guassian noise
    iaa.Add((-5, 5), per_channel=0.5),                                  # randomly change of the pixel values
    iaa.Multiply((0.8, 1.2), per_channel=0.5),                          # changing the intensity of channel
    iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0))),                # change images to grayscale overlayed with the original   iaa.Sometimes(0.50, iaa.Grayscale(alpha=1.)),
    iaa.Fliplr(0.5),                                                    # left/right flip (50% of cases)
    iaa.Flipud(0.5),                                                    # up/down flip (50% of cases)
    iaa.Affine(                                                         # affine transformations 
        rotate=(0, 359),
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, 
        mode=ia.ALL, cval=(0, 255))
    ], random_order=True)


# Grayscale augmentation takes fewer transformations, because all three channels are the same
seqGray = iaa.Sequential([
    iaa.GammaContrast(gamma=(0.97,1.03)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        rotate=(0, 359),
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, 
        mode=ia.ALL, cval=(255, 255))
    ], random_order=True)

#################################################################
# augment_N takes the image array and augment it several times. 
# N number of images are then taken randomly and returned as output.
def augment_N(images, labels, pgcIDs, filter, N, isGray=True, seed=0):
    '''
        images, labels, pgcIDs: input data
        filer: the band-pass of the input data
        N: the number of the output augmented images
        isGray: True: grayscale images, False: colorful images 
        seed: the randomness state
    '''
   
    if isGray:
        images_aug = seqGray(images=images)
    else:
        images_aug = seq(images=images)
    labels_aug = labels
    pgcIDs_aug = pgcIDs
    filter_aug = filter
    
    ii = 0 
    while len(labels_aug)<N:   # The image array is augmented until its size grows bigger than N
        if isGray:
            images_aug = np.concatenate((images_aug,seqGray(images=images)))
        else:
            images_aug = np.concatenate((images_aug,seq(images=images)))         
        labels_aug = np.concatenate((labels_aug, labels))
        pgcIDs_aug = np.concatenate((pgcIDs_aug, pgcIDs))
        filter_aug = np.concatenate((filter_aug, filter))
        
    
    # returning N number of galaxies randomly
    indx = esn_shuffle(np.arange(len(labels_aug)), seed=seed)
    images_aug = images_aug[indx][:N]
    labels_aug = labels_aug[indx][:N]
    pgcIDs_aug = pgcIDs_aug[indx][:N]
    filter_aug = filter_aug[indx][:N]
    
    return images_aug, labels_aug, pgcIDs_aug, filter_aug

#################################################################

def uniformer(images, labels, pgcIDs, filter, N=1, isGray=True, seed=0):

    '''
    images, labels, pgcIDs: input data
    filer: the band-pass of the input data
    N: the number of the augmented images within 5 degrees of inclination interval
    isGray: True: grayscale images, False: colorful images 
    seed: the randomness state
    '''

    n = len(labels)
    indices = np.arange(n)

    first_time = True
    
    for i in range(45,90,5):

        if i==45:   # taking care of boundaries
            idx = indices[((labels>=i)&(labels<=i+5))]
        else:
            idx = indices[((labels>i)&(labels<=i+5))]

        ## augmenting each 5-deg interval to hold N galaxies

        if len(idx)!=0:
            images_aug, labels_aug, pgcIDs_aug, filter_aug = augment_N(images[idx], 
                                                                    labels[idx], 
                                                                    pgcIDs[idx], 
                                                                    filter[idx], N, isGray=True, seed=seed)
            if not first_time:
                images_aug_ = np.concatenate((images_aug_, images_aug))
                labels_aug_ = np.concatenate((labels_aug_, labels_aug))
                pgcIDs_aug_ = np.concatenate((pgcIDs_aug_, pgcIDs_aug))
                filter_aug_ = np.concatenate((filter_aug_, filter_aug))
            else:
                images_aug_ = copy.deepcopy(images_aug)
                labels_aug_ = copy.deepcopy(labels_aug)
                pgcIDs_aug_ = copy.deepcopy(pgcIDs_aug)
                filter_aug_ = copy.deepcopy(filter_aug)
                first_time = False

    return images_aug_, labels_aug_, pgcIDs_aug_, filter_aug_

#################################################################
def main(infolder, size, outfolder, m_iter=3, n_batches=3, batch_size=1000, verbose=False):

    pixels = str(size)+'x'+str(size)

    for jj in range(m_iter+1):

        ## Importing all training samples
        data = np.load(infolder+'/RGB_'+pixels+'_train_'+'%03d'%(jj)+'.npz')
        images_RGB = data['images'].astype(np.uint8)
        labels_RGB = data['labels'] 
        pgcIDs_RGB = data['pgcIDs']
        N = images_RGB.shape[0]
        filter_RGB = np.chararray(N)
        filter_RGB[:] = 'c'
        data.close() 
        del data


        data = np.load(infolder+'/g_'+pixels+'_train_'+'%03d'%(jj)+'.npz')
        images_g = data['images'].astype(np.uint8)
        labels_g = data['labels'] 
        pgcIDs_g = data['pgcIDs']
        N = images_g.shape[0]
        filter_g = np.chararray(N)
        filter_g[:] = 'g'
        data.close() 
        del data

        data = np.load(infolder+'/r_'+pixels+'_train_'+'%03d'%(jj)+'.npz')
        images_r = data['images'].astype(np.uint8)
        labels_r = data['labels'] 
        pgcIDs_r = data['pgcIDs']
        N = images_r.shape[0]
        filter_r = np.chararray(N)
        filter_r[:] = 'r'
        data.close() 
        del data

        data = np.load(infolder+'/i_'+pixels+'_train_'+'%03d'%(jj)+'.npz')
        images_i = data['images'].astype(np.uint8)
        labels_i = data['labels'] 
        pgcIDs_i = data['pgcIDs']
        N = images_i.shape[0]
        filter_i = np.chararray(N)
        filter_i[:] = 'i'
        data.close() 
        del data

        ## Concatenating all grayscale images
        images_gri = np.concatenate((images_g, images_r, images_i))
        labels_gri = np.concatenate((labels_g, labels_r, labels_i))
        pgcIDs_gri = np.concatenate((pgcIDs_g, pgcIDs_r, pgcIDs_i))
        filter_gri = np.concatenate((filter_g, filter_r, filter_i))


        ## Generating n batches of augmented training sample
        for i in range(n_batches):

            ia.seed(2*i+12)
            
            ## The gri ensebmle is almost 3 times larger than the RGB sample, so N=3000 is three times larger
            ## than that for the RGB images
            images_aug_gri, labels_aug_gri, pgcIDs_aug_gri, filter_aug_gri = uniformer(images_gri, 
                                                                                    labels_gri, 
                                                                                    pgcIDs_gri, 
                                                                                    filter_gri, 
                                                                                    N=3*batch_size, isGray=True, seed=3*i+36)
            ia.seed(5*i+25)
            
            images_aug_RGB, labels_aug_RGB, pgcIDs_aug_RGB, filter_aug_RGB = uniformer(images_RGB, 
                                                                                    labels_RGB, 
                                                                                    pgcIDs_RGB, 
                                                                                    filter_RGB, 
                                                                                    N=batch_size, isGray=False, seed=3*i+41)
            N_RGB = len(labels_aug_RGB)
            N_gri = len(labels_aug_gri)

            indx = esn_shuffle(np.arange(N_gri), seed=6*i+40)
            images_aug = images_aug_gri[indx][:N_RGB]
            labels_aug = labels_aug_gri[indx][:N_RGB]
            pgcIDs_aug = pgcIDs_aug_gri[indx][:N_RGB]
            filter_aug = filter_aug_gri[indx][:N_RGB]
            
            ## half of the grayscale images are drawn randomly and inverted
            n = len(images_aug)
            p = int(n/2)
            images_aug[:p] = 255 - images_aug[:p]

            ia.seed(2*i+51)
            images_aug = np.concatenate((images_aug_RGB,images_aug))
            labels_aug = np.concatenate((labels_aug_RGB,labels_aug))
            pgcIDs_aug = np.concatenate((pgcIDs_aug_RGB,pgcIDs_aug))
            filter_aug = np.concatenate((filter_aug_RGB,filter_aug))
            
            indx = np.arange(len(images_aug))
            indx = esn_shuffle(indx, seed=32*i+13)
            images_aug = images_aug[indx]
            labels_aug = labels_aug[indx]
            pgcIDs_aug = pgcIDs_aug[indx]
            filter_aug = filter_aug[indx]
            

            if not os.path.exists(outfolder):
                xcmd('mkdir '+outfolder, verbose)
            
            outData = outfolder+'/'+'Uset_'+'%03d'%(jj)+'_npz'
            if not os.path.exists(outData):
                xcmd('mkdir '+outData, verbose)   
            ## saving the output product in npz format
            npzname = outData+'/'+pixels+'_train_aug_'+'%02d'%(i+1)+'.npz'
            np.savez_compressed(npzname, 
                                    images=images_aug, 
                                    labels=labels_aug, 
                                    pgcIDs=pgcIDs_aug,
                                    filter=filter_aug
                            )
            
            if verbose: print(npzname+' ... saved.')


 ###############################################

def arg_parser():
    parser = OptionParser(usage="""\
\n
 - Augmenting training sets and storing them on the disk to be used during the traning process
 - Generating augmented samples with uniform distribution of inclinations 

 - How to run: 
 
    $ %prog -i <input_folder_path> -o <output_folder_path> -s <image_size> 
        -n <m_iter> -b <n_batches> -z <batch_size> -v <verbose>

    - m_iter is the number of subsamples, each with the size of 67% of the entire dataset
    - All arrays of images are taken, and they are divided into the inclination bins of size 5, starting at 45 degrees. 
    - Each of the 5-degree sub-samples are augmented separately
    - <batch_size>=1000 means that each of the 5 degree intervals have 1,000 galaxies after the augmentation process.
 
 - Example:
    $ python data_augment_uniform.py -i samples -o augmented -s 128 -n 3 -b 5 -v

    - output format for 128x128 images:
        - <band>_128x128_test_000.npz
        - <band>_128x128_train_000.npz

        - <band>_128x128_test_xxx.npz   (67% of data)
        - <band>_128x128_train_xxx.npz  (67% of data)
        where <xxx> is the iteration number. 

   
 - Author: "Ehsan Kourkchi"
 - Copyright 2021
""")

    parser.add_option('-i', '--infolder',
                      type='string', action='store', default="./samples/",
                      help="""folder of resized images""")

    
    parser.add_option("-s", "--size",
                      type='int', action='store',
                      help="number of pixels on each side (e.g. 128)", default=128)

    parser.add_option("-o", "--outfolder",
                      type='string', action='store',
                      help="the path of the output folder", default='./augmented')

    parser.add_option("-n", "--niter",
                      type='int', action='store',
                      help="number of iterations", default='3')

    parser.add_option("-b", "--nbatch",
                      type='int', action='store',
                      help="number of batches", default='3')

    parser.add_option("-z", "--batchsize",
                      type='int', action='store',
                      help="""
                      The nominal size of batches within each 5 degrees of inclnation interval.
                      Total size is estimated to be 18*batchsize in full production.
                      """, default='3')

    parser.add_option("-v", "--verbose", action="store_true",
                      help="verbose (default=False)", default=False)

    (opts, args) = parser.parse_args()
    return opts, args
########

#################################################################
if __name__ == '__main__':

    opts, args = arg_parser()

    if opts.verbose: 
        print("\n------------------------------------")
        print("Input Arguments (provided by User)")
        print("------------------------------------")
        print(opts)
        print("------------------------------------")

    if not os.path.exists(opts.infolder):
        print('Error with input folder: '+opts.infolder+" doesn't exist !!!")
        sys.exit()

    try:
        main(
            opts.infolder, 
            opts.size, 
            opts.outfolder, 
            m_iter=opts.niter, 
            n_batches=opts.nbatch, 
            batch_size=opts.batchsize,
            verbose=opts.verbose
            )
    except:
        print("Error: use \"python "+sys.argv[0]+" -h\" for help ...  \n", file=sys.stderr)
    
