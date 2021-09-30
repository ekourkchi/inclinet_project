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

#################################################################
def esn_shuffle(array, seed=None):
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(array)
        return array

#################################################################
def train_test_creation(npzFile, outFileRoot, m_iter=3, verbose=False):
    '''
        npzFile: name of the input npz file that holds the original sample at each band (g, r, i, RGB)
        outFileRoot: the root name of the output products
        m_iter: the number training sub-samples, each consisting of 67% of the whole training sample
    '''    
    data = np.load(npzFile)
    images_ = data['images'].astype(np.uint8)
    labels_ = data['labels'] 
    data.close() 
    del data
    
    ix, = np.where(labels_[1,:]>=45)

    images_c = images_[ix]
    labels_c = labels_[1,:][ix]
    pgcIDs_c = labels_[0,:][ix]

    N = images_c.shape[0]
    idx = np.arange(N)
    idx = esn_shuffle(idx, seed=0)
    images_c = images_c[idx]
    labels_c = labels_c[idx]
    pgcIDs_c = pgcIDs_c[idx]

    images_train, images_test, labels_train, labels_test, pgcIDs_train,  pgcIDs_test = train_test_split(images_c, 
                                                                                                        labels_c, 
                                                                                                        pgcIDs_c, 
                                                                                                        test_size=0.15,
                                                                                                        random_state=100
                                                                                                       )
    ## Storing train/test samples
    ###############################################
    npzTest = outFileRoot+'test_000.npz'
    np.savez_compressed(npzTest, 
                        images=images_test, 
                        labels=labels_test, 
                        pgcIDs=pgcIDs_test)
    
    if verbose: print('created ...', npzTest)
    
    
    npzTrain = outFileRoot+'train_000.npz'
    np.savez_compressed(npzTrain, 
                        images=images_train, 
                        labels=labels_train, 
                        pgcIDs=pgcIDs_train)
    
    if verbose: print('created ...', npzTrain)
    ###############################################

    del images_
    del labels_

    del images_c
    del labels_c
    del pgcIDs_c

    
    ## Storing m_iter training sub-samples, each holding 2/3 of the entire training set
    ###############################################
    N = images_train.shape[0]
    n = N*2//3
    pgcIDs_train_list = []
    
    for jj in range(m_iter):
        idx = np.arange(N)
        idx = esn_shuffle(idx, seed=100*(jj+1))
        images_train_ = images_train[idx][:n]
        labels_train_ = labels_train[idx][:n]
        pgcIDs_train_ = pgcIDs_train[idx][:n]
        
        images_test_ = images_train[idx][n:]
        labels_test_ = labels_train[idx][n:]
        pgcIDs_test_ = pgcIDs_train[idx][n:]
        
        npzTrain = outFileRoot+'train_'+'%03d'%(jj+1)+'.npz'
        np.savez_compressed(npzTrain, 
                            images=images_train_, 
                            labels=labels_train_, 
                            pgcIDs=pgcIDs_train_)
        if verbose: print('created ...', npzTrain)
        
        
        npzTest = outFileRoot+'test_'+'%03d'%(jj+1)+'.npz'
        np.savez_compressed(npzTest, 
                            images=images_test_, 
                            labels=labels_test_, 
                            pgcIDs=pgcIDs_test_)
        
        if verbose: print('created ...', npzTest)
        pgcIDs_train_list.append(pgcIDs_train_)
        
        del images_train_
        
    return pgcIDs_test, pgcIDs_train_list

#################################################################
def train_test_replication(npzFile, outFileRoot, pgcIDs_test, pgcIDs_train_list, verbose=False):
    
    m_iter = len(pgcIDs_train_list)
    
    data = np.load(npzFile)
    images_ = data['images'].astype(np.uint8)
    labels_ = data['labels'] 
    data.close() 
    del data
    
    ix, = np.where(labels_[1,:]>=45)
    images_c = images_[ix]
    labels_c = labels_[1,:][ix]
    pgcIDs_c = labels_[0,:][ix]
    
    idx = np.isin(pgcIDs_c, pgcIDs_test)
    npzTest = outFileRoot+'test_000.npz'
    np.savez_compressed(npzTest, 
                        images=images_c[idx], 
                        labels=labels_c[idx], 
                        pgcIDs=pgcIDs_c[idx])  
    if verbose: print('created ...', npzTest)
    
    
    idx = np.logical_not(np.isin(pgcIDs_c, pgcIDs_test))
    npzTrain = outFileRoot+'train_000.npz'
    np.savez_compressed(npzTrain, 
                        images=images_c[idx], 
                        labels=labels_c[idx], 
                        pgcIDs=pgcIDs_c[idx])  
    if verbose: print('created ...', npzTrain)
    
    
    for jj in range(m_iter):
        
        idx = np.isin(pgcIDs_c, pgcIDs_train_list[jj])
        npzTrain = outFileRoot+'train_'+'%03d'%(jj+1)+'.npz'
        np.savez_compressed(npzTrain, 
                            images=images_c[idx], 
                            labels=labels_c[idx], 
                            pgcIDs=pgcIDs_c[idx])  
        if verbose: print('created ...', npzTrain)
        
        
        idx = np.logical_not(np.isin(pgcIDs_c, pgcIDs_train_list[jj]))
        npzTest = outFileRoot+'test_'+'%03d'%(jj+1)+'.npz'
        np.savez_compressed(npzTest, 
                            images=images_c[idx], 
                            labels=labels_c[idx], 
                            pgcIDs=pgcIDs_c[idx])  
        if verbose: print('created ...', npzTest)
        
    del images_
    del labels_

    del images_c
    del labels_c
    del pgcIDs_c

 ###############################################

def main(infolder, size, outfolder, m_iter=3, verbose=False):

    pixels = str(size)+'x'+str(size)

    bands = ['r', 'i', 'g', 'RGB']
    main_band = None

    for band in bands:    
        npzfile = infolder+'/'+pixels+'_'+band+'.npz'

        if os.path.exists(npzfile):
            pgcIDs_test, pgcIDs_train_list = train_test_creation(npzfile, 
                                                        outfolder+'/'+band+'_'+pixels+'_',
                                                        m_iter=m_iter, verbose=verbose)
            main_band = band
            break
    
    if main_band is not None:
        for band in bands:
            if band != main_band:
                train_test_replication(infolder+'/'+pixels+'_'+band+'.npz', 
                                outfolder+'/'+band+'_'+pixels+'_', 
                                pgcIDs_test, 
                                pgcIDs_train_list, verbose=verbose)
    else:
        print('Warning: nothing to be done')
        return None

 ###############################################

def arg_parser():
    parser = OptionParser(usage="""\
\n
 - generating multiple data samples, each set is spitted to training/testing subsets
 - testing sample doesn't overlap with any of the training samples

 - How to run: 
 
    $ %prog -i <input_folder_path> -o <output_folder_path> -s <image_size> -n <m_iter> -v <verbose>

    - m_iter is the number of subsamples, each with the size of 67% of the entire dataset
 
 - Example:
    $ python data_split.py -i ./compressed/ -o samples/ -n 3 -v

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
                      type='string', action='store', default="./compressed/",
                      help="""folder of resized images""")

    
    parser.add_option("-s", "--size",
                      type='int', action='store',
                      help="number of pixels on each side (e.g. 128)", default=128)

    parser.add_option("-o", "--outfolder",
                      type='string', action='store',
                      help="the path of the output folder", default='./samples')

    parser.add_option("-n", "--niter",
                      type='int', action='store',
                      help="number of iterations", default='3')

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

    if not os.path.exists(opts.outfolder):
        print('Error with output folder: '+opts.outfolder+" doesn't exist !!!")
        sys.exit()

    try:
        main(opts.infolder, opts.size, opts.outfolder, m_iter=opts.niter, verbose=opts.verbose)
    except:
        print("Error: use \"python "+sys.argv[0]+" -h\" for help ...  \n", file=sys.stderr)

