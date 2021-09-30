import sys
import os
import subprocess
import glob
from math import *
import numpy as np
from datetime import *
from pylab import *
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
from astropy.table import Table, Column 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optparse import OptionParser
from PIL import Image#, ImageTk
from subprocess import Popen, PIPE
import matplotlib.patches as patches
import scipy.misc as scimisc
import scipy.ndimage



## labeling inclinations with numerical numbers
def inc_class(inc):
    
    if inc=='0':
        return 0
    elif inc=='F':
        return 1
    else:
        return int(inc)

#################################################################

def main(infolder, size, outfolder, verbose=False):

    files = glob.glob(infolder+'/*jpg')
    N = len(files)

    images = np.zeros((N, size, size, 3), dtype=np.dtype('>i4'))

    labels = np.zeros((2, N), dtype=np.dtype('>i4'))

    for i, fname in enumerate(files):

        froot = [x for x in fname.split('/') if x!=''][-1]
        pgcID =  int(froot.split('_')[0][3:])
        
        try:
            inc  =  fname.split("_")[3].split('.')[0]
        except:
            print('[failed] filename: ' + fname)
            sys.exit()  
            
        try:   # colorful images, RGB with 3 channels
            img = np.asarray(Image.open(fname))
            images[i] = img
            clss = inc_class(inc)
            labels[1][i] = clss
            labels[0][i]  = pgcID
        except: # grayscales, all channels are similar
            tmp = np.asarray(Image.open(fname))
            img = np.zeros((128, 128, 3), dtype=np.dtype('>i4'))
            img[:,:,0] = tmp
            img[:,:,1] = tmp
            img[:,:,2] = tmp           
            images[i] = img
            clss = inc_class(inc)
            labels[1][i] = clss
            labels[0][i]  = pgcID

    infolder_root = [x for x in infolder.split('/') if x!=''][-1]
    compressed_file = outfolder+'/'+infolder_root+'.npz'
    try:
        np.savez_compressed(compressed_file, images=images, labels=labels)
        if verbose:
            print("compressed file: "+compressed_file)
    except:
        pass

#################################################################

def arg_parser():
    parser = OptionParser(usage="""\
\n
 - Compressing all images in a folder
 - all images must have have the same size provided in the command line
 - make sure that the output folder exists, otherwise this code doesn't work

 - How to run: 
 
    $ %prog -i <input_folder_path> -o <output_folder_path> -s <image_size> -v <verbose>
 
 - Example:
    $ python data_compress.py -i 128x128_RGB -o compressed -s 128 -v

   
 - Author: "Ehsan Kourkchi"
 - Copyright 2021
""")

    parser.add_option('-i', '--infolder',
                      type='string', action='store', default="./128x128_RGB/",
                      help="""folder of resized images""")

    
    parser.add_option("-s", "--size",
                      type='int', action='store',
                      help="number of pixels on each side (e.g. 128)", default=128)

    parser.add_option("-o", "--outfolder",
                      type='string', action='store',
                      help="the path of the output folder", default='./')

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
        main(opts.infolder, opts.size, opts.outfolder, verbose=opts.verbose)
    except:
        print("Error: use \"python "+sys.argv[0]+" -h\" for help ...  \n", file=sys.stderr)

