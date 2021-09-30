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
# Rotating and zoomming based of the user inputs
def convertIMAGE(im_path, pgcID, im_root, band, angle=0., scale=1., size=128, suffix=None, verbose=False):
    
    if scale<1.:
        scale=1

    if band != 'gri':
        try:
            img = Image.open(im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.png')
        except:
            if verbose:
                print("Error: "+im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.png')
            return None
    else:
        try:
            img = Image.open(im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.sdss.jpg')
        except:
            try:
                img = Image.open(im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.jpg')
            except:
                if verbose:
                    print("Error: "+im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.sdss.jpg')
                    print("Error: "+im_path + 'pgc'+str(pgcID)+'_'+im_root+'_'+band+'.jpg')
                return None

    
    # image rotation
    img_rot = scipy.ndimage.rotate(img, -angle)
    
    img_rot = np.asarray(img_rot)
    
    N = img_rot.shape[0]

    # image scaling
    
    d = N
    p =  int(d/scale)
    d1 = int(np.round(d/2-p/2))
    d2 = int(np.round(d1 + p))
    
    if band == 'gri':
        band = 'RGB'
    
    if band != 'RGB':

        try:
            img_rot = np.mean(img_rot, axis=2)
            img_rot = img_rot.astype(np.uint8)
        except:
            pass
       
    
    img_cut = img_rot[d1:d2, d1:d2]
    
    if band != 'RGB':
        img = Image.fromarray(img_cut, 'L').resize((size,size))
    else:
        img = Image.fromarray(img_cut, 'RGB').resize((size,size))

    outDIR = './' + str(size)+'x'+str(size) + '_'+band
    if not os.path.exists(outDIR):
        xcmd('mkdir '+outDIR, verbose)
    
    if not suffix is None:
        outName = outDIR+'/pgc'+str(pgcID)+'_'+str(size)+'x'+str(size)+'_'+suffix+'.'+band+'.jpg'
    else:
        outName = outDIR+'/pgc'+str(pgcID)+'_'+str(size)+'x'+str(size)+'.'+band+'.jpg'

    try:
        img.save(outName, "JPEG")
        if verbose: print("saved: ", outName)
    except:
        if verbose: print("couldn't save: ", outName)
        pass
    
    return img_cut

######################################

def main(im_path, im_root='d25x2_rot', band='i', catalog='catalog.csv', size=128, verbose=False):

    inFile = catalog
    table  = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None, encoding=None)
    pgc  = table['pgc']
    inc  = table['inc']
    face_on  = table['fon']
    inc_note = table['inc_note']
    inc_flg = table['inc_flg']

    for j, id in enumerate(pgc):
        
        if band!='gri':
            im_full_path =  im_path+'/'+'pgc'+str(id)+'_'+im_root+'_'+band+'.png'
        else:
            im_full_path = im_path + 'pgc'+str(id)+'_'+im_root+'_'+band+'.sdss.jpg'
            if not os.path.exists(im_full_path):
                im_full_path = im_path + 'pgc'+str(id)+'_'+im_root+'_'+band+'.jpg'
        
        available = os.path.exists(im_full_path)
        
        ## 'F' stands for face-on
        ## '0' spirals with inclinations less than 45 deg from face-on
        ## '45'-'90' spirals with inclinations between 45 and 90 deg
        try:
            fon = " ".join(face_on[j].split())
        except:
            fon = ""

        if ((fon == 'F' and inc_flg[j]>0) or (inc_flg[j]>0 and 'face_on' in inc_note[j])):
            suffix = 'F'
        elif inc_flg[j]>0:
            suffix = '0'
        else:
            suffix = "%d"%inc[j]
        
        if available:
            try: 
                img = convertIMAGE(im_path, id, im_root, band, size=size, suffix=suffix, verbose=verbose)
            except:
                if verbose:
                    print('Problem: ' + im_full_path)
                pass
            

#################################################

def arg_parser():
    parser = OptionParser(usage="""\
\n
 - Resizing the original images that reside in a specific folder (image_path), e.g. ./galaxies
 - File names are formatted as pgc<xxx>_<image_root>_<image_root>_<band>.png
 - or pgc<xxx>_<image_root>_<image_root>_gri.sdss.jpg (directly from the SDSS server)
 - or pgc<xxx>_<image_root>_<image_root>_gri.jpg (processed locally)
 - <xxx> stands for the ID of galaxy

 - bands: They specify the desired waveband: 'g', 'r', 'i' for grayscale, and 'gri' for colorful
 
 - How to run: 
 
    $ %prog -p <image_path> -r <image_root> -b <band> -c <catalog_name] -s <output_size> -v <verbose>
 
 - Example:
    $ python data_prep.py -p ./galaxies/ -r d25x2_rot -c catalogs/catalog.csv -b gri -s 64 -v

    This looks for images like:
        galaxies/pgc44182_d25x2_rot_gri.sdss.jpg
        or 
        galaxies/pgc44182_d25x2_rot_gri.jpg


    $ python data_prep.py --catalog catalogs/catalog.csv --band i

    This requires images like:
        galaxies/pgc44182_d25x2_rot_i.png


    
 - Author: "Ehsan Kourkchi"
 - Copyright 2021
""")

    parser.add_option('-p', '--impath',
                      type='string', action='store', default="./galaxies/",
                      help="""images path""")

    parser.add_option('-r', '--imroot',
                      type='string', action='store',
                      help="""images name root""", default="d25x2_rot")                     
    
    parser.add_option("-c", "--catalog",
                      type='string', action='store',
                      help="catalog name, e.g. catalog.csv", default="catalog.csv")

    parser.add_option("-b", "--band",
                      type='string', action='store',
                      help=""" waveband, e.g. "g", "r", or  "i" """)
    

    parser.add_option("-s", "--size",
                      type='int', action='store',
                      help="number of pixels on each side (e.g. 128)", default=128)

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

    if not os.path.exists(opts.impath):
        print('Error: '+opts.impath+" doesn't exist !!!")

    try:
        main(opts.impath, im_root=opts.imroot, band=opts.band, catalog=opts.catalog, size=opts.size, verbose=opts.verbose)
    except:
        print("Error: use \"python "+sys.argv[0]+" -h\" for help ...  \n", file=sys.stderr)
