import sys, os, gc
import subprocess
from math import *
import numpy as np
from datetime import *
from pylab import *
import random
from datetime import datetime
import json
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from optparse import OptionParser
import CNN_models
import pandas as pd
import matplotlib.pyplot as plt
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

#################################################################

def esn_shuffle(array, seed=None):
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(array)
        return array


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#################################################################

def augment_test_data(infolder, pixels):

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
        iaa.GammaContrast(gamma=(0.97,1.03)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3), per_channel=0.5),
        iaa.Add((-5, 5), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.5),
        iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.50, iaa.Grayscale(alpha=1.)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(0, 359),
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, 
            mode=ia.ALL, cval=(0, 255))
        ], random_order=True)


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
  
    data = np.load(infolder+'/RGB_'+pixels+'_test_000.npz')
    images_RGB = data['images'].astype(np.uint8)
    labels_RGB = data['labels'] 
    pgcIDs_RGB = data['pgcIDs']
    N = images_RGB.shape[0]
    filter_RGB = np.chararray(N)
    filter_RGB[:] = 'c'
    data.close() 
    del data


    data = np.load(infolder+'/g_'+pixels+'_test_000.npz')
    images_g = data['images'].astype(np.uint8)
    labels_g = data['labels'] 
    pgcIDs_g = data['pgcIDs']
    N = images_g.shape[0]
    filter_g = np.chararray(N)
    filter_g[:] = 'g'
    data.close() 
    del data

    data = np.load(infolder+'/r_'+pixels+'_test_000.npz')
    images_r = data['images'].astype(np.uint8)
    labels_r = data['labels'] 
    pgcIDs_r = data['pgcIDs']
    N = images_r.shape[0]
    filter_r = np.chararray(N)
    filter_r[:] = 'r'
    data.close() 
    del data

    data = np.load(infolder+'/i_'+pixels+'_test_000.npz')
    images_i = data['images'].astype(np.uint8)
    labels_i = data['labels'] 
    pgcIDs_i = data['pgcIDs']
    N = images_i.shape[0]
    filter_i = np.chararray(N)
    filter_i[:] = 'i'
    data.close() 
    del data

    images_gri = np.concatenate((images_g, images_r, images_i))
    labels_gri = np.concatenate((labels_g, labels_r, labels_i))
    pgcIDs_gri = np.concatenate((pgcIDs_g, pgcIDs_r, pgcIDs_i))
    filter_gri = np.concatenate((filter_g, filter_r, filter_i))

    N_RGB = len(labels_RGB)
    N_gri = len(labels_gri)

    ia.seed(100)

    indx = esn_shuffle(np.arange(N_gri), seed=200)
    images_aug = seqGray(images=images_gri[indx][:N_RGB,:,:,:])
    labels_aug = labels_gri[indx][:N_RGB]
    pgcIDs_aug = pgcIDs_gri[indx][:N_RGB]
    filter_aug = filter_gri[indx][:N_RGB]

    n = len(images_aug)
    p = int(n/2)
    images_aug[:p] = 255 - images_aug[:p]

    ia.seed(200)

    images_aug = np.concatenate((seq(images=images_RGB),images_aug))
    labels_aug = np.concatenate((labels_RGB,labels_aug))
    pgcIDs_aug = np.concatenate((pgcIDs_RGB,pgcIDs_aug))
    filter_aug = np.concatenate((filter_RGB,filter_aug))

    indx = np.arange(len(images_aug))
    indx = esn_shuffle(indx, seed=100)
    images_test_aug = images_aug[indx]
    labels_test_aug = labels_aug[indx]
    pgcIDs_test_aug = pgcIDs_aug[indx]
    filter_test_aug = filter_aug[indx]

    labels_test_aug = 2.*(labels_test_aug-45.)/45. - 1.

    return images_test_aug, labels_test_aug, pgcIDs_test_aug, filter_test_aug

#################################################################

def trainer(iter, batchNo, model_function, samples, size, zp_dir, ckpt_dir, 
        training_batch_size=64, suffix=None, model_name='',  verbose=False):

    
    pixels = str(size)+'x'+str(size)
    images_test_aug, labels_test_aug, pgcIDs_test_aug, filter_test_aug = augment_test_data(samples, pixels)

    if suffix is None:
        suffix = ''

    vgg_model = model_function()
    vgg_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse', 'mae'])

    r = [x for x in zp_dir.split('/') if x!=''][-1][:-4]
    evalJSON = ckpt_dir+r+'_'+model_name+'_evalDict'+suffix+'.json'
    evalFig  = ckpt_dir+r+'_'+model_name+'_evalFig'+suffix+'.jpg'

    if iter>0:

        vgg_model = model_function()
        vgg_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse', 'mae'])
        vgg_model.load_weights(ckpt_dir+str(iter-1)+suffix+".ckpt")

        with open(evalJSON) as json_file:
            evalDict = json.load(json_file)
    else:
        evalDict = {}

    batchFile = pixels+'_train_aug_'+'%02d'%(batchNo+1)+'.npz'
    data = np.load(zp_dir + '/' + batchFile)
    images_train_aug = data['images'].astype(np.uint8)
    labels_train_aug = data['labels']
    pgcIDs_train_aug = data['pgcIDs']
    data.close() 
    del data

    if verbose:
        print("batch file: ", zp_dir + '/' + batchFile)
        print("evaluation file: ", evalJSON)
        print("evaluation figure: ", evalFig)

    labels_train_aug = 2.*(labels_train_aug-45.)/45. - 1.

    n_epochs=1
    vgg_model.fit(images_train_aug/255., labels_train_aug, 
                                        epochs=n_epochs, 
                                        batch_size=training_batch_size, 
                                        validation_data=(images_test_aug/255., labels_test_aug),
                                        verbose=verbose, shuffle=True)
    evalDict[iter] = {} 
    for key in vgg_model.history.history:
        evalDict[iter][key] = vgg_model.history.history[key][0]
    evalDict[iter]["batchNo"] = batchNo
    evalDict[iter]["batchFile"] = batchFile

    with open(evalJSON, "w", encoding ='utf8') as outfile:
        json.dump(evalDict, outfile, allow_nan=True, cls=NpEncoder)

    ## plotting the valuation metrics
    plot_dict(evalDict, evalFig)

    vgg_model.save_weights(ckpt_dir+str(iter)+suffix+".ckpt")

    del vgg_model
    tf.keras.backend.clear_session()
    del images_train_aug 
    del labels_train_aug
    del pgcIDs_train_aug

    gc.collect()

    return evalDict

#################################################################
def plot_dict(evalDict, figname):

    # with open(evalJSON) as json_file:
    #     evalDict = json.load(json_file)
        
    df =  pd.DataFrame.from_dict(evalDict).T

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    epochs = np.arange(len(df))

    ax[0].plot(epochs, df.mse.values, label='Training')
    ax[0].plot(epochs, df.val_mse, label='Validation', alpha=0.5)
    ax[0].set_title('Mean Square Error')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(epochs, df.mae.values, label='Training')
    ax[1].plot(epochs, df.val_mae, label='Validation', alpha=0.5)
    ax[1].set_title('Mean Absolute Error Curves')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Mean Absolute Error')

    plt.savefig(figname)
    plt.close()

    return ax

#################################################################

def main(infolder, sample, size, model_foder, model_name, m_iter=3, 
        n_batches=3, n_batch_iterations=2, 
        training_batch_size=2, 
        verbose=False):

    ## training for all data subsamples
    for jj in range(m_iter+1): 

        myModels = CNN_models.Models(input_shape=(size, size, 3))

        batches = esn_shuffle(np.arange(n_batches), seed=50)
        for i in range(1, n_batch_iterations):
            batches = np.concatenate((batches, esn_shuffle(np.arange(n_batches), seed=i*50*jj))) 

        # VGG function
        model_function = myModels.getModel(model_name)

        ## Folder to save the weight numbers of the network after each iteration
        ckpt_dir = model_foder+'/'+'Uset_'+'%03d'%(jj)+'_'+model_name+'_ckpt'

        ## Folder that contains all n_batches training batches in npz format
        zp_dir = infolder+'/'+'Uset_'+'%03d'%(jj)+'_npz'

        if not os.path.exists(model_foder):
            xcmd('mkdir '+ model_foder, verbose)

        if not os.path.exists(ckpt_dir):
            xcmd('mkdir '+ ckpt_dir, verbose)

        N = n_batches*n_batch_iterations

        for iter in range(N):
            batchNo = batches[iter]

            if verbose:
                print("iteration number: {}/{}".format(iter+1, N))
                print("batch number: ", batchNo)

            trainer(iter, batchNo, model_function, sample, size, zp_dir+'/', ckpt_dir+'/', 
                        training_batch_size=training_batch_size, model_name=model_name, verbose=verbose)

#################################################################

def arg_parser():
    parser = OptionParser(usage="""\
\n
 - Training a VGG model using the augmented data. 
 - Data augmentation has been done separately in another code and the output data has been stored on disk for this purpose

 - How to run: 
 
    $ %prog -a <augmented_images_folder> -i <resized_images_path>
         -s <image_size> -o <output_models_path> -m <model_name_to_train>
        -n <m_iter> -b <n_batches> -N <totl_coverage_of_batches>
        -z <training_batch_size> -v <verbose>

    - m_iter is the number of subsamples, each with the size of 67% of the entire dataset
    - n_batches is the number of augmented batches as stored in npz format
    - totl_coverage_of_batches is the total number of repetition over all batches
    - output_models_path is the folder to save the weight numbers of the network after each iteration

 - Example:
    $ python batch_training_regr.py \
        -a augmented -i samples -s 128 \
        -o models -m model4 -n 3 -b 3 -N 1 -z 64 -v

    - outputs are model snapshots (weight numbers) and evaluation metrics:
        - <output_models_path>/Uset_<m_iter>_<model_name>_ckpt
        - <output_models_path>/***.json  (training/testing metrics)
        - <output_models_path>/***.jpg   (a figure plotting metrics vs. epoch number)

   
 - Author: "Ehsan Kourkchi"
 - Copyright 2021
""")

    parser.add_option('-a', '--augmented',
                      type='string', action='store', default="./augmented/",
                      help="""folder of augmented images""")

    parser.add_option('-i', '--samples',
                      type='string', action='store', default="./samples/",
                      help="""folder of resized images""")
    
    parser.add_option("-s", "--size",
                      type='int', action='store',
                      help="number of pixels on each side (e.g. 128)", default=128)

    parser.add_option("-o", "--outModels",
                      type='string', action='store',
                      help="the path of the output folder to store models", default='./models/')

    parser.add_option('-m', '--modelName',
                      type='string', action='store', default="model4",
                      help="""name of the model (e.g. model4)""")

    parser.add_option("-n", "--niter",
                      type='int', action='store',
                      help="number of data subsets", default='3')

    parser.add_option("-b", "--nbatch",
                      type='int', action='store',
                      help="number of batches (stored npz files)", default='3')

    parser.add_option("-N", "--N_batch_iter",
                      type='int', action='store',
                      help="total number of iterations over all stored bath files", default='1')

    parser.add_option("-z", "--training_batch_size",
                      type='int', action='store',
                      help="""
                      Batch size at each epoch.
                      """, default='3')

    parser.add_option("-v", "--verbose", action="store_true",
                      help="verbose (default=False)", default=False)

    (opts, args) = parser.parse_args()
    return opts, args

#################################################################

if __name__ == '__main__':

    opts, args = arg_parser()

    if opts.verbose: 
        print("\n------------------------------------")
        print("Input Arguments (provided by User)")
        print("------------------------------------")
        print(opts)
        print("------------------------------------")

    if not os.path.exists(opts.augmented):
        print('Error with input folder: '+opts.augmented+" doesn't exist !!!")
        sys.exit()
    
    if not os.path.exists(opts.samples):
        print('Error with input folder: '+opts.samples+" doesn't exist !!!")
        sys.exit()

    if True: # try:
        main(opts.augmented, 
        opts.samples, 
        opts.size, 
        opts.outModels, 
        opts.modelName, 
        m_iter=opts.niter, 
        n_batches=opts.nbatch, 
        n_batch_iterations=opts.N_batch_iter, 
        training_batch_size=opts.training_batch_size, 
        verbose=opts.verbose)
    # except:
        # print("Error: use \"python "+sys.argv[0]+" -h\" for help ...  \n", file=sys.stderr)
    

