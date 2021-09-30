#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Ehsan Kourkchi

# =============================================================================
# DOCS
# =============================================================================

"""The inclinet server side in Python

This package uses ``Flask`` to implement the API functionalities in Python.

Visit the online GUI at http://edd.ifa.hawaii.edu/inclinet/

For documentations check:
    https://github.com/ekourkchi/inclinet_deployment_repo/

"""

# =============================================================================
# IMPORTS
# =============================================================================

from flask import render_template
from flask import Flask, jsonify, request
import connexion
from datetime import date, timedelta
import json, base64
import pandas as pd
import sys, os, time
import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from PIL import Image
from optparse import OptionParser
from copy import deepcopy
import CNN_models
from incNET import *

# =============================================================================
# Flask initializations
# =============================================================================

# Create the application instance
app = connexion.App(__name__, specification_dir='./')
flask_app = app.app
flask_app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # set the maximum upload size to 1 MB

##########################################################################

## This function allows to execute the OS commands
def xcmd(cmd, verbose=True):
    """Runs an OS command

    :param cmd: terminal command
    :type cmd: ``str``
    :param verbose: printing the details, default True 
    :type verbose: ``boolean``
    :return: OS outputs
    :rtype: ``str``
    """

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

##########################################################################

def createDir(folderPath):
    """generating a directory/folder if it doesn't exist

    :param folderPath: path to the desired folder
    :type folderPath: ``str``
    :return: True is created, False if the folder already exists
    :rtype: ``str``
    """

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        return True
    else:
        False

##########################################################################

# Read the swagger.yml file to configure the endpoints
# app.add_api('swagger.yml')

# Create a URL route in our application for "/"

## Old models used for Demo purpose
# regression = load_model("./models/CNN_inc_VGG6_regr_seed100.h5")
# binary = load_model("./models/CNN_inc_VGG6_binary.h5")

def loadModels():
    """Loading all ML models (deployment versions)

    :return: a Python dictionary that holds all models in groups
    :rtype: python ``Dictionary``
    """

    myModels = []
    myModel = []

    classify_models = CNN_models.Models(n_classes=2)
    regression_models = CNN_models.Models(input_shape=(128, 128, 3))

    ckpt_dir = "./models/U0_model04_binary_ckpt/"
    MODEL4 = classify_models.getModel("model4")()
    MODEL4.load_weights(ckpt_dir+"200.ckpt")
    myModel.append(("model4-binary" , MODEL4))

    ckpt_dir = "./models/U0_model05_binary_ckpt/"
    MODEL5 = classify_models.getModel("model5")()
    MODEL5.load_weights(ckpt_dir+"200.ckpt")
    myModel.append(("model5-binary" , MODEL5))

    ckpt_dir = "./models/U0_model06_binary_ckpt/"
    MODEL6 = classify_models.getModel("model6")()
    MODEL6.load_weights(ckpt_dir+"200.ckpt")
    myModel.append(("model6-binary" , MODEL6))

    ################################################
    myModels.append(myModel)
    myModel = []

    model4 = regression_models.getModel("model4")

    ckpt_dir = "./models/U0_model04_ckpt/"
    MODEL = model4()
    MODEL.load_weights(ckpt_dir+"2999.ckpt")
    myModel.append(("model4" , MODEL))

    ckpt_dir = "./models/U1_model04_ckpt/"
    MODEL = model4()
    MODEL.load_weights(ckpt_dir+"1503.ckpt")
    myModel.append(("model41" , MODEL))

    ckpt_dir = "./models/U2_model04_ckpt/"
    MODEL = model4()
    MODEL.load_weights(ckpt_dir+"1800.ckpt")
    myModel.append(("model42" , MODEL))

    ckpt_dir = "./models/U3_model04_ckpt/"
    MODEL = model4()
    MODEL.load_weights(ckpt_dir+"1000.ckpt")
    myModel.append(("model43" , MODEL))


    ################################################

    myModels.append(myModel)
    myModel = []

    model5 = regression_models.getModel("model5")

    ckpt_dir = "./models/U0_model05_ckpt/"
    MODEL = model5()
    MODEL.load_weights(ckpt_dir+"700.ckpt")
    myModel.append(("model5" , MODEL))

    ckpt_dir = "./models/U1_model05_ckpt/"
    MODEL = model5()
    MODEL.load_weights(ckpt_dir+"700.ckpt")
    myModel.append(("model51" , MODEL))

    ckpt_dir = "./models/U2_model05_ckpt/"
    MODEL = model5()
    MODEL.load_weights(ckpt_dir+"700.ckpt")
    myModel.append(("model52" , MODEL))

    ckpt_dir = "./models/U3_model05_ckpt/"
    MODEL = model5()
    MODEL.load_weights(ckpt_dir+"700.ckpt")
    myModel.append(("model53" , MODEL))

    ################################################
    myModels.append(myModel)
    myModel = []

    model6 = regression_models.getModel("model6")

    ckpt_dir = "./models/U0_model06_ckpt/"
    MODEL = model6()
    MODEL.load_weights(ckpt_dir+"1200.ckpt")
    myModel.append(("model6" , MODEL))

    ckpt_dir = "./models/U1_model06_ckpt/"
    MODEL = model6()
    MODEL.load_weights(ckpt_dir+"1200.ckpt")
    myModel.append(("model61" , MODEL))

    ckpt_dir = "./models/U2_model06_ckpt/"
    MODEL = model6()
    MODEL.load_weights(ckpt_dir+"1200.ckpt")
    myModel.append(("model62" , MODEL))

    ckpt_dir = "./models/U3_model06_ckpt/"
    MODEL = model6()
    MODEL.load_weights(ckpt_dir+"1500.ckpt")
    myModel.append(("model63" , MODEL))

    myModels.append(myModel)

    return myModels


##########################################################################
def loadLeda():
    """Loading the HyperLeda catalog: 
    - http://leda.univ-lyon1.fr/
    - This catalog tabulates the proper information on local galaxies

    :return: the Leda catalog in the Pandas dataFrame format
    :rtype: Pandas ``dataFrame``
    """

    df = pd.read_csv('./data/LEDA.csv', delimiter=',')
    df = df.rename(columns=lambda x: x.strip())
    return df.set_index('PGC')
##########################################################################

def mySoup(url):
    """Loading an online webpage using the Beautiful soup
    client, in order to parse the html contents 
    
    Args:
        url (str): url of the online page

    Returns:
        <class 'bs4.BeautifulSoup'>: the parsed html text of the requested url

    >>> mySoup("https://edd.ifa.hawaii.edu")
    
    """
    
    uClient = uReq(url)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, "html.parser")    
    
    return page_soup

##########################################################################
def getPGCid(objname):
    """extracting the PGC ID of a galaxy given its name

    Args:
        objname  (str): galaxy name

    Returns:
        int: PGC ID
    """

    
    my_url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-objsearch?objname='+objname+'&extend=no&of=xml_names&extend=no&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES'
    tag = mySoup(my_url).findAll('a', href="/cgi-bin/catdef?prefix=PGC")[0].parent
    pgc = int(str(tag).split("</a>")[1].split('</td>')[0].strip())
   
    return pgc

##########################################################################
def addGalaxyInfo(Leda, pgc, response=None, objname=None):
    """Exctracting the information of a galaxy from the LEDA catalog 
    given the corresponsing PGC ID

    :param Leda: Leda catalog
    :type Leda: ``Pandas dataFrame``
    :param pgc: pgc ID
    :type pgc: ``int``
    :param response: response to be modified, defaults to None
    :type response: python ``dictionary``, optional
    :param objname: galaxy name, defaults to None
    :type objname: ``str``, optional
    :return: response dictionary that holds the requested galaxy information
    :rtype: python dictionary
    """

    if response is None:
        response = {}

    myGalaxy = Leda.loc[int(pgc)]
    response["pgc"]     = pgc
    response["ra"]      = "%.4f"%((myGalaxy.al2000)*15)         # deg
    response["dec"]     = "%.4f"%(myGalaxy.de2000)              # deg
    response["fov"]     = "%.2f"%(1.5*0.1*10**myGalaxy.logd25)  # arcmin
    response["pa"]      = str(myGalaxy.pa)

    if objname is None:
        response["objname"] = str(myGalaxy.objname)
    else:
        response["objname"] = objname

    return response


##########################################################################

@app.route('/getObj', methods=['POST'])
def getObj():
    """``API`` getObj rule  
    Object name is provided in ``json`` format and it returns the information about 
    the galaxy in ``jason``

    :return: galaxy information
    
    * ``ra, dec``: galaxy coordinates
    * ``fov``: field of view
    * ``pa``: position angle
    * ``objname``: galaxy name
    * ``status``: success/fail depending on the status of the query

    :rtype: ``json``

    **Example:**

    ::  

        $ curl -X POST <inclinet_url>/getObj -d '{"objname":"M31"}' -H 'Content-Type: application/json'
        {
        "dec": "41.2689", 
        "fov": "266.74", 
        "objname": "M31", 
        "pa": "35.0", 
        "pgc": 2557, 
        "ra": "10.6848", 
        "status": "success"
        } 
    """
    response = {"status": "success"}   # default
    if request.method == "POST":

        try:
            params = request.get_json()
            objname = params['objname']
            pgc = getPGCid(objname)
            response = addGalaxyInfo(Leda, pgc, response=response, objname=objname)
        except:
            response = {"status": "error"}

    return jsonify(response)


##########################################################################
@app.route('/getPGC', methods=['POST'])
def getPGC():
    """``API`` getPGC rule  
    The PGC number of galaxy is provided in ``json`` format and it returns the information about 
    the galaxy in ``json``

    :return: galaxy information

    * ``ra, dec``: galaxy coordinates
    * ``fov``: field of view
    * ``pa``: position angle
    * ``objname``: galaxy name
    * ``status``: success/fail depending on the status of the query

    :rtype: ``json``

    **Example:**

    ::  

        $ curl -X POST <inclinet_url>/getPGC -d '{"pgc":"2557"}' -H 'Content-Type: application/json'
        {
        "dec": "41.2689", 
        "fov": "266.74", 
        "objname": "NGC0224", 
        "pa": "35.0", 
        "pgc": "2557", 
        "ra": "10.6848", 
        "status": "success"
        } 
    """
    response = {"status": "success"}   # default
    if request.method == "POST":

        try:
            params = request.get_json()
            response = addGalaxyInfo(Leda, params['pgc'], response=response)
        except:
            response = {"status": "error"}

    return jsonify(response)

##########################################################################


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """An ``API`` rule that accepts galaxy images
    and returns a summary of all results

    :return: evaluated inclination(s) and other evaluations/predictions 
    :rtype: ``json``
    """

    response = {"status": "success"}   # default

    if request.method == "POST":
        params = request.get_json()

        key = "fileName"
        if key in params:
            params[key] = './' + params[key]

        response['output'],  response['scaledImage']= model2html(params, myModels)
    else:
        response['status'] = 'error'
    

    return jsonify(response)

##########################################################################
def allowedFile(fileName, suffixes=['PNG', 'JPG', 'JPEG', 'GIF']):
    """Determining is the provided input file is acceptable

    :param fileName: name of the input file
    :type fileName: ``str``
    :param suffixes: valid file extensions, defaults to ``['PNG', 'JPG', 'JPEG', 'GIF']``
    :type suffixes: ``list``, optional
    :return: True if acceptable, False if invalid
    :rtype: ``boolean``
    """

    try:
        suffix = fileName.split('.')[-1]
        if suffix.upper() in suffixes:
            return True
    except:
        return False
    
    return False

##########################################################################
def expand2square(pil_img, background_color):
    """transforms images to square shapes
    If a rectangular image is provided, the output image is square in size.
    **Note:** image is NOT stretched, just the smaller dimension is padded

    :param pil_img: input image
    :type pil_img: ``PIL`` image object
    :param background_color: background color of the augmented part
    :type background_color: tuple ``(R, G, B)``
    :return: an square image
    :rtype: ``PIL`` image object
    """

    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
##########################################################################
def addUnits(params):
    """adding units to the parameters of the provided dictionary

    :param params: a dictionary that holds galaxy information
    :type params: ``dict``
    :return: revised dictionary
    :rtype: ``dict``
    """

    outDict = deepcopy(params)
    outDict["ra"]      += " deg"
    outDict["dec"]     += " deg"
    outDict["fov"]     += " arcmin"
    outDict["pa"]      += " deg"

    return outDict

##########################################################################

def modelParams(params):
    """an auxillary function for additional book keepings
    The conventional extracted galaxy information from the LEDA catalog need to be
    revised to be compatible with the API of the SDSS image server

    :param params: input parameters
    :type params: ``dict``
    :return: augmented/adjusted information
    :rtype: ``dict``
    """


    mParams = {}
    d=60.
    pix = 0.25
    npix = int(float(params["fov"])*d / pix)   ## fov*d in arcsec

    while (npix>2048):
      pix = pix*2
      npix = int(float(params["fov"])*d / pix)

    if (npix<64): 
        npix=64

    mParams['alfa']  = params["ra"]
    mParams['delta'] = params["dec"]
    mParams['npix']  = npix
    mParams['scale'] = 1
    mParams['angle'] = float(params["pa"]) + 90
    mParams['pix']   = pix

    return mParams

##########################################################################

@app.route('/api/pgc/<pgcID>')
def pgc_api(pgcID):
    """the URL ``API`` that evaluates inclinations by providing the ``PGC ID`` in the URL

    :param pgcID: PGC ID
    :type pgcID:  ``int``
    :return: the summary of all evaluated inclinations
    :rtype: ``json``

    The following example shows that the output ``JSON`` contains the detailed evaluation of the available models.
    The ``summary`` fields hold the statistical summary of the outputs of all models.

    ::  

        $ curl <inclinet_url>/api/pgc/2557'   
        {
        "status": "success",
        "galaxy": {
            "pgc": "2557",
            "ra": "10.6848 deg",
            "dec": "41.2689 deg",
            "fov": "266.74 arcmin",
            "pa": "35.0 deg",
            "objname": "NGC0224"
        },
        "inclinations": {
            "Group_0": {
            "model4": 69.0,
            "model41": 72.0,
            "model42": 76.0,
            "model43": 71.0
            },
            "Group_1": {
            "model5": 73.0,
            "model51": 73.0,
            "model52": 74.0,
            "model53": 74.0
            },
            "Group_2": {
            "model6": 73.0,
            "model61": 76.0,
            "model62": 76.0,
            "model63": 67.0
            },
            "summary": {
            "mean": 72.83333333333333,
            "median": 73.0,
            "stdev": 2.6718699236468995
            }
        },
        "rejection_likelihood": {
            "model4-binary": 50.396937131881714,
            "model5-binary": 20.49814760684967,
            "model6-binary": 65.37048816680908,
            "summary": {
            "mean": 45.42185763518015,
            "median": 50.396937131881714,
            "stdev": 18.65378065042258
            }
        }
        }


    """

    response = {}
    response["status"] = "failed"

    try:
        params = addGalaxyInfo(Leda, pgcID)
        response['galaxy'] = addUnits(params)
    except:
        response["message"] = "Could not find PGC"+pgcID+' in the database. '
        return jsonify(response)
    
    ## evaluating inclinations
    try:
        response['inclinations'], response['rejection_likelihood'] = model(modelParams(params), myModels)
        response["status"] = "success"
    except:
        if not 'message' in response: response['message']=""
        response['message'] += 'Could not evaluate inclination! ' 

    return json.dumps(response, cls=NpEncoder, indent=2)+"\n"

##########################################################################

@app.route('/api/objname/<objname>')
def obj_api(objname):
    """the URL ``API`` that evaluates inclinations by providing the ``PGC ID`` in the URL

    :param objname: galaxy name
    :type objname:  ``string``
    :return: the summary of all evaluated inclinations
    :rtype: ``json``

    The following example shows the output ``JSON`` that contains the detailed evaluation of the available models.
    The ``summary`` fields hold the statistical summary of the outputs of all models.

    ::  

        $ curl  <inclinet_url>/api/objname/M33
        {
        "status": "success",
        "galaxy": {
            "pgc": 5818,
            "ra": "23.4621 deg",
            "dec": "30.6599 deg",
            "fov": "92.49 arcmin",
            "pa": "22.5 deg",
            "objname": "M33"
        },
        "inclinations": {
            "Group_0": {
            "model4": 54.0,
            "model41": 58.0,
            "model42": 54.0,
            "model43": 52.0
            },
            "Group_1": {
            "model5": 54.0,
            "model51": 55.0,
            "model52": 52.0,
            "model53": 55.0
            },
            "Group_2": {
            "model6": 56.0,
            "model61": 57.0,
            "model62": 55.0,
            "model63": 53.0
            },
            "summary": {
            "mean": 54.583333333333336,
            "median": 54.5,
            "stdev": 1.753963764987432
            }
        },
        "rejection_likelihood": {
            "model4-binary": 41.287994384765625,
            "model5-binary": 4.068142548203468,
            "model6-binary": 55.70455193519592,
            "summary": {
            "mean": 33.686896289388336,
            "median": 41.287994384765625,
            "stdev": 21.754880108256657
            }
        }
        }

    """

    response = {}
    response["status"] = "failed"

    try:
        pgcID = getPGCid(objname)
        params = addGalaxyInfo(Leda, pgcID, objname=objname)
        response['galaxy'] = addUnits(params)
    except:
        response["message"] = "Could not find oject "+objname+'.'
        return jsonify(response)
    
    ## evaluating inclinations
    try:
        response['inclinations'], response['rejection_likelihood'] = model(modelParams(params), myModels)
        response["status"] = "success"
    except:
        if not 'message' in response: response['message']=""
        response['message'] += 'Could not evaluate inclination! '  

    return json.dumps(response, cls=NpEncoder, indent=2)+"\n"


##########################################################################
@app.route('/api/file', methods=["POST"])
def file_api():
    """``API`` function, evaluating the inclination by providing the galaxy image

    :return: summary of the results, inclinations, statistics
    :rtype: ``json``

    ::

        $ curl -F 'file=@/path/to/image/NGC_4579.jpg' <inclinet_url>/api/file
        {
        "status": "success",
        "filename": "NGC_4579.jpg",
        "inclinations": {
            "Group_0": {
            "model4": 47.0,
            "model41": 51.0,
            "model42": 50.0,
            "model43": 47.0
            },
            "Group_1": {
            "model5": 49.0,
            "model51": 49.0,
            "model52": 51.0,
            "model53": 52.0
            },
            "Group_2": {
            "model6": 50.0,
            "model61": 49.0,
            "model62": 49.0,
            "model63": 48.0
            },
            "summary": {
            "mean": 49.333333333333336,
            "median": 49.0,
            "stdev": 1.49071198499986
            }
        },
        "rejection_likelihood": {
            "model4-binary": 84.28281545639038,
            "model5-binary": 94.24970746040344,
            "model6-binary": 88.11054229736328,
            "summary": {
            "mean": 88.88102173805237,
            "median": 88.11054229736328,
            "stdev": 4.105278145778375
            }
        }
        }


    """

    response = {}
    response["status"] = "failed"

    if request.method == 'POST':

        thisFile = request.files["file"]
        fileName = thisFile.filename
        response["filename"] = fileName
        if allowedFile(fileName):
            try:
                imPath = "./static/tempImages/uploads/" + str(time.time()).replace('.','_') + '.' + fileName.split('.')[-1]
                thisFile.save(imPath)
                imSquarify(imPath)
                response['inclinations'], response['rejection_likelihood'] = model({'fileName': imPath}, myModels)
            except:
                response["message"] = "Could not handle the image. Model(s) failed to make prediction(s)."
                return jsonify(response)
        else:
            response["message"] = "[Note] Please upload jpg, png, jpeg, and gif giles no larger than 1 MB !"
            return jsonify(response)

    response["status"] = "success"

    return json.dumps(response, cls=NpEncoder, indent=2)+"\n"

##########################################################################

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

##########################################################################
def imSquarify(imPath, outPath=None):
    """padding an image to square shape

    :param imPath: image path
    :type imPath: ``str``
    :param outPath: output path, defaults to None
    :type outPath: ``str``, optional
    :return: output path, if the``outPath`` is not given the output is stored in the same locations
    as that of the input image
    :rtype: ``str``
    """

    if outPath is None:
        outPath = imPath

    im = Image.open(imPath)
    width, height = im.size
    size = max([width, height])
    im_thumb = expand2square(im, (0, 0, 0)).resize((size, size), Image.LANCZOS)
    im_thumb.save(outPath, quality=100)

    return outPath

##########################################################################
@app.route('/upload', methods=['POST'])
def IM_upload():
    """An image uploader ``API`` for the use in the online GUI
    This API is called from the online application, through ``AJAX`` calls.
    The uploaded image is stored on the server for further analysis.

    :return: the status of the process, and the public address of the images path for the online GUI
    :rtype: ``json``
    """

    response = {}
      
    if request.method == 'POST':
        thisFile = request.files["fileToUpload"]
        fileName = thisFile.filename
        
        htmlPath = "/static/tempImages/uploads/"  + str(time.time()).replace('.','_') + '.' + fileName.split('.')[-1]
        imPath = './' + htmlPath
 
        #size = os.stat(imPath).st_size
        if allowedFile(fileName):
            response["status"] = 'success'
            response["htmlPath"] = htmlPath
            thisFile.save(imPath)
            imSquarify(imPath)
        else:
            response["status"] = 'error'

    return jsonify(response)

##########################################################################
def arg_parser():

    """
    Parsing the command line arguments

    :return: parsed arguments ``(opts, args)``

    opts: the 

    :rtype: ``tuple``
    """    

    parser = OptionParser(usage="""\
\n
 - starting up the service on the desired host:port
 
 - How to run: 
 
    $ python server.py -t <host IP> -p <port_number> -d <debugging_mode>

- To get help
    $ python server.py -h
 
""")

    parser.add_option('-p', '--port',
                      type='int', action='store', default=3030,
                      help="the port number to run the service on")

    parser.add_option('-t', '--host',
                      type='string', action='store',
                      help="service host", default="0.0.0.0")                     
    
    parser.add_option("-d", "--debug", action="store_true",
                      help="debugging mode", default=False)

    (opts, args) = parser.parse_args()

    return opts
##########################################################################

@app.route('/api/docs')
def get_docs():
    print('sending docs')
    return render_template('swaggerui.html')

##########################################################################
@app.route('/')
def home():
    """mounting the server home

    :return: the rendered template ``index.html``
    """

    webDict = {"webroot": WEBROOT}

    return render_template('index.html', webDict=webDict)

##########################################################################
### GLOBAL Variables ###
# WEBROOT is an environmental variable
WEBROOT = os.getenv('WEBROOT')  
if WEBROOT is None:
    WEBROOT = './'

myModels = None
Leda = None

##########################################################################

#### autopep8 -i server.py
# If we're running in stand-alone mode, run the application
if __name__ == '__main__':

    opts = arg_parser()

    print("\n------------------------------------")
    print(" To get HELP enter:")
    print(" python server.py -h")
    print("------------------------------------")
    print("\n------------------------------------")
    print(" Input Arguments (provided by User)")
    print("------------------------------------")
    print(" ", opts)
    print("------------------------------------")

    ## loading models, Leda catalog
    myModels = loadModels()
    Leda = loadLeda()

    ## generating the necessary folders
    createDir('./static')
    xcmd("cp -rf swagger/static/* ./static/.")
    xcmd("cp -rf swagger/templates/* ./templates/.")
    xcmd("cp -rf docs/build/html ./static/.")
    createDir('./static/tempImages')
    createDir('./static/tempImages/uploads')
    createDir('./static/tempImages/resized')

    ## starting up the service
    app.run(host=opts.host, port=opts.port, debug=opts.debug)

##########################################################################
##########################################################################
##########################################################################
