#!/usr/bin/env python3
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from coreLib.utils import readJson,LOG_INFO,readh5,saveh5,DataSet,createH5Data,create_dir
#-----------------------------------------------------Load Config-------
config_data=readJson('config.json')

class FLAGS:
    SRC_DIR         = config_data['SRC_DIR']
    DS_DIR          = config_data['DS_DIR']
    IMAGE_DIM       = config_data['IMAGE_DIM']
    NB_CHANNELS     = config_data['NB_CHANNELS']
    ROT_START       = config_data['ROT_START']
    ROT_STOP        = config_data['ROT_STOP']
    ROT_STEP        = config_data['ROT_STEP']
    FID             = config_data['FID']
    DATA_COUNT      = config_data['DATA_COUNT']
    NB_EVAL         = config_data['NB_EVAL']
    NB_TRAIN        = config_data['NB_TRAIN']
    ID_START        = config_data['ID_START']
    ID_END          = config_data['ID_END']
#--------------------------------------------------------------------------
def plot_data(img,gt) :
    plt.figure('test')
    plt.subplot(121)
    plt.imshow(img)
    plt.title(' image')
    plt.subplot(122)
    plt.title('ground truth')
    plt.imshow(gt)
    plt.show()
    plt.clf()
    plt.close()
#--------------------------------------------------------------------------
def h5_debug(h5_dir):
    data=readh5(h5_dir)
    for dat in data:
        img= dat[:,:FLAGS.IMAGE_DIM]
        gt = dat[:,FLAGS.IMAGE_DIM:]
        plot_data(img,gt)

h5_dir=os.path.join(FLAGS.DS_DIR,'DataSet','H5','Eval','Eval_0.h5')

