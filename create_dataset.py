#!/usr/bin/env python3
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
from glob import glob
from coreLib.utils import readJson,LOG_INFO,readh5,saveh5,DataSet
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
def main(FLAGS):
    st=time.time()
    DS=DataSet(FLAGS)
    DS.createMasks()
    DS.create('eval')
    DS.create('test')
    DS.create('train')
    LOG_INFO('Time Taken:{}s'.format(time.time()-st))
if __name__=='__main__':
    main(FLAGS)