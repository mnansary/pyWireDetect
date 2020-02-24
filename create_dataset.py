#!/usr/bin/env python3
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
import random
import shutil
from tqdm import tqdm
from coreLib.utils import readJson,LOG_INFO,DataSet,create_dir,to_tfrecord
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
    TRAIN_COUNT     = config_data['TRAIN_COUNT']
    EVAL_COUNT      = config_data['EVAL_COUNT']
    REPLICATE       = config_data['REPLICATE']
    
#--------------------------------------------------------------------------

def genTFRecords(mode,FLAGS):
    rec_dir=create_dir(FLAGS.DS_DIR,'TFRecords')
    mode_dir=create_dir(rec_dir,mode)
    LOG_INFO("Creating TFRecords:{}".format(mode_dir))
    data_dir=os.path.join(FLAGS.DS_DIR,'DataSet',mode,'images')
    __paths=[os.path.join(data_dir,_file) for _file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,_file))]
    random.shuffle(__paths)
    for i in tqdm(range(0,len(__paths),FLAGS.DATA_COUNT)):
        image_paths= __paths[i:i+FLAGS.DATA_COUNT]
        random.shuffle(image_paths)        
        r_num=i // FLAGS.DATA_COUNT
        to_tfrecord(image_paths,mode_dir,mode,r_num)
              
def main(FLAGS):
    st=time.time()
    
    DS=DataSet(FLAGS)
    DS.createMasks()
    DS.baseData()
    DS.create('eval')
    DS.create('train')
    genTFRecords('Train',FLAGS)
    genTFRecords('Eval',FLAGS)
    
    LOG_INFO('Time Taken:{}s'.format(round(time.time()-st)))


if __name__=='__main__':
    main(FLAGS)