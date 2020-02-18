#!/usr/bin/env python3
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
import random
import shutil
from tqdm import tqdm
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
def genH5(mode,FLAGS):
    # h5 dir
    h5_dir=create_dir(os.path.join(FLAGS.DS_DIR,'DataSet'),'H5')
    mode_dir=create_dir(h5_dir,mode)
    LOG_INFO("Creating H5s:{}".format(mode_dir))
    data_dir=os.path.join(FLAGS.DS_DIR,'DataSet',mode)
    __paths=[os.path.join(data_dir,_file) for _file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,_file))]
    random.shuffle(__paths)
    for i in range(0,len(__paths),FLAGS.DATA_COUNT):
        image_paths= __paths[i:i+FLAGS.DATA_COUNT]
        random.shuffle(image_paths)        
        r_num=i // FLAGS.DATA_COUNT
        if len(image_paths)==FLAGS.DATA_COUNT:
            createH5Data(image_paths,r_num,mode_dir,mode)
        else:
            LOG_INFO('Testing Data Addition:{}'.format(mode))
            random.shuffle(image_paths)         
            for __path in tqdm(image_paths):
                base_name=os.path.basename(__path)
                dest_path=os.path.join(FLAGS.DS_DIR,'DataSet','Test',base_name)
                shutil.copy(__path,dest_path)
                

def main(FLAGS):
    st=time.time()
    DS=DataSet(FLAGS)
    DS.createMasks()
    DS.create('eval')
    DS.create('test')
    DS.create('train')
    genH5('Eval',FLAGS)
    genH5('Train',FLAGS)
    LOG_INFO('Time Taken:{}s'.format(round(time.time()-st)))


if __name__=='__main__':
    main(FLAGS)