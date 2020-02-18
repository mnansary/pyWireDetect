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
import h5py
from coreLib.utils import readJson,LOG_INFO,readh5,saveh5,DataSet,createH5Data,create_dir
import tensorflow as tf
tf.enable_eager_execution()
import imageio
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

#--------------------------------------------------------------------------
BATCH_SIZE=128
NB_CHANNEL=3
EPOCH=2
IMG_DIM=256
BUFFER_SIZE=1024
h5_dir=os.path.join(FLAGS.DS_DIR,'DataSet','H5','Eval')

class generator:
    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            for data in hf["data"]:
                data=data.astype('float32')/255.0
                img=data[:,:IMG_DIM]
                tgt=data[:,IMG_DIM:]
                yield img,tgt

def data_input_fn(h5_dir):
    flist=[os.path.join(h5_dir,_h5) for _h5 in os.listdir(h5_dir) if os.path.isfile(os.path.join(h5_dir,_h5))]
    dataset = tf.data.Dataset.from_tensor_slices(flist)
    dataset = dataset.repeat(EPOCH)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.interleave(lambda filename: tf.data.Dataset.from_generator(
                                generator(), 
                                (tf.float32,tf.float32), 
                                (tf.TensorShape([IMG_DIM,IMG_DIM,NB_CHANNEL]), 
                                tf.TensorShape([IMG_DIM,IMG_DIM,NB_CHANNEL])),
                                args=(filename,)))
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

ds=data_input_fn(h5_dir)
batch_num=0
epch=0
tryout_dir=create_dir(FLAGS.DS_DIR,'TRYOUT')
epdir=create_dir(tryout_dir,str(epch))
for imgs,tgts in tqdm(ds):
    batch_num+=1
    if batch_num==24:
        epch+=1
        batch_num=0
        epdir=create_dir(tryout_dir,str(epch))
    for x,y,n in zip(imgs,tgts,range(imgs.shape[0])):
        dat=np.concatenate((x,y),axis=1)
        dat=dat*255
        dat=dat.astype('uint8')
        dpath=os.path.join(epdir,'{}_{}.png'.format(batch_num,n))
        imageio.imsave(dpath,dat)

