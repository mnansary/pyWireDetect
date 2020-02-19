#!/usr/bin/env python3
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
import random
import shutil
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from coreLib.utils import readJson,LOG_INFO,readh5,saveh5,DataSet,create_dir
import tensorflow as tf
#tf.enable_eager_execution()
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
EPOCH=3
IMG_DIM=256
BUFFER_SIZE=1024

tf_dir=os.path.join(FLAGS.DS_DIR,'TFRecords')

def data_input_fn(tf_dir,mode,img_dim=256): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'target' : tf.io.FixedLenFeature([],tf.string)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=3)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(img_dim,img_dim,3))
        
        target_raw=parsed_example['target']
        target=tf.image.decode_png(target_raw,channels=1)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(img_dim,img_dim,1))
        
        return image,target

    file_paths=glob(os.path.join(tf_dir,mode,'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat(EPOCH)
    dataset = dataset.batch(BATCH_SIZE)
    iterator= dataset.make_one_shot_iterator()
    return iterator




eval_iter=data_input_fn(tf_dir,"Eval")
data=eval_iter.get_next()
batch_num=0
epch=0
tryout_dir=create_dir(FLAGS.DS_DIR,'TRYOUT')
epdir=create_dir(tryout_dir,str(epch))

with tf.Session() as sess:
    for _ in tqdm(range(72)):
        imgs,tgts=sess.run(data)
        batch_num+=1
        if batch_num==25:
            epch+=1
            batch_num=0
            epdir=create_dir(tryout_dir,str(epch))
        for x,y,n in zip(imgs,tgts,range(imgs.shape[0])):
            x=x*255
            x=x.astype('uint8')
            y=y*255
            y=y.astype('uint8')
            
            xpath=os.path.join(epdir,'{}_{}.png'.format(batch_num,n))
            ypath=os.path.join(epdir,'{}_{}_mask.png'.format(batch_num,n))
            
            imageio.imsave(xpath,x)
            imageio.imsave(ypath,y)
            
        


