# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
# TODO: add execption handling
from __future__ import print_function
from termcolor import colored
import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image as imgop
import imageio
from glob import glob
import json
import random
from tqdm import tqdm
from skimage.draw import polygon
import cv2
from PIL import Image as imgop
import random
import shutil
import tensorflow as tf
import itertools
import math

#--------------------------helper functions----------------------------------------
def readJson(file_name):
    '''
        JSON file read
    '''
    return json.load(open(file_name))

def LOG_INFO(log_text,p_color='green',rep=True):
    '''
        Terminal LOG
    '''
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')

def create_dir(base_dir,ext_name):
    '''
        creates a new dir with ext_name in base_dir and returns the path
    '''
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def nCr(n,r):
    '''
        returns the number of combinations taking r objects from n objects
    '''
    f = math.factorial
    return f(n) // (f(r)*f(n-r))

#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    """
    This class is used to create the complete dataset
    """
    def __init__(self,FLAGS):
        # source dirs
        self.img_dir    =   os.path.join(FLAGS.SRC_DIR,'images')
        self.ant_path   =   os.path.join(FLAGS.SRC_DIR,'annotations','via_region_data.json')
        
        # create needed paths:
        self.mask_dir   =   create_dir(FLAGS.SRC_DIR,'masks')
        self.ds_dir     =   create_dir(FLAGS.DS_DIR,'DataSet')
        # train
        self.train_dir  =   create_dir(self.ds_dir,'Train')
        self.train_img  =   create_dir(self.train_dir,'images')
        self.train_mask =   create_dir(self.train_dir,'masks')
        # eval
        self.eval_dir   =   create_dir(self.ds_dir,'Eval')
        self.eval_img  =   create_dir(self.eval_dir,'images')
        self.eval_mask =   create_dir(self.eval_dir,'masks')
        # test
        self.test_dir   =   create_dir(self.ds_dir,'Test')
        self.test_img   =   create_dir(self.test_dir,'images')
        self.test_mask  =   create_dir(self.test_dir,'masks')
        # image params
        self.img_dim    =   FLAGS.IMAGE_DIM
        self.nb_channels=   FLAGS.NB_CHANNELS
        # Train,Eval,Test split
        self.__nb_train =   FLAGS.NB_TRAIN
        self.__nb_eval  =   FLAGS.NB_EVAL
        # allowed rotations for augmentation
        self.rot_angles =   [angle for angle in range(FLAGS.ROT_START,
                                                      FLAGS.ROT_STOP+FLAGS.ROT_STEP,
                                                      FLAGS.ROT_STEP)]
        # flips and saver identifier
        self.fid        = FLAGS.FID
        self.rs_start   = FLAGS.ID_END
        # source IDS for Train,Test,Eval
        self.ids        = [i for i in range(FLAGS.ID_START,FLAGS.ID_END+1)]
        if FLAGS.REPLICATE==0:
            random.shuffle(self.ids)
            
            self.__train_data = self.ids[:self.__nb_train]
            self.__eval_data  = self.ids[self.__nb_train:self.__nb_train+self.__nb_eval]
            self.__test_data  = self.ids[self.__nb_train+self.__nb_eval:]
        else:
            self.__eval_data  = [13,28,32,43,45,62,68,94,95,97]
            self.__test_data  = [1,9,15,40,47,50,52,57,75,76,77]
            self.__train_data = [i  for i in self.ids if i not in self.__eval_data and i not in self.__test_data]

        self.nb_train     = FLAGS.TRAIN_COUNT
        self.nb_eval      = FLAGS.EVAL_COUNT
        

    def __modifyMask(self,y):
        '''
            creates a binary mask from 3 channel mask
        '''
        # binary holder
        by=np.zeros(y.shape[:2])
        # grey scale
        gy=np.dot(y[...,:3], [0.299, 0.587, 0.114])
        # thresholding
        by[gy>0]=255
        # uint conversion
        by=by.astype('uint8')
        return by

    def __createBase(self,idens,img_save,mask_save,mode):
        '''
            Resizes the original Source images and masks and saves them according to mode
        '''
        LOG_INFO('Base Data:{}'.format(mode))

        for iden in tqdm(idens):    
            # paths
            img_path  =   os.path.join(self.img_dir,'{}.jpg'.format(iden))
            gt_path   =   os.path.join(self.mask_dir,'{}.png'.format(iden))
            # read
            x=imgop.open(img_path)
            y=imgop.open(gt_path)
            # process
            x=np.array(x.resize((self.img_dim,self.img_dim)))
            y=np.array(y.resize((self.img_dim,self.img_dim)))
            y=self.__modifyMask(y)
            # save paths
            __img_path=   os.path.join(img_save,'{}.png'.format(iden))
            __gt_path =   os.path.join(mask_save,'{}.png'.format(iden))  
            # save
            imageio.imsave(__img_path,x)
            imageio.imsave(__gt_path,y)


    def baseData(self):
        '''
            A routine wrapper for createBase
        '''
        # create test train and eval data
        self.__createBase(self.__train_data,self.train_img,self.train_mask,'train')
        self.__createBase(self.__eval_data,self.eval_img,self.eval_mask,'eval')
        self.__createBase(self.__test_data,self.test_img,self.test_mask,'test')

    def __getFlipDataById(self,img,gt,fid):
        '''
            Returns flipped image and mask numpy data based on fid
        '''
        if fid==0:# ORIGINAL
            x=np.array(img)
            y=np.array(gt)
        elif fid==1:# Left Right Flip
            x=np.array(img.transpose(imgop.FLIP_LEFT_RIGHT))
            y=np.array(gt.transpose(imgop.FLIP_LEFT_RIGHT))
        elif fid==2:# Up Down Flip
            x=np.array(img.transpose(imgop.FLIP_TOP_BOTTOM))
            y=np.array(gt.transpose(imgop.FLIP_TOP_BOTTOM))
        else: # Mirror Flip
            x=img.transpose(imgop.FLIP_TOP_BOTTOM)
            x=np.array(x.transpose(imgop.FLIP_LEFT_RIGHT))
            y=gt.transpose(imgop.FLIP_TOP_BOTTOM)
            y=np.array(y.transpose(imgop.FLIP_LEFT_RIGHT))
        return x,y

    

    def __saveTransposedData(self,rot_img,rot_gt,_img_path,_mask_path):
        '''
            saves the rotated and flipped data
        '''
        _fid = random.randint(0,self.fid) 
        self.rs_start+=1
        x,y=self.__getFlipDataById(rot_img,rot_gt,_fid)
        file_name='{}.png'.format(self.rs_start)
        imageio.imsave(os.path.join(_img_path,file_name) ,x)
        imageio.imsave(os.path.join(_mask_path,file_name) ,y)
    
    def __createDatafromComb(self,comb,img_paths,_dpath):
        '''
            image collage from 4 unique images that works as a completely new image
        '''
        x0=np.array(imgop.open(img_paths[comb[0]]))
        x1=np.array(imgop.open(img_paths[comb[1]]))
        x2=np.array(imgop.open(img_paths[comb[2]]))
        x3=np.array(imgop.open(img_paths[comb[3]]))
        
        x=np.concatenate((np.concatenate((x0,x1),axis=1),np.concatenate((x2,x3),axis=1)),axis=0)
        x=imgop.fromarray(x)
        x=x.resize((self.img_dim,self.img_dim))
        
        y0=np.array(imgop.open(str(img_paths[comb[0]]).replace('images','masks')))
        y1=np.array(imgop.open(str(img_paths[comb[1]]).replace('images','masks')))
        y2=np.array(imgop.open(str(img_paths[comb[2]]).replace('images','masks')))
        y3=np.array(imgop.open(str(img_paths[comb[3]]).replace('images','masks')))
        
        y=np.concatenate((np.concatenate((y0,y1),axis=1),np.concatenate((y2,y3),axis=1)),axis=0)
        y=imgop.fromarray(y)
        y=y.resize((self.img_dim,self.img_dim))
        
        rot_angle=random.choice(self.rot_angles)
        
        x  =   x.rotate(rot_angle)
        y  =   y.rotate(rot_angle)
        
        self.__saveTransposedData(x,y,os.path.join(_dpath,'images'),os.path.join(_dpath,'masks'))   

   

    def create(self,mode):
        if mode=='train':
            _dpath=self.train_dir
        elif mode=='eval':
            _dpath=self.eval_dir
        
        LOG_INFO('Creating DataSet:{}'.format(mode))
        
        LOG_INFO('This will take quite some time. Thank you for your patience.')
            
        img_paths=[_path for _path in glob(os.path.join(_dpath,'images','*.*'))]
        
        random.shuffle(img_paths)

        vals=[i for i in range(len(img_paths))]
        
        if mode=='eval':

            for comb in tqdm(itertools.combinations(vals,4),total=nCr(len(vals),4)):
                self.__createDatafromComb(comb,img_paths,_dpath)
            # eval data not sufficient enough
            needed_data=self.nb_eval- nCr(len(vals),4)- self.__nb_eval
            _paths=[_path for _path in glob(os.path.join(_dpath,'images','*.*'))]
            count=0
            for _path in tqdm(_paths):
                x=imgop.open(_path)
                y=imgop.open(str(_path).replace('images','masks'))
                for rot_angle in self.rot_angles:
                    x  =   x.rotate(rot_angle)
                    y  =   y.rotate(rot_angle)
                    self.__saveTransposedData(x,y,os.path.join(_dpath,'images'),os.path.join(_dpath,'masks'))   
                    count+=1
                    if count==needed_data:
                        LOG_INFO('Generated Necessary Data !')
                        break
                if count==needed_data:
                        break
                    
        elif mode=='train':
            needed_data=self.nb_train-self.__nb_train
            count=0
            for comb in tqdm(itertools.combinations(vals,4),total=needed_data):
                self.__createDatafromComb(comb,img_paths,_dpath)
                count+=1
                if count==needed_data:
                    LOG_INFO('Generated Necessary Data !')
                    break

            
             
    
            

    
    def __draw_contour(self,img_shape,x,y,channel):
        
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        x = np.array(x)
        x = [min(i, img_shape[1]) for i in x]
        
        y = np.array(y)
        y = [min(i, img_shape[0]) for i in y]
        
        width, height = max(x)-min(x), max(y)-min(y)
        
        sq = width*height
        
        mask_x, mask_y = polygon(x, y)
        mask[mask_y, mask_x, channel] = 255
        
        return mask

    def createMasks(self):
        meta=readJson(self.ant_path)
        masks = []
        filenames = []
        missed = []
        LOG_INFO('Creating Source Masks!')
        for iden in tqdm(meta):
            try:
                __filename = iden.split('.jpg')[0] + '.jpg'
                img = cv2.imread(os.path.join(self.img_dir,__filename))
                # get image shape
                img_height, img_weight = img.shape[:2]
                # init a mask
                mask = np.zeros((img_height, img_weight, 3), dtype=np.uint8)
                # iter through marked contours in file
                file_contours = meta[iden]['regions']
                for _,cnt_meta  in file_contours.items():
                    cnt_x   = cnt_meta['shape_attributes']['all_points_x']
                    cnt_y   = cnt_meta['shape_attributes']['all_points_y']
                    # start with a mask
                    channel = 0
                    mask += self.__draw_contour(mask.shape, cnt_x, cnt_y, channel)
                    
                if mask.sum() > 0:
                    masks.append(mask)
                    filenames.append(__filename)
                    imageio.imsave(os.path.join(self.mask_dir,'{}.png'.format(__filename.replace(".jpg",""))),mask)
                else:
                    missed.append(__filename)
                    
            except Exception as e:
                missed.append(__filename)
                LOG_INFO('ERROR:{}'.format(__filename),p_color='red')
#--------------------------------------------------------------------------------------------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,save_dir,mode,r_num):

    tfrecord_name='{}_{}.tfrecord'.format(mode,r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in image_paths:
            target_path=str(image_path).replace('images','masks')
            with(open(image_path,'rb')) as fid:
                image_png_bytes=fid.read()
            with(open(target_path,'rb')) as fid:
                target_png_bytes=fid.read()
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes)
            }
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)   
#--------------------------------------------------------------------------------------------------------------------------------------------------

