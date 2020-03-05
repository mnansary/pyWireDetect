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

def show_data(data):
    plt.imshow(np.array(data))
    plt.show()

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
        # cropped data
        self.temp_dir   =   create_dir(FLAGS.DS_DIR,'temp')
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
        self.dim_factor =   FLAGS.DIM_FACTOR
        # Train,Eval,Test split
        self.__nb_train =   FLAGS.NB_TRAIN
        self.__nb_eval  =   FLAGS.NB_EVAL
        
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
        y=np.array(y)
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
        ORIG_DIM=256
        LOG_INFO('Base Data:{}'.format(mode))

        for iden in tqdm(idens):    
            # paths
            img_path  =   os.path.join(self.img_dir,'{}.jpg'.format(iden))
            gt_path   =   os.path.join(self.mask_dir,'{}.png'.format(iden))
            # read
            x=imgop.open(img_path)
            y=imgop.open(gt_path)
            # process
            x=x.resize((ORIG_DIM,ORIG_DIM))
            y=y.resize((ORIG_DIM,ORIG_DIM))
            y=imgop.fromarray(self.__modifyMask(y))
            
            if mode=='test':
                self.__saveTransposedData(x,y,img_save,mask_save,flag=False)
            else:
                #RE=np.zeros(np.array(x).shape)
                #show_data(x)
                #show_data(y)
                for pxv in range(0,ORIG_DIM,self.img_dim):
                    for pxl in range(0,ORIG_DIM,self.img_dim):
                        self.rs_start+=1
                        l=pxv
                        r=pxv+self.img_dim
                        t=pxl
                        b=pxl+self.img_dim
                        bbox=(l,t,r,b)
                        _x=   x.crop(bbox)
                        _y=   y.crop(bbox)
                        self.__saveTransposedData(_x,_y,img_save,mask_save,flag=False)
                        #RE[t:b,l:r,:]=np.array(_x)[:,:,:]
                        #RE=RE.astype('uint8')
                        #show_data(RE)
    
    def __createCrop(self,mode_path):
        '''
            crops the train and eval data in a temp dir
        '''
        LOG_INFO('Temp Data for:{}'.format(mode_path))
        temp_mode=create_dir(self.temp_dir,os.path.basename(mode_path))
        img_save =create_dir(temp_mode,'images')
        mask_save=create_dir(temp_mode,'masks')
        for img_path in tqdm(glob(os.path.join(mode_path,'images','*.png'))):    
            # paths
            gt_path   =   str(img_path).replace('images','masks')
            # read
            x=imgop.open(img_path)
            y=imgop.open(gt_path)
            
            for pxv in range(0,self.img_dim,self.img_dim//2):
                for pxl in range(0,self.img_dim,self.img_dim//2):
                    self.rs_start+=1
                    l=pxv
                    r=pxv+self.img_dim//2
                    t=pxl
                    b=pxl+self.img_dim//2
                    bbox=(l,t,r,b)
                    _x=   x.crop(bbox)
                    _y=   y.crop(bbox)
                    self.__saveTransposedData(_x,_y,img_save,mask_save,flag=False)
                    
    def cropTempData(self,mode):
        if mode=='train':
            self.__createCrop(self.train_dir)
        elif mode=='eval':
            self.__createCrop(self.eval_dir)
                       
    
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
            Takes Numpy
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

    def __saveTransposedData(self,_img,_gt,_img_path,_mask_path,flag=True):
        '''
            Takes IMAGE OBJECT
        '''
        _fid = random.randint(0,4) 
        self.rs_start+=1
        if flag:
            _img,_gt=self.__getFlipDataById(_img,_gt,_fid)
        self.rs_start+=1
        file_name='{}.png'.format(self.rs_start)
        imageio.imsave(os.path.join(_img_path,file_name) ,np.array(_img))
        imageio.imsave(os.path.join(_mask_path,file_name) ,np.array(_gt))
    
    def __checkFeasible(self,y):
        y=np.array(y)
        tp=float(y.shape[0]*y.shape[1])
        nzp=np.count_nonzero(y==0)
        if (nzp*100.0)/tp < 30:
            return False
        else:
            return True


    def __createDatafromComb(self,comb,img_paths,_spath):
        '''
            image collage from 4 unique images that works as a completely new image
        '''
        x0=np.array(imgop.open(img_paths[comb[0]]))
        x1=np.array(imgop.open(img_paths[comb[1]]))
        x2=np.array(imgop.open(img_paths[comb[2]]))
        x3=np.array(imgop.open(img_paths[comb[3]]))
        
        x=np.concatenate((np.concatenate((x0,x1),axis=1),np.concatenate((x2,x3),axis=1)),axis=0)
        x=imgop.fromarray(x)
        

        y0=np.array(imgop.open(str(img_paths[comb[0]]).replace('images','masks')))
        y1=np.array(imgop.open(str(img_paths[comb[1]]).replace('images','masks')))
        y2=np.array(imgop.open(str(img_paths[comb[2]]).replace('images','masks')))
        y3=np.array(imgop.open(str(img_paths[comb[3]]).replace('images','masks')))
        
        y=np.concatenate((np.concatenate((y0,y1),axis=1),np.concatenate((y2,y3),axis=1)),axis=0)
        y=imgop.fromarray(y)
        
        if self.__checkFeasible(y):
            self.__saveTransposedData(x,y,os.path.join(_spath,'images'),os.path.join(_spath,'masks'))   
            return True
        else:
            return False
   
    def create(self,mode):
        if mode=='train':
            _dpath=os.path.join(self.temp_dir,'Train')
            _spath=self.train_dir
        elif mode=='eval':
            _dpath=os.path.join(self.temp_dir,'Eval')
            _spath=self.eval_dir

        LOG_INFO('Creating DataSet:{}'.format(mode))
        
        LOG_INFO('This will take quite some time. Thank you for your patience.')
            
        img_paths=[_path for _path in glob(os.path.join(_dpath,'images','*.*'))]
        
        random.shuffle(img_paths)

        vals=[i for i in range(len(img_paths))]
        
        if mode=='eval':
            needed_data=self.nb_eval- (self.__nb_eval*self.dim_factor)
            
        elif mode=='train':
            needed_data=self.nb_train-(self.__nb_train*self.dim_factor)
        
        counter=0
        for comb in tqdm(itertools.combinations(vals,4),total=nCr(len(vals),4)):
            if self.__createDatafromComb(comb,img_paths,_spath):
                counter+=1
            if counter==needed_data:
                LOG_INFO('Data Generation Successful')
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

