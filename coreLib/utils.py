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
import h5py
import json
import random
from tqdm import tqdm
from skimage.draw import polygon
import cv2
from PIL import Image as imgop
import random
import shutil
import tensorflow as tf
#---------------------------------------------------------------------------
def readJson(file_name):
    return json.load(open(file_name))

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',np.shape(data),h5py.h5t.STD_U8BE,data=data)
    hf.close()
    
def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def LOG_INFO(log_text,p_color='green',rep=True):
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')

def create_dir(base_dir,ext_name):
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir
#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    """
    This class is used to create the complete dataset
    """
    def __init__(self,FLAGS):
        self.img_dir    =   os.path.join(FLAGS.SRC_DIR,'images')
        self.ant_path   =   os.path.join(FLAGS.SRC_DIR,'annotations','via_region_data.json')
        self.mask_dir   =   create_dir(FLAGS.SRC_DIR,'masks')
        
        self.ds_dir     =   create_dir(FLAGS.DS_DIR,'DataSet')
        self.train_dir  =   create_dir(self.ds_dir,'Train')
        self.train_img  =   create_dir(self.train_dir,'images')
        self.train_mask =   create_dir(self.train_dir,'masks')
        
        self.eval_dir   =   create_dir(self.ds_dir,'Eval')
        self.eval_img  =   create_dir(self.eval_dir,'images')
        self.eval_mask =   create_dir(self.eval_dir,'masks')
        
        self.test_dir   =   create_dir(self.ds_dir,'Test')
        self.test_img   =   create_dir(self.test_dir,'images')
        self.test_mask  =   create_dir(self.test_dir,'masks')
        
        self.img_dim    =   FLAGS.IMAGE_DIM
        self.nb_channels=   FLAGS.NB_CHANNELS
        
        self.__nb_train =   FLAGS.NB_TRAIN
        self.__nb_eval  =   FLAGS.NB_EVAL
        
        self.rot_angles =   [angle for angle in range(FLAGS.ROT_START,
                                                      FLAGS.ROT_STOP+FLAGS.ROT_STEP,
                                                      FLAGS.ROT_STEP)]
        self.fid        = FLAGS.FID
        
        self.ids        = [i for i in range(FLAGS.ID_START,FLAGS.ID_END+1)]
        random.shuffle(self.ids)
        
        self.__train_data = self.ids[:self.__nb_train]
        self.__eval_data  = self.ids[self.__nb_train:self.__nb_train+self.__nb_eval]
        self.__test_data  = self.ids[self.__nb_train+self.__nb_eval:]

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

    def __getFlipDataById(self,img,gt,fid):
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

    def __modifyMask(self,y):
        by=np.zeros(y.shape[:2])
        gy=np.dot(y[...,:3], [0.299, 0.587, 0.114])
        by[gy>0]=255
        by=by.astype('uint8')
        return by

    def __saveTransposedData(self,rot_img,rot_gt,base_name,rot_angle,mode):
        for _fid in range(self.fid):
            
            x,y=self.__getFlipDataById(rot_img,rot_gt,_fid)
            y  =self.__modifyMask(y)
            file_name='{}_fid-{}_angle-{}.png'.format(base_name,_fid,rot_angle)

            if mode=='train':
                _img_path   =   os.path.join(self.train_img,file_name)
                _mask_path  =   os.path.join(self.train_mask,file_name)
            
            elif mode=='eval':
                _img_path   =   os.path.join(self.eval_img,file_name)
                _mask_path  =   os.path.join(self.eval_mask,file_name)
            
            else:
                pass

            imageio.imsave(_img_path ,x)
            imageio.imsave(_mask_path ,y)
            


    def create(self,mode):
        if mode=='train':
            __idens=self.__train_data
            dpath=os.path.join(self.train_dir)
        elif mode=='eval':
            __idens=self.__eval_data
            dpath=os.path.join(self.eval_dir)
        else:
            __idens=self.__test_data
            dpath=os.path.join(self.test_dir)
        LOG_INFO('Creating DataSet:{}'.format(mode))
        LOG_INFO('Directory:{}'.format(dpath))
        LOG_INFO('This will take quite some time. Thank you for your patience.')
        if mode!='test': 
            for iden in tqdm(__idens):
                img_path    =   os.path.join(self.img_dir,'{}.jpg'.format(iden))
                gt_path   =   os.path.join(self.mask_dir,'{}.png'.format(iden))
                # Load IMAGE  
                IMG=imgop.open(img_path)
                # Load GROUNDTRUTH
                GT=imgop.open(gt_path)
                # Create Segments
                _height,_width = np.array(IMG).shape[:2]
                for pxv in [0,_width//2,'AC']:
                    for pxl in [0,_height//2,'AC']:
                        if (pxv!='AC' and pxl!='AC'):
                            left    =   pxv
                            right   =   pxv+_width//2
                            top     =   pxl
                            bottom  =   pxl+_height//2
                            bbox    =   (left,top,right,bottom)
                            _IMG    =   IMG.crop(bbox).resize((self.img_dim,self.img_dim))
                            _GT     =   GT.crop(bbox).resize((self.img_dim,self.img_dim))
                        elif (pxv=='AC' and pxl!='AC'): 
                            continue
                        elif (pxl=='AC' and pxv!='AC'):
                            continue
                        elif (pxv=='AC' and pxl=='AC'):
                            _GT=GT.resize((self.img_dim,self.img_dim))
                            _IMG=IMG.resize((self.img_dim,self.img_dim))
                        else:
                            continue
                        # Create Rotations
                        
                        for rot_angle in self.rot_angles:
                            rot_img =   _IMG.rotate(rot_angle)
                            rot_gt  =   _GT.rotate(rot_angle)
                            self.__saveTransposedData(rot_img,rot_gt,'{}_{}_{}'.format(iden,pxv,pxl),rot_angle,mode)    
        else:
            for iden in tqdm(__idens):
                img_path  =   os.path.join(self.img_dir,'{}.jpg'.format(iden))
                gt_path   =   os.path.join(self.mask_dir,'{}.png'.format(iden))
                __img_path=   os.path.join(self.test_img,'{}.jpg'.format(iden))
                __gt_path =   os.path.join(self.test_mask,'{}.png'.format(iden))  
                shutil.copy(img_path,__img_path)
                shutil.copy(gt_path,__gt_path)

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
    '''
    Creates tfrecords from Provided Image Paths
    Arguments:
    image_paths = List of Image Paths with Fixed Size (NOT THE WHOLE Dataset)
    save_dir    = Tfrecords saving dir
    mode        = Mode of data to be created
    r_num       = number of record
    '''
    tfrecord_name='{}_{}.tfrecord'.format(mode,r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    LOG_INFO(tfrecord_path) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in tqdm(image_paths):
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
def data_input_fn(tf_dir,mode,BUFFER_SIZE,BATCH_SIZE,img_dim=256): 
    
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
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
