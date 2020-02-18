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
#---------------------------------------------------------------------------
def readJson(file_name):
    return json.load(open(file_name))

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
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
        self.eval_dir   =   create_dir(self.ds_dir,'Eval')
        self.test_dir   =   create_dir(self.ds_dir,'Test')
        self.test_img   =   create_dir(self.test_dir,'images')
        self.test_mask  =   create_dir(self.test_dir,'mask')
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

    def __saveTransposedData(self,rot_img,rot_gt,base_name,rot_angle,mode):
        for _fid in range(self.fid):
            x,y=self.__getFlipDataById(rot_img,rot_gt,_fid)
            data=np.concatenate((x,y),axis=1)
            file_name='{}_fid-{}_angle-{}.png'.format(base_name,_fid,rot_angle)
            if mode=='train':
                dpath=os.path.join(self.train_dir,file_name)
            elif mode=='eval':
                dpath=os.path.join(self.eval_dir,file_name)
            else:
                dpath=os.path.join(self.test_dir,file_name)
            imageio.imsave(dpath,data)


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
def createH5Data(data_paths,counter,save_dir,mode):
    holder=[]
    _dpath=os.path.join(save_dir,'{}_{}.h5'.format(mode,counter))
    LOG_INFO(_dpath)
    for __path in tqdm(data_paths):
        data=np.array(imgop.open(__path))
        data=np.expand_dims(data,axis=0)
        holder.append(data)
    _data=np.vstack(holder)
    saveh5(_dpath,_data) 
#--------------------------------------------------------------------------------------------------------------------------------------------------
