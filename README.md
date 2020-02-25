# pyWireDetect
Wire detection using synthetic data and dilated convolutional networks

Ref: UPWORK


    Version: 0.0.2   
    Author : Erenus Yildiz ,Md. Nazmuddoha Ansary
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
Major libraries:
*   image-classifiers==1.0.0
*   imageio==2.6.1
*   Keras==2.3.1
*   matplotlib==3.1.3
*   numpy==1.18.1
*   opencv-python==4.2.0.32
*   Pillow==7.0.0
*   scikit-image==0.16.2
*   scikit-learn==0.22.1
*   scipy==1.4.1
*   segmentation-models==1.0.1
*   tensorflow==2.1.0
*   termcolor==1.1.0
*   tqdm==4.42.1

Setup:

* create a virtualenvironment: for example-  ```$:~ virtualenv venv```
* activate the virtualenvironment: ```$:~ source venv/bin/activate```
* ```pip3 install -r requirements.txt```: depending on permission **--user** flag may be needed 

#  DataSet

Change The following Values in ***config.json***:
* ```SRC_DIR``` : The absolute path to the source folder which should contain the **images** and **annotations** folder which would have the following tree:
        
        source
        ├── annotations
        │   ├── ready_cleaned.csv
        │   └── via_region_data.json
        └── images
            ├── 100.jpg
            ├── 101.jpg
            ├── 10.jpg
            ├── 11.jpg
            ├── 12.jpg
            ├── .......

* ```DS_DIR``` : Location where the **DataSet** should be created with **Test**,**Train** and **Eval** split.

* An example config will look like as follows:

        {
            "SRC_DIR"        : "/media/ansary/DriveData/UPWORK/WireDetection/source/",
            "DS_DIR"         : "/media/ansary/DriveData/UPWORK/WireDetection/",
            "IMAGE_DIM"      : 512,
            "NB_CHANNELS"    : 3,
            "ROT_START"      : 0,
            "ROT_STOP"       : 45,
            "ROT_STEP"       : 5,
            "FID"            : 4,
            "DATA_COUNT"     : 256,
            "NB_EVAL"        : 10,
            "NB_TRAIN"       : 80,
            "ID_START"       : 1,
            "ID_END"         : 101,
            "TRAIN_COUNT"    : 20480,
            "EVAL_COUNT"     : 2048,
            "REPLICATE"      : 1
        }

* run ```python3 create_dataset.py``` within the activated virtual environment
> Ignore: Corrupt EXIF data Warning

* upon successfull execution the following number of images will be created in **{DS_DIR}/DataSet** folder: 
    * Train Data = 20480 
    * Test Data  = 11
    * Eval Data  = 2048

* The **DataSet** folder will have the following folder tree:

        DataSet
        ├── Eval
        │   ├── images
        │   └── masks
        ├── Test
        │   ├── images
        │   └── masks
        └── Train
            ├── images
            └── masks


* The **{DS_DIR}/TFRecords** folder contains all the **Train** and **Eval** data in **.tfrecord** format

        TFRecords
        ├── Eval
        │   ├── Eval_0.tfrecord
        │   ├── Eval_1.tfrecord
        │   ...................
        |   ...................
        │   ├── Eval_6.tfrecord
        │   └── Eval_7.tfrecord
        └── Train
            ├── Train_0.tfrecord
            ├── Train_10.tfrecord
            .....................
            .....................
            ├── Train_7.tfrecord
            ├── Train_8.tfrecord
            └── Train_9.tfrecord


**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  

