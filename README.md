# pyWireDetect
Wire detection using synthetic data and dilated convolutional networks
Ref: UPWORK
    
    Version: 0.0.1   
    Author : Erenus Yildiz 
             Md. Nazmuddoha Ansary
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
*   numpy==1.18.1
*   opencv-python==4.2.0.32
*   Python == 3.6.9

> Create a Virtualenv and *pip3 install -r requirements.txt*

#  DataSet

**config.json**
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
            "IMAGE_DIM"      : 256,
            "NB_CHANNELS"    : 3,
            "ROT_START"      : 0,
            "ROT_STOP"       : 45,
            "ROT_STEP"       : 5,
            "FID"            : 4,
            "DATA_COUNT"     : 1024,
            "NB_EVAL"        : 20,
            "NB_TRAIN"       : 70,
            "ID_START"       : 1,
            "ID_END"         : 101
        }

* run ```python3 create_dataset.py```

![](/info/exec.png?raw=true)

# Information
* Train Data = 14000 
* Test Data  = 11
* Eval Data  = 4000
* DataSet Size = 8.6 GiB
* DataSet Creation Time ~= 2228s

**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  

