# pyWireDetect
Wire detection using synthetic data and dilated convolutional networks



    Version: 1.0.0   
    Author : Erenus Yildiz ,Md. Nazmuddoha Ansary
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# SETUP
* Min Version: **python > 3.5**
* Tested on  : **python 3.6.9**

## IMPORTANT
Before cloning or executing any script the following has to be ensured in the **parent system**:
*  There should not exist any virtualenv named **wirevenv**
*  To ensure this,install jupyter in the **parent system** (not in any **virtualenv**) by:
    * ```$:~  pip3 install --upgrade pip``` 
    * ```$:~  pip3 install jupyter```
* See the list of kernels 
    * ```$:~  jupyter kernelspec list```
* **IF THERE EXISTS A PRE-EXISTING KERNEL NAMED** -->  **wirevenv**
    * ```$:~  jupyter kernelspec remove wirevenv```
* Confirm Deletion by :
    * ```$:~  jupyter kernelspec list```

## CLONING:
* Git clone this repo to a location where there is no additional read-write permission conflict **(THIS WILL HIGHLY DEPEND ON THE PARENT SYSTEM)**
* An example place can be: ```/home/{user_name}/WireDetect/```
* ```$:~  git clone https://github.com/mnansary/pyWireDetect.git```

## Environment Setup:
* create a virtualenvironment in the cloned repo   
    * ```WireDetect/pyWireDetect$:~    virtualenv wirevenv```
* activate the virtualenvironment: 
    * ```(wirevenv)WireDetect/pyWireDetect$:~    source wirevenv/bin/activate```
* Upgrade pip:
    * ```(wirevenv)WireDetect/pyWireDetect$:~    pip3 install --upgrade pip``` 
* Install requirements:
    * ```(wirevenv)WireDetect/pyWireDetect$:~    pip3 install -r requirements.txt```
    * depending on permission **--user** flag may be needed 
* Add **wirevenv** to jupyter:
    * ```(wirevenv)WireDetect/pyWireDetect$:~    python3 -m ipykernel install --user --name=wirevenv``` 
* Confirm Addition by :
    * ```(wirevenv)WireDetect/pyWireDetect$:~    jupyter kernelspec list```
 
     

#  DataSet Creation

* Download the [FINAL DATASET]() <-- TO BE ADDED AFTER FINILISING DATASET
* The extracted folder should have a **wire_set** folder which has the following tree: 
  
        wire_set
        ├── test
        │   ├── images
        │   │   ├── xx.jpg
        ..................
        │   │   ├── xx.jpg
        │   │   └── xx.jpg
        │   └── masks
        │       ├── xx.jpg
        ..................
        │       ├── xx.jpg
        │       └── xx.jpg
        ├── train
        │   ├── images
        │   │   ├── xx.jpg
        ..................
        │   │   ├── xx.jpg
        │   │   └── xx.jpg
        │   └── masks
        │       ├── xx.jpg
        ....................
        │       ├── xx.jpg
        │       └── xx.jpg
        └── val
            ├── images
            │   ├── xx.jpg
            ...............
            │   └── xx.jpg
            └── masks
                ├── xx.jpg
                ...........
                └── xx.jpg

* Copy the **wire_set** folder to the git repo (**pyWireDetect**) so that the **datagen.ipynb** and **wire_set** are under the same folder. I.E- the repo tree should update to show as: 

        ├── datagen.ipynb
        ├── wire_set
        ............

* Move The **test** folder in **wire_set** to **TPU_COLAB** folder. The **wire_set** folder should now only contain **train** and **val** folders

* Open the repo in **jupyter-notebook** within the activated virtualenv:
    * ```(wirevenv)WireDetect/pyWireDetect$:~  jupyter-notebook```
* Select **wirevenv** as the kernel for **datagen.ipynb**

* Run all the blocks in **datagen.ipynb**

* Upon Successfull Execution The **DataSet** folder now created should have the following tree with populated data:

        DataSet
        ├── Eval
        │   ├── images
        │   └── masks
        ├── Train
        │   ├── images
        │   └── masks
        └── WireDTF
            ├── Eval
            └── Train

# Training:

## TPU COLAB
![](/info/src_img/tpu.ico?raw=true) 
> TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the Tesla K80 available in Google Colab delivers respectable 1.87 TFlops and has 12GB RAM, the TPUv2 available from within Google Colab comes with a whopping 180 TFlops, give or take. It also comes with 64 GB High Bandwidth Memory (HBM). 

* Create A folder named **WIRE_DETECTION** (in all caps) in you google drive
* Upload the folder **TPU_COLAB** into **WIRE_DETECTION** folder. The **TPU_COLAB** should have the following folder tree before Uploding:

        ├── model1.ipynb
        ├── model2.ipynb
        ├── model3.ipynb
        ├── model4.ipynb
        ├── model_weights
        ├── test
        └── train.ipynb

## SOTA MODELS (SELECTED)
* The selected SOTA models for training and scoring are as follows:
    * **model1.ipynb** : 'efficientnetb7'
    * **model2.ipynb** : 'inceptionv3'
    * **model3.ipynb** : 'inceptionresnetv2'
    * **model4.ipynb** : 'densenet201'

* In order to train and evaluate a **selected SOTA** model, open the uploaded **model{x}.ipynb** in google colab
* Go To **Edit**>**Notebook Settings** > **Hardware Accelerator** select **TPU**
* Run the cell **MOUNT GOOGLE Drive** only

![](/info/mount.png?raw=true)

* Upon Successfull mount run the cell **Change your working directory**

![](/info/cd.png?raw=true)

* If both **Mounting** and **Directory Change** are successfull, Go To **Runtime**>**Restart and run all**

* Upon successfull run the model predictions on test data will be displayed within the notebook with **SSIM** and **IoU/F1** score summary

* The trained model will be saved at **WIRE_DETECTION**/**TPU_COLAB**/**model_weights**/**model{x}.h5** in **google drive**

* The predicted masks can be found at **WIRE_DETECTION**/**TPU_COLAB**/**test**/**preds**/**model{x}**/ in **google drive**

## train.ipynb
* The **train.ipynb** is used to train other models 
* Available SOTA model identifiers:

| Model Type    | Identifiers   |
| ------------- | ------------- |
|   VGG	        |'vgg16' 'vgg19'|
|   ResNet      |	'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'|
|   SE-ResNet   |	'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'|
|   ResNeXt     |	'resnext50' 'resnext101'|
|   SE-ResNeXt  |	'seresnext50' 'seresnext101'|
|   SENet154    |	'senet154'|
|   DenseNet    |	'densenet121' 'densenet169' 'densenet201'|
|   Inception   |	'inceptionv3' 'inceptionresnetv2'|
|   MobileNet   |	'mobilenet' 'mobilenetv2'|
|   EfficientNet|	'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' 'efficientnetb6' 'efficientnetb7'|

The process for training is same as before with exception of **MODEL SPEC** cell execution

* Select The model you want to train  
![](/info/specB.png?raw=true)
* Copy the model **identifier** with **single quotes** to **model_name**. i.e- for example to train the **ResNet50** model you need to copy **'resnet50'** (with quotes)
![](/info/specC.png?raw=true)

* After selection:
    *  **MOUNT GOOGLE Drive** 
    *  **Change your working directory**
    *  **Runtime**>**Restart and run all**

* The trained model will be saved at **WIRE_DETECTION**/**TPU_COLAB**/**model_weights**/**{identifier}.h5** in **google drive**

* The predicted masks can be found at **WIRE_DETECTION**/**TPU_COLAB**/**test**/**preds**/**{IDENTIFIER}**/ in **google drive**


* The model arch is the famous SOTA arch **Unet**

![](/info/arch.png?raw=true)

# EXECUTION ENVIRONMENT 

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  


