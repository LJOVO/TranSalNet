# TranSalNet: Towards perceptually relevant visual saliency prediction <br>
  
This repository provides the Pytorch implementation of **TranSalNet: Towards perceptually relevant visual saliency prediction** published in the [Neurocomputing paper](https://doi.org/10.1016/j.neucom.2022.04.080).<br>

***Overview***: <br>
  
![arch](https://github.com/LJOVO/TranSalNet/blob/master/Figures/arch.png?raw=true)
  

## Requirements
- Python 3.8
- Pytorch 1.7.1
- Torchvision 0.8.2
- OpenCV-Python 4.5.1
- SciPy 1.6.0
- tqdm 4.56.0
  
## Pretrained Models
TranSalNet has been implemented in two variants: **TranSalNet_Res** with the CNN backbone of **ResNet-50** and **TranSalNet_Dense** with the CNN backbone of **DenseNet-161**.  
Pre-trained models on SALICON training set for the above two variants can be download at:  
 - TranSalNet_Res: [Google Drive](https://drive.google.com/file/d/14czAAQQcRLGeiddPOM6AaTJTieu6QiHy/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1bDSCyM6BWJrhpLaUL9CIhg) (access code: 1234)
 - TranSalNet_Dense: [Google Drive](https://drive.google.com/file/d/1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1uSl8YTnPwgWZPWt35mav6A) (access code: 1234)
  
It is also necessary to download ResNet-50 (for TranSalNet_Res) and DenseNet-161 (TranSalNet_Dense) pre-trained models on ImageNet. These models can be download at:
 - ResNet-50: [Google Drive](https://drive.google.com/file/d/1AdTljzE3tvTIgTxWCEdf0g9ZWt68RCVD/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1UbZwKAaHGamBu2zg_0pWMg) (access code: 1234)
 - DenseNet-161: [Google Drive](https://drive.google.com/file/d/1IZ8EtoM7Ui8QA_MlX7lqcIhusLa3ddl6/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/18VRdKRPBefFCdtK68OsJUQ) (access code: 1234)
  

  
## Quick Start
The pre-trained models should be downloaded and put in the folder named `pretrained_models` in the code folder first, then the following example codes can be used smoothly.  
We have prepared two Jupyter Notebook files (.ipynb) for usage of TranSalNet.  
- Testing: `testing.ipynb`. It can be used to compute and obtain the visual saliency maps of input images.     
By default, the test image and the corresponding output are in the folder named `testing`, and the models are loaded with parameters pre-trained on the [SALCON](http://salicon.net/challenge-2017/) training set.   
- Fine-tuning or Training from scratch: `training&fine-tuning.ipynb`  
  ```
  Data prepare for fine-tuning and training:
  │ dataset/
  ├── train_ids.csv
  ├── val_ids.csv
  ├── train/
  │   ├── train_stimuli/
  │   │   ├── ......
  │   ├── train_saliency/
  │   │   ├── ......
  │   ├── train_fixation/
  │   │   ├── ......
  ├── val/
  │   ├── val_stimuli/
  │   │   ├── ......
  │   ├── val_saliency/
  │   │   ├── ......
  │   ├── val_fixation/
  │   │   ├── ......
  ```  
In the above two .ipynb files, it is possible to choose whether TranSalNet_Res or TranSalNet_Dense is used, depending on the needs and preferences.
  
___Please note: The spatial size of inputs should be 384×288 (width×height).___
  
## Citation
If this work is helpful, please consider citing:  
  ```
@article{TranSalNet,
title = {TranSalNet: Towards perceptually relevant visual saliency prediction},
journal = {Neurocomputing},
year = {2022},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2022.04.080},
author = {Jianxun Lou and Hanhe Lin and David Marshall and Dietmar Saupe and Hantao Liu},
}
  ```


