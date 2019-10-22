Paper
===============
* Mumford–Shah Loss Functional for Image Segmentation With Deep Learning
  * Authors: Boah Kim and Jong Chul Ye
  * published in IEEE Transactions on Image Processing (TIP)

Implementation
===============
A PyTorch implementation of deep-learning-based segmentation based on original cycleGAN code.
[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] 
(*Thanks for Jun-Yan Zhu and Taesung Park, and Tongzhou Wang.)

* Requirements
  * Python 2.7
  * PyTorch 1.1.0

Main
===============
* Training: LiTS_train_unet.py which is handled by scripts/LiTS_train_unet.sh
* A code for Mumford-Shah loss functional is in models/loss.py.
  * 'levelsetLoss' and 'gradientLoss2d' classes compose our Mumford-Shah loss function.
