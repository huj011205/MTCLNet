

@@ -1,69 +0,0 @@
# MTCL:Multi-Task Consistency Learning for Semi-Supervised 3D Medical Image
Segmentation
by Jiale Hu, Yuze Hu, Changfang Chen, Yushu Zhang, Rensong Liu, and Zhaoyang Liu. 

### Code owner

```
Yuze Hu
```

### News

```
<28.09.2025> We provided our pre-trained models on the LA and Pancreas-CT ;
```
```
<28.09.2025> We released the codes;
```
### Introduction
This repository is for our paper: '[MTCL:Multi-Task Consistency Learning for Semi-Supervised 3D Medical Image Segmentation]
### Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.2 and Python 3.8.10; All experiments in our paper were conducted on a single NVIDIA Tesla V100 GPU.

### Usage
1. Clone the repo.;
```
git clone https://github.com/huj011205/MTCLNet.git
```
2. Put the data in './MTCLNet/data';

3. Train the model;
```
cd MTCLNet
# e.g., for 20% labels on LA
python ./code/train_mtclnet_3d.py --dataset_name LA --model mtclnet3d_v1 --labelnum 16 --gpu 0 --temperature 0.1
```
4. Test the model;
```
cd MTCLNet
# e.g., for 20% labels on LA
python ./code/test_mtclnet_3d.py --dataset_name LA --model mtclnet3d_v1 --exp MTCLNET --labelnum 16 --gpu 0
```



### Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) , [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and[MC-Net+](https://github.com/ycwu1997/MC-Net). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at '1406811561@qq.com'