# UString
This repo contains code for our following paper:

Wentao Bao, Qi Yu and Yu Kong, Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning, submitted to ACM Multimedia 2020.

## Contents
0. [Overview](#overview)
0. [Dataset Preparation](#dataset)
0. [Pre-trained Models](#models)
0. [Installation Guide](#install)
0. [Train and Test](#traintest)
0. [Citation](#citation)

<a name="overview"></a>
## :globe_with_meridians:  Overview 
<div align=center>
  <img src="demo/000821_vis.gif" alt="Visualization Demo" width="800"/>
</div>

We propose an uncertainty-based traffic accident anticipation model for dashboard camera videos. The task aims to accurately identify traffic accidents and anticipate them as early as possible. We first use Cascade R-CNN to detect bounding boxes of each frame as risky region proposals. Then, the features of these proposals are fed into our model to predict accident scores (red curve). In the same time, both aleatoric (brown region) and epistemic (yellow region) uncertainties are predicted by Bayesian neural networks.

<a name="dataset"></a>
## :file_cabinet:  Dataset Preparation

The code currently supports three datasets., DAD, A3D, and CCD. These datasets need to be prepared under the folder `data/`. 

> * For CCD dataset under the folder `data/crash/`, please refer to the [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. 
> * For DAD dataset under the folder `data/dad/`, you can acquire it from [DAD official](https://github.com/smallcorgi/Anticipating-Accidents). The officially provided features are grouped into batches while it is more standard to split them into separate files for training and testing. To this end, you can use the script `./script/split_dad.py`. 
> * For A3D dataset under the folder `data/a3d/`, the annotations and videos are obtained from [A3D official](https://github.com/MoonBlvd/tad-IROS2019). Since it is sophiscated to process it for traffic accident anticipation with the same setting as DAD, you can directly download our processed A3D dataset from Google Drive: [A3D processed](https://drive.google.com/drive/folders/1loK_Cr1UHZGJpetUIQCSI3NlBQWynK3v?usp=sharing).

<a name="models"></a>
## :file_cabinet:  Pre-trained Models

Choose the following files according to your need.

> * [**Cascade R-CNN**](https://drive.google.com/drive/folders/1fbjKrzgXv_FobuIAS37k9beCkxYzVavi?usp=sharing): The pre-trained Cascade R-CNN model files and modified source files. Please download and extract them under `lib/mmdetection/`.
> * *链接提供的 Cascade R-CNN 配置文件 config.py 与当前版本的 mmdetection(v2.26.0) 不兼容, 可以选择重新训练*


> * [**Pre-trained UString Models**](https://drive.google.com/drive/folders/1yUJnxwDtn2JGDOf_weVMDOwywdkWULG2?usp=sharing): The pretrained model weights for testing and demo usages. Download them and put them anywhere you like.
> * *链接提供了三个权重文件分别针对三个数据集, 不同数据集的权重似乎存在差异，不可以混用*

<a name="install"></a>
## :file_cabinet: Installation Guide

**Note**: 在项目根目录执行所有代码

**Note**: 服务器环境 `CUDA=11.7.99`, `cuDNN=8.6.0`, `Nvidia GPU Driver=515.65.01`

Please follow the [official mmdetection installation guide](https://mmdetection.readthedocs.io/en/stable/get_started.html) to setup an mmdetection environment.

### 1. Setup Python & MMDetection Environment (ZZJ)

```shell
# create python environment
conda create -n UString python=3.8

# activate environment
conda activate UString

# install dependencies (MMDetection, Pytorch and PyG are NOT included)
pip install -r requirements.txt

# intsall Pytorch (2022.11 Pytorch=1.13[stable] with CUDA=11.7)[https://pytorch.org/get-started/locally/]
pip3 install torch torchvision torchaudio

# intsall PyG [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html]
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# install MMDetection (https://mmdetection.readthedocs.io/en/stable/get_started.html)
# Install MMCV
pip install -U openmim
mim install mmcv-full

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# compile & install
pip install -v -e .
python setup.py install

```

<a name="traintest"></a>
## :rocket: Train and Test

**Note**: 在项目根目录执行所有代码

### 1. Demo

We provide an end-to-end demo to predict accident curves with given video. Note that before you run the following script, both the python and mmdetection environments above are needed. The following command is an example using the pretrained model on CCD dataset. 

请在所有脚本中重新设定 UString 模型权重的位置

```shell
bash run_demo.sh demo/000821.mp4 # 尚未验证通过 
```
Results will be saved in the same folder `demo/`.

### 2. Test the pre-trained UString model

Take the DAD dataset as an example, after the DAD dataset is correctly configured, run the following command. By default the model file is placed at `output/UString/vgg16/snapshot/final_model.pth`.
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh test 0 dad 10 # 验证通过
```
The evaluation results on test set will be reported, and visualization results will be saved in `output/UString/vgg16/test/`.

### 3. Train UString from scratch.

To train UString model from scratch, run the following commands for DAD dataset:
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh train 0 dad 10 # 尚未验证
```
By default, the snapshot of each checkpoint file will be saved in `output/UString/vgg16/snapshot/`.


<a name="citation"></a>
## :bookmark_tabs:  Citation

Please cite our paper if you find our code useful.

```
@InProceedings{BaoMM2020,
    author = {Bao, Wentao and Yu, Qi and Kong, Yu},
    title  = {Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM ’20)},
    month  = {October},
    year   = {2020}
}
```

If you have any questions, please feel free to leave issues in this repo or contact [Wentao Bao](mailto:wb6219@rit.edu) by email. Note that part of codes in `src/` are referred from [VGRNN](https://github.com/VGraphRNN/VGRNN) project.

