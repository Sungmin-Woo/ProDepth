# ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion
### [Project Page](https://sungmin-woo.github.io/prodepth/) | [Paper](https://arxiv.org/pdf/2407.09303)

Official PyTorch implementation for the ECCV 2024 paper: "ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion". 

Codes will be released soon.

## 👀 Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [KITTI](#-KITTI)
  - [Cityscapes](#-Cityscapes)
- [Pretrianed weights](#-pretrained-weights)
- [Training](#training)
  - [Single-GPU](#-single-gpu-training)
  - [Multi-GPU](#-multi-gpu-training)
- [Ground Truth Data Prepare and Evaluation](#evaluation)
- [Citation](#citation)

## ⚙️ Installation
You can install the dependencies with:
```
git clone https://github.com/Sungmin-Woo/ProDepth.git
cd ProDepth/
conda create -n prodepth python=3.9.13
conda activate prodepth
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install numpy==1.23.4 matplotlib==3.5.3 opencv-python==4.7.0.72 tqdm scikit-image timm==0.9.7 tensorboardX==1.4
```
We ran out experiments with PyTorch 1.7.1, CUDA 11.0, Pyhton 3.9.13 and Ubuntu 18.04.

## 💾 Data Prepare
### 🔹 KITTI

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```

You should be able to obtain following directory structure.
```
data_dir/kitti_data/
  |-- 2011_09_26
    |-- calib_cam_to_cam.txt
    |-- calib_imu_to_velo.txt
    |-- calib_velo_to_cam.txt
    |-- 2011_09_26_drive_0001_sync
      |-- image_00
      |-- image_01
      |-- image_02
      |-- image_03
      |-- oxts
      |-- velodyne_points
    |-- 2011_09_26_drive_0002_sync
    |-- ...
  |-- 2011_09_28
  |-- 2011_09_29
  |-- 2011_09_30
  |-- 2011_10_03
```

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) for detail instructions.

### 🔹 Cityscapes

From [Cityscapes official website](https://www.cityscapes-dataset.com/) download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip` into the `CS_RAW` folder.

Preprocess the Cityscapes dataset using the `prepare_train_data.py`(from SfMLearner) script with following command:
```bash
cd CS_RAW
unzip leftImg8bit_sequence_trainvaltest.zip
unzip camera_trainvaltest.zip
cd ..

python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir CS_RAW \
    --dataset_name cityscapes \
    --dump_root CS \
    --seq_length 3 \
    --num_threads 8
```

You should be able to obtain following directory structure ./CS_RAW and ./CS as...
```
data_dir/CS_RAW/
  |--camera
    |--test
    |--train
    |--val
  |--leftImg8bit_sequence
    |--test
    |--train
    |--val
  |--license.txt
  |--ReadMe

data_dir/CS/
  |--aachen
  |--bochum
```

You can also place the Cityscapes dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

## 📦 Pretrained Weights

You can download weights for some pretrained models here:

🔹 KITTI
| CNN Backbone      | Input size  | Cityscapes AbsRel | Link                                                               |
|-------------------|-------------|:-----------------------------------:|----------------------------------------------------------------------------------------------|
| ResNet 18         | 640 x 192   |      0.091         | [Download 🔗]           |
| LiteMono          | 640 x 192   |      0.086         | [Download 🔗]           |

🔹 Cityscapes
| CNN Backbone      | Input size  | Cityscapes AbsRel | Link                                                               |
|-------------------|-------------|:-----------------------------------:|----------------------------------------------------------------------------------------------|
| ResNet 18         | 512 x 192   |      0.099         | [Download 🔗]           |
| LiteMono          | 512 x 192   |      0.095         | [Download 🔗]           |


<!--
| CNN Backbone      | Input size  | Cityscapes AbsRel | Link                                                               |
|-------------------|-------------|:-----------------------------------:|----------------------------------------------------------------------------------------------|
| ResNet 18         | 640 x 192   |      0.104         | [Download 🔗](https://drive.google.com/file/d/1k3-7nki-v6k111wBZ-7pQNVU8QU7Nemx/view?usp=sharing)           |
-->

## ⏳ Training

Training can be done with a single GPU or multiple GPUs (via `torch.nn.parallel.DistributedDataParallel`).
By default models and log event files are saved to ./log.

## Note
- If you want to run Prodepth with Lite-Mono backbone, please download the ImageNet-1K pretrained [Lite-Mono](https://surfdrive.surf.nl/files/index.php/s/oil2ME6ymoLGDlL) and place to './pretrained/'.
- For better training, we recommend to freeze single-frame depth estimation during training. Here, we provide checkpoints for single-frame depth estimation for both [KITTI] and [Cityscapes]. Please download the given checkpoints and place to './pretrained/<CS> or <KIITI>/'.
### 🔹 Single GPU Training

For instance, to train w/ 1 GPU on Cityscapes Dataset:

Change $GPU_NUM and $BS in train_cs_prodepth.sh to 1 and 24
Then run,
```
bash ./train_cs_prodepth.sh <model_name> <port_num>
```

### 🔹 Multi-GPU Training

For instance, to train w/ 4 GPUs on Cityscapes Dataset:
Change $GPU_NUM and $BS in train_cs_prodepth.sh to 4 and 6
Then run,
```
CUDA_VISIBLE_DEVICES=<your_desired_GPU> bash ./train_cs_prodepth.sh <model_name> <port_num>
```
Note: Learning rate and scheduler step size should be adjusted accordingly when training with a single GPU (See [options.py] for details).

## 📊 Ground Truth Data Prepare and Evaluation

🔹 KITTI
To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```

...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

To evaluate a model on KITTI, run:
```
bash ./test_kitti_prodepth.sh <model_name>
```


🔹 Cityscapes
Download cityscapes depth ground truth(provided by manydepth) for evaluation:
```bash
cd ..
cd splits/cityscapes/
wget https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip
unzip gt_depths_cityscapes.zip
cd ../..
```
To evaluate a model on KITTI, run:
```
bash ./test_cs_prodepth.sh <model_name>
```

## ✏️ 📄 Citation
If you find our work useful or interesting, please cite our paper:

```
@article{woo2024prodepth,
  title={ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion},
  author={Woo, Sungmin and Lee, Wonjoon and Kim, Woo Jin and Lee, Dogyoon and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2407.09303},
  year={2024}
}
```
