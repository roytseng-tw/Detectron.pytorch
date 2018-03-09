# A Pytorch Implementation of Mask R-CNN
This repository is built on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). Million thanks to Jianwei Yang and Jiasen Lu's awesome work.

This implementation preservers the following features from jwyang et al's implementation:
- **It is pure Pytorch code**.

- **It supports multi-image batch training**. To be more precise, 1 image for batch size 1 and n images for batch size n.

- **It supports multiple GPUs training**. I use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

- **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. More importantly, they modify all of them to support multi-image batch training.

- **It is memory efficient**. We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. For the mask part, the gt_masks only retain the corresponding the gt mask for each roi. For the pose part, only the keypoint positions are passed and are kept in the form of a few numbers all the way. These is no heatmaps involved in the computation of pose loss.

## Progress
- Training of **bbox + mask** and **bbox + mask + pose** is done.
- Use `mask_trainval_net.py` to train on bbox and mask. Use `pose_mask_trainval_net.py` to train on bbox, mask and keypoint.
  Refer to original README from jwyan below for the command line arguments.

## Requirements
Either python 2 or 3 should work. Tested under python 3 (recommend).
- python packages
    - pytorch  
    - torchvision  
    - numpy  
    - scipy  
    - scikit-image  
    - opencv  
    - easydict
    - pyyaml
    - tqdm
    - pycocotools

- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

## Getting Started
Clone the repo
```
git clone https://github.com/roytseng-tw/mask-rcnn.pytorch.git
```

Then, create a data folder under the repo
```
cd mask-rcnn.pytorch
mkdir data
```

### Compilation
As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Change the [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/blob/master/lib/make.sh#L16) in `lib/mask.sh` accordingly.

And then compile the cuda dependencies using following commands:
```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop.

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compile the code.**

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

### Pretrained Model

For mask-rcnn only the resnet backbone is implemented, so you may only download the caffe resnet101 weight file.

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the `data/pretrained_model/`.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results. 

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Data Preparation

For mask-rcnn, only **COCO** is used. Recommend to use coco2017 datasets instead of coco2014.

In coco2014, some mask annotations have different (h,w) shape to the corresponding images. Maybe `instances_minival2014.json` and `instances_valminusminival2014.json` contains corrupted mask annotations, which should have been fixed in the official coco annotation file.

- **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

- **COCO**: 
Download the coco images and annotations from [coco website](http://cocodataset.org/#download).

  And make sure to put the files as the following structure:
  ```
  coco
  ├── annotations
  |   ├── instances_minival2014.json
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   ├── instances_valminusminival2014.json
  │   ├── person_keypoints_train2014.json
  │   ├── person_keypoints_train2017.json
  │   ├── person_keypoints_val2014.json
  │   └── person_keypoints_val2017.json
  └── images
      ├── train2014
      ├── train2017
      ├── val2014
      └── val2017
  ```
  Download link for [instances_minival2014.json](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0) and [instances_valminusminival2014.json](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  
   Feel free to put the dataset at any place you want. Recommend to put the images on a SSD for possible better training performance
   
   At last, remember to soft link the dataset under the `data/` folder:
   ```
   ln -s path/to/coco data/coco
   ```

- **Visual Genome**: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to prepare Visual Genome dataset. You need to download the images and object annotation files first, and then perform proprecessing to obtain the vocabulary and cleansed annotations based on the scripts provided in this repository.

## Train
Take `mask_trainval_net.py` for example.

Before training, set the right directory to save and load the trained models. Change the argument value of "save_dir" to adapt to your environment, which defaults to `~/models`.

To train a mask R-CNN model with res101 on coco2017 using a single gpu, simply run:
```
python mask_trainval_net.py \
    --dataset coco2017 --net res101 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $LR_DECAY_STEP \
    --cuda
```

Above, `BATCH_SIZE` and `WORKER_NUMBER` can be set adaptively according to your GPU memory size. **On 1080ti with 11G memory, BATCH_SIZE can be up to 3**.

For `LEARNING_RATE` and `LR_DECAY_STEP`, you can refer to the [below section](#benchmarking-of-faster-rcnn).
**Note that, you should change learning rete according to batch size. Smaller batch size, smaller learning rate.**

If you have multiple GPUs, add `--mGPUs` to the arguments.
If you want to record the losses during the training a the tensorboard log file, add `--use_tfboard`.
Use `--epochs N` to train for N epochs. N defaults to 10.

To load a pretrained checkpoint file and continue training, use the following command:
```
python mask_trainval_net.py --load_ckpt path/to/the/checkpoint_file ...
```

## Notes
- For now, the gt roi mask is cropped using **rounded** bbox coord. Maybe it's better to do something like roi-align ?

### TODOs
- Update the testing scripts and testing results.

- Improve the flexibility of dataloader. Make number of images used per batch configurable. Now, it's hard coded to 1 per batch.

- Implement a customized `nn.DataParallel` which can pass through gpu tensors along with cpu tensors. Pytorch's `nn.DataParallel` will scatter all the inputs to gpu memory if cuda is available, but some inputs don't need to be on gpu so early. For better gpu memory efficiency, only part of original inputs, which will reduce in size, have to be put on gpu.  


## Benchmarking of faster-rcnn
Copied from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

We benchmark our code thoroughly on three datasets: pascal voc, coco and imagenet-200, using two different network architecture: vgg16 and resnet101. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|-----
VGG-16     | 1 | 1 | 1e-3 | 5   | 7   |  0.76 hr | 3265MB   | 70.2
VGG-16     | 1 | 4 | 4e-3 | 8   | 10  |  0.50 hr | 9083MB   | 70.7
[VGG-16](https://www.dropbox.com/s/1a31y7vicby0kvy/faster_rcnn_1_10_625.pth?dl=0)     | 8 | 16| 1e-2 | 8   | 10  |  0.19 hr | 5291MB   | 69.4
[VGG-16](https://www.dropbox.com/s/hkj7i6mbhw9tq4k/faster_rcnn_1_11_416.pth?dl=0)     | 8 | 24| 1e-2 | 10  | 11  |  0.16 hr | 11303MB  | 69.2
[Res-101](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0)   | 1 | 1 | 1e-3 | 5   | 7   |  0.88 hr | 3200 MB  | 75.2
[Res-101](https://www.dropbox.com/s/8bhldrds3mf0yuj/faster_rcnn_1_10_2504.pth?dl=0)    | 1 | 4 | 4e-3 | 8   | 10  |  0.60 hr | 9700 MB  | 74.9
[Res-101](https://www.dropbox.com/s/5is50y01m1l9hbu/faster_rcnn_1_10_625.pth?dl=0)    | 8 | 16| 1e-2 | 8   | 10  |  0.23 hr | 8400 MB  | 75.2 
[Res-101](https://www.dropbox.com/s/cn8gneumg4gjo9i/faster_rcnn_1_12_416.pth?dl=0)    | 8 | 24| 1e-2 | 10  | 12  |  0.17 hr | 10327MB  | 75.1   


2). COCO (Train/Test: coco_train/coco_test, scale=800, max_size=1200, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
VGG-16     | 8 | 16    |1e-2| 4   | 6  |  4.9 hr | 7192 MB  | 29.2 
[Res-101](https://www.dropbox.com/s/5if6l7mqsi4rfk9/faster_rcnn_1_6_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 6  |  6.0 hr    |10956 MB  | 36.2
[Res-101](https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 10  |  6.0 hr    |10956 MB  | 37.0

**NOTE**. Since the above models use scale=800, you need add "--ls" at the end of test command.

3). COCO (Train/Test: coco_train/coco_test, scale=600, max_size=1000, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
[Res-101](https://www.dropbox.com/s/y171ze1sdw1o2ph/faster_rcnn_1_6_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 6  |  5.4 hr    |10659 MB  | 33.9
[Res-101](https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 10  |  5.4 hr    |10659 MB  | 34.5

4). Visual Genome (Train/Test: vg_train/vg_test, scale=600, max_size=1000, ROI Align, category=2500)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
[Res-101](http://data.lip6.fr/cadene/faster-rcnn.pytorch/faster_rcnn_1_19_48611.pth)    | 1 P100 | 4    |1e-3| 5   | 20  |  3.7 hr    |12707 MB  | 4.4

Thanks to [Remi](https://github.com/Cadene) for providing the pretrained detection model on visual genome!

* Click the links in the above tables to download our pre-trained faster r-cnn models.
* If not mentioned, the GPU we used is NVIDIA Titan X Pascal (12GB).
