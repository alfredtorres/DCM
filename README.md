# DCM
This repository is the official implementation of [Rethinking RoI Strategy in Interactive 3D Segmentation for Medical Images](https://openreview.net/forum?id=jospESnUL9).

![model](.\figs\model.png)

## Environments and Setup

For inference or evaluation, you can directly download the docker file: [coreset](https://huggingface.co/yuyi1005/yiooo/blob/main/yiooo_coreset.tar.gz) and [alldata](https://huggingface.co/yuyi1005/yiooo/blob/main/yiooo_alldata.tar.gz).

For training, please download the pretrained weights: [SAM-Med3D](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo_cvpr_coreset.pth) and [VISTA3D](https://drive.google.com/file/d/1hQ8imaf4nNSg_43dYbPSJT0dr7JgAKWX/view?usp=sharing) and install the requirements.

```
pip install -r requirements.txt
```

## Training

1. train the global roi branch:

   ```
   cd train_global_roi
   torchrun --nnodes=1 --nproc_per_node=2 train.py
   ```

2. train the local roi branch:

   ```
   cd train_local_roi
   torchrun --nnodes=1 --nproc_per_node=3 train_cvpr_ddp_interactive.py
   ```

## Inference and Evaluation

```
docker load -i yiooo_coreset.tar.gz
docker container run --gpus "device=0" -m 32G --name yiooo_coreset --rm -v $PWD/PathToTestSet/:/workspace/inputs/ -v $PWD/yiooo_coreset_outputs/:/workspace/outputs/ yiooo_coreset:latest /bin/bash -c "sh predict.sh"
```

## Results

Our method achieves the following performance on the challenge coreset.

| Modality   | DSC AUC | NSD AUC | DSC Final | NSD Final |
| ---------- | :-----: | :-----: | :-------: | :-------: |
| CT         | 3.3461  | 3.4719  |  0.8462   |  0.8797   |
| MRI        | 2.7133  | 3.0852  |  0.6809   |  0.7714   |
| Microscopy | 2.2917  | 3.0618  |  0.5871   |  0.7743   |
| PET        | 3.0188  | 2.8778  |  0.7691   |  0.7440   |
| Ultrasound | 3.6741  | 3.7096  |  0.9299   |  0.9440   |

