# Rethinking RoI Strategy in Interactive 3D Segmentation for Medical Images

This repository is the official implementation of [Rethinking RoI Strategy in Interactive 3D Segmentation for Medical Images](https://openreview.net/forum?id=jospESnUL9).

![model](/figs/model.png)

## 🏆 Achievements
Our solution achieved remarkable results in the CVPR 2025 Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge:
- 🥈 **Second place** in AllData Track
- 🥉 **Third place** in Coreset Track

## 📋 Overview
This repository implements the **DCM (DualClickMed)** approach for interactive 3D medical image segmentation. Our dual-expert architecture features both global and local Region-of-Interest (RoI) strategies:
- **Global-RoI expert**: Processes the entire organ based on user prompts to provide comprehensive anatomical context
- **Local-RoI expert**: Focuses on high-resolution patches centered on specific user clicks for precise segmentation of fine structures

## 🔧 Environment Setup
### Using Docker (For Inference/Evaluation)
Download the docker images:
- [Coreset model (4.93 GB)](https://huggingface.co/yuyi1005/yiooo/blob/main/yiooo_coreset.tar.gz)
- [AllData model (4.93 GB)](https://huggingface.co/yuyi1005/yiooo/blob/main/yiooo_alldata.tar.gz)

### Manual Setup (For Training)
1. Download pretrained weights:
   - [SAM-Med3D (402 MB)](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo_cvpr_coreset.pth)
   - [VISTA3D](https://drive.google.com/file/d/1hQ8imaf4nNSg_43dYbPSJT0dr7JgAKWX/view?usp=sharing)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 🚀 Training

### Global RoI Branch
```bash
cd train_global_roi
torchrun --nnodes=1 --nproc_per_node=2 train.py
```

### Local RoI Branch
```bash
cd train_local_roi
torchrun --nnodes=1 --nproc_per_node=3 train_cvpr_ddp_interactive.py
```

## 🔍 Inference and Evaluation
```bash
# Load the docker image
docker load -i yiooo_coreset.tar.gz

# Run inference
docker container run --gpus "device=0" -m 32G --name yiooo_coreset --rm \
  -v $PWD/PathToTestSet/:/workspace/inputs/ \
  -v $PWD/yiooo_coreset_outputs/:/workspace/outputs/ \
  yiooo_coreset:latest /bin/bash -c "sh predict.sh"
```
## 📊 Results
Our method achieves the following performance on the challenge coreset.

| Modality   | DSC AUC | NSD AUC | DSC Final | NSD Final |
| ---------- | :-----: | :-----: | :-------: | :-------: |
| CT         | 3.3461  | 3.4719  |  0.8462   |  0.8797   |
| MRI        | 2.7133  | 3.0852  |  0.6809   |  0.7714   |
| Microscopy | 2.2917  | 3.0618  |  0.5871   |  0.7743   |
| PET        | 3.0188  | 2.8778  |  0.7691   |  0.7440   |
| Ultrasound | 3.6741  | 3.7096  |  0.9299   |  0.9440   |

## 🙏 Acknowledgements
We sincerely thank the competition organizers for providing this valuable research platform. We also acknowledge the excellent work and open-source contributions from:
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D/tree/CVPR25_3DFM)
- [VISTA3D](https://github.com/Project-MONAI/VISTA/tree/main/vista3d/cvpr_workshop)

## 📝 Citation
If you find our work useful, please consider citing our paper:
```
@inproceedings{zhang2025rethinking,
  title={Rethinking RoI Strategy in Interactive 3D Segmentation for Medical Images},
  author={Zhang, Ziyu and Yu, Yi and Xue, Yuan},
  booktitle={CVPR Workshop on Foundation Models for Medical Vision},
  year={2025}
}
```
