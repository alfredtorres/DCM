import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import numpy as np
from glob import glob
from collections import defaultdict
import torch
import torchio as tio
import cv2
import multiprocessing


def roi_extractor(image, segs, center, size):
    label = np.zeros_like(segs)
    label[round(center[0]), round(center[1]), round(center[2])] = 1
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image[None]),
        segs=tio.LabelMap(tensor=segs[None]),
        label=tio.LabelMap(tensor=label[None]))
    
    infer_transform = tio.Compose([
        tio.CropOrPad(mask_name='label', target_shape=size.clip(8)),
        tio.Resize((64, 64, 64), label_interpolation='linear'),
    ])
    subject_roi = infer_transform(subject)
    return subject_roi
    

def unified_preprocess(img_file, gts_file, out_file):
    print(img_file, flush=True)
    # if 'CT_AbdomenAtlas' in img_file:
    #     return

    img_data = np.load(img_file, allow_pickle=True)
    imgs = img_data.get('imgs', None).astype(np.float32)
    gts_data = np.load(gts_file, allow_pickle=True)
    gts = gts_data.get('gts', None)
    spacing = img_data.get('spacing', None)
    
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=imgs[None]), 
        label=tio.LabelMap(tensor=gts[None]))
    
    bboxes_3d = []
    gts = subject['label'].data[0].numpy()
    for cls in sorted(np.unique(gts)[1:]):
        z_indices = np.where(gts == cls)[0]
        z0, z1 = np.min(z_indices), np.max(z_indices)
        z_indices = np.unique(z_indices)
        z_mid = z_indices[len(z_indices)//2]
        coords = np.argwhere((gts == cls)[z_mid])
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        y0, y1 = min_coords[0], max_coords[0]
        x0, x1 = min_coords[1], max_coords[1]
        D, H, W = gts.shape
        bbox_shift_x = int(2.5 * W / 256)
        bbox_shift_y = int(2.5 * H / 256)
        x0 = max(0, x0 - bbox_shift_x)
        x1 = min(W - 1, x1 + bbox_shift_x)
        y0 = max(0, y0 - bbox_shift_y)
        y1 = min(H - 1, y1 + bbox_shift_y)
        bboxes_3d.append([z0, z1, y0, y1, x0, x1])

    if len(bboxes_3d) == 0:
        return
    bboxes_3d = np.float32(bboxes_3d)
    points_3d = (bboxes_3d[:, 0::2] + bboxes_3d[:, 1::2]) / 2
    
    img = subject['image'].data[0].numpy()
    gts = subject['label'].data[0].numpy()
    for cls, box, pnt in zip(sorted(np.unique(gts)[1:]), bboxes_3d, points_3d):
        gts_cls = (gts == cls).astype(np.uint8)
        box = box.reshape(3, 2) - pnt[:, None]
        subject_roi = roi_extractor(img, gts_cls.astype(float), pnt, np.round((box[:, 1] - box[:, 0] + 1) * 1.8))
        np.savez_compressed(f'{out_file[:-4]}_{cls}.npz',
                            imgs=subject_roi['image'].data[0].numpy(),
                            gts=subject_roi['segs'].data[0].numpy())
        
        # image = subject_roi['image'].data[0].numpy()
        # segs = subject_roi['segs'].data[0].numpy()
        # image1 = image.mean(0)
        # image2 = image.mean(1)
        # vis = np.concatenate(((image1 - image1.min()) / (image1.max() - image1.min()) * 255, 
        #                       np.ones((segs.shape[1], 1)) * 255,
        #                       segs.max(0) * 255,
        #                       np.ones((segs.shape[1], 1)) * 255,
        #                       (image2 - image2.min()) / (image2.max() - image2.min()) * 255, 
        #                       np.ones((segs.shape[1], 1)) * 255,
        #                       segs.max(1) * 255
        #                       ), 1)
        # cv2.imwrite(f'debug/{os.path.basename(out_file)[:-4]}_{cls}.png', vis)

        if cls >= 99:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('--img-path', default='/fs/CVPR-BiomedSegFM/3D_train_npz', type=str)
    parser.add_argument('--gts-path', default='/fs/CVPR-BiomedSegFM/3D_train_npz', type=str)
    parser.add_argument('--out-path', default='/fs/CVPR-BiomedSegFM/preprocessed_train', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    img_files = glob(os.path.join(args.img_path, '**/*.npz'), recursive=True)
    gts_files = []
    out_files = []
    for img_file in img_files:
        gts_file = img_file.replace(args.img_path, args.gts_path)
        out_file = os.path.join(args.out_path, os.path.basename(img_file))
        gts_files.append(gts_file)
        out_files.append(out_file)

    with multiprocessing.Pool(8) as p:
        p.starmap(unified_preprocess, zip(img_files, gts_files, out_files))


if __name__ == '__main__':
    main()
