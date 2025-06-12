# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Example code for inference with MedIM
'''

import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os
import cv2
from glob import glob
from collections import defaultdict
from functools import partial
from sam_med3d_64 import SAM_Med3D64, ImageEncoderViT3D, PromptEncoder3D, MaskDecoder3D
from medim.models._pretrain import load_pretrained_weights


def build_sam3D_vit_b_ori(
    pretrained: bool = False,
    checkpoint_path: str = '',
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_global_attn_indexes=[2, 5, 8, 11],
    prompt_embed_dim=384,
    image_size=64,
    vit_patch_size=8,
    **kwargs,
) -> SAM_Med3D64:
    image_embedding_size = image_size // vit_patch_size

    model = SAM_Med3D64(
        image_encoder=ImageEncoderViT3D(depth=encoder_depth,
                                        embed_dim=encoder_embed_dim,
                                        img_size=image_size,
                                        mlp_ratio=4,
                                        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                        num_heads=encoder_num_heads,
                                        patch_size=vit_patch_size,
                                        qkv_bias=True,
                                        use_rel_pos=True,
                                        global_attn_indexes=encoder_global_attn_indexes,
                                        window_size=14,
                                        out_chans=prompt_embed_dim,
                                        **kwargs),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size,
                                  image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=0,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    if pretrained:
        load_pretrained_weights(model,
                                checkpoint_path,
                                mode='torch',
                                state_dict_key='model_state_dict')
    return model


def read_data_from_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    imgs = data.get('imgs', None)

    spacing = data.get('spacing', None)
    spacing = [spacing[2], spacing[0], spacing[1]]
    imgs = imgs.astype(np.float32)

    boxes = data.get('boxes', None)

    # boxes buffer
    if os.path.exists('info_buffer.npy'):
        info_buffer = np.load('info_buffer.npy', allow_pickle=True).item()
    else:
        info_buffer = {}

    if boxes is not None:
        info_buffer[npz_file] = boxes
        np.save('info_buffer.npy', info_buffer)
    elif npz_file in info_buffer.keys():
        boxes = info_buffer[npz_file]
    else:
        raise Exception()
        prev_pred = data.get('prev_pred', np.zeros_like(imgs, dtype=np.uint8))
        boxes = []
        if prev_pred.sum() == 0:
            clicks = data.get('clicks', None)
            print(spacing)
            bsize = 64 / np.float32(spacing) * 1.5
            for cls_idx, cls_click_dict in enumerate(clicks):
                click = cls_click_dict['fg'][0]
                box = {'z_min': click[0] - bsize[0] / 2, 'z_max': click[0] + bsize[0] / 2,
                    'z_mid_y_min': click[1] - bsize[1] / 2, 'z_mid_y_max': click[1] + bsize[1] / 2,
                    'z_mid_x_min': click[2] - bsize[2] / 2, 'z_mid_x_max': click[2] + bsize[2] / 2}
                boxes.append(box)
        else:
            clicks = data.get('clicks', None)
            for cls in sorted(np.unique(prev_pred)[1:]):
                add_indices = np.round(np.float32(clicks[cls - 1]['fg'])).astype(int)
                prev_pred[add_indices[:, 0], add_indices[:, 1], add_indices[:, 2]] = cls
                z_indices = np.where(prev_pred == cls)[0]
                z0, z1 = np.min(z_indices), np.max(z_indices)
                z_mid = z_indices[len(z_indices)//2]
                coords = np.argwhere((prev_pred == cls)[z_mid])
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                y0, y1 = min_coords[0], max_coords[0]
                x0, x1 = min_coords[1], max_coords[1]
                D, H, W = prev_pred.shape
                bbox_shift_x = int(2.5 * W / 256)
                bbox_shift_y = int(2.5 * H / 256)
                x0 = max(0, x0 - bbox_shift_x)
                x1 = min(W - 1, x1 + bbox_shift_x)
                y0 = max(0, y0 - bbox_shift_y)
                y1 = min(H - 1, y1 + bbox_shift_y)
                box = {'z_min': z0, 'z_max': z1,
                    'z_mid_y_min': y0, 'z_mid_y_max': y1,
                    'z_mid_x_min': x0, 'z_mid_x_max': x1}
                boxes.append(box)
    
    # parsing boxes/clicks tensor, allow category to has more than 1 clicks
    all_clicks = defaultdict(list)
    for cls_idx, bbox in enumerate(boxes):
        all_clicks[cls_idx + 1].append(((
            bbox['z_min'],
            bbox['z_max'],
            bbox['z_mid_y_min'],
            bbox['z_mid_y_max'],
            bbox['z_mid_x_min'],
            bbox['z_mid_x_max']),
            1))

    prev_pred = data.get('prev_pred', np.zeros_like(imgs, dtype=np.uint8))
    clicks = data.get('clicks', None)
    if (clicks is not None):
        for cls_idx, cls_click_dict in enumerate(clicks):
            for click in cls_click_dict['fg']:
                all_clicks[cls_idx + 1].append((click, 1))
            for click in cls_click_dict['bg']:
                all_clicks[cls_idx + 1].append((click, 0))

    return imgs, spacing, all_clicks, prev_pred


def roi_extractor64(image, segs, center, size):
    label = np.zeros_like(segs)
    label[round(center[0]), round(center[1]), round(center[2])] = 1
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image[None]),
        segs=tio.LabelMap(tensor=segs[None]),
        label=tio.LabelMap(tensor=label[None]))
    
    target_shape = size.clip(8).astype(int).tolist()
    crop_transform = tio.CropOrPad(mask_name='label', target_shape=target_shape)
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.Resize((64, 64, 64)),
        tio.ZNormalization(masking_method=partial(torch.gt, other=0)),
    ])
    subject_roi = infer_transform(subject)

    roi_image = subject_roi.image.data.clone().detach()
    roi_segs = subject_roi.segs.data.clone().detach()
    offset_params = (
        cropping_params[0],
        cropping_params[0] + target_shape[0] - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + target_shape[1] - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + target_shape[2] - padding_params[4] - padding_params[5],
    )
    meta_info = {
        'target_shape': target_shape,
        'padding_params': padding_params,
        'offset_params': offset_params
    }
    return roi_image, roi_segs, meta_info


def data_postprocess(roi_pred, full_segs, meta_info):
    target_shape = meta_info['target_shape']
    roi_pred = F.interpolate(roi_pred,
                            size=target_shape,
                            mode='trilinear',
                            align_corners=False)
    roi_pred = roi_pred.cpu().numpy().squeeze()
    roi_pred = (roi_pred > 0.5).astype(np.uint8)

    pad_x1, pad_x2, pad_y1, pad_y2, pad_z1, pad_z2 = meta_info['padding_params']
    start_x, end_x, start_y, end_y, start_z, end_z = meta_info['offset_params']
    unpadded_pred = roi_pred[pad_x1:target_shape[0] - pad_x2,
                             pad_y1:target_shape[1] - pad_y2,
                             pad_z1:target_shape[2] - pad_z2]
    full_segs[start_x:end_x, start_y:end_y, start_z:end_z] = unpadded_pred
    return full_segs


def sam_model_infer(model, image, boxes_3d, click, label, prev_low_res_mask):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_tensor = image.to(device)
        image_embeddings = model.image_encoder(input_tensor)
        
        points_coords = torch.Tensor([click]).to(device)
        points_labels = torch.Tensor([label]).long().to(device)
        prev_low_res_mask = prev_low_res_mask[..., ::2, ::2, ::2].float().to(device)

        # print(prev_low_res_mask[0, 0, 15, ::4, ::4])
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
            points=[points_coords, points_labels],
        )

    return low_res_masks


def single_eval(model, image, segs, boxes_3d, click, label):
    seg_prob = sam_model_infer(model,
                           image[None],
                           boxes_3d=boxes_3d,
                           click=click,
                           label=label,
                           prev_low_res_mask=segs[None])
    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(seg_prob)  # (1, 1, 64, 64, 64)
    return medsam_seg_prob


def interactive_eval(model, image, all_clicks, prev_pred, casename):
    pred = np.zeros(image.shape, dtype=np.uint8)

    for cls, clicks in all_clicks.items():
        bbox = np.float32(clicks[0][0]).reshape(3, 2)
        cent = np.mean(bbox, 1)
        size = bbox[:, 1] - bbox[:, 0] + 1
        pnts = clicks[1:]
        labels = list(map(lambda x: x[1], [(cent, 1)] + pnts))
        pnts = np.float32(list(map(lambda x: x[0], [(cent, 1)] + pnts)))

        # print(bbox, cent, pnts, labels)
        segs = (prev_pred == cls).astype(np.uint8)
        roi_image, roi_segs, meta_info = roi_extractor64(image, segs, cent, np.round(size * 1.8))
        
        ratio = 64 / np.float32(meta_info['target_shape'])
        pnts = (pnts - cent) * ratio + 31

        roi_segs = single_eval(model, roi_image, roi_segs, bbox, pnts, labels)
        segs = data_postprocess(roi_segs, segs, meta_info)
        pred[segs != 0] = cls
            
    vis = np.concatenate((image.mean(0), 
                            np.ones((segs.shape[1], 1)) * 255,
                            segs.max(0) * 255), 1)
    cv2.imwrite(f'debug/{casename}-{cls}.png', vis)
    return pred


if __name__ == "__main__":
    try:
        npz_file = glob("inputs/*.npz")[0]
        out_dir = "./outputs"
        imgs, spacing, all_clicks, prev_pred = read_data_from_npz(npz_file)
        ckpt_path = "yiooo_model_full_latest.pth"
        model = build_sam3D_vit_b_ori(pretrained=True, checkpoint_path=ckpt_path)
        final_pred = interactive_eval(model, imgs, all_clicks, prev_pred, os.path.basename(npz_file))
        output_path = os.path.join(out_dir, os.path.basename(npz_file))
        np.savez_compressed(output_path, segs=final_pred)
        print("result saved to", output_path)

    except:
        os.system('python local_roi_expert.py --test_img_path inputs/ --save_path outputs/ --model yiooo_model_full_latest.pth')
