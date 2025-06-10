# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

ENABLE_SPECIAL = True
SPECIAL_INDEX = (23, 24, 25, 26, 27, 57, 128)
MERGE_LIST = {
    1: [25, 26],  # hepatic tumor and vessel merge into liver
    4: [24],  # pancreatic tumor merge into pancreas
    132: [57],  # overlap with trachea merge into airway
}

__all__ = ["sample_prompt_pairs"]


def _get_point_label(id: int) -> tuple[int, int]:
    if id in SPECIAL_INDEX and ENABLE_SPECIAL:
        return 2, 3
    else:
        return 0, 1


def sample_prompt_pairs(
    labels: Tensor,
    label_set: Sequence[int],
    max_prompt: int | None = None,
    max_foreprompt: int | None = None,
    max_backprompt: int = 1,
    max_point: int = 20,
    include_background: bool = False,
    drop_label_prob: float = 0.2,
    drop_point_prob: float = 0.2,
    point_sampler: Callable | None = None,
    **point_sampler_kwargs: Any,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """
    Sample training pairs for VISTA3D training.

    Args:
        labels: [1, 1, H, W, D], ground truth labels.
        label_set: the label list for the specific dataset. Note if 0 is included in label_set,
            it will be added into automatic branch training. Recommend removing 0 from label_set
            for multi-partially-labeled-dataset training, and adding 0 for finetuning specific dataset.
            The reason is region with 0 in one partially labeled dataset may contain foregrounds in
            another dataset.
        max_prompt: int, max number of total prompt, including foreground and background.
        max_foreprompt: int, max number of prompt from foreground.
        max_backprompt: int, max number of prompt from background.
        max_point: maximum number of points for each object.
        include_background: if include 0 into training prompt. If included, background 0 is treated
            the same as foreground and points will be sampled. Can be true only if user want to segment
            background 0 with point clicks, otherwise always be false.
        drop_label_prob: probability to drop label prompt.
        drop_point_prob: probability to drop point prompt.
        point_sampler: sampler to augment masks with supervoxel.
        point_sampler_kwargs: arguments for point_sampler.

    Returns:
        tuple:
            - label_prompt (Tensor | None): Tensor of shape [B, 1] containing the classes used for
              training automatic segmentation.
            - point (Tensor | None): Tensor of shape [B, N, 3] representing the corresponding points
              for each class. Note that background label prompts require matching points as well
              (e.g., [0, 0, 0] is used).
            - point_label (Tensor | None): Tensor of shape [B, N] representing the corresponding point
              labels for each point (negative or positive). -1 is used for padding the background
              label prompt and will be ignored.
            - prompt_class (Tensor | None): Tensor of shape [B, 1], exactly the same as label_prompt
              for label indexing during training. If label_prompt is None, prompt_class is used to
              identify point classes.

    """

    # class label number
    if not labels.shape[0] == 1:
        raise ValueError("only support batch size 1")
    labels = labels[0, 0]
    device = labels.device
    unique_labels = labels.unique().cpu().numpy().tolist()
    if include_background:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)))
    else:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)) - {0})
    background_labels = list(set(label_set) - set(unique_labels))
    # during training, balance background and foreground prompts
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            background_labels = background_labels[:max_backprompt]

    if max_foreprompt is not None:
        if len(unique_labels) > max_foreprompt:
            random.shuffle(unique_labels)
            unique_labels = unique_labels[:max_foreprompt]

    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                unique_labels = random.sample(unique_labels, max_prompt)
                background_labels = []
            else:
                background_labels = random.sample(background_labels, max_prompt - len(unique_labels))
    _point = []
    _point_label = []
    # if use regular sampling
    if point_sampler is None:
        num_p = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
        num_n = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
        for id in unique_labels:
            neg_id, pos_id = _get_point_label(id)
            plabels = labels == int(id)
            nlabels = ~plabels
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
            # final sampled positive points
            num_pa = min(len(plabelpoints), num_p)
            # final sampled negative points
            num_na = min(len(nlabelpoints), num_n)
            _point.append(
                torch.stack(
                    random.choices(plabelpoints, k=num_pa)
                    + random.choices(nlabelpoints, k=num_na)
                    + [torch.tensor([0, 0, 0], device=device)] * (num_p + num_n - num_pa - num_na)
                )
            )
            _point_label.append(
                torch.tensor([pos_id] * num_pa + [neg_id] * num_na + [-1] * (num_p + num_n - num_pa - num_na)).to(
                    device
                )
            )
        for _ in background_labels:
            # pad the background labels
            _point.append(torch.zeros(num_p + num_n, 3).to(device))  # all 0
            _point_label.append(torch.zeros(num_p + num_n).to(device) - 1)  # -1 not a point
    else:
        _point, _point_label = point_sampler(unique_labels, **point_sampler_kwargs)
        for _ in background_labels:
            # pad the background labels
            _point.append(torch.zeros(len(_point_label[0]), 3).to(device))  # all 0
            _point_label.append(torch.zeros(len(_point_label[0])).to(device) - 1)  # -1 not a point
    if len(unique_labels) == 0 and len(background_labels) == 0:
        # if max_backprompt is 0 and len(unique_labels), there is no effective prompt and the iteration must
        # be skipped. Handle this in trainer.
        label_prompt, point, point_label, prompt_class = None, None, None, None
    else:
        label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
        point = torch.stack(_point)
        point_label = torch.stack(_point_label)
        prompt_class = copy.deepcopy(label_prompt)
        if random.uniform(0, 1) < drop_label_prob and len(unique_labels) > 0:
            label_prompt = None
            # If label prompt is dropped, there is no need to pad with points with label -1.
            pad = len(background_labels)
            point = point[: len(point) - pad]  # type: ignore
            point_label = point_label[: len(point_label) - pad]
            prompt_class = prompt_class[: len(prompt_class) - pad]
        else:
            if random.uniform(0, 1) < drop_point_prob:
                point = None
                point_label = None
    return label_prompt, point, point_label, prompt_class



from scipy.ndimage import distance_transform_edt 
import cc3d

def sample_coord(edt):
    # Find all coordinates with max EDT value
    np.random.seed(42)

    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)

    # Uniformly choose one of them
    chosen_index = max_coords[np.random.choice(len(max_coords))]

    center = tuple(chosen_index)
    return center

# Compute the EDT with same shape as the image
def compute_edt(error_component):
    # Get bounding box of the largest error component to limit computation
    coords = np.argwhere(error_component)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_shape = max_coords - min_coords

    # Compute padding (25% of crop size in each dimension)
    padding =  np.maximum((crop_shape * 0.25).astype(int), 1)


    # Define new padded shape
    padded_shape = crop_shape + 2 * padding

    # Create new empty array with padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)

    # Fill center region with actual cropped data
    center_crop[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    large_roi = False
    if center_crop.shape[0] * center_crop.shape[1] * center_crop.shape[2] > 60000000:
        from skimage.measure import block_reduce
        print(f'ROI too large {center_crop.shape} --> 2x downsampling for EDT')
        center_crop = block_reduce(center_crop, block_size=(2, 2, 2), func=np.max)
        large_roi = True

    # Compute EDT on the padded array
    if torch.cuda.is_available() and not large_roi: # GPU available
        import cupy as cp
        from cucim.core.operations import morphology
        error_mask_cp = cp.array(center_crop)
        edt_cp = morphology.distance_transform_edt(error_mask_cp, return_distances=True)
        edt = cp.asnumpy(edt_cp)
    else: # CPU available only
        edt = distance_transform_edt(center_crop)
    
    if large_roi: # upsample
        edt = edt.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)

    # Crop out the center (remove padding)
    dist_cropped = edt[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ]

    # Create full-sized EDT result array and splat back 
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ] = dist_cropped

    dist_transformed = dist_full

    return dist_transformed

def sample_prompt_point_interactive(
    labels: Tensor,
    pred_mask: Tensor = None,  
    label_set: Sequence[int] = None,
    max_prompt: int | None = None,
    max_foreprompt: int | None = None,
    max_backprompt: int = 1,
    max_point: int = 20,
    include_background: bool = False,
    drop_label_prob: float = 0.2,
    drop_point_prob: float = 0.2,
    point_sampler: Callable | None = None,
    use_error_guided: bool = True,  
    **point_sampler_kwargs: Any,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """
    Error region analysis-based prompt point sampling strategy
    
    Args:
        labels: [1, 1, H, W, D], ground truth labels
        pred_mask: [1, 1, H, W, D], model prediction mask, used for error region analysis if provided
        ...other parameters same as original function...
        use_error_guided: whether to use error-guided point sampling strategy
    """
    # Basic logic remains unchanged
    if not labels.shape[0] == 1:
        raise ValueError("only support batch size 1")
    labels = labels[0, 0]
    device = labels.device
    
    # When pred_mask is None, create an all-zero prediction mask
    if use_error_guided and pred_mask is None:
        pred_mask = torch.zeros_like(labels).unsqueeze(0).unsqueeze(0)
    
    unique_labels = labels.unique().cpu().numpy().tolist()
    if include_background:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)))
    else:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)) - {0})
    background_labels = list(set(label_set) - set(unique_labels))
    
    # Handle maximum prompt quantity limitations
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            background_labels = background_labels[:max_backprompt]

    if max_foreprompt is not None:
        if len(unique_labels) > max_foreprompt:
            random.shuffle(unique_labels)
            unique_labels = unique_labels[:max_foreprompt]

    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                unique_labels = random.sample(unique_labels, max_prompt)
                background_labels = []
            else:
                background_labels = random.sample(background_labels, max_prompt - len(unique_labels))
    
    _point = []
    _point_label = []
    
   # Use error-guided sampling strategy
    if use_error_guided:
        pred_mask = pred_mask.cpu().numpy()
        gt_mask = labels.cpu().numpy()
        
        for id in unique_labels:
            neg_id, pos_id = _get_point_label(id)
            
            # Extract class mask
            pred_cls = (pred_mask == int(id)).astype(np.uint8)
            gt_cls = (gt_mask == int(id)).astype(np.uint8)
            # Calculate error mask
            error_mask = (pred_cls != gt_cls).astype(np.uint8)
            
            # Sample foreground points (from error regions)
            p_points = []
            n_points = []
            
            if np.sum(error_mask) > 0:
                # Use connected component analysis to find the largest error region
                errors = cc3d.connected_components(error_mask, connectivity=26)
                
                # Calculate the size of each error region
                component_sizes = np.bincount(errors.flat)
                component_sizes[0] = 0  # Ignore non-error regions
                
                # Find the largest error component
                largest_component_error = np.argmax(component_sizes)
                # Find the voxel coordinates of the largest error component
                largest_component = (errors == largest_component_error)
                
                edt = compute_edt(largest_component)
                center = sample_coord(edt)
                # center = np.unravel_index(center_idx, edt.shape)
                
                # Determine if this is a foreground or background point based on GT mask
                if gt_cls[center] == 1:  # This should be foreground but was predicted as background
                    p_points.append(torch.tensor([center[0], center[1], center[2]]).to(device))
                else:  # This should be background but was predicted as foreground
                    n_points.append(torch.tensor([center[0], center[1], center[2]]).to(device))

            
            # If error region analysis didn't produce enough points, supplement with random points
            if use_error_guided:
                while len(p_points) + len(n_points) < 1:
                    p_points.append(torch.tensor([0, 0, 0]).to(device))
                
                # Build point set and labels
                all_points = p_points + n_points
                all_labels = [pos_id] * len(p_points) + [neg_id] * len(n_points)
                
                # Add -1 labels (for padding)
                all_labels += [-1] * (len(all_points) - len(all_labels))
                
                _point.append(torch.stack(all_points))
                _point_label.append(torch.tensor(all_labels).to(device))
            else:
                # Fall back to original random sampling
                num_p = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
                num_n = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
                
                plabels = labels == int(id)
                nlabels = ~plabels
                plabelpoints = torch.nonzero(plabels)
                nlabelpoints = torch.nonzero(nlabels)
                
                num_pa = min(len(plabelpoints), num_p)
                num_na = min(len(nlabelpoints), num_n)
                
                _point.append(
                    torch.stack(
                        random.choices(plabelpoints, k=num_pa)
                        + random.choices(nlabelpoints, k=num_na)
                        + [torch.tensor([0, 0, 0], device=device)] * (num_p + num_n - num_pa - num_na)
                    )
                )
                _point_label.append(
                    torch.tensor([pos_id] * num_pa + [neg_id] * num_na + [-1] * (num_p + num_n - num_pa - num_na)).to(device)
                )
    elif point_sampler is not None:
        # Use custom point sampler
        _point, _point_label = point_sampler(unique_labels, **point_sampler_kwargs)
    else:
         # Use original random sampling
        num_p = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
        num_n = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
        
        for id in unique_labels:
            neg_id, pos_id = _get_point_label(id)
            plabels = labels == int(id)
            nlabels = ~plabels
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
            
            num_pa = min(len(plabelpoints), num_p)
            num_na = min(len(nlabelpoints), num_n)
            
            _point.append(
                torch.stack(
                    random.choices(plabelpoints, k=num_pa)
                    + random.choices(nlabelpoints, k=num_na)
                    + [torch.tensor([0, 0, 0], device=device)] * (num_p + num_n - num_pa - num_na)
                )
            )
            _point_label.append(
                torch.tensor([pos_id] * num_pa + [neg_id] * num_na + [-1] * (num_p + num_n - num_pa - num_na)).to(device)
            )
    
    # Handle background labels
    for _ in background_labels:
        if _point and _point_label:
           # Get the shape of the first point set to determine padding size
            pad_shape = _point[0].shape
            _point.append(torch.zeros(pad_shape[0], 3).to(device))  #  All zeros
            _point_label.append(torch.zeros(pad_shape[0]).to(device) - 1)  #  -1 is not a point
    
    # return values
    if len(unique_labels) == 0 and len(background_labels) == 0:
        print(f'no valid label')
        label_prompt, point, point_label, prompt_class = None, None, None, None
    else:
        label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
        point = torch.stack(_point)
        point_label = torch.stack(_point_label)
        prompt_class = copy.deepcopy(label_prompt)
        
        # Randomly drop labels or points
        if random.uniform(0, 1) < drop_label_prob and len(unique_labels) > 0:
            label_prompt = None
            # If label prompts are dropped, no need to pad with points with label -1
            pad = len(background_labels)
            point = point[: len(point) - pad]
            point_label = point_label[: len(point_label) - pad]
            prompt_class = prompt_class[: len(prompt_class) - pad]
        else:
            if random.uniform(0, 1) < drop_point_prob:
                point = None
                point_label = None
    
    return label_prompt, point, point_label, prompt_class
