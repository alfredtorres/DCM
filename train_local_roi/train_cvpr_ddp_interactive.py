import json
import os
import sys
import warnings

import monai
import monai.transforms
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from sampler import sample_prompt_point_interactive
from monai.data import DataLoader
from monai.networks.nets import vista3d132
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.simplefilter("ignore")
# Custom dataset for .npz files

import matplotlib.pyplot as plt

NUM_PATCHES_PER_IMAGE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"


def plot_to_tensorboard(writer, epoch, inputs, labels, points, outputs):
    """
    Plots B figures, where each figure shows the slice where the point is located
    and overlays the point on this slice.

    Args:
        writer: TensorBoard writer
        epoch: Current epoch number
        inputs: Tensor [1, 1, H, W, D] - Input image
        labels: Tensor [1, 1, H, W, D] - Ground truth segmentation
        points: Tensor [B, N, 3] - Foreground object points (z, y, x)
        outputs: Tensor [B, 1, H, W, D] - Model outputs
    """
    B, N, _ = points.shape  # B objects, N click points per object
    inputs_np = inputs[0, 0].cpu().numpy()  # [H, W, D]
    labels_np = labels[0, 0].cpu().numpy()  # [H, W, D]

    for b in range(B):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Select the first click point in (z, y, x) format
        x, y, z = points[b, 0].cpu().numpy().astype(int)

        # Extract the corresponding slice
        input_slice = inputs_np[:, :, z]  # Get slice at depth z
        label_slice = labels_np[:, :, z]
        output_slice = outputs[b, 0].cpu().detach().numpy()[:, :, z] > 0

        # Plot input with point overlay
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].scatter(y, x, c="red", marker="x", s=50)
        axes[0].set_title(f"Input (Slice {z})")

        # Plot label
        axes[1].imshow(label_slice, cmap="gray")
        axes[0].scatter(y, x, c="red", marker="x", s=50)
        axes[1].set_title(f"Ground Truth (Slice {z})")

        # Plot output
        axes[2].imshow(output_slice, cmap="gray")
        axes[0].scatter(y, x, c="red", marker="x", s=50)
        axes[2].set_title(f"Model Output (Slice {z})")

        plt.tight_layout()

        # Log figure to TensorBoard
        writer.add_figure(f"Object_{b}_Segmentation", fig, epoch)
        plt.close(fig)


class NPZDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.file_paths = json.load(f)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx], allow_pickle=True)
        img_array = torch.from_numpy(img["imgs"]).unsqueeze(0).to(torch.float32)
        label = torch.from_numpy(img["gts"].astype(np.int32)).unsqueeze(0)
        data = {"image": img_array, "label": label, "filename": self.file_paths[idx]}
        affine = np.diag(img["spacing"].tolist() + [1])  # 4x4 affine matrix
        transforms = monai.transforms.Compose(
            [
                monai.transforms.ScaleIntensityRangePercentilesd(
                    keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True
                ),
                monai.transforms.SpatialPadd(
                    mode=["constant", "constant"],
                    keys=["image", "label"],
                    spatial_size=[128, 128, 128],
                ),
                monai.transforms.RandCropByLabelClassesd(
                    spatial_size=[128, 128, 128],
                    keys=["image", "label"],
                    label_key="label",
                    num_classes=label.max() + 1,
                    num_samples=NUM_PATCHES_PER_IMAGE,
                ),
                monai.transforms.RandScaleIntensityd(
                    factors=0.2, prob=0.2, keys="image"
                ),
                monai.transforms.RandShiftIntensityd(
                    offsets=0.2, prob=0.2, keys="image"
                ),
                monai.transforms.RandGaussianNoised(
                    mean=0.0, std=0.2, prob=0.2, keys="image"
                ),
                monai.transforms.RandFlipd(
                    spatial_axis=0, prob=0.2, keys=["image", "label"]
                ),
                monai.transforms.RandFlipd(
                    spatial_axis=1, prob=0.2, keys=["image", "label"]
                ),
                monai.transforms.RandFlipd(
                    spatial_axis=2, prob=0.2, keys=["image", "label"]
                ),
                monai.transforms.RandRotate90d(
                    max_k=3, prob=0.2, keys=["image", "label"]
                ),
            ]
        )
        data = transforms(data)
        return data


# Training function
def train():
    json_file = "train_subsetall.json"  # Update with your JSON file
    epoch_number = 20
    start_epoch = 0
    lr = 2e-7
    checkpoint_dir = "checkpoints_interactive_sample"
    start_checkpoint = "./checkpoints/CPRR25_vista3D_model_final_10percent_data.pth"

    os.makedirs(checkpoint_dir, exist_ok=True)
    dist.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dataset = NPZDataset(json_file)
    print(f'dataset length: {len(dataset)} ')
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank
    )
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=32)
    model = vista3d132(in_channels=1).to(device)
    pretrained_ckpt = torch.load(start_checkpoint, map_location=device)
    model.load_state_dict(pretrained_ckpt, strict=True)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-05)
    lr_scheduler = monai.optimizers.WarmupCosineSchedule(
        optimizer=optimizer,
        t_total=epoch_number + 1,
        warmup_multiplier=0.1,
        warmup_steps=0,
    )
    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "Events"))

    step = start_epoch * len(dataloader) * NUM_PATCHES_PER_IMAGE
    n_clicks = 5
    for epoch in range(start_epoch, epoch_number):
        sampler.set_epoch(epoch)
        for batch in tqdm(dataloader):
            image_l = batch["image"]
            label_l = batch["label"]
            for _k in range(image_l.shape[0]):
                inputs = image_l[[_k]].to(device)
                labels = label_l[[_k]].to(device)
                pred_mask= torch.zeros_like(labels[0,0], dtype=torch.int64)
                for click in range(n_clicks):
                    label_prompt, point, point_label, prompt_class = sample_prompt_point_interactive(
                        labels,
                        pred_mask,
                        list(set(labels.unique().tolist()) - {0}),
                        max_point=5,
                        max_prompt=10,
                        drop_label_prob=1,
                        drop_point_prob=0,
                    )
                    skip_update = torch.zeros(1, device=device)
                    if point is None:
                        print(
                            f"Iteration skipped due to None prompts at {batch['filename']}"
                        )
                        skip_update = torch.ones(1, device=device)
                    if world_size > 1:
                        dist.all_reduce(skip_update, op=dist.ReduceOp.SUM)
                    if skip_update[0] > 0:
                        continue  # some rank has no foreground, skip this batch
                    optimizer.zero_grad()
                    outputs = model(
                        input_images=inputs, point_coords=point, point_labels=point_label
                    )
                    if local_rank == 0 and step % 50 == 0:
                        plot_to_tensorboard(writer, step, inputs, labels, point, outputs)

                    loss, loss_n = torch.tensor(0.0, device=device), torch.tensor(
                        0.0, device=device
                    )
                    if prompt_class is not None:
                        for idx in range(len(prompt_class)):
                            if prompt_class[idx] == 0:
                                continue  # skip background class
                            loss_n += 1.0
                            gt = labels == prompt_class[idx]
                            loss += monai.losses.DiceCELoss(
                                include_background=False,
                                sigmoid=True,
                                smooth_dr=1.0e-05,
                                smooth_nr=0,
                                softmax=False,
                                squared_pred=True,
                                to_onehot_y=False,
                            )(outputs[[idx]].float(), gt.float())
                    loss /= max(loss_n, 1.0)
                    print(loss)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if local_rank == 0:
                        writer.add_scalar("loss", loss.item(), step)
        if local_rank == 0 and epoch % 1 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pth")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "step": step},
                checkpoint_path,
            )
            print(
                f"Rank {local_rank}, Epoch {epoch}, Loss: {loss.item()}, Checkpoint saved: {checkpoint_path}"
            )
        lr_scheduler.step()

    dist.destroy_process_group()



if __name__ == "__main__":
    train()
    # torchrun --nnodes=1 --nproc_per_node=3 train_cvpr.py
