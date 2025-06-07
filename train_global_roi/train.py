# set up environment
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
from os.path import join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from click_method import get_next_click3D_gaussian
from data_loader import SegFM3D_Dataset, Union_Dataloader
from sam_med3d_64 import SAM_Med3D64, ImageEncoderViT3D, PromptEncoder3D, MaskDecoder3D
from medim.models._pretrain import load_pretrained_weights
import cv2
from functools import partial


logger = logging.getLogger(__name__)


def show_clicks(prev_masks, gt3D, points_input, labels_input):
    prev_masks = prev_masks[0, 0].detach().cpu().numpy()
    gt3D = gt3D[0, 0].detach().cpu().numpy()
    points_input = points_input[0].detach().cpu().numpy()
    labels_input = labels_input[0].detach().cpu().numpy()
    images = []
    for point, label in zip(points_input, labels_input):
        pd2d = prev_masks[round(point[0])]
        gt2d = gt3D[round(point[0])]
        img = np.stack((gt2d * 0.5, gt2d * 0.5, gt2d * 0.5 + pd2d * 0.5), -1)
        img[round(point[1]), round(point[2]), 0] = 1
        img[round(point[1]), round(point[2]), 2] = label
        images.append(img)
        images.append(np.ones((img.shape[0], 1, 3)))
    images = np.concatenate(images, 1)
    return images


def show_clicks_all(images):
    images = np.concatenate(images, 0)
    cv2.imwrite(f'debug/1.png', images * 255)


class SAM_Med3D64_Train(SAM_Med3D64):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def mask_forward(self, image_embedding, prev_masks, boxes=None, points=None):
        low_res_masks = prev_masks[..., ::2, ::2, ::2]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            points=points
        )
        prev_masks = F.interpolate(low_res_masks, size=(64, 64, 64), mode='trilinear', align_corners=False)
        return prev_masks

    def interaction(self, image_embedding, gt3D, boxes, num_clicks):
        prev_masks = torch.zeros_like(gt3D).float().to(gt3D.device)
        init_points = torch.tensor((((31, 31, 31),),)).expand(gt3D.shape[0], 1, 3).to(gt3D.device)
        init_labels = torch.tensor(((1,),)).expand(gt3D.shape[0], 1).to(gt3D.device)
        prev_masks = self.mask_forward(image_embedding, prev_masks, points=[init_points, init_labels])
        return_loss = self.seg_loss(prev_masks, gt3D)
        prev_masks = prev_masks.sigmoid().round().detach()
        prev_masks_init = prev_masks.clone()

        # images = []
        num_randoms = 6
        for num_click in range(num_clicks):
            points_input, labels_input = get_next_click3D_gaussian(prev_masks, gt3D, num_randoms)
            batch_points_input = points_input.view(-1, 1, 3)
            batch_labels_input = labels_input.view(-1, 1)
            # images.append(show_clicks(prev_masks, gt3D, points_input, labels_input))
            # images.append(np.ones((1, images[-1].shape[1], 3)))

            batch_image_embedding = image_embedding.repeat_interleave(num_randoms, dim=0)
            batch_prev_masks = prev_masks.repeat_interleave(num_randoms, dim=0)
            batch_gt3D = gt3D.repeat_interleave(num_randoms, dim=0)

            batch_points_input = torch.cat((init_points.repeat_interleave(num_randoms, dim=0), batch_points_input), 1)
            batch_labels_input = torch.cat((init_labels.repeat_interleave(num_randoms, dim=0), batch_labels_input), 1)
            batch_prev_masks = self.mask_forward(batch_image_embedding, batch_prev_masks, points=[batch_points_input, batch_labels_input])
            return_loss += num_randoms * self.seg_loss(batch_prev_masks, batch_gt3D)
            prev_masks = batch_prev_masks[0::num_randoms].sigmoid().round().detach()
            init_points = batch_points_input[0::num_randoms]
            init_labels = batch_labels_input[0::num_randoms]
        # show_clicks_all(images)

        return_loss /= num_clicks + 1
        return prev_masks_init, prev_masks, return_loss

    def forward(self, image3D, gt3D):
        image_embedding = self.image_encoder(image3D)
        prev_masks_init, prev_masks, loss = self.interaction(image_embedding, gt3D, None, num_clicks=5)
        return prev_masks_init, prev_masks, loss


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
) -> SAM_Med3D64_Train:
    image_embedding_size = image_size // vit_patch_size

    model = SAM_Med3D64_Train(
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


def build_model(args):
    sam_model = build_sam3D_vit_b_ori().to(args.device)
    sam_model = DDP(sam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    return sam_model


def get_dataloaders(args):
    transforms = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, label_interpolation='linear'),
        tio.RandomAffine(scales=(0.2, 0.5, 0.5), degrees=0, translation=(4, 4, 4), isotropic=False, default_pad_value=0, label_interpolation='linear'),
        tio.transforms.OneOf([
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
            tio.RandomBlur(std=(0, 1)),
            tio.RandomNoise(std=(0, 0.25))]),
        tio.ZNormalization(masking_method=partial(torch.gt, other=0)),
    ])
    train_dataset = SegFM3D_Dataset(paths=args.data_path, transform=transforms, threshold=0)

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint()
        
    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.module.image_encoder.parameters(), 'lr': self.args.lr},
            {'params': self.model.module.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': self.model.module.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                  lr_lambda=lambda step: min(1.0, (step + 1) / 50))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            self.args.step_size,
                                                            self.args.gamma)

    def init_checkpoint(self):
        if args.resume:
            ckp_path = join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth')
        else:
            ckp_path = self.args.checkpoint
        
        last_ckpt = None
        if os.path.exists(ckp_path):
            dist.barrier()
            last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
        
        if last_ckpt:
            model_state = self.model.module.state_dict()
            filtered_state = {k: v for k, v in last_ckpt['model_state_dict'].items()
                if k in model_state and v.shape == model_state[k].shape}
            self.model.module.load_state_dict(filtered_state, strict=False)
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
            logger.info(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")
            logger.info(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": args.data_path,
        }, join(self.args.work_dir, self.args.task_name, f"sam_model_{describe}.pth"))
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                print('dice calculation: NaN')
                return volume_sum
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item() 

    def train_epoch(self, epoch):
        self.model.train()
        tbar = tqdm(self.dataloaders) if self.args.rank == 0 else self.dataloaders
        self.optimizer.zero_grad()
        epoch_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            acc_context = self.model.no_sync if (step + 1) % self.args.accumulation_steps != 0 else nullcontext
            with acc_context():
                image3D = data3D["image"].to(self.args.device)
                gt3D = data3D["label"].to(self.args.device).round().long()
                
                if torch.isnan(image3D).any() or torch.isinf(image3D).any() or (image3D <= 0).all():
                    print('Invalid detected:', torch.isnan(image3D).any(), torch.isinf(image3D).any(), (image3D <= 0).all(), flush=True)
                    image3D.zero_()
                    gt3D.zero_()

                with torch.amp.autocast('cuda'):
                    prev_masks_init, prev_masks, loss = self.model(image3D, gt3D)

                self.scaler.scale(loss / self.args.accumulation_steps).backward()

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)

            # image = image3D[0, 0].detach().cpu().numpy()
            # segs = prev_masks[0, 0].detach().cpu().numpy()
            # gts_cls = gt3Df[0, 0].detach().cpu().numpy()
            # image = image.mean(0)
            # vis = np.concatenate(((image - image.min()) / (image.max() - image.min()) * 255, 
            #                     np.ones((segs.shape[1], 1)) * 255,
            #                     segs.max(0) * 255, 
            #                     np.ones((segs.shape[1], 1)) * 255,
            #                     gts_cls.max(0) * 255), 1)
            # import cv2
            # cv2.imwrite(f'debug/{step}.png', vis)            

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.args.rank == 0:
                    print_loss = loss.item()
                    print_dice0 = self.get_dice_score(prev_masks_init, gt3D)
                    print_dice1 = self.get_dice_score(prev_masks, gt3D)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print_str = f'Epoch: {epoch}, Step: {step}, LR: {current_lr:.1e}, Loss: {print_loss:.4f}, Dice: {print_dice0:.4f} | {print_dice1:.4f}'
                    print(print_str, flush=True)
                    logger.info(print_str)
            
                if epoch == 0:
                    self.warmup_scheduler.step()

        epoch_loss /= len(tbar)
        epoch_dice /= len(tbar)
        return epoch_loss, epoch_dice

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(self.args.work_dir, self.args.task_name, f'{save_name}.png'))
        plt.close()

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            dist.barrier()
            self.dataloaders.sampler.set_epoch(epoch)
            epoch_loss, epoch_dice = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            dist.barrier()
        
            if self.args.rank == 0:
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                state_dict = self.model.module.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')
                
                # save train dice best checkpoint
                if epoch_dice > self.best_dice: 
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {args.data_path}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        

def main(args):
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}")
    
    print(f"MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, RANK={args.rank}, LOCAL_RANK={args.local_rank}, WORLD_SIZE={args.world_size}")
    dist.init_process_group(backend='nccl')

    init_seeds(2023 + args.local_rank)

    os.makedirs(join(args.work_dir, args.task_name), exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(args.work_dir, args.task_name, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='train-full')
    parser.add_argument('--model_type', type=str, default='vit_b_ori')
    parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d_turbo_bbox_cvpr.pth')
    parser.add_argument('--work_dir', type=str, default='work_dir')
    
    # train
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true', default=False)
    
    # lr_scheduler
    parser.add_argument('--step_size', type=list, default=[98, 132])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=144)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.data_path = [
        '/fs/CVPR-BiomedSegFM/preprocessed_train'
        ]
    
    main(args)
