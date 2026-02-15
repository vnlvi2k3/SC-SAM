import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.sam.build_sam import sam_model_registry
import torch.optim as optim
from utils.losses import KDLoss, DiceLoss
import logging
from utils.utils import dice_coef, eval

import numpy as np
import matplotlib.pyplot as plt

import random

from Model.model import SamUnet
from prediction import sample_points_from_mask

ce_loss = torch.nn.CrossEntropyLoss()

GPUdevice = torch.device('cuda', 0)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def sigmoid_rampup(current, rampup_length=10000):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.criterion_mse = nn.MSELoss()
        self.KDLoss = KDLoss(T=10)
        self.dice_loss = DiceLoss(args.num_classes)

        self.sam_model = sam_model_registry[args.model_type](args=args, checkpoint=args.sam_checkpoint).to(args.device).train()
        self.Unet = SamUnet(args).cuda().train()

        self.optimizer_sam = optim.Adam(self.sam_model.parameters(), lr=args.lr)
        self.optimizer_Unet = torch.optim.SGD(self.Unet.parameters(), lr=args.UNet_lr, momentum=0.9,
                                              weight_decay=0.0001)

        self.best_performance_sam = 0.0
        self.best_performance_Unet = 0.0

        for n, value in self.sam_model.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            elif "super_prompt" in n:
                value.requires_grad = True
            elif "prompt_encoder" in n:
                value.requires_grad = True
            elif "mask_decoder" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def entropy_loss(self, p, C=2):
        # p N*C*W*H*D
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
             torch.tensor(np.log(C)).cuda()
        ent = torch.mean(y1)
        return ent

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)


    def train(self, volume_batch, label_batch, iter_num):
        image_embeddings = self.sam_model.image_encoder(volume_batch)
        pred_UNet, pred_UNet_soft = self.Unet(volume_batch)
      
        points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)
        low_res_masks_all = torch.empty((self.args.batch_size, 0, int(self.args.image_size/4), int(self.args.image_size/4)), device=self.args.device)
        pred_classes = torch.argmax(pred_UNet_soft, dim=1)
        for i in range(self.args.num_classes):
            mask_class_i = (pred_classes == i).float()  
            sampled_points = sample_points_from_mask(mask_class_i, num_points=5)

            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=sampled_points,
                boxes=boxes_embedding[i],
                masks=None
            )

            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )

            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

        pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size), mode="bilinear", align_corners=False)
        pred_sam_soft = torch.softmax(pred_sam, dim=1)

        sam_pseudo_label = torch.argmax(pred_sam_soft, dim=1).detach()
        unet_pseudo_label = torch.argmax(pred_UNet_soft, dim=1).detach()

        # losses

        UNet_sup_loss = ce_loss(pred_UNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_UNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        UNet_enp_loss = self.entropy_loss(pred_UNet_soft, C=2)
        UNet_unsup_loss = ce_loss(pred_UNet[self.args.labeled_bs:], sam_pseudo_label[self.args.labeled_bs:].long()) + self.dice_loss(pred_UNet_soft[self.args.labeled_bs:], sam_pseudo_label[self.args.labeled_bs:]) 

        sam_sup_loss = ce_loss(pred_sam[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_sam_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])

        UNet_loss = UNet_sup_loss + UNet_unsup_loss + 0.9 * UNet_enp_loss

        unsup_weight = self.sigmoid_rampup(iter_num, self.args.mixed_iterations)
        sam_unsup_loss = ce_loss(pred_sam[self.args.labeled_bs:], unet_pseudo_label[self.args.labeled_bs:].long()) + self.dice_loss(pred_sam_soft[self.args.labeled_bs:], unet_pseudo_label[self.args.labeled_bs:])
        sam_loss = sam_sup_loss + sam_unsup_loss * unsup_weight

        self.optimizer_sam.zero_grad()
        self.optimizer_Unet.zero_grad()

        sam_loss.backward()
        UNet_loss.backward()

        self.optimizer_sam.step()
        self.optimizer_Unet.step()

        lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        UNet_lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)

        for param_group in self.optimizer_sam.param_groups:
            param_group['lr'] = lr_
        for param_group in self.optimizer_Unet.param_groups:
            param_group['lr'] = UNet_lr_

        logging.info('iteration %d : '
                     '  sam_loss : %f'
                     '  sam_lr_ : %10f'
                     
                     '  Unet_loss : %f'
                     '  UNet_lr_ : %10f'

                     % (iter_num, sam_loss.item(), lr_,
                        UNet_loss.item(), UNet_lr_,
                        ))
        
    def val(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.Unet.eval()

        avg_dice_sam = 0.0
        avg_dice_unet = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image_embeddings = self.sam_model.image_encoder(val_image)
            pred_UNet, pred_UNet_soft = self.Unet(val_image)

            points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(self.args.image_size / 4), int(self.args.image_size / 4)),
                device=self.args.device)
            with torch.no_grad():
                pred_classes = torch.argmax(pred_UNet_soft, dim=1)
                for i in range(self.args.num_classes):
                    mask_class_i = (pred_classes == i).float()  
                    sampled_points = sample_points_from_mask(mask_class_i, num_points=5)
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=sampled_points,
                        boxes=boxes_embedding[i],
                        masks=None
                    )
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
            pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            dice_sam = dice_coef(val_label, pred_sam_soft, thr=0.5)
            avg_dice_sam += dice_sam

            dice_unet = dice_coef(val_label, pred_UNet_soft, thr=0.5)
            avg_dice_unet += dice_unet

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_unet = avg_dice_unet / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  unet_mean_dice : %f '
                    % (iter_num, avg_dice_sam, avg_dice_unet))
        


        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)

        if avg_dice_unet > self.best_performance_Unet:
            self.best_performance_Unet = avg_dice_unet
            save_best_UNet = os.path.join(snapshot_path, 'Unet_best_model.pth')
            torch.save(self.Unet.state_dict(), save_best_UNet)
        self.sam_model.train()
        self.Unet.train()

    def test(self, val_loader):
        self.Unet.load_state_dict(torch.load(self.args.unet_checkpoint, map_location=self.args.device), strict=True)
        self.sam_model.eval()
        self.Unet.eval()

        avg_dice_sam = 0.0
        avg_iou_sam = 0.0
        avg_hd95_sam = 0.0
        avg_asd_sam = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image_embeddings = self.sam_model.image_encoder(val_image)
            pred_UNet, pred_UNet_soft = self.Unet(val_image)

            points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(self.args.image_size / 4), int(self.args.image_size / 4)),
                device=self.args.device)
            with torch.no_grad():
                pred_classes = torch.argmax(pred_UNet_soft, dim=1)
                for i in range(self.args.num_classes):
                    mask_class_i = (pred_classes == i).float()  
                    sampled_points = sample_points_from_mask(mask_class_i, num_points=5)
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=sampled_points,
                        boxes=boxes_embedding[i],
                        masks=None
                    )
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
            pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            metrics = eval(val_label, pred_sam_soft)
            avg_dice_sam += metrics[0]
            avg_iou_sam += metrics[1]
            avg_hd95_sam += metrics[2]
            avg_asd_sam += metrics[3]

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_iou_sam = avg_iou_sam / len(val_loader)
        avg_hd95_sam = avg_hd95_sam / len(val_loader)
        avg_asd_sam = avg_asd_sam / len(val_loader)

        print('  sam_mean_dice : %f ' % (avg_dice_sam))
        print('  sam_mean_iou : %f ' % (avg_iou_sam))
        print('  sam_mean_hd95 : %f ' % (avg_hd95_sam))
        print('  sam_mean_asd : %f ' % (avg_asd_sam))
