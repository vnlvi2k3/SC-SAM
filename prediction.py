from medpy import metric
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn.functional as F


def sample_points_from_mask(mask, num_points=5):
    B, H, W = mask.shape
    coords_tensor = torch.full((B, 2 * num_points, 2), -1, device=mask.device, dtype=torch.int)
    labels_tensor = torch.full((B, 2 * num_points), -1, device=mask.device, dtype=torch.int)

    for b in range(B):
        pos_coords = (mask[b] > 0).nonzero(as_tuple=False)  # foreground
        neg_coords = (mask[b] == 0).nonzero(as_tuple=False)  # background

        # ----- POSITIVE -----
        Np = pos_coords.size(0)
        if Np >= num_points:
            idx = torch.randperm(Np, device=mask.device)[:num_points]
            pos_sampled = pos_coords[idx]
            pos_labels = torch.ones(num_points, device=mask.device, dtype=torch.int)
        else:
            pos_sampled = pos_coords
            pos_labels = torch.ones(Np, device=mask.device, dtype=torch.int)
            # pad
            pad_coords = torch.full((num_points - Np, 2), -1, device=mask.device, dtype=torch.int)
            pad_labels = torch.full((num_points - Np,), -1, device=mask.device, dtype=torch.int)
            pos_sampled = torch.cat([pos_sampled, pad_coords], dim=0)
            pos_labels = torch.cat([pos_labels, pad_labels], dim=0)

        # ----- NEGATIVE -----
        Nn = neg_coords.size(0)
        if Nn >= num_points:
            idx = torch.randperm(Nn, device=mask.device)[:num_points]
            neg_sampled = neg_coords[idx]
            neg_labels = torch.zeros(num_points, device=mask.device, dtype=torch.int)
        else:
            neg_sampled = neg_coords
            neg_labels = torch.zeros(Nn, device=mask.device, dtype=torch.int)
            # pad
            pad_coords = torch.full((num_points - Nn, 2), -1, device=mask.device, dtype=torch.int)
            pad_labels = torch.full((num_points - Nn,), -1, device=mask.device, dtype=torch.int)
            neg_sampled = torch.cat([neg_sampled, pad_coords], dim=0)
            neg_labels = torch.cat([neg_labels, pad_labels], dim=0)

        # positive + negative
        coords_tensor[b] = torch.cat([pos_sampled, neg_sampled], dim=0)
        labels_tensor[b] = torch.cat([pos_labels, neg_labels], dim=0)

    return coords_tensor, labels_tensor


def sample_bbox_from_mask(mask):
    """
    [x_min, y_min, x_max, y_max].

    Returns:
        bboxes: Tensor shape (B, 4)
    """
    B, H, W = mask.shape
    bboxes = torch.zeros((B, 4), device=mask.device, dtype=torch.int)

    for b in range(B):
        coords = (mask[b] > 0).nonzero(as_tuple=False)  # [N,2], 
        if coords.size(0) == 0:
            bboxes[b] = torch.tensor([-1, -1, -1, -1], device=mask.device, dtype=torch.int)
        else:
            y_min = coords[:, 0].min()
            x_min = coords[:, 1].min()
            y_max = coords[:, 0].max()
            x_max = coords[:, 1].max()
            bboxes[b] = torch.tensor([x_min, y_min, x_max, y_max], device=mask.device, dtype=torch.int)

    return bboxes


def getLargestCC(segmentation):
    from skimage.measure import label
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC

def calculate_metric_percase(sam_pred, SGDL_pred, gt):
    sam_pred[sam_pred > 0] = 1
    SGDL_pred[SGDL_pred > 0] = 1
    gt[gt > 0] = 1
    dice_res = []
    if sam_pred.sum() > 0:
        dice_res.append(metric.binary.dc(sam_pred, gt))
    else:
        dice_res.append(0)

    if SGDL_pred.sum() > 0:
        dice_res.append(metric.binary.dc(SGDL_pred, gt))
    else:
        dice_res.append(0)

    return dice_res

def calculate_metric_sam(sam_pred, gt):
    sam_pred[sam_pred > 0] = 1
    gt[gt > 0] = 1
    dice_res = []
    if sam_pred.sum() > 0:
        dice_res.append(metric.binary.dc(sam_pred, gt))
    else:
        dice_res.append(0)

    return dice_res

def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map
