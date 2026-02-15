import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloader.promise12_dataset import Promise12Dataset
from torch.utils.data import DataLoader
from dataloader.transforms import build_weak_strong_transforms
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler

from trainer_cotraining import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./PROMISE12/processed',
                    help='Path to the processed PROMISE12 dataset')
parser.add_argument('--labeled_num', type=int, default=28, #5% label
                    help='Number of labeled cases')
parser.add_argument('--dataset', type=str, default='PROMISE12',
                    help='Dataset name')

# PROMISE12 is binary segmentation (background + prostate)
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of classes (background + prostate)')
parser.add_argument('--in_channels', type=int, default=3,
                    help='Number of input channels (grayscale)')

parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-UNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--image_size', type=int, default=256, help='image_size')
parser.add_argument('--point_nums', type=int, default=5, help='points number')
parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
parser.add_argument('-thd', type=bool, default=False, help='3d or not')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')

parser.add_argument('--mixed_iterations', type=int, default=10000,
                    help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=40000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--multimask", type=bool, default=False, help="output multimask")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="sam checkpoint")
parser.add_argument("--unet_checkpoint", type=str, default="", help="unet checkpoint")
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

parser.add_argument("--mode", type=str, default="train", help="train or test")


args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)



def train(args, snapshot_path):
    max_iterations = args.max_iterations
    
    # Initialize trainer
    trainer = Trainer(args)

    data_transforms = build_weak_strong_transforms(args)
    train_dataset = Promise12Dataset(args=args, base_dir=args.data_path, split="train", transform=data_transforms)
    val_dataset = Promise12Dataset(args=args, base_dir=args.data_path, split="val", transform=data_transforms["valid_test"])

    total_slices = len(train_dataset)
    labeled_count = args.labeled_num

    labeled_idxs = list(range(0, labeled_count))
    unlabeled_idxs = list(range(labeled_count, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} iterations per epoch".format(len(train_loader)))
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            trainer.train(volume_batch, label_batch, iter_num)
            iter_num = iter_num + 1
            
            if iter_num > 0 and iter_num % 200 == 0:
                trainer.val(val_loader, snapshot_path, iter_num)


    trainer.val(val_loader, snapshot_path, iter_num)
    
def test(args):
    
    trainer = Trainer(args)
    
    data_transforms = build_weak_strong_transforms(args)
    test_dataset = Promise12Dataset(args=args, base_dir=args.data_path, split="test", transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    trainer.test(test_loader)


if __name__ == '__main__':
    if args.mode == "train":
        for fold in range(args.n_fold):
            torch.autograd.set_detect_anomaly(True)
            random.seed(2024)
            np.random.seed(2024)
            torch.manual_seed(2024)
            torch.cuda.manual_seed(2024)

            snapshot_path = f"./Outputs/results_PROMISE_{args.labeled_num}/fold_{fold}"
            os.makedirs(snapshot_path, exist_ok=True)

            logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            logging.info(str(args))

            train(args, snapshot_path)
    else:
        test(args)