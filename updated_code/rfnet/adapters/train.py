#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from split_protocol import (add_split_arguments, resolve_split_files, record_split_manifest,
                            internal_validation, save_best_checkpoint)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--datapath', default='/scratch_net/maus/mrostamian/processed_BRATS2023', type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default='/scratch_net/maus/mrostamian/project/RFNet_main_org/checkpoints_miss_modality_v6_BSA', type=str)
# parser.add_argument('--resume', default='/scratch_net/maus/mrostamian/project/RFNet-main_org/checkpoints_adapter_scaling/model_last.pth', type=str)
# parser.add_argument('--resume', default='/scratch_net/maus/mrostamian/project/RFNet_main_org/checkpoints_org/model_300_adapter.pth', type=str)
# parser.add_argument('--resume', default='/scratch_net/maus/mrostamian/project/RFNet_main_org/checkpoints_adapter_v1/model_last.pth', type=str)
parser.add_argument('--resume', default=None, type=str)


# parser.add_argument('--resume', default=None, type=str)
# parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=500, type=int)
parser.add_argument('--region_fusion_start_epoch', default=100, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)





def print_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}\nTrainable params: {trainable:,}")



## parse arguments
add_split_arguments(parser)
args = parser.parse_args()
split_files = resolve_split_files(args)
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

# ###modality missing mask
masks = [[True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['flairt1cet1t2']
print (masks_torch.int())


###modality missing mask
# masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]]
# masks_torch = torch.from_numpy(np.array(masks))
# mask_name = ['t2', 't1c', 't1', 'flair', 
#             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
#             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
#             'flairt1cet1t2']



# --- Tracking Callback for Scaling Layers ---
def print_scaling_parameters(model):
    print("Scaling parameters:")
    for name, module in model.named_modules():
        if isinstance(module, models.ScalingLayer3d):  # Adjust if ScalingLayer3d is imported from models
            print(f"  {name}: alpha = {module.alpha.item():.4f}, beta = {module.beta.item():.4f}")

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018', 'BRATS2023']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    print (model)

    print_params(model)
    
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    # train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    # optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ########## Setting Optimizer with Conditional L2 ##########
    wd_params, no_wd_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1) No weight‑decay on biases
        if name.endswith('.bias'):
            no_wd_params.append(p)
        # 2) No weight‑decay on norm-layer weights
        elif 'bn' in name.lower() or 'norm' in name.lower():
            no_wd_params.append(p)
        else:
            # everything else (conv, linear, scaling_layers.alpha/beta, etc.) gets decay
            wd_params.append(p)

    optimizer = torch.optim.Adam([
        { 'params': wd_params,    'weight_decay': args.weight_decay },
        { 'params': no_wd_params, 'weight_decay': 0.0 }
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)


    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015', 'BRATS2023']:
        # train_file = 'train_full.txt'
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'

    train_file = split_files['train']
    test_file = split_files['test'] if args.evaluate_test else split_files['val']
    split_manifest = record_split_manifest(split_files, args.savepath, args.seed)

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax(
                                test_loader,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask,
                                mask_name = mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
            exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training org############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    best_validation_score = -float('inf')
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))
        
        # --- Track and print scaling parameters from ScalingLayer3d ---
        print_scaling_parameters(model)

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.num_epochs:
            validation_score = internal_validation(model, test_loader, test_softmax, args, masks)
            best_validation_score = save_best_checkpoint(
                validation_score, best_validation_score, model, optimizer, epoch, ckpts, split_manifest)
            logging.info('Internal-validation mean Dice: %.6f; best: %.6f',
                         validation_score, best_validation_score)

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
