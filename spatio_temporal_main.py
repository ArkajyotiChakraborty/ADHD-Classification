#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["RANK"] ="0"
os.environ['WORLD_SIZE'] = "1"
os.environ['LOCAL_RANK'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1111111'


# In[2]:


import os
import random
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import sys
import pickle
import itertools
import datetime
import time
import math
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import models.vision_transformer as vits
from models.vision_transformer import DINOHead
from models import build_model

from timm.data import create_transform
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing
from timm.data import Mixup

from config import config
from config import update_config
from config import save_config

from datasets import build_dataloader

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


# In[5]:


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'cvt_small', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'vil_14121', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (deit_tiny, deit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=768, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.95, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, nargs='+', default=(8,), help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--local_crops_size', type=int, nargs='+', default=(96,), help="""Crop region size of local views to generate.
        When disabling multi-crop we recommend to use "--local_crops_size 96." """)


    # Augmentation parameters
    parser.add_argument('--aug-opt', type=str, default='dino_aug', metavar='NAME',
                        help='Use different data augmentation policy. [deit_aug, dino_aug, mocov2_aug, basic_aug] \
                             "(default: dino_aug)')    
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # * Mixup params
    parser.add_argument('--use_mixup', type=utils.bool_flag, default=False, help="""Whether or not to use mixup/mixcut for self-supervised learning.""")  
    parser.add_argument('--num_mixup_views', type=int, default=10, help="""Number of views to apply mixup/mixcut """)
      
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')

    # Dataset
    parser.add_argument('--dataset', default="imagenet1k", type=str, help='Pre-training dataset.')
    parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not to use zip file.""")
    parser.add_argument('--tsv_mode', type=utils.bool_flag, default=False, help="""Whether or not to use tsv file.""")
    parser.add_argument('--sampler', default="distributed", type=str, help='Sampler for dataloader.')


    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--outfits_json', default='/path/to/outfits/json/', type=str,
        help='Please specify path to json file having outfits')
    
    parser.add_argument('--pretrained_weights_ckpt', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--all_categories", default=0, type=int, help="path to list having all categories")
    parser.add_argument("--cat2items", default=0, type=int, help="Dict : category--> item")

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    
    args = parser.parse_args()
    return args

sys.argv = ['foo']
# parser = ArgumentParser( parents=[parse_args()])
args = parse_args()


# In[6]:


args.arch = 'vit_small'
args.data_path ='/efs/manugaur/spatio-temporal/project/data/images'
args.outfits_json= '/efs/manugaur/temporal/project/1lac_clean_amazon.json'

args.cat2items = '/efs/manugaur/temporal/project/cat2items_train.pkl'
args.all_categories = '/efs/manugaur/temporal/project/total_cat.json'

args.output_dir ='/efs/manugaur/esvit/output/temporal/ssl/error'
args.batch_size_per_gpu= 1
args.epochs =300
args.teacher_temp =0.07
args.warmup_epochs =10
args.warmup_teacher_temp_epochs =30 
args.norm_last_layer= 'false' 
args.use_dense_prediction =False 
args.cfg = 'experiments/imagenet/vit_small/swin_tiny_patch4_window7_224.yaml' 


# In[7]:


import os
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


# ## train_DINO

# In[9]:


import cv2
import matplotlib.pyplot as plt

def display(image):
# read image 
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

#     cv2.destroyAllWindows()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import cv2
def print_img(img):
    plt.imshow(img.numpy())
    plt.show()


# In[11]:



def train_esvit(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    # ============ preparing data ... ============
    data_loader = build_dataloader(args)
        
#     for i in enumerate(data_loader):
#         index = i[0]
#         data = i[1]
#         import pdb;pdb.set_trace()
#         label = data[1][index]
#         views = []
#         #extracting data for first image instance 
#         for img in data[0]: # data[1] has 16 labels, data[0] has ten lists : Each list has 16 views            
#             view = img[index]
#             views.append(view)
#         for x in views:
#             display(x)
#         return

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active and args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.batch_size_per_gpu)

    # ============ building student and teacher networks ... ============

    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = nn.Identity()
        teacher.head = nn.Identity()

    # if the network is a 4-stage vision transformer (i.e. longformer)
    if 'vil' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.out_planes,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.out_planes, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.out_planes,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.out_planes, args.out_dim, args.use_bn_in_head)


    # if the network is a 4-stage conv vision transformer (i.e. CvT)
    if 'cvt' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        fea_dim = config.MODEL.SPEC.DIM_EMBED[-1]
        # print(fea_dim)
        student.head = DINOHead(
            fea_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(fea_dim, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                fea_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(fea_dim, args.out_dim, args.use_bn_in_head)


    # if the network is a vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            use_dense_prediction=args.use_dense_prediction,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.embed_dim, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.embed_dim, args.out_dim, args.use_bn_in_head)

    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]

        use_dense_prediction = args.use_dense_prediction
        if use_dense_prediction: 
            head_dense_student = DINOHead(
                embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            head_dense_teacher = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
        else:
            head_dense_student, head_dense_teacher = None, None
            
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ), head_dense=head_dense_student, use_dense_prediction=use_dense_prediction)
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
            head_dense=head_dense_teacher,
            use_dense_prediction=use_dense_prediction
        )


    else:
        print(f"Unknow architecture: {args.arch}")

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============

    if args.use_dense_prediction: 
        # Both view and region level tasks are considered
        dino_loss = DDINOLoss(
            args.out_dim,
            sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()
    else:
        # Only view level task is considered
        dino_loss = DINOLoss(
            args.out_dim,
            sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")


    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}

    if args.pretrained_weights_ckpt:
        utils.restart_from_checkpoint(
            os.path.join(args.pretrained_weights_ckpt),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        print(f'Resumed from {args.pretrained_weights_ckpt}')

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]


    start_time = time.time()
    print(f"Starting training of EsViT ! from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of EsViT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, mixup_fn, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# ## train one epoch

# In[12]:


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, mixup_fn, 
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images,items_per_outfit) in enumerate(metric_logger.log_every(data_loader, 1000, header)):
        
        #images is a list of 10 lists. 1 sublist for each view. 
        #Each sublist has len = batch_size. ith view for each sample in the batch is stored at images[i] 
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
                
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
#         print(items_per_outfit)
        
    
        # mixup for teacher model output
        teacher_input = images[:2]
        if mixup_fn is not None:
            student_input = []
            targets_mixup = []
            n_mix_views = 0
            # print(f'number of images {len(images)}')
            for samples in images:
                targets = torch.arange(0, args.batch_size_per_gpu, dtype=torch.long).cuda(non_blocking=True)
                if n_mix_views < args.num_mixup_views:
                    samples, targets = mixup_fn(samples, targets)
                    n_mix_views = n_mix_views + 1
                else:
                    targets = torch.eye(args.batch_size_per_gpu).cuda(non_blocking=True)

                student_input.append(samples)
                targets_mixup.append(targets)

            del images, targets, samples

        else:
            student_input = images
            targets_mixup = None

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(teacher_input)  # only the 2 global views pass through the teacher
            student_output = student(student_input)
            loss = dino_loss(student_output, teacher_output,items_per_outfit, epoch, targets_mixup)
            
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)

            # ============ writing logs on a NaN for debug ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_NaN.pth'))

            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            torch.cuda.synchronize()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            torch.cuda.synchronize()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ## DinoLoss

# In[13]:


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, items_per_outfit, epoch, targets_mixup):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if targets_mixup:
                    # print(targets_mixup[v])
                    loss = -torch.sum( targets_mixup[v] * torch.mm(q, F.log_softmax(student_out[v], dim=-1).t()), dim=-1)
                else:
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        import pdb;pdb.set_trace()
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# # L-margin Softmax Loss

# In[14]:


class L_softmax(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.lin_proj = nn.Linear(1536,2)
#         self.CEloss = torch.nn.CrossEntropyLoss()
        self.target = torch.Tensor([0,1])
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_output, teacher_output, items_per_outfit, epoch, targets_mixup):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        m = 2 # number of negatives per outift
        n = 2 # num_items sampled for negative pairs
        student_out = student_output / self.student_temp

        student_out = student_out.chunk(self.ncrops)
        
        # teacher centering and sharpening
        temp = 0.02
#         temp = self.teacher_temp_schedule[epoch]

        teacher_out = (teacher_output - self.center) / temp
        teacher_out = teacher_out.detach().chunk(2)
        
        
        
        # this should be before any for loop, since for any Vi = [n1 + n2 + n3, out_dim]
        # Its invariant to the views. It depends on the outfit
        S = [[] for _ in range(len(items_per_outfit))]
        N= []
        negative_pairs = [[] for _ in range(len(items_per_outfit))]
        abs_index = 0
        for outfit_idx in range(len(items_per_outfit)):
            
            N.append([abs_index + items_per_outfit[outfit_idx]-1,abs_index + items_per_outfit[outfit_idx]-2])
            
            index = np.random.randint(abs_index, abs_index + items_per_outfit[outfit_idx]-2)
            #             for i in range(n-1):     # for n negative items in an outfit
            S[outfit_idx].append(index)

            while index in S[outfit_idx]:

                index = np.random.randint(abs_index, abs_index + items_per_outfit[outfit_idx]-2)
            S[outfit_idx].append(index)
            
            negative_pairs[outfit_idx] = list(itertools.product(S[outfit_idx],N[outfit_idx]))

            abs_index+=items_per_outfit[outfit_idx]
            
        
        total_loss = [[0,0] for _ in range(args.batch_size_per_gpu)]  #for all the outfits : [neg,pos]
        
        for t_idx in range(len(teacher_out)):
            for s_idx in range(len(student_out)):
                pos_loss = 0
                neg_loss = 0
                
                outfit = 0
                traversed =0
                
                
                
                for i in range(len(teacher_out[t_idx])):   #iterating teacher: v1 and v2 
                    
                    j=traversed
                
                    while(j< items_per_outfit[outfit] + traversed):
                        
                        #same item
                        if i==j:
                            j+=1
                            continue
                        
                        # (vi shirt, vi pant) == (vi pant, vi shirt)
                        if s_idx==t_idx and j<i:
                            j+=1
                            continue
                        
                        cat_feat = torch.cat((teacher_out[t_idx][i],student_out[s_idx][j]), dim = -1)
                        cat_feat = self.lin_proj(cat_feat)
                        if (i,j) in negative_pairs[outfit]:
                            
                            target = torch.nn.functional.one_hot(torch.as_tensor([0]), num_classes=2).squeeze()
                            neg_loss += torch.sum(-target.cuda()* torch.log(cat_feat), dim=-1)

                            
                        else:
                            target = torch.nn.functional.one_hot(torch.as_tensor([1]), num_classes=2)
                            pos_loss += torch.sum(-target.cuda()* torch.log(cat_feat), dim=-1)

                            
                        j+=1
                        
                    # all the losses have been calculated for the last item in the teacher view
                    #update traversed, i goes to next outfit in the batch
                    
                    if i == items_per_outfit[outfit] + traversed -1 :
                        traversed+= items_per_outfit[outfit]
                        
                        # New loss calculated for the next outfit in the batch
                        total_loss[outfit][0]+=pos_loss
                        total_loss[outfit][1]+=neg_loss                        
                        outfit+=1

                        pos_loss = 0
                        neg_loss = 0
                #for 2 views

                

        
        for ith_outfit in range(len(total_loss)):
            n_outfit = items_per_outfit[ith_outfit]
            total_loss[ith_outfit] = total_loss[ith_outfit][0]/((n_outfit -m)*(n_outfit-m-1)) + total_loss[ith_outfit][1]/(2*n*m) 
        """
        average the loss of all the outfits in the batch.
        Scale it by 20, since 20 combination of views in student & teacher
        """
        
        out_loss = 0 
        for i in total_loss:
            out_loss+=i/args.batch_size_per_gpu
        self.update_center(teacher_output)
#        return out_loss  
        return out_loss/20  

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)        


# # Proposed loss : penalization

# In[15]:


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    
    def forward(self, student_output, teacher_output, items_per_outfit, epoch, targets_mixup):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        m = 5 # number of negatives per outift
        n = 2 # num_items sampled for negative pairs
        student_out = student_output / self.student_temp

        student_out = student_out.chunk(self.ncrops)
        
        # teacher centering and sharpening
        temp = 0.02
#         temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        cos= torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        minx = -1 
        maxx = 1
        
        # this should be before any for loop, since for any Vi = [n1 + n2 + n3, out_dim]
        # Its invariant to the views. It depends on the outfit
        S = [[] for _ in range(len(items_per_outfit))]
        N= []
        negative_pairs = [[] for _ in range(len(items_per_outfit))]
        abs_index = 0
        for outfit_idx in range(len(items_per_outfit)):
            
            N.append([abs_index + items_per_outfit[outfit_idx]-1,abs_index + items_per_outfit[outfit_idx]-2])
            
            index = np.random.randint(abs_index, abs_index + items_per_outfit[outfit_idx]-2)
            #             for i in range(n-1):     # for n negative items in an outfit
            S[outfit_idx].append(index)

            while index in S[outfit_idx]:

                index = np.random.randint(abs_index, abs_index + items_per_outfit[outfit_idx]-2)
            S[outfit_idx].append(index)
            
            negative_pairs[outfit_idx] = list(itertools.product(S[outfit_idx],N[outfit_idx]))

            abs_index+=items_per_outfit[outfit_idx]
            
        
        total_loss = [[0,0] for _ in range(args.batch_size_per_gpu)]  #for all the outfits : [CEloss,cos]
        
        for t_idx in range(len(teacher_out)):
            for s_idx in range(len(student_out)):
                pos_loss = 0
                neg_loss = 0
                
                outfit = 0
                traversed =0
                
                for i in range(len(teacher_out[t_idx])):   #iterating teacher: v1 and v2 
                    
                    j=traversed
                
                    while(j< items_per_outfit[outfit] + traversed):
                        import pdb;pdb.set_trace()
                        #same item
                        if i==j:
                            j+=1
                            continue
                        
                        # (vi shirt, vi pant) == (vi pant, vi shirt)
                        if s_idx==t_idx and j<i:
                            j+=1
                            continue
                            
                        if (i,j) in negative_pairs[outfit]:
#                             cos_sim += (cos(teacher_out[t_idx][i],student_out[s_idx][j])- minx)/(maxx-minx)
                            neg_loss += torch.abs(cos(teacher_out[t_idx][i],student_out[s_idx][j]))      
                        else:
                            # loss between two views of different items in the same outfit
             
                            pos_loss += torch.abs((1 - cos(teacher_out[t_idx][i],student_out[s_idx][j])))     

#                             CEloss += torch.sum(-teacher_out[t_idx][i] * F.log_softmax(student_out[s_idx][j], dim=-1), dim=-1)
                            
                        j+=1
                        
                    # all the losses have been calculated for the last item in the teacher view
                    #update traversed, i goes to next outfit in the batch
                    
                    if i == items_per_outfit[outfit] + traversed -1 :
                        traversed+= items_per_outfit[outfit]
                        
                        # New loss calculated for the next outfit in the batch
                        total_loss[outfit][0]+=pos_loss
                        total_loss[outfit][1]+=neg_loss                        
                        outfit+=1

                        pos_loss = 0
                        neg_loss = 0
                #for 2 views

                

        
        for ith_outfit in range(len(total_loss)):
            n_outfit = items_per_outfit[ith_outfit]
            total_loss[ith_outfit] = total_loss[ith_outfit][0]/((n_outfit -m)*(n_outfit-m-1)) + total_loss[ith_outfit][1]/(2*n*m) 
        """
        average the loss of all the outfits in the batch.
        Scale it by 20, since 20 combination of views in student & teacher
        """
        
        out_loss = 0 
        for i in total_loss:
            out_loss+=i/args.batch_size_per_gpu
        self.update_center(teacher_output)
#         return out_loss/20  
        return out_loss
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)        
        


# ## main

# In[1]:


if __name__ == '__main__':
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_esvit(args)

