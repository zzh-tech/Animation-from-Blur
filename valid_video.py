import sys

sys.path.append('./model/RAFT/core')

import yaml
import random
import torch
import torchmetrics
import lpips
import time
import cv2
import os.path as osp
import numpy as np
import torch.distributed as dist
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model.MBD import MBD
from model.utils import AverageMeter
from os.path import join
from logger import Logger
from tqdm import tqdm
from raft import RAFT
from utils.utils import InputPadder

loss_fn_alex = lpips.LPIPS(net='alex').to('cuda:0')


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation(local_rank, configs):
    # Preparation
    torch.backends.cudnn.benchmark = True

    # model init
    model = MBD(local_rank=local_rank, configs=configs)

    # dataset init
    dataset_args = configs['dataset_args']
    valid_dataset = BDDataset(set_type='valid', **dataset_args)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=configs['num_workers'],
                              pin_memory=True)
    evaluate(model, valid_loader, local_rank)


@torch.no_grad()
def _gen_flow(img0, img1):
    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)
    flow_low, flow_up = raft(img0, img1, iters=30, test_mode=True)
    flow_up = padder.unpad(flow_up)

    return flow_up[0].permute(1, 2, 0).cpu().numpy()


def gen_flow(img_ref, img_tgt):
    flow = _gen_flow(img_tgt, img_ref)  # backward flow
    flow = flow * (-1.)
    size = (int(flow_ratio * flow.shape[1]), int(flow_ratio * flow.shape[0]))
    # ! resizing flow needs to time ratio
    flow = flow_ratio * cv2.resize(flow, size, interpolation=cv2.INTER_AREA)
    trend_x = flow[:, :, 0::2]
    trend_y = flow[:, :, 1::2]
    trend_x = np.mean(trend_x, axis=-1, keepdims=True)
    trend_y = np.mean(trend_y, axis=-1, keepdims=True)
    trend_x_temp = trend_x.copy()
    trend_y_temp = trend_y.copy()
    trend_x[np.sqrt((trend_x_temp ** 2) + (trend_y_temp ** 2)) < threshold] = 0
    trend_y[np.sqrt((trend_x_temp ** 2) + (trend_y_temp ** 2)) < threshold] = 0

    trend_x[trend_x > 0] = 1
    trend_x[trend_x < 0] = -1
    trend_y[trend_y > 0] = 1
    trend_y[trend_y < 0] = -1
    trend_x[(trend_x == 0) & (trend_y == 1)] = 1
    trend_x[(trend_x == 0) & (trend_y == -1)] = -1
    trend_y[(trend_y == 0) & (trend_x == 1)] = -1
    trend_y[(trend_y == 0) & (trend_x == -1)] = 1

    trend = np.concatenate([trend_x, trend_y], axis=-1)
    trend = trend.astype(np.int8)
    return torch.from_numpy(trend).permute(2, 0, 1)[None].float()


@torch.no_grad()
def evaluate(model, valid_loader, local_rank):
    # Preparation
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    time_stamp = time.time()

    # One epoch validation
    for i, tensor in enumerate(tqdm(valid_loader, total=len(valid_loader))):
        tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)

        img_ref = tensor['inp'][:, 0]
        img_tgt = tensor['inp'][:, 1]
        trend = gen_flow(img_ref, img_tgt)
        tensor['trend'] = trend.unsqueeze(dim=1).to(device)  # (b, 1, 2, h, w)
        tensor['inp'] = img_tgt.unsqueeze(dim=1).to(device)  # (b, 1, 3, h, w)
        tensor['gt'] = tensor['gt'][:, 7:].to(device)  # (b, num_gts, 3, h, w)

        out_tensor = model.update(inp_tensor=tensor, training=False)
        pred_imgs = out_tensor['pred_imgs']  # pred_imgs shape (b, num_gts, 3, h, w)
        gt_imgs = out_tensor['gt_imgs']  # gt_imgs shape (b, num_gts, 3, h, w)
        loss = out_tensor['loss']

        # Record loss and metrics
        pred_imgs = pred_imgs.to('cuda:0')
        gt_imgs = gt_imgs.to('cuda:0')
        pred_imgs = pred_imgs[:, [0, 3, 6]]
        gt_imgs = gt_imgs[:, [0, 3, 6]]
        b, num_gts, c, h, w = pred_imgs.shape
        pred_imgs = pred_imgs.reshape(num_gts * b, c, h, w)
        gt_imgs = gt_imgs.reshape(num_gts * b, c, h, w)
        psnr_val = torchmetrics.functional.psnr(pred_imgs, gt_imgs, data_range=255)
        ssim_val = torchmetrics.functional.ssim(pred_imgs, gt_imgs, data_range=255)
        pred_imgs = (pred_imgs - (255. / 2)) / (255. / 2)
        gt_imgs = (gt_imgs - (255. / 2)) / (255. / 2)
        lpips_val = loss_fn_alex(pred_imgs, gt_imgs)
        psnr_meter.update(psnr_val, num_gts * b)
        ssim_meter.update(ssim_val, num_gts * b)
        lpips_meter.update(lpips_val.mean().detach(), num_gts * b)
        loss_meter.update(loss.item(), b)
        # print('{}/{}'.format(i + 1, len(valid_loader)), psnr_meter.avg, ssim_meter.avg, lpips_meter.avg)

    # Ending of validation
    eval_time_interval = time.time() - time_stamp
    msg = 'eval time: {:.4f} sec, loss: {:.4f}, psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'.format(
        eval_time_interval, loss_meter.avg, psnr_meter.avg, ssim_meter.avg, lpips_meter.avg
    )
    logger(msg, prefix='[valid]')
    logger.close()


if __name__ == '__main__':
    # load args & configs
    parser = ArgumentParser(description='Blur Decomposition')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--log_dir', default='logs', help='path of log')
    parser.add_argument('--log_name', default='valid', help='log name')
    parser.add_argument('--resume_dir', help='path of checkpoint dir', required=True)
    parser.add_argument('--data_dir', nargs='+', required=True)
    parser.add_argument('--num_iters', type=int, default=1, help='number of iters')
    parser.add_argument('--verbose', action='store_true', help='whether to print out logs')

    # arguments for RAFT
    parser.add_argument('--model_path', default='./checkpoints/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    args.config = osp.join(args.resume_dir, 'cfg.yaml')
    with open(args.config) as f:
        configs = yaml.full_load(f)
    configs['resume_dir'] = args.resume_dir
    configs['num_iterations'] = args.num_iters
    device = torch.device("cuda", args.local_rank)
    flow_ratio = 1
    threshold = 0.5 * flow_ratio

    # Import blur decomposition dataset
    is_gen_blur = True
    for root_dir in configs['dataset_args']['root_dir']:
        if 'b-aist++' in root_dir:
            is_gen_blur = False
    if is_gen_blur:
        from data.dataset import GenBlur as BDDataset

        configs['dataset_args']['aug_args']['valid']['image'] = {}
    else:
        from data.dataset import BAistPP as BDDataset

        configs['dataset_args']['aug_args']['valid']['image']['NearBBoxResizedSafeCrop']['max_ratio'] = 0
    configs['dataset_args']['root_dir'] = args.data_dir
    configs['dataset_args']['num_past'] = 1
    configs['dataset_args']['num_fut'] = 0
    configs['dataset_args']['use_trend'] = False

    # DDP init
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    init_seeds(seed=rank)

    # Logger init
    logger = Logger(file_path=join(args.log_dir, '{}.txt'.format(args.log_name)),
                    verbose=args.verbose)

    # model init
    raft = torch.nn.DataParallel(RAFT(args))
    raft.load_state_dict(torch.load(args.model_path))
    raft = raft.to(device)
    raft.eval()

    # Training model
    validation(local_rank=args.local_rank, configs=configs)

    # Tear down the process group
    dist.destroy_process_group()
