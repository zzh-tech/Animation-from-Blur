import yaml
import random
import torch
import torchmetrics
import lpips
import time
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model.MBD import MBD
from model.utils import AverageMeter
from os.path import join
from logger import Logger
from model.bicyclegan.global_model import GuidePredictor as GP
from tqdm import tqdm

loss_fn_alex = lpips.LPIPS(net='alex').to('cuda:0')


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation(local_rank, d_configs, p_configs):
    # Preparation and backup
    torch.backends.cudnn.benchmark = True

    # model init
    d_model = MBD(local_rank=local_rank, configs=d_configs)
    p_model = GP(local_rank=local_rank, configs=p_configs)

    # dataset init
    dataset_args = d_configs['dataset_args']
    valid_dataset = BDDataset(set_type='valid', **dataset_args)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=d_configs['num_workers'],
                              pin_memory=True)
    evaluate(d_model, p_model, valid_loader, local_rank)


@torch.no_grad()
def evaluate(d_model, p_model, valid_loader, local_rank):
    # Preparation
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    psnr_meter_better = AverageMeter()
    ssim_meter_better = AverageMeter()
    lpips_meter_better = AverageMeter()
    time_stamp = time.time()

    # One epoch validation
    for i, tensor in enumerate(tqdm(valid_loader, total=len(valid_loader))):
        tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)
        tensor['trend'] = torch.zeros_like(tensor['inp'])[:, :, :2]
        tensor['gt'] = tensor['gt'].to(device)  # (b, num_gts, 3, h, w)
        b, num_gts, c, h, w = tensor['gt'].shape

        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        psnr_vals_better = []
        ssim_vals_better = []
        lpips_vals_better = []

        for _ in range(num_sampling):
            p_tensor = {}
            p_tensor['inp'] = F.interpolate(tensor['inp'], size=(3, 256, 256))
            p_tensor['trend'] = F.interpolate(tensor['trend'], size=(2, 256, 256))
            p_tensor['video'] = ''
            out_tensor = p_model.test(inp_tensor=p_tensor, zeros=False)
            tensor['trend'] = out_tensor['pred_trends'].to(device)
            tensor['trend'] = F.interpolate(tensor['trend'], size=(h, w))[None]
            out_tensor = d_model.update(inp_tensor=tensor, training=False)
            pred_imgs = out_tensor['pred_imgs']  # pred_imgs shape (b, num_gts, 3, h, w)
            gt_imgs = out_tensor['gt_imgs']  # gt_imgs shape (b, num_gts, 3, h, w)

            # Record loss and metrics
            pred_imgs = pred_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            pred_imgs_rev = torch.flip(pred_imgs, dims=[1, ])
            pred_imgs = pred_imgs.reshape(num_gts * b, c, h, w)
            pred_imgs_rev = pred_imgs_rev.reshape(num_gts * b, c, h, w)
            gt_imgs = gt_imgs.reshape(num_gts * b, c, h, w)
            psnr_val = torchmetrics.functional.psnr(pred_imgs, gt_imgs, data_range=255)
            psnr_val_rev = torchmetrics.functional.psnr(pred_imgs_rev, gt_imgs, data_range=255)
            ssim_val = torchmetrics.functional.ssim(pred_imgs, gt_imgs, data_range=255)
            ssim_val_rev = torchmetrics.functional.ssim(pred_imgs_rev, gt_imgs, data_range=255)
            pred_imgs = (pred_imgs - (255. / 2)) / (255. / 2)
            pred_imgs_rev = (pred_imgs_rev - (255. / 2)) / (255. / 2)
            gt_imgs = (gt_imgs - (255. / 2)) / (255. / 2)
            lpips_val = loss_fn_alex(pred_imgs, gt_imgs)
            lpips_val_rev = loss_fn_alex(pred_imgs_rev, gt_imgs)

            psnr_vals.append(psnr_val)
            psnr_vals_better.append(max(psnr_val, psnr_val_rev))
            ssim_vals.append(ssim_val)
            ssim_vals_better.append(max(ssim_val, ssim_val_rev))
            lpips_vals.append(lpips_val.mean().detach())
            lpips_vals_better.append(min(lpips_val.mean().detach(), lpips_val_rev.mean().detach()))

        psnr_meter.update(max(psnr_vals), num_gts * b)
        ssim_meter.update(max(ssim_vals), num_gts * b)
        lpips_meter.update(min(lpips_vals), num_gts * b)
        psnr_meter_better.update(max(psnr_vals_better), num_gts * b)
        ssim_meter_better.update(max(ssim_vals_better), num_gts * b)
        lpips_meter_better.update(min(lpips_vals_better), num_gts * b)
        # print('{}/{}'.format(i + 1, len(valid_loader)), psnr_meter.avg, ssim_meter.avg, lpips_meter.avg)
        # print('{}/{}'.format(i + 1, len(valid_loader)), psnr_meter_better.avg, ssim_meter_better.avg,
        #       lpips_meter_better.avg)

    # Ending of validation
    eval_time_interval = time.time() - time_stamp
    msg = 'eval time: {} sec, psnr: {:.5f}, ssim: {:.5f}, lpips: {:.4f}'.format(
        eval_time_interval, psnr_meter.avg, ssim_meter.avg, lpips_meter.avg
    )
    logger(msg, prefix='[valid]')
    msg = 'eval time: {:.4f} sec, psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'.format(
        eval_time_interval, psnr_meter_better.avg, ssim_meter_better.avg, lpips_meter_better.avg
    )
    logger(msg, prefix='[valid max.]')
    logger.close()


if __name__ == '__main__':
    # load args & configs
    parser = ArgumentParser(description='Guidance prediction & Blur Decomposition')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--log_dir', default='logs', help='path of log')
    parser.add_argument('--log_name', default='valid', help='log name')
    parser.add_argument('--predictor_resume_dir', help='path of log', required=True)
    parser.add_argument('--decomposer_resume_dir', help='path of log', required=True)
    parser.add_argument('--data_dir', nargs='+', required=True)
    parser.add_argument('--num_iters', type=int, default=1, help='number of iters')
    parser.add_argument('-ns', '--num_sampling', type=int, default=1, help='number of sampling times')
    parser.add_argument('--verbose', action='store_true', help='whether to print out logs')
    args = parser.parse_args()
    num_sampling = args.num_sampling

    # Para for decomposer
    decomposer_config = join(args.decomposer_resume_dir, 'cfg.yaml')
    with open(decomposer_config) as f:
        d_configs = yaml.full_load(f)
    d_configs['resume_dir'] = args.decomposer_resume_dir
    d_configs['num_iterations'] = args.num_iters

    # Para for predictor
    predictor_config = join(args.predictor_resume_dir, 'cfg.yaml')
    with open(predictor_config, 'rt', encoding='utf8') as f:
        p_configs = yaml.full_load(f)
    p_configs['bicyclegan_args']['checkpoints_dir'] = args.predictor_resume_dir
    p_configs['bicyclegan_args']['isTrain'] = False
    p_configs['bicyclegan_args']['epoch'] = 'latest'
    p_configs['bicyclegan_args']['no_encode'] = False

    # Import blur decomposition dataset
    is_gen_blur = True
    for root_dir in d_configs['dataset_args']['root_dir']:
        if 'b-aist++' in root_dir:
            is_gen_blur = False
    if is_gen_blur:
        from data.dataset import GenBlur as BDDataset

        d_configs['dataset_args']['aug_args']['valid']['image'] = {}
    else:
        from data.dataset import BAistPP as BDDataset

        d_configs['dataset_args']['aug_args']['valid']['image'] = {
            'NearBBoxResizedSafeCrop': {'max_ratio': 0, 'height': 192, 'width': 160}
        }
    d_configs['dataset_args']['root_dir'] = args.data_dir
    d_configs['dataset_args']['use_trend'] = False

    # DDP init
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    init_seeds(seed=rank)

    # Logger init
    logger = Logger(file_path=join(args.log_dir, '{}.txt'.format(args.log_name)), verbose=args.verbose)

    # Training model
    validation(local_rank=args.local_rank, d_configs=d_configs, p_configs=p_configs)

    # Tear down the process group
    dist.destroy_process_group()
