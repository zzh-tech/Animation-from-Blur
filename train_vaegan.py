import yaml
import random
import torch
import time
import numpy as np
import torch.distributed as dist
from datetime import datetime
from argparse import ArgumentParser
from data.flow_viz import trend_plus_vis
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from model.bicyclegan.global_model import GuidePredictor as GP
from model.utils import AverageMeter
from os.path import join
from logger import Logger


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(local_rank, configs, log_dir):
    # Preparation and backup
    device = torch.device("cuda", args.local_rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        writer = SummaryWriter(log_dir)
        configs_bp = join(log_dir, 'cfg.yaml')
        with open(configs_bp, 'w') as f:
            yaml.dump(configs, f)
    else:
        writer = None
    step = 0
    num_eval = 0

    # model init
    model = GP(local_rank=local_rank, configs=configs)

    # dataset init
    dataset_args = configs['dataset_args']
    train_batch_size = configs['bicyclegan_args']['batch_size']
    valid_batch_size = configs['bicyclegan_args']['batch_size']
    num_workers = configs['bicyclegan_args']['num_threads']
    train_dataset = BDDataset(set_type='train', **dataset_args)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    valid_dataset = BDDataset(set_type='valid', **dataset_args)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=valid_batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    # training looping
    step_per_epoch = len(train_loader)
    time_stamp = time.time()
    start_epoch = configs['bicyclegan_args']['epoch_count']
    end_epoch = configs['bicyclegan_args']['niter'] + configs['bicyclegan_args']['niter_decay'] + 1
    for epoch in range(start_epoch, end_epoch):
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch)
        for i, tensor in enumerate(train_loader):
            # Record time after loading data
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # Update model
            tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)
            tensor['trend'] = tensor['trend'].to(device)  # (b, 1, 2, h, w)
            out_tensor = model.update(inp_tensor=tensor, training=True)
            if out_tensor is None:
                print("skip this batch")
                continue
            loss = out_tensor['loss']
            # Record time after updating model
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # Print training info
            if step % 100 == 0:
                if rank == 0:
                    writer.add_scalar('learning_rate', model.get_lr(), step)
                    msg = 'epoch: {:>3}, batch: [{:>5}/{:>5}], time: {:.2f} + {:.2f} sec, '
                    msg = msg.format(epoch,
                                     i + 1,
                                     step_per_epoch,
                                     data_time_interval,
                                     train_time_interval)
                    for key, val in loss.items():
                        writer.add_scalar('train/loss_{}'.format(key), val, step)
                        msg += 'loss_{}: {:.5f} '.format(key, val)
                    msg += 'loss: {:.5f}'.format(sum(loss.values()))
                    logger(msg, prefix='[train]')

            if (rank == 0) and (step % 500 == 0):
                inp_img = out_tensor['inp_img']  # inp_img shape (b, c, h, w)
                pred_trends = out_tensor['pred_trends']  # pred_imgs shape (b, 2, h, 2*w)
                gt_trend = out_tensor['gt_trend']  # gt_imgs shape (b, 2, h, w)

                # Prepare recorded results
                inp_img = inp_img.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
                pred_trends = pred_trends.permute(0, 2, 3, 1).cpu().detach().numpy()
                gt_trend = gt_trend.permute(0, 2, 3, 1).cpu().detach().numpy()

                pred_trends_rgb = []
                gt_trend_rgb = []
                for pred_item, gt_item in zip(pred_trends, gt_trend):
                    pred_trends_rgb.append(trend_plus_vis(pred_item))
                    gt_trend_rgb.append(trend_plus_vis(gt_item))

                b = inp_img.shape[0]
                # Record each sample results in the batch
                for j in range(b):
                    # Record predicted images pair
                    cat_imgs = np.concatenate([inp_img[j], pred_trends_rgb[j], gt_trend_rgb[j]],
                                              axis=1)  # (h, 4 * w, c)
                    writer.add_image('train/imgs_results_{}'.format(j), cat_imgs, step, dataformats='HWC')

            # Ending of a batch
            step += 1

        # Ending of an epoch
        num_eval += 1
        if num_eval % 5 == 0:
            evaluate(model, valid_loader, num_eval, local_rank, writer)
            if rank == 0:
                model.save_model(epoch)
                model.save_model('latest')
        model.scheduler_step()
        dist.barrier()


@torch.no_grad()
def evaluate(model, valid_loader, num_eval, local_rank, writer):
    # Preparation
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)
    loss_meter = AverageMeter()
    time_stamp = time.time()

    # One epoch validation
    random_idx = random.randint(0, len(valid_loader))
    for i, tensor in enumerate(valid_loader):
        tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)
        tensor['trend'] = tensor['trend'].to(device)  # (b, 1, 2, h, w)
        out_tensor = model.update(inp_tensor=tensor, training=False)
        if out_tensor is None:
            print("skip this batch")
            continue
        pred_trends = out_tensor['pred_trends']  # pred_imgs shape (b, 2, h, 2*w)
        gt_trend = out_tensor['gt_trend']  # gt_imgs shape (b, 2, h, w)
        loss = out_tensor['loss']
        loss = sum(loss.values())

        # Record loss and metrics
        pred_trends = pred_trends.detach()
        gt_trend = gt_trend.detach()
        b = pred_trends.size(0)
        loss_meter.update(loss, b)

        # Record image results
        if rank == 0 and i == random_idx:
            inp_img = out_tensor['inp_img']  # inp_img shape (b, c, h, w)
            inp_img = inp_img.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
            pred_trends = pred_trends.permute(0, 2, 3, 1).cpu().detach().numpy()
            gt_trend = gt_trend.permute(0, 2, 3, 1).cpu().detach().numpy()

            pred_trends_rgb = []
            gt_trend_rgb = []
            for pred_item, gt_item in zip(pred_trends, gt_trend):
                pred_trends_rgb.append(trend_plus_vis(pred_item))
                gt_trend_rgb.append(trend_plus_vis(gt_item))

            for j in range(b):
                # Record predicted images pair
                cat_imgs = np.concatenate([inp_img[j], pred_trends_rgb[j], gt_trend_rgb[j]],
                                          axis=1)  # (h, 4 * w, c)
                writer.add_image('valid/imgs_results_{}'.format(j), cat_imgs, num_eval, dataformats='HWC')

    # Ending of validation
    eval_time_interval = time.time() - time_stamp
    if rank == 0:
        writer.add_scalar('valid/loss', loss_meter.avg, num_eval)
        msg = 'eval time: {} sec, loss: {:.5f}'.format(
            eval_time_interval, loss_meter.avg
        )
        logger(msg, prefix='[valid]')


if __name__ == '__main__':
    # load args & configs
    parser = ArgumentParser(description='Motion Guide Prediction')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/cfg.yaml', help='path of config')
    parser.add_argument('--log_dir', default='log', help='path of log')
    parser.add_argument('--verbose', action='store_true', help='whether to print out logs')

    args = parser.parse_args()
    with open(args.config, 'rt', encoding='utf8') as f:
        configs = yaml.full_load(f)
    configs['bicyclegan_args']['checkpoints_dir'] = args.log_dir

    # Import blur decomposition dataset
    is_gen_blur = True
    for root_dir in configs['dataset_args']['root_dir']:
        if 'b-aist++' in root_dir:
            is_gen_blur = False
    if is_gen_blur:
        from data.dataset import GenBlur as BDDataset
    else:
        from data.dataset import BAistPP as BDDataset

    # DDP init
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    init_seeds(seed=rank)

    # Logger init
    if rank == 0:
        logger = Logger(file_path=join(args.log_dir, 'log_{}.txt'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),
                        verbose=args.verbose)

    # Training model
    train(local_rank=args.local_rank,
          configs=configs,
          log_dir=args.log_dir)

    # Tear down the process group
    dist.destroy_process_group()
