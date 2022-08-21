import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os.path import join
from model.flow_estimator import FlowEstimator
from model.decomposer import Decomposer
from model.embedder import get_embedder
from model.utils import ckpt_convert, Vgg19
from torch.nn.parallel import DistributedDataParallel as DDP


class MBD:
    """
    Multimodal blur decomposition
    """

    def __init__(self, local_rank, configs):
        self.configs = configs
        self.val_range = configs['val_range']
        self.num_iters = configs['num_iterations']
        self.num_gts = configs['dataset_args']['num_gts']
        self.hybrid = True
        if 'hybrid' in configs:
            self.hybrid = configs['hybrid']
        self.residual = False
        if 'residual' in configs:
            self.residual = configs['residual']
        self.residual_blur = False
        if 'residual_blur' in configs:
            self.residual_blur = configs['residual_blur']
        self.flow_to_s2 = True
        if 'flow_to_s2' in configs:
            self.flow_to_s2 = configs['flow_to_s2']
        self.s1_to_s2 = False
        if 's1_to_s2' in configs:
            self.s1_to_s2 = configs['s1_to_s2']
        self.use_trend = True
        if 'use_trend' in configs:
            self.use_trend = configs['use_trend']
        self.perc_ratio = 0
        if 'perc_ratio' in configs:
            self.perc_ratio = configs['perc_ratio']
        # print('residual: {}\nresidual_blur: {}\nflow_to_s2: {}\ns1_to_s2: {}\nhybrid: {}\nuse_trend: {}'.format(
        #     self.residual,
        #     self.residual_blur,
        #     self.flow_to_s2,
        #     self.s1_to_s2,
        #     self.hybrid,
        #     self.use_trend))
        # print('percptual loss ratio: {}'.format(self.perc_ratio))
        self.device = torch.device("cuda", local_rank)

        # Init modules
        if self.num_iters > 0:
            self.flow_estimator = FlowEstimator(device=self.device, **configs['flow_estimator_args'])
        self.trend_embedder, trend_embed_dim = get_embedder(**configs['trend_embedder_args'])
        if self.use_trend:
            configs['decomposer_s1_args']['in_channels'] += trend_embed_dim
        self.flow_embbedder, flow_embed_dim = get_embedder(**configs['flow_embedder_args'])
        if self.flow_to_s2:
            configs['decomposer_s2_args']['in_channels'] += (self.num_gts - 1) * flow_embed_dim
        if self.s1_to_s2:
            configs['decomposer_s2_args']['in_channels'] += self.num_gts * 3
        self.decomposer_s1 = Decomposer(**configs['decomposer_s1_args'])
        self.decomposer_s2 = Decomposer(**configs['decomposer_s2_args'])

        # Replace BN as SyncBN
        self.decomposer_s1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s1)
        self.decomposer_s2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s2)

        # Move modules to GPU
        self.decomposer_s1.to(device=self.device)
        self.decomposer_s2.to(device=self.device)

        # Load checkpoints
        if (local_rank == 0) and (configs['resume_dir'] is not None):
            self.load_model(configs['resume_dir'], self.device)

        # DDP wrapper
        # https://github.com/open-mmlab/mmdetection/issues/2539
        self.decomposer_s1 = DDP(self.decomposer_s1, device_ids=[local_rank], output_device=local_rank,
                                 broadcast_buffers=False)
        self.decomposer_s2 = DDP(self.decomposer_s2, device_ids=[local_rank], output_device=local_rank,
                                 broadcast_buffers=False)

        # Init optimizer, learning rate scheduler, and loss function
        self.optimizer = optim.AdamW(itertools.chain(self.decomposer_s1.parameters(),
                                                     self.decomposer_s2.parameters()), **configs['optimizer'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs['epoch'],
                                                              **configs['scheduler'])
        self.l2 = nn.MSELoss()
        # self.l1 = nn.L1Loss()
        if self.perc_ratio > 0:
            self.vgg = Vgg19().to(self.device)

    def train(self):
        """
        Open training mode
        """
        self.decomposer_s1.train()
        self.decomposer_s2.train()

    def eval(self):
        """
        Open evaluating mode
        """
        self.decomposer_s1.eval()
        self.decomposer_s2.eval()

    def scheduler_step(self):
        """
        Update scheduler after each epoch
        """
        self.scheduler.step()

    def get_lr(self):
        """
        Get learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def load_model(self, log_dir, device, name=None):
        """
        Load checkpoints for decomposer_s1 and decomposer_s2
        """
        if name is None:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1.pth'), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2.pth'), map_location=device)))
        else:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1_{}.pth'.format(name)), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2_{}.pth'.format(name)), map_location=device)))

    def save_model(self, log_dir, name=None):
        """
        Save checkpoints for decomposer_s1 and decomposer_s2
        """
        if name is None:
            # todo: save optimizer and other info for restarting training from specific epoch
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1.pth'))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2.pth'))
        else:
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1_{}.pth'.format(name)))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2_{}.pth'.format(name)))

    def update(self, inp_tensor, hybrid_flag=0, training=True):
        """
        Forward propagation, and backward propagation (if training == True) for a batch of data
        If hybrid_flag is 1, use gt-flow as input in second stage instead of predicted one in the first stage
        Shape of blur_img is (b, 1, 3, h, w), val range from 0~255
        Shape of sharp_imgs is (b, num_gts, 3, h, w), val range from 0~255
        shape of trend is (b, 1, 2, h, w), val -1. or 1.
        Return pred_imgs of last iteration as shape of (b, num_gts, 3, h, w)
        """

        # Preparation
        if training:
            self.train()
        else:
            self.eval()
        out_tensor = {}
        blur_img, sharp_imgs, trend_img = inp_tensor['inp'], inp_tensor['gt'], inp_tensor['trend']
        b, n, c, h, w = blur_img.shape
        blur_img = blur_img.reshape(b, n * c, h, w)  # shape (b, 1 * 3, h, w)
        b, n, tc, h, w = trend_img.shape
        trend_img = trend_img.reshape(b, n * tc, h, w)  # shape (b, 1 * 2, h, w)
        blur_img = blur_img / self.val_range
        sharp_imgs = sharp_imgs / self.val_range

        # Forward propagation
        all_pred_imgs = []
        # First stage
        # pred_imgs shape (b, num_gts, c, h, w)
        if self.residual_blur:
            pred_imgs = self.first_stage(blur_img, trend_img) + blur_img.unsqueeze(dim=1)
        else:
            pred_imgs = self.first_stage(blur_img, trend_img)
        all_pred_imgs.append(pred_imgs)

        for _ in range(self.num_iters):
            # pred_imgs shape (b, num_gts, c, h, w)
            if training:
                if (hybrid_flag == 0) or (not self.hybrid):
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs)
                    else:
                        pred_imgs = self.second_stage(blur_img, pred_imgs)
                elif hybrid_flag == 1:
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, sharp_imgs)
                    else:
                        pred_imgs = self.second_stage(blur_img, sharp_imgs)
                else:
                    raise ValueError
            else:
                if self.residual:
                    pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs)
                else:
                    pred_imgs = self.second_stage(blur_img, pred_imgs)
            all_pred_imgs.append(pred_imgs)

        # Calculate losses
        # Loss for predicted images
        loss = 0.
        for pred_imgs in all_pred_imgs:
            # Reconstruction loss
            loss += self.l2(pred_imgs, sharp_imgs)
            if self.perc_ratio > 0:
                b, num_gts, c, h, w = pred_imgs.shape
                # Perceptual loss
                x_vgg = self.vgg(pred_imgs.reshape(b * num_gts, c, h, w))
                y_vgg = self.vgg(sharp_imgs.reshape(b * num_gts, c, h, w))
                for i in range(len(x_vgg)):
                    loss += self.perc_ratio * torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            # # Reblur loss (self-supervised)
            # loss += self.l1(torch.mean(pred_imgs, dim=1), blur_img)
            # # Pairwise ordering-invariant loss
            # for i in range(num_gts // 2):
            #     loss += self.l1(torch.abs(pred_imgs[:, i] + pred_imgs[:, num_gts - 1 - i]),
            #                     torch.abs(sharp_imgs[:, i] + sharp_imgs[:, num_gts - 1 - i]))
            #     loss += self.l1(torch.abs(pred_imgs[:, i] - pred_imgs[:, num_gts - 1 - i]),
            #                     torch.abs(sharp_imgs[:, i] - sharp_imgs[:, num_gts - 1 - i]))

        # Backward propagation
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Register tensors
        # total loss
        out_tensor['loss'] = loss
        # input blurry image (b, c, h, w)
        out_tensor['inp_img'] = self.val_range * blur_img[:, :c].detach()
        # input trend guidance (b, 2, h, w)
        out_tensor['trend_img'] = trend_img.detach()
        # last predicted image sequence and the corresponding ground truth, (b, num_gts, c, h, w)
        out_tensor['pred_imgs'] = torch.clamp(self.val_range * all_pred_imgs[-1].detach(), min=0,
                                              max=self.val_range)
        out_tensor['gt_imgs'] = self.val_range * sharp_imgs

        return out_tensor

    def first_stage(self, blur_img, trend_img):
        """
        :param blur_img: blurry image (b, c, h, w), c = 3
        :param trend_img: blurry image (b, tc, h, w), tc = 2
        :return: sharp image sequence (b, num_gts, c, h, w)
        """
        if not self.use_trend:
            return self.decomposer_s1(blur_img)

        num_gts = self.configs['dataset_args']['num_gts']
        b, tc, h, w = trend_img.shape
        trend_static = torch.zeros(b, 1, h, w).to(self.device)  # (0, 0) black
        trend_static[(trend_img[:, 0:1] == 0) & (trend_img[:, 1:2] == 0)] = 1
        trend_static = trend_static.unsqueeze(dim=1)  # (b, 1, 1, h, w)
        trend_dynamic = 1 - trend_static  # (b, 1, 1, h, w)

        inp_img = torch.cat([blur_img, trend_img], dim=1)  # shape (b, 5, h, w)
        pred_imgs = trend_dynamic * self.decomposer_s1(inp_img)
        pred_imgs += trend_static * blur_img.unsqueeze(dim=1).repeat(1, num_gts, 1, 1, 1)
        torch.cuda.synchronize()
        return pred_imgs

    def second_stage(self, blur_img, pred_imgs):
        """
        :param blur_img: blurry image (b, c, h, w)
        :param pred_imgs: predicted images (b, num_gts, c, h, w)
        :return: predicted images after refinement (b, num_gts, c, h, w)
        """
        b, num_gts, c, h, w = pred_imgs.shape

        tensor_inp = [blur_img, ]

        if self.s1_to_s2:
            tensor_inp.append(pred_imgs.reshape(b, num_gts * c, h, w))

        if self.flow_to_s2:
            ref_imgs = pred_imgs[:, :-1]  # (b, num_gts - 1, c, h, w)
            ref_imgs = self.val_range * ref_imgs.reshape(b * (num_gts - 1), c, h, w)
            tgt_imgs = pred_imgs[:, 1:]  # (b, num_gts - 1, c, h, w)
            tgt_imgs = self.val_range * tgt_imgs.reshape(b * (num_gts - 1), c, h, w)
            flows = self.flow_estimator.batch_multi_inference(ref_imgs=ref_imgs, tgt_imgs=tgt_imgs)
            flows = flows.reshape(b, (num_gts - 1), 2, h, w)
            flows = flows.reshape(b, (num_gts - 1) * 2, h, w)
            tensor_inp.append(flows)

        tensor_inp = torch.cat(tensor_inp, dim=1)
        tensor_out = self.decomposer_s2(tensor_inp)

        return tensor_out

    @torch.no_grad()
    def inference(self, blur_img, trend, num_iters=0, full_results=False):
        """
        :param blur_img: blurry image (h, w, 3), tensor read by cv2.imread(), 0~255, RGB
        :param trend: flow trend guidance (2, ) or (h, w, 2)
        :return: predicted images (num_gts, h, w, 3)
        """
        self.eval()
        h, w, _ = blur_img.shape
        blur_img = torch.from_numpy(blur_img).float().to(self.device)
        blur_img = blur_img.permute(2, 0, 1).unsqueeze(dim=0) / self.val_range  # (1, 3, h, w)

        if len(trend.shape) == 1:
            trend = torch.from_numpy(trend).float().to(self.device)
            trend = trend[None, None, None].repeat(1, h, w, 1).permute(0, 3, 1, 2)  # (1, 2, h ,w)
        elif len(trend.shape) == 3:
            trend = torch.from_numpy(trend).float().to(self.device)
            trend = trend[None].permute(0, 3, 1, 2)  # (1, 2, h ,w)
        else:
            raise NotImplementedError

        pred_imgs_s1 = self.first_stage(blur_img=blur_img, trend_img=trend)
        torch.cuda.synchronize()
        for i in range(num_iters):
            if self.residual:
                pred_residual = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1)
                pred_imgs_s2 = pred_imgs_s1 + pred_residual
            else:
                pred_imgs_s2 = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1)
            pred_imgs_s1 = pred_imgs_s2
        if num_iters == 0:
            pred_imgs_s2 = pred_imgs_s1
        pred_imgs_s2 = pred_imgs_s2.squeeze(dim=0)  # (num_gts, 3, h, w)
        pred_imgs_s2 = torch.clamp(pred_imgs_s2 * self.val_range, 0, 255)
        pred_imgs_s2 = pred_imgs_s2.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)  # (num_gts, h, w, 3)

        if not full_results:
            return pred_imgs_s2
        else:
            pred_imgs_s1 = pred_imgs_s1.squeeze(dim=0)  # (num_gts, 3, h, w)
            pred_imgs_s1 = torch.clamp(pred_imgs_s1 * self.val_range, 0, 255)
            pred_imgs_s1 = pred_imgs_s1.permute(0, 2, 3, 1).detach().cpu().numpy().astype(
                np.uint8)  # (num_gts, h, w, 3)
            return pred_imgs_s2, pred_imgs_s1
