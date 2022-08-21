import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from .local_model import BiCycleGANModel


class GuidePredictor:
    '''
    Model for motion guide prediction (BiCycleGAN)
    Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency (NeurIPS 2017)
    https://proceedings.neurips.cc/paper/2017/file/819f46e52c25763a55cc642422644317-Paper.pdf
    '''

    def __init__(self, local_rank, configs):
        self.device = torch.device("cuda", local_rank)
        self.val_range = configs['val_range']

        opt = edict(configs['bicyclegan_args'])
        opt.gpu_ids = [local_rank, ]
        self.model = BiCycleGANModel(opt)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.setup(opt)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save_model(self, name):
        self.model.save_networks(name)

    def scheduler_step(self):
        self.model.update_learning_rate()

    def get_lr(self):
        return self.model.get_lr()

    def guidance2label(self, trend_img):
        """
        (0, 0) black -> 0
        (1, -1) red -> 1
        (-1,-1) green -> 2
        (-1, 1) yellow -> 3
        (1, 1) blue -> 4
        :param trend_img: shape (b, 2, h, w)
        :return: trend_label: shape (b, 1, h, w)
        """
        b, _, h, w = trend_img.shape
        trend_label = torch.zeros(b, 1, h, w).to(self.device)
        # (0, 0) black -> 0
        trend_label = torch.where((trend_img[:, 0:1] == 0) & (trend_img[:, 1:2] == 0),
                                  torch.tensor(0.).to(self.device), trend_label)
        # (1, -1) red -> 1
        trend_label = torch.where((trend_img[:, 0:1] == 1) & (trend_img[:, 1:2] == -1),
                                  torch.tensor(1.).to(self.device), trend_label)
        # (-1,-1) green -> 2
        trend_label = torch.where((trend_img[:, 0:1] == -1) & (trend_img[:, 1:2] == -1),
                                  torch.tensor(2.).to(self.device), trend_label)
        # (-1, 1) yellow -> 3
        trend_label = torch.where((trend_img[:, 0:1] == -1) & (trend_img[:, 1:2] == 1),
                                  torch.tensor(3.).to(self.device), trend_label)
        # (1, 1) blue -> 4
        trend_label = torch.where((trend_img[:, 0:1] == 1) & (trend_img[:, 1:2] == 1),
                                  torch.tensor(4.).to(self.device), trend_label)

        return trend_label

    def label2guidance(self, pred_label):
        """
        0 -> (0, 0) black
        1 -> (1, -1) red
        2 -> (-1,-1) green
        3 -> (-1, 1) yellow
        4 -> (1, 1) blue
        :param pred_label: shape (b, 1, h, w)
        :return: shape (b, 2, h, w)
        """
        b, _, h, w = pred_label.shape
        pred_trend_img = torch.zeros(b, 2, h, w).to(self.device)  # shape (b, 2, h, w)
        # 0 -> (0, 0) black
        pred_trend_img[:, 0:1] = torch.where(pred_label == 0, torch.tensor(0.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 0, torch.tensor(0.).to(self.device), pred_trend_img[:, 1:2])
        # 1 -> (1, -1) red
        pred_trend_img[:, 0:1] = torch.where(pred_label == 1, torch.tensor(1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 1, torch.tensor(-1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (-1,-1) green
        pred_trend_img[:, 0:1] = torch.where(pred_label == 2, torch.tensor(-1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 2, torch.tensor(-1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (-1, 1) yellow
        pred_trend_img[:, 0:1] = torch.where(pred_label == 3, torch.tensor(-1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 3, torch.tensor(1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (1, 1) blue
        pred_trend_img[:, 0:1] = torch.where(pred_label == 4, torch.tensor(1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 4, torch.tensor(1.).to(self.device), pred_trend_img[:, 1:2])

        return pred_trend_img

    def tensor_preprocess(self, inp_tensor):
        # preprocess data
        blur_img = inp_tensor['inp']  # (b, n, c, h, w), n=1, c=3
        assert blur_img.size(1) == 1  # (b, c, h, w)
        blur_img = blur_img.squeeze(dim=1) / self.val_range
        trend_img = inp_tensor['trend']  # (b, n, tc, h, w), n=1, tc=2
        trend_img = trend_img.squeeze(dim=1)  # (b, tc, h, w)
        trend_rev_img = -1 * trend_img.clone()
        label_img = self.guidance2label(trend_img)
        label_img = F.one_hot(label_img.long(), 5).squeeze(dim=1).permute(0, 3, 1, 2).float()  # (b, 5, h, w)
        label_rev_img = self.guidance2label(trend_rev_img)
        label_rev_img = F.one_hot(label_rev_img.long(), 5).squeeze(dim=1).permute(0, 3, 1, 2).float()  # (b, 5, h, w)
        tensor = {}
        tensor['A'] = blur_img  # (b, c, h, w)
        tensor['B'] = label_img
        tensor['B_rev'] = label_rev_img
        tensor['A_paths'] = inp_tensor['video']
        tensor['B_paths'] = inp_tensor['video']

        return tensor

    def return_tensor(self):
        out_tensor = {}
        # out_tensor['loss'] = sum(self.model.get_current_losses().values())
        out_tensor['loss'] = self.model.get_current_losses()
        out_tensor['inp_img'] = self.val_range * self.model.real_A_encoded
        _, fake_B_random = torch.max(self.model.fake_B_random, dim=1, keepdim=True)
        fake_B_random = self.label2guidance(fake_B_random)  # (b, 2, h, w)
        _, fake_B_encoded = torch.max(self.model.fake_B_encoded, dim=1, keepdim=True)
        fake_B_encoded = self.label2guidance(fake_B_encoded)
        out_tensor['pred_trends'] = torch.cat([fake_B_encoded, fake_B_random], dim=-1)  # (b, 2, h, 2 * w)
        _, real_B_encoded = torch.max(self.model.real_B_encoded, dim=1, keepdim=True)
        real_B_encoded = self.label2guidance(real_B_encoded)
        out_tensor['gt_trend'] = real_B_encoded

        return out_tensor

    def update(self, inp_tensor, training=True):
        tensor = self.tensor_preprocess(inp_tensor)
        self.model.set_input(tensor)
        if not self.model.is_train():
            return None
        if training:
            self.train()
            # calculate loss functions, get gradients, update network weights
            self.model.optimize_parameters()
        else:
            with torch.no_grad():
                self.eval()
                # calculate loss functions, get gradients, update network weights
                self.model.forward()

        return self.return_tensor()

    @torch.no_grad()
    def test(self, inp_tensor, zeros=False):
        tensor = self.tensor_preprocess(inp_tensor)
        self.model.set_input(tensor)
        real_A, fake_B, real_B = self.model.test(encode=False, turbulent=False, zeros=zeros)
        real_A = self.val_range * real_A
        _, fake_B = torch.max(fake_B, dim=1, keepdim=True)
        fake_B = self.label2guidance(fake_B)
        _, real_B = torch.max(real_B, dim=1, keepdim=True)
        real_B = self.label2guidance(real_B)
        out_tensor = {'inp_img': real_A, 'pred_trends': fake_B, 'gt_trend': real_B}

        return out_tensor
