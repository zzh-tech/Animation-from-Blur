import torch
from model.RAFT.core.raft import RAFT
from easydict import EasyDict as edict
from model.utils import ckpt_convert


class FlowEstimator:
    """
    This estimator is relied on RAFT
    RAFT: https://github.com/princeton-vl/RAFT
    """

    def __init__(self, device, checkpoint, iters=20):
        self.iters = iters
        self.device = device
        args = edict({'mixed_precision': False, 'small': False, 'alternate_corr': False})
        self.model = RAFT(args).to(device)
        self.model.load_state_dict(ckpt_convert(torch.load(checkpoint)))
        self.model.eval()

    @torch.no_grad()
    def batch_inference(self, ref_img, tgt_imgs):
        """
        infer the optical flows from tgt_imgs to ref_img
        value range of ref_img and tgt_imgs should be 0 ~ 255
        :param ref_img: referred image (b, c, h, w)
        :param tgt_imgs: target images (b * num_gts, c, h, w)
        :return flows: optical flows from target images to referred image (b * num_gts, 2, h, w)
        """
        ref_img = ref_img.detach()
        tgt_imgs = tgt_imgs.detach()
        ref_imgs = ref_img.repeat(int(tgt_imgs.size(0) / ref_img.size(0)), 1, 1, 1)  # (b * num_gts, c, h, w)
        _, flows = self.model(tgt_imgs, ref_imgs, iters=self.iters, test_mode=True)
        return flows.detach()

    @torch.no_grad()
    def batch_multi_inference(self, ref_imgs, tgt_imgs):
        '''
        infer the optical flows from tgt_imgs to ref_imgs
        value range of ref_img and tgt_imgs should be 0 ~ 255
        :param ref_imgs: referred images (b * num_gts, c, h, w)
        :param tgt_imgs: target images (b * num_gts, c, h, w)
        :return: optical flows from target images to referred image (b * num_gts, 2, h, w)
        '''
        ref_imgs = ref_imgs.detach()
        tgt_imgs = tgt_imgs.detach()
        _, flows = self.model(tgt_imgs, ref_imgs, iters=self.iters, test_mode=True)  # (b * (num_gts-1), c, h, w)
        return flows.detach()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_estimator = FlowEstimator(device, checkpoint='../checkpoints/raft-sintel.pth', iters=20)
    # ref_img = torch.randn(4, 3, 192, 160).to(device)
    # tgt_imgs = torch.randn(7 * 4, 3, 192, 160).to(device)
    # flows = flow_estimator.batch_inference(ref_img, tgt_imgs)
    # print(flows.shape)

    import cv2
    import numpy as np
    from os.path import join
    from data.flow_viz import flow_to_image

    data_root = './RAFT/demo-frames'
    img_path = join(data_root, 'frame_{:04d}.png'.format(20))
    ref_img = np.ascontiguousarray(cv2.resize(cv2.imread(img_path), (800, 400), cv2.INTER_AREA)[:, :, ::-1])
    ref_img = ref_img[np.newaxis, ...]
    ref_img = torch.from_numpy(ref_img).permute(0, 3, 1, 2).float().to(device)
    ref_img = ref_img.repeat(10, 1, 1, 1)

    tgt_imgs = []
    for i in range(16, 26):
        img_path = join(data_root, 'frame_{:04d}.png'.format(i))
        tgt_img = np.ascontiguousarray(cv2.resize(cv2.imread(img_path), (800, 400), cv2.INTER_AREA)[:, :, ::-1])
        tgt_imgs.append(tgt_img)
    tgt_imgs = np.stack(tgt_imgs, axis=0)
    tgt_imgs = torch.from_numpy(tgt_imgs).permute(0, 3, 1, 2).float().to(device)

    flows = flow_estimator.batch_inference(ref_img, tgt_imgs)
    for flow in flows:
        flow = flow.permute(1, 2, 0).cpu().numpy()
        flow_vis = flow_to_image(flow)
        cv2.imshow('flow', flow_vis / 255.0)
        cv2.waitKey(100)
