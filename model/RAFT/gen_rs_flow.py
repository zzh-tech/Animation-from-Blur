import sys

sys.path.append('core')

import glob
import argparse
import torch
import cv2
import numpy as np
import os
from os.path import join
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

args.model = 'models/raft-things.pth'
args.path = '/home/zhong/Dataset/valid/'
# args.path = '/home/zhong/Dataset/RS-GOPRO/'
device = 'cuda'

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))
model = model.cuda()
model.eval()


def viz(img, flo):
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    cv2.imshow('img-flo', img_flo[:, :, ::-1] / 255.0)
    cv2.waitKey(1)


@torch.no_grad()
def gen_flow(img0, img1):
    img0 = torch.from_numpy(img0).permute(2, 0, 1).float()[None].to(device)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)
    flow_low, flow_up = model(img0, img1, iters=20, test_mode=True)
    flow_up = padder.unpad(flow_up)

    return flow_up[0].permute(1, 2, 0).cpu().numpy()


def main_dir(dir_path):
    rs_path = join(dir_path, 'RS')
    gs_path = join(dir_path, 'GS')
    flow_path = join(dir_path, 'FL')
    os.makedirs(flow_path, exist_ok=True)
    num_imgs = int(len(os.listdir(rs_path)) // 2)
    gs_frames = [0, 2, 4, 6, 8]
    for i in range(num_imgs):
        img0_t2b_file = join(rs_path, '{:08d}_rs_t2b.png'.format(i))
        img0_t2b = np.ascontiguousarray(cv2.imread(img0_t2b_file)[:, :, ::-1])
        img0_b2t_file = join(rs_path, '{:08d}_rs_b2t.png'.format(i))
        img0_b2t = np.ascontiguousarray(cv2.imread(img0_b2t_file)[:, :, ::-1])
        for j in gs_frames:
            img1_file = join(gs_path, '{:08d}_gs_{:03d}.png'.format(i, j))
            img1 = np.ascontiguousarray(cv2.imread(img1_file)[:, :, ::-1])
            flow_t2b = gen_flow(img1, img0_t2b)
            flow_b2t = gen_flow(img1, img0_b2t)
            np.save(join(flow_path, '{:08d}_fl_t2b_{:03d}.npy'.format(i, j)), flow_t2b)
            np.save(join(flow_path, '{:08d}_fl_b2t_{:03d}.npy'.format(i, j)), flow_b2t)
            viz(img0_t2b, flow_t2b)


def main(args):
    for dir in os.listdir(args.path):
        dir_path = join(args.path, dir)
        for sub_dir in [sub_dir for sub_dir in os.listdir(dir_path) if os.path.isdir(join(dir_path, sub_dir))]:
            sub_dir_path = join(dir_path, sub_dir)
            main_dir(sub_dir_path)


if __name__ == '__main__':
    main(args)
