import cv2
import torch


def _flow_rot_minus90(flow):
    rot_flow = flow.clone()
    # spatial rotation (-90 degree)
    rot_flow = torch.rot90(rot_flow, k=-1, dims=[-2, -1])
    rot_flow = rot_flow[:, [1, 0]]
    rot_flow[:, 0] = -1 * rot_flow[:, 0]
    return rot_flow


def _flow_rot_plus90(flow):
    rot_flow = flow.clone()
    # spatial rotation (+90 degree)
    rot_flow = torch.rot90(rot_flow, k=1, dims=[-2, -1])
    rot_flow = rot_flow[:, [1, 0]]
    rot_flow[:, 1] = -1 * rot_flow[:, 1]
    return rot_flow


def flow_rot90(flow, n=1):
    assert len(flow.shape) == 4
    if n == 0:
        return flow
    rot_func = _flow_rot_plus90 if n > 0 else _flow_rot_minus90
    for _ in range(abs(n)):
        flow = rot_func(flow)
    return flow


def flow_lr_flip(flow):
    assert len(flow.shape) == 4
    flip_flow = flow.clone()
    flip_flow = torch.flip(flip_flow, dims=[-1, ])
    flip_flow[:, 0] = -1 * flip_flow[:, 0]
    return flip_flow


def flow_ud_flip(flow):
    flip_flow = flow.clone()
    flip_flow = torch.flip(flip_flow, dims=[-2, ])
    flip_flow[:, 1] = -1 * flip_flow[:, 1]
    return flip_flow


def flow_diagonal_reverse(flow):
    reverse_flow = flow.clone()
    reverse_flow = -1 * reverse_flow
    return reverse_flow
