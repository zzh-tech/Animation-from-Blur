import cv2
import random
import torch
import numpy as np
import albumentations as A

try:
    # from trend import trend_rot90, trend_lr_flip, trend_ud_flip, trend_diagonal_reverse
    from flow import flow_rot90, flow_lr_flip, flow_ud_flip, flow_diagonal_reverse
except:
    # from data.trend import trend_rot90, trend_lr_flip, trend_ud_flip, trend_diagonal_reverse
    from data.flow import flow_rot90, flow_lr_flip, flow_ud_flip, flow_diagonal_reverse


class Compose:
    """
    Compose a series of image augmentations
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, list) and len(transforms) > 0
        self.tranforms = transforms

    def __call__(self, image, bbox, flow=False, trend=False, replay_args=None):
        # Copy the list
        replay_args = list(replay_args) if isinstance(replay_args, list) else None
        out = {'image': image, 'bbox': bbox}
        replay_args_record = []
        for transform in self.tranforms:
            args = replay_args.pop(0) if isinstance(replay_args, list) else None
            out, args = transform(**out, flow=flow, trend=trend, args=args)
            replay_args_record.append(args)

        out['replay_args'] = replay_args_record
        return out


class NearBBoxResizedSafeCrop:
    """
    Crop near the outside of the bbox and resize the cropped image
    max_ratio means the max valid crop ratio from boundary of the bbox to boundary of the image, (0, 1)
    bbox format is pascal_voc, a bounding box looks like [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212]
    """

    def __init__(self, height, width, max_ratio=0.1):
        self.height = height
        self.width = width
        self.max_ratio = max_ratio

    def __call__(self, image, bbox, flow=False, trend=False, args=None):
        if len(bbox) == 5:
            x_min, y_min, x_max, y_max, _ = bbox
        else:
            x_min, y_min, x_max, y_max = bbox
        img_h, img_w, _ = image.shape

        # Prepare args
        if args is None:
            args = {}
            args['ratio'] = self.max_ratio * random.uniform(0, 1)

        # Crop image
        ratio = args['ratio']
        x_min = int((1 - ratio) * x_min)
        y_min = int((1 - ratio) * y_min)
        x_max = int(x_max + ratio * (img_w - x_max))
        y_max = int(y_max + ratio * (img_h - y_max))
        image = image[y_min:y_max, x_min:x_max]
        if trend:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
        else:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Adjust the values based on the size if it is optical flow
        if flow:
            image[:, :, 0] *= self.width / float(x_max - x_min)
            image[:, :, 1] *= self.height / float(y_max - y_min)

        # args: arguments for replaying this augmentation
        # todo: re-calculate the bbox
        return {'image': image, 'bbox': None}, args


class RandomCrop:
    """
    Randomly crop the image based on the given cropping size
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, bbox=None, flow=False, trend=False, args=None):
        img_h, img_w, _ = image.shape

        # Prepare args
        if args is None:
            args = {}
            args['x_min'] = random.randint(0, img_w - self.width)
            args['y_min'] = random.randint(0, img_h - self.height)

        # Crop image
        x_min = args['x_min']
        y_min = args['y_min']
        x_max = x_min + self.width
        y_max = y_min + self.height
        image = image[y_min:y_max, x_min:x_max]

        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class Resize:
    """
    resize the image based on the given size
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, bbox=None, flow=False, trend=False, args=None):
        img_h, img_w, _ = image.shape

        # Resize image
        if trend:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
        else:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if flow:
            image[:, :, 0] *= self.width / img_w
            image[:, :, 1] *= self.height / img_h

        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class Rot90:
    '''
    Rotate n times 90 degree for the input tensor (counter-wise)
    '''

    def __call__(self, image, bbox, flow=False, trend=False, args=None):
        # Prepare args
        if args is None:
            args = {}
            args['rot_num'] = random.randint(-2, 2)

        # Rotate image
        n = args['rot_num']
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # (1, c, h, w)
        if trend or flow:
            image = flow_rot90(image, n=n)
            # image = trend_rot90(image, n=n)
        else:
            image = torch.rot90(image, k=n, dims=[-2, -1])
        image = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float)

        return {'image': image, 'bbox': bbox}, args


class Flip:
    '''
    Flip the input tensor
    flip_flag == 0: keep original
    flip_flag == 1: left-right flipping
    flip_flog == 2: up-down flipping
    '''

    def __call__(self, image, bbox, flow=False, trend=False, args=None):
        # Prepare args
        if args is None:
            args = {}
            args['flip_flag'] = random.randint(0, 2)

        # Flip image
        flip_flag = args['flip_flag']
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # (1, c, h, w)
        if trend or flow:
            if flip_flag == 0:
                pass
            elif flip_flag == 1:
                image = flow_lr_flip(image)
                # image = trend_lr_flip(image)
            elif flip_flag == 2:
                image = flow_ud_flip(image)
                # image = trend_ud_flip(image)
            else:
                raise NotImplementedError('flip_flag: {}'.format(flip_flag))
        else:
            if flip_flag == 0:
                pass
            elif flip_flag == 1:
                image = torch.flip(image, dims=[-1, ])
            elif flip_flag == 2:
                image = torch.flip(image, dims=[-2, ])
            else:
                raise NotImplementedError('flip_flag: {}'.format(flip_flag))
        image = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float)

        return {'image': image, 'bbox': bbox}, args


class ColorJitter:
    def __init__(self, **kwargs):
        self.transform = A.ReplayCompose([A.ColorJitter(**kwargs), ])

    def __call__(self, image, bbox, flow=False, trend=False, args=None):
        if flow or trend:
            return {'image': image, 'bbox': bbox}, args

        # Prepare args
        if args is None:
            tsf_image = self.transform(image=image.astype(np.uint8))
            args = tsf_image['replay']
            image = tsf_image['image'].astype(np.float)
        else:
            image = A.ReplayCompose.replay(args, image=image.astype(np.uint8))['image'].astype(np.float)

        return {'image': image, 'bbox': bbox}, args


class ComposeV:
    """
    Compose a series of video augmentations
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, list) and len(transforms) > 0
        self.tranforms = transforms

    def __call__(self, images, flow=False, trend=False, replay_args=None):
        # Copy the list
        replay_args = list(replay_args) if isinstance(replay_args, list) else None
        out = {'images': images, }
        replay_args_record = []
        for transform in self.tranforms:
            args = replay_args.pop(0) if isinstance(replay_args, list) else None
            out, args = transform(**out, flow=flow, trend=trend, args=args)
            replay_args_record.append(args)

        out['replay_args'] = replay_args_record
        return out


class Reverse:
    """
    Temporal flipping
    """

    def __call__(self, images, flow=False, trend=False, args=None):
        # Prepare args
        if args is None:
            args = {}
            args['reverse_flag'] = random.randint(0, 1)

        # Reverse images
        reverse_flag = args['reverse_flag']
        if reverse_flag == 0:
            pass
        elif reverse_flag == 1:
            if trend or flow:
                images = images[::-1]
                for i, image in enumerate(images):
                    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # (1, c, h, w)
                    image = flow_diagonal_reverse(image)
                    # image = trend_diagonal_reverse(image)
                    images[i] = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float)
            else:
                images = images[::-1]
        else:
            raise NotImplementedError

        return {'images': images, }, args
