import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.nn import BatchNorm2d, InstanceNorm2d
from skimage.metrics import structural_similarity as compare_ssim, peak_signal_noise_ratio as compare_psnr


class AverageMeter(object):
    """
    Compute and store the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def psnr_calculate(x, y):
    """
    Calculate PSNR value between two images
    Input range must be 0~255
    x,y shape (h,w,c)
    """
    return compare_psnr(y, x, data_range=255)


def ssim_calculate(x, y):
    """
    Calculate SSIM value between two images
    Input range must be 0~255
    x,y shape (h,w,c)
    """
    ssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=255)
    return ssim


def pad(x, pad_size):
    """
    Pad x to be divided by pad_size with no remainder
    """
    assert len(x.shape) == 4, 'len(x.shape) should be 4, but given {}!'.format(len(x.shape))
    n, c, h, w = x.shape
    ph = ((h - 1) // pad_size + 1) * pad_size
    pw = ((w - 1) // pad_size + 1) * pad_size
    padding = (0, pw - w, 0, ph - h)
    return F.pad(x, padding)


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']  # (b,n,2)

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())  # (h,w,2)
    number_of_leading_dimensions = len(mean.shape) - 1  # number of learning dimensions are 2
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape  # (1,1,h,w,2)
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)  # (b,n,1,1,1)
    coordinate_grid = coordinate_grid.repeat(*repeats)  # (b,n,h,w,2)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)  # (b,n,1,1,2)
    mean = mean.view(*shape)  # (b,n,1,1,2)

    mean_sub = (coordinate_grid - mean)  # (b,n,h,w,2)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out  # (b,n,h,w)


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)  # size -> (h, w, 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, use_norm=True, norm_type='batch'):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.use_norm = use_norm
        if self.use_norm:
            if norm_type == 'batch':
                self.norm1 = BatchNorm2d(in_features, affine=True)
                self.norm2 = BatchNorm2d(in_features, affine=True)
            elif norm_type == 'instance':
                self.norm1 = InstanceNorm2d(in_features, affine=False, track_running_stats=True)
                self.norm2 = InstanceNorm2d(in_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        if self.use_norm:
            out = self.norm1(x)
            out = F.relu(out)
        else:
            out = F.relu(x)
        out = self.conv1(out)
        if self.use_norm:
            out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_norm=True, norm_type='batch'):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.use_norm = use_norm
        if self.use_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm2d(out_features, affine=True)
            elif norm_type == 'instance':
                self.norm = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        if self.use_norm:
            out = self.norm(out)
        out = F.relu(out)
        return out


class UpBlockTrans2d(nn.Module):
    """
    Upsampling block for use in decoder using ConvTranspose.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_norm=True, norm_type='batch'):
        super(UpBlockTrans2d, self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=4, stride=2,
                                        padding=1)
        self.conv = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.use_norm = use_norm
        if self.use_norm:
            if norm_type == 'batch':
                self.norm1 = BatchNorm2d(out_features, affine=True)
                self.norm2 = BatchNorm2d(out_features, affine=True)
            elif norm_type == 'instance':
                self.norm1 = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
                self.norm2 = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        out = self.trans(x)
        if self.use_norm:
            out = self.norm1(out)
        out = F.relu(out)
        out = self.conv(out)
        if self.use_norm:
            out = self.norm2(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_norm=True, norm_type='batch'):
        super(DownBlock2d, self).__init__()
        self.use_norm = use_norm
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        if self.use_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm2d(out_features, affine=True)
            elif norm_type == 'instance':
                self.norm = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        out = self.conv(x)
        if self.use_norm:
            out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlockConv2d(nn.Module):
    """
    Downsampling block for use in encoder using conv instead of pooling.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_norm=True, norm_type='batch'):
        super(DownBlockConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding, groups=groups)
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size,
                               stride=2, padding=padding, groups=groups)
        self.use_norm = use_norm
        if self.use_norm:
            if norm_type == 'batch':
                self.norm1 = BatchNorm2d(out_features, affine=True)
                self.norm2 = BatchNorm2d(out_features, affine=True)
            elif norm_type == 'instance':
                self.norm1 = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
                self.norm2 = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        out = self.conv1(x)
        if self.use_norm:
            out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_norm:
            out = self.norm2(out)
        out = F.relu(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, use_norm=True, norm_type='batch'):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.use_norm = use_norm
        if self.use_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm2d(out_features, affine=True)
            elif norm_type == 'instance':
                self.norm = InstanceNorm2d(out_features, affine=False, track_running_stats=True)
            else:
                raise ValueError

    def forward(self, x):
        out = self.conv(x)
        if self.use_norm:
            out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


def positional_encoding_2d(d_model, height, width):
    """
    d_model: dimension of the model
    height: height of the positions
    width: width of the positions
    return d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def st_encoding_2d(x, t):
    """
    Adding spatial-temporal 2d encoding matrix
    x shape (b, c, h, w)
    """
    pe = positional_encoding_2d(*x.shape[-3:]) * t
    pe = pe.unsqueeze(dim=0).to(x.device)
    x += pe
    return x


def ckpt_convert(param):
    return {
        k.replace('module.', ''): v
        for k, v in param.items()
        if 'module.' in k
    }


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3 in FOMM
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# ----------------------------------------------------------------------------------------------
# Residual dense block
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


class DenseLayer(nn.Module):
    """
    Dense layer for residual dense block
    """

    def __init__(self, in_chs, growth_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.conv = conv3x3(in_chs, growth_rate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ResDenseBlock(nn.Module):
    """
    Residual Dense Block
    """

    def __init__(self, in_chs, growth_rate, num_layer, activation='relu'):
        super(ResDenseBlock, self).__init__()
        in_chs_acc = in_chs
        op = []
        for i in range(num_layer):
            op.append(DenseLayer(in_chs_acc, growth_rate, activation))
            in_chs_acc += growth_rate
        self.dense_layers = nn.Sequential(*op)
        self.conv1x1 = conv1x1(in_chs_acc, in_chs)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    """
    Middle network of residual dense blocks
    """

    def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
        self.conv3x3 = conv3x3(in_chs, in_chs)
        self.act = actFunc(activation)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.act(self.conv1x1(out))
        out = self.act(self.conv3x3(out))
        return out
# ----------------------------------------------------------------------------------------------
