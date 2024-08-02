import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import timm
from typing import Tuple, Union
from enum import Enum
from timm.layers import LayerNorm2d, SelectAdaptivePool2d, fast_layer_norm, is_fast_norm
from timm.layers.grn import GlobalResponseNorm
from timm.models.convnext import ConvNeXtBlock


_int_tuple_3_t = Union[int, Tuple[int, int, int]]


# Format

class Format(str, Enum):
    NCHWD = 'NCHWD'
    NHWDC = 'NHWDC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWDC:
        dim = (1, 2, 3)
    else:
        dim = (2, 3, 4)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWDC:
        dim = 4
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


# Selective Adaptive Pooling

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def adaptive_avgmax_pool3d(x, output_size: _int_tuple_3_t = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool3d(x, output_size: _int_tuple_3_t = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool3d(x, pool_type='avg', output_size: _int_tuple_3_t = 1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool3d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool3d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool3d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool3d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: F = 'NCHWD'):
        super(FastAdaptiveAvgPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten)


class FastAdaptiveMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.amax(self.dim, keepdim=not self.flatten)


class FastAdaptiveAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim, keepdim=not self.flatten)
        x_max = x.amax(self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max


class FastAdaptiveCatAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveCatAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim_reduce, keepdim=not self.flatten)
        x_max = x.amax(self.dim_reduce, keepdim=not self.flatten)
        return torch.cat((x_avg, x_max), self.dim_cat)


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: _int_tuple_3_t = 1):
        super(AdaptiveAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool3d(x, self.output_size)


class AdaptiveCatAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: _int_tuple_3_t = 1):
        super(AdaptiveCatAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool3d(x, self.output_size)


class SelectAdaptivePool3d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_3_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHWD',
    ):
        super(SelectAdaptivePool3d, self).__init__()
        assert input_fmt in ('NCHWD', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        pool_type = pool_type.lower()
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHWD':
            assert output_size == 1, 'Fast pooling and non NCHWD input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type == 'fast' or pool_type.endswith('avg'):
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.Identity()
        else:
            assert input_fmt == 'NCHWD'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool3d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool3d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool3d(output_size)
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool3d(output_size)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

# Utils

def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True, conv_type=nn.Conv2d):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    if new_in_channels == default_in_channels:
        return

    # get first conv
    for module in model.modules():
        if isinstance(module, conv_type) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()
    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


class LayerNorm3d(nn.LayerNorm):
    """ LayerNorm for channels of '3D' spatial NCHWD tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x
    

class GlobalResponseNorm3d(nn.Module):
    """ Global Response Normalization layer for 3d data
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2, 3)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3, 4)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)


def ConvNeXtBlock_forward_3d(self, x):
    shortcut = x
    x = self.conv_dw(x)
    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 4, 1, 2, 3)
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))

    x = self.drop_path(x) + self.shortcut(shortcut)
    return x


def repeat_last_size(size, ndim):
    if isinstance(size, int):
        return (size, ) * ndim
    if ndim < len(size):
        return size
    return (*size, *(size[-1] for _ in range(ndim - len(size))))


def convert_2d_to_3d(layer):
    for child_layer_name, child_layer in layer.named_children():
        new_child_layer = None
        if isinstance(child_layer, nn.Conv2d):
            assert child_layer.weight.shape[-1] == child_layer.weight.shape[-2]
            new_dim = child_layer.weight.shape[-1]

            new_child_layer = nn.Conv3d(
                in_channels=child_layer.in_channels, 
                out_channels=child_layer.out_channels, 
                kernel_size=repeat_last_size(child_layer.kernel_size, ndim=3), 
                stride=repeat_last_size(child_layer.stride, ndim=3), 
                padding=repeat_last_size(child_layer.padding, ndim=3), 
                dilation=repeat_last_size(child_layer.dilation, ndim=3), 
                groups=child_layer.groups, 
                bias=child_layer.bias is not None,
            )
            new_child_layer.weight = nn.Parameter(
                (
                    (child_layer.weight[..., None, :, :] / new_dim) + 
                    (child_layer.weight[..., :, None, :] / new_dim) + 
                    (child_layer.weight[..., :, :, None] / new_dim)
                ) / 3,
                requires_grad=child_layer.weight.requires_grad
            )
            if child_layer.bias is not None:
                new_child_layer.bias = child_layer.bias
        elif isinstance(child_layer, LayerNorm2d):
            new_child_layer = LayerNorm3d(
                num_channels=child_layer.normalized_shape, 
                eps=child_layer.eps, 
                affine=child_layer.elementwise_affine
            )
            new_child_layer.weight = child_layer.weight
            new_child_layer.bias = child_layer.bias
        elif isinstance(child_layer, GlobalResponseNorm):
            new_child_layer = GlobalResponseNorm3d(
                dim=child_layer.weight.shape[0], 
                eps=child_layer.eps, 
                channels_last=child_layer.channel_dim == -1,
            )
            new_child_layer.weight = child_layer.weight
            new_child_layer.bias = child_layer.bias
        elif isinstance(child_layer, SelectAdaptivePool2d):
            input_fmt = child_layer.pool.input_fmt if hasattr(child_layer.pool, 'input_fmt') else 'NCHWD'
            new_child_layer = SelectAdaptivePool3d(
                output_size=repeat_last_size(child_layer.pool.output_size, ndim=3), 
                pool_type=child_layer.pool_type, 
                flatten=isinstance(child_layer.pool, nn.Flatten),
                input_fmt=input_fmt,
            )
        else:
            # TODO: move to context manager
            if isinstance(child_layer, ConvNeXtBlock):
                setattr(child_layer, 'forward', types.MethodType(ConvNeXtBlock_forward_3d, child_layer))
            convert_2d_to_3d(child_layer)

        if new_child_layer is not None:
            setattr(layer, child_layer_name, new_child_layer)


class TimmUniversalEncoder3d(nn.Module):
    def __init__(
        self, 
        name, 
        pretrained=True, 
        in_channels=3, 
        depth=5, 
        output_stride=32, 
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    ):
        super().__init__()
        kwargs = dict(
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)
        patch_first_conv(self.model, in_channels)
        convert_2d_to_3d(self.model)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride
        
        self.strides = strides

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
