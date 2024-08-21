import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import timm
from typing import Tuple, Union, List, Optional, Callable
from enum import Enum
from timm.layers import (
    LayerNorm2d, 
    SelectAdaptivePool2d, 
    BatchNormAct2d, 
    Conv2dSame, 
    PatchEmbed, 
    RotaryEmbeddingCat,
    fast_layer_norm, 
    is_fast_norm, 
    get_same_padding, 
    to_3tuple, 
    Format as Format2d, 
    apply_keep_indices_nlc, 
    trunc_normal_,
    build_fourier_pos_embed,
)
from timm.layers.grn import GlobalResponseNorm
from timm.models.convnext import ConvNeXtBlock
from timm.models.eva import feature_take_indices, Eva
from timm.layers.trace_utils import _assert
from timm.layers.norm_act import _create_act

from src.utils.utils import efficientnet_init_weights_3d


_logger = logging.getLogger(__name__)
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


class BatchNormAct3d(nn.BatchNorm3d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=nn.ReLU,
            act_kwargs=None,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None,
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        _assert(x.ndim == 5, f'expected 5D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


def pad_same_3d(
        x,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1, 1),
        value: float = 0,
):
    ih, iw, id = x.size()[-3:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    pad_d = get_same_padding(id, kernel_size[2], stride[2], dilation[2])
    x = F.pad(
        x, 
        (
            pad_w // 2, 
            pad_w - pad_w // 2, 
            pad_h // 2, 
            pad_h - pad_h // 2,
            pad_d // 2, 
            pad_d - pad_d // 2,
        ), 
        value=value
    )
    return x


def conv3d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1, 1),
    padding: Tuple[int, int] = (0, 0, 0),
    dilation: Tuple[int, int] = (1, 1, 1),
    groups: int = 1,
):
    x = pad_same_3d(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, 0, dilation, groups, bias,
        )

    def forward(self, x):
        return conv3d_same(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


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


def conv_2d_to_3d(child_layer):
    assert child_layer.weight.shape[-1] == child_layer.weight.shape[-2]
    new_dim = child_layer.weight.shape[-1]
    new_conv_class = Conv3dSame if isinstance(child_layer, Conv2dSame) else nn.Conv3d

    new_child_layer = new_conv_class(
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

    return new_child_layer


def convert_2d_to_3d(layer):
    for child_layer_name, child_layer in layer.named_children():
        new_child_layer = None
        if isinstance(child_layer, (nn.Conv2d, Conv2dSame)):
            new_child_layer = conv_2d_to_3d(child_layer)
        elif isinstance(child_layer, LayerNorm2d):
            new_child_layer = LayerNorm3d(
                num_channels=child_layer.normalized_shape, 
                eps=child_layer.eps, 
                affine=child_layer.elementwise_affine
            )
            new_child_layer.weight = child_layer.weight
            new_child_layer.bias = child_layer.bias
        elif isinstance(child_layer, BatchNormAct2d):
            new_child_layer = BatchNormAct3d(
                num_features=child_layer.num_features, 
                eps=child_layer.eps, 
                momentum=child_layer.momentum, 
                affine=child_layer.affine, 
                track_running_stats=child_layer.track_running_stats, 
                apply_act=True, 
                act_layer=nn.ReLU, 
                act_kwargs=None,
                inplace=True, 
                drop_layer=None, 
            )
            new_child_layer.weight = child_layer.weight
            new_child_layer.bias = child_layer.bias
            new_child_layer.running_mean = child_layer.running_mean
            new_child_layer.running_var = child_layer.running_var
            new_child_layer.num_batches_tracked = child_layer.num_batches_tracked
            new_child_layer.act = child_layer.act
            new_child_layer.drop = child_layer.drop
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
        elif isinstance(child_layer, PatchEmbed):
            assert child_layer.output_fmt is None or child_layer.output_fmt == Format2d.NCHW, \
                f'PatchEmbed.output_fmt other than NCHW(D) is not supported for 3D data, got {child_layer.output_fmt}'
            assert child_layer.proj.weight.shape[-1] == child_layer.proj.weight.shape[-2]
            new_child_layer = PatchEmbed3d(
                img_size=repeat_last_size(child_layer.img_size, ndim=3),
                patch_size=repeat_last_size(child_layer.patch_size, ndim=3),
                in_chans=child_layer.proj.in_channels,
                embed_dim=child_layer.proj.out_channels,
                norm_layer=not isinstance(child_layer.norm, nn.Identity),
                flatten=child_layer.flatten,
                output_fmt=None,
                bias=child_layer.proj.bias is not None,
                strict_img_size=child_layer.strict_img_size,
                dynamic_img_pad=child_layer.dynamic_img_pad,
            )

            # Replace conv & norm weights
            new_child_layer.proj = conv_2d_to_3d(child_layer.proj)
            if not isinstance(child_layer.norm, nn.Identity):
                new_child_layer.norm.weight = child_layer.norm.weight
                new_child_layer.norm.bias = child_layer.norm.bias

            # Replace pos_embed
            layer.pos_embed = nn.Parameter(
                torch.zeros(1, new_child_layer.num_patches + layer.num_prefix_tokens, layer.embed_dim)) if layer.pos_embed is not None else None
            
            # Replace rotary embed
            if layer.rope is not None:
                num_heads = layer.blocks[0].attn.num_heads
                layer.rope = RotaryEmbeddingCat(
                    layer.embed_dim // num_heads,
                    in_pixels=False,
                    feat_shape=new_child_layer.grid_size,
                    ref_feat_shape=None,
                )
        else:
            # TODO: move to context manager
            if isinstance(child_layer, ConvNeXtBlock):
                setattr(child_layer, 'forward', types.MethodType(ConvNeXtBlock_forward_3d, child_layer))
            elif isinstance(child_layer, Eva):
                setattr(child_layer, 'forward_intermediates', types.MethodType(Eva_forward_intermediates_3d, child_layer))
                setattr(child_layer, '_pos_embed', types.MethodType(Eva__pos_embed_3d, child_layer))
                setattr(child_layer, 'init_weights', types.MethodType(Eva_init_weights_3d, child_layer))
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
        create_model_kwargs=None,
    ):
        assert not pretrained, 'Pretrained models could not be used for 3D data'

        super().__init__()
        kwargs = dict(
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
            ref_feat_shape=None,
        ) | (create_model_kwargs if create_model_kwargs is not None else {})

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)
        patch_first_conv(self.model, in_channels, pretrained=pretrained)
        convert_2d_to_3d(self.model)

        if not pretrained:
            if isinstance(self.model, timm.models.efficientnet.EfficientNet):
                efficientnet_init_weights_3d(self.model)
            elif isinstance(self.model, timm.models._features.FeatureGetterNet):
                self.model.model.init_weights()
            else:
                raise NotImplementedError(f'Initialization for {name} is not implemented, got class {self.model}')

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



class TimmUniversalEncoderEva3d(TimmUniversalEncoder3d):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        aux_create_model_kwargs = dict(
            img_size=112,
            embed_dim=792,
            out_indices=[3, 5, 7, 9, 11],
        )
        super().__init__(
            *args,
            **kwargs,
            create_model_kwargs=aux_create_model_kwargs,
        )

    def forward(self, x):
        features = self.model(x)
        # Eva's features are of same spatial size (// 4 w. r. t. input)
        # so we need to upsample last feature manually
        features[0] = F.interpolate(features[0], scale_factor=2, mode='nearest')
        features = [
            x,
        ] + features
        return features


def nchwd_to(x, fmt: FormatT):
    raise NotImplementedError()


class PatchEmbed3d(nn.Module):
    """ 3D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHWD
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_3tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv3d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed_3d(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1]), math.ceil(img_size[2] / self.patch_size[2])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], img_size[2] // self.patch_size[2]

    def forward(self, x):
        B, C, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
                _assert(D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
                _assert(
                    D % self.patch_size[2] == 0,
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[2]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        elif self.output_fmt != Format.NCHWD:
            x = nchwd_to(x, self.output_fmt)
        x = self.norm(x)
        return x



def resample_patch_embed_3d(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int, int): target shape (height, width, depth)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 5, "Five dimensions expected"
    assert len(new_size) == 3, "New shape should only be hw"
    old_size = patch_embed.shape[-3:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


def Eva_resample_abs_pos_embed_3d(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] * new_size[2] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1] == new_size[2]:
        return posemb

    if old_size is None:
        hwd = int((num_pos_tokens - num_prefix_tokens) ** (1. / 3))
        old_size = hwd, hwd, hwd

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], -1).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


def Eva_forward_intermediates_3d(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHWD',
        intermediates_only: bool = False,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """ Forward features that returns intermediates.
    Args:
        x: Input image tensor
        indices: Take last n blocks if an int, if is a sequence, select by matching indices
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
    """
    # NOTE: here we expect NCHWD format, but NCHW is ok because the method called from 2D model
    assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
    reshape = output_fmt == 'NCHW'
    intermediates = []
    take_indices, max_index = feature_take_indices(len(self.blocks), indices)

    # forward pass
    B, _, height, width, depth = x.shape
    x = self.patch_embed(x)
    x, rot_pos_embed = self._pos_embed(x)
    if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
        blocks = self.blocks
    else:
        blocks = self.blocks[:max_index + 1]
    for i, blk in enumerate(blocks):
        x = blk(x, rope=rot_pos_embed)
        if i in take_indices:
            intermediates.append(self.norm(x) if norm else x)

    # process intermediates
    if self.num_prefix_tokens:
        # split prefix (e.g. class, distill) and spatial feature tokens
        prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
        intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
    if reshape:
        # reshape to BCHW output format
        H, W, D = self.patch_embed.dynamic_feat_size((height, width, depth))
        intermediates = [y.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous() for y in intermediates]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(intermediates, prefix_tokens))

    if intermediates_only:
        return intermediates

    x = self.norm(x)

    return x, intermediates


def Eva__pos_embed_3d(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if self.dynamic_img_size:
        B, H, W, D, C = x.shape
        if self.pos_embed is not None:
            pos_embed = Eva_resample_abs_pos_embed_3d(
                self.pos_embed,
                (H, W, D),
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            pos_embed = None
        x = x.view(B, -1, C)
        rot_pos_embed = self.rope.get_embed(shape=(H, W, D)) if self.rope is not None else None
    else:
        pos_embed = self.pos_embed
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

    if self.cls_token is not None:
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

    if pos_embed is not None:
        x = x + pos_embed

    if self.reg_token is not None:
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)

    x = self.pos_drop(x)

    # obtain shared rotary position embedding and apply patch dropout
    if self.patch_drop is not None:
        x, keep_indices = self.patch_drop(x)
        if rot_pos_embed is not None and keep_indices is not None:
            rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
    return x, rot_pos_embed


def Eva_init_weights_3d(self):
    self.apply(self._init_weights)
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=.02)
    if self.cls_token is not None:
        trunc_normal_(self.cls_token, std=.02)
    if self.reg_token is not None:
        trunc_normal_(self.reg_token, std=.02)

    head_init_scale = 0.001
    self.fix_init_weight()
    if isinstance(self.head, nn.Linear):
        trunc_normal_(self.head.weight, std=.02)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


def Eva_build_rotary_pos_embed_3d(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 6,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb
