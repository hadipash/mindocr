from typing import Tuple, List
from mindspore import nn, ops, Tensor
from .mindcv_models.resnet import ResNet, BasicBlock, Bottleneck, default_cfgs
from .mindcv_models.utils import load_pretrained
from ._registry import register_backbone, register_backbone_class

__all__ = ['DetResNet', 'det_resnet50', 'det_resnet18']


class DeformConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, group: int = 1, deform_group: int = 1, has_bias: bool = False, modulated: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, group=group,
                         has_bias=has_bias, weight_init='normal', bias_init='zeros')

        self.stride = (1, 1, stride, stride)
        self.padding = (padding,) * 4
        self.dilation = (1, 1, dilation, dilation)
        self.deform_group = deform_group
        self.modulated = modulated

        # FIXME: weight initialization here may cause a problem on Ascend architecture,
        # namely it should be a float number with the fractional part, i.e. `numpy.ones()` will not work.
        self.offset_conv = nn.Conv2d(in_channels, 3 * deform_group * (kernel_size ** 2), kernel_size,
                                     stride, pad_mode='pad', padding=self.padding, has_bias=True, weight_init='zeros')
        self._mask_offset = self.offset_conv.out_channels // 3

    def construct(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        offset[:, -self._mask_offset:] = ops.sigmoid(offset[:, -self._mask_offset:])
        return ops.deformable_conv2d(x, self.weight, offset, self.kernel_size, self.stride, self.padding,
                                     self.bias, self.dilation, self.group, self.deform_group, self.modulated)


class BottleneckDCN(Bottleneck):
    """
    ResNet Bottleneck with 3x3 2D convolution replaced by
    `Modulated Deformable Convolution <https://arxiv.org/abs/1811.11168>`__ .
    """
    def __init__(self, in_channels: int, channels: int, stride: int = 1, groups: int = 1, base_width: int = 64,
                 **kwargs):
        super().__init__(in_channels, channels, stride, groups, base_width, **kwargs)

        width = int(channels * (base_width / 64.0)) * groups
        self.conv2 = DeformConv2d(width, width, kernel_size=3, stride=stride, padding=1, group=groups)


@register_backbone_class
class DetResNet(ResNet):
    def __init__(self, block, layers, dcn=False, **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion
        self.out_channels = [ch * block.expansion for ch in [64, 128, 256, 512]]

        if dcn:
            self.input_channels = 64 * block.expansion  # reset the input channels counter. TODO: fix it?
            self.layer2 = self._make_layer(BottleneckDCN, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(BottleneckDCN, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(BottleneckDCN, 512, layers[3], stride=2)

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        '''
        ftrs = []
        for i, layer in enumerate([self.layer1, self.layer2,  self.layer3, self.layer4]):
            x = layer(x)
            if i in self.out_indices:
                ftrs.append(x)
                self.out_channels.append()
        '''
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


# TODO: load pretrained weight in build_backbone or use a unify wrapper to load


@register_backbone
def det_resnet18(pretrained: bool = True, **kwargs):
    model = DetResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet18']
        load_pretrained(model, default_cfg)

    return model


@register_backbone
def det_resnet50(pretrained: bool = True, **kwargs):
    model = DetResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return model
