# Accuracy on imagenet validation set (single model)
# SE-ResNeXt101_32x4d  Acc1 = 80.236
# SE-ResNeXt50_32x4d Acc1 = 79.076
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from pretrainedmodels.models.senet import SENet, initialize_pretrained_model
from pretrainedmodels.models.senet import SEBottleneck
from pretrainedmodels.models.senet import SEResNetBottleneck
from pretrainedmodels.models.senet import SEResNeXtBottleneck
from pretrainedmodels.models.senet import pretrained_settings


class SENetEncoder(SENet):

    def __init__(self, input_channel=1, inplanes=64, input_3x3=True, *args, **kwargs):
        super().__init__(inplanes=inplanes, *args, **kwargs)
        self.pretrained = False
        self.inplanes = inplanes
        self.input_channel = input_channel
        if input_channel != 3:
            if input_3x3:
                layer0_modules = [
                    ('conv1', nn.Conv2d(input_channel, 64, 3, stride=2, padding=1,
                                        bias=False)),
                    ('bn1', nn.BatchNorm2d(64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                        bias=False)),
                    ('bn2', nn.BatchNorm2d(64)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                        bias=False)),
                    ('bn3', nn.BatchNorm2d(inplanes)),
                    ('relu3', nn.ReLU(inplace=True)),
                ]
            else:
                layer0_modules = [
                    ('conv1', nn.Conv2d(input_channel, inplanes, kernel_size=7, stride=2,
                                        padding=3, bias=False)),
                    ('bn1', nn.BatchNorm2d(inplanes)),
                    ('relu1', nn.ReLU(inplace=True)),
                ]
                # To preserve compatibility with Caffe weights `ceil_mode=True`
                # is used instead of `padding=1`.
            layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        if self.input_channel != 3:
            state_dict.pop('layer0.conv1.weight')
        super().load_state_dict(state_dict, **kwargs)


def se_resnext50_32x4d(num_classes=4, in_channels=1, pretrained='imagenet'):
    model = SENetEncoder(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=num_classes, input_channel=in_channels)
    if pretrained:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    return model