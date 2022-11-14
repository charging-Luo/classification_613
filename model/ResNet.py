import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from utils import patch_first_conv

layers = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3], "resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3]}
block = {"resnet18": BasicBlock, "resnet34": BasicBlock, "resnet50": Bottleneck, "resnet101": Bottleneck}


class ResNetEncoder(ResNet):
    def __init__(self, model_name, input_channel=1, num_classes=2, *args, **kwargs):
        super().__init__(block[model_name], layers[model_name], num_classes=num_classes,
                         *args, **kwargs)
        self.input_channel = input_channel

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)
        if self.input_channel != 3:
            patch_first_conv(self, self.input_channel)

    def load_state_dict_test(self, state_dict, **kwargs):
        if self.input_channel != 3:
            patch_first_conv(self, self.input_channel)
        super().load_state_dict(state_dict, **kwargs)


def resnet(name='resnet18', num_classes=4, in_channels=1, pretrained='imagenet'):
    model = ResNetEncoder(model_name=name, num_classes=num_classes, input_channel=in_channels)
    if pretrained:
        settings = pretrained_settings[name][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    net = model = resnet(name="resnet18", num_classes=4, in_channels=1).cuda()  # 注意需要把模型放到cuda上 .cuda()
    summary(net, (1, 256, 512))
