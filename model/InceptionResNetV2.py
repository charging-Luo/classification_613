# Accuracy on imagenet validation set (single model)
# InceptionResNetV2  Acc1 = 80.170
# Xception    Acc1 = 78.888
# model_name = "InceptionResNetV2"
# model = pretrainedmodels.__dict__[model_name](num_classes=4, pretrained='imagenet')
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2, BasicConv2d
from pretrainedmodels.models.inceptionresnetv2 import pretrained_settings


class InceptionResNetV2Encoder(InceptionResNetV2):

    def __init__(self, input_channel=1, num_classes=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channel = input_channel
        if self.input_channel != 3:
            self.conv2d_1a = BasicConv2d(input_channel, 32, kernel_size=3, stride=2)
        self.last_linear = nn.Linear(1536, num_classes)

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        if self.input_channel != 3:
            state_dict.pop('conv2d_1a.conv.bias')
            state_dict.pop('conv2d_1a.conv.weight')
        super().load_state_dict(state_dict, **kwargs)


def inceptionresnetv2(num_classes=4, in_channels=1, pretrained='imagenet'):
    model = InceptionResNetV2Encoder(num_classes=num_classes, input_channel=in_channels)
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)

    return model