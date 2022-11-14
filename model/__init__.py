from model.InceptionResNetV2 import inceptionresnetv2
from model.SE_ResNeXt50_32x4d import se_resnext50_32x4d
from model.ResNet import resnet
from efficientnet_pytorch import EfficientNet

input_size = {"InceptionResNetV2": (299, 299),
              "SE-ResNeXt50_32x4d": (224, 224),
              "efficientnet": (224, 224),
              "resnet":(224,224)}  # 使用的AdaptiveAvgPool2d，未固定池化大小，但论文中基础b0模型使用的是224


def get_resize_shape(model_name):
    if 'resnet' in model_name:
        return input_size["SE-ResNeXt50_32x4d"]
    else:
        return input_size[model_name]


def get_model(model_name, num_classes, in_channels=3, pretrained=False):
    if 'efficientnet' in model_name:
        if pretrained:
            model = EfficientNet.from_pretrained(model_name, num_classes=num_classes, in_channels=in_channels)
        else:
            model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
            # from_name原码不支持改in_channels
            if in_channels != 3:
                from efficientnet_pytorch.utils import round_filters, get_same_padding_conv2d
                Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
                out_channels = round_filters(32, model._global_params)
                model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    elif 'resnet' in model_name:
        assert model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]
        if pretrained:
            pretrained = 'imagenet'
        model = resnet(name=model_name, num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    elif model_name == 'InceptionResNetV2':
        if pretrained:
            pretrained = 'imagenet'
        model = inceptionresnetv2(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    elif model_name == 'SE-ResNeXt50_32x4d':
        if pretrained:
            pretrained = 'imagenet'
        model = se_resnext50_32x4d(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    else:
        raise RuntimeError("{} not support".format(model_name))
    return model


def finetune_schedule(model, mode):
    import torch.nn as nn

    def _01():
        """freeze backbone的参数，只更新（预训练）新的fc layer的参数（更新的参数量少，训练更快）到收敛为止，之后再放开所有层的参数，再一起训练"""
        for k in model.children():
            if not isinstance(k, nn.Linear):
                for param in k.parameters():
                    param.requires_grad = False
            else:
                return k.parameters()  # 仅返回fc层参数

    def _02():
        """差分学习率: 即更新backbone参数和新fc layer的参数所使用的学习率是不一致的，一般可选择差异10倍"""
        params_list = []
        for k in model.children():
            if isinstance(k, nn.Linear):
                ignored_params = list(map(id, k.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
                params_list.append({'params': base_params})
                params_list.append({'params': k.parameters(), 'lr': 0.01})
                return params_list

    def _03():
        """freeze浅层，训练深层（如可以不更新resnet前两个resnet block的参数，只更新其余的参数，以增强泛化，减少过拟合）
        此功能仿佛要根据具体网络单独写
        """
        raise Exception('This function do not implement yet!')

    def default():
        return model.parameters()

    switch = {"00": default(),
              "01": _01(),
              "02": _02(),
              "03": _03()}
    switch.get(model)


if __name__ == '__main__':
    model_name = 'resnet18'
    model = get_model(model_name, num_classes=2, in_channels=3, pretrained=True).cuda()
    from torchsummary import summary  # 类似keras model.summary()
    h, w = get_resize_shape(model_name)
    summary(model, (1, h, w))




