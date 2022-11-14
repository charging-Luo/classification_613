from torch.utils import model_zoo
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters, get_same_padding_conv2d, url_map


class EfficientNetEncoder(EfficientNet):

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        load_pretrained_weights2(model, model_name, in_channels=in_channels, load_fc=(num_classes == 1000))
        return model


def load_pretrained_weights2(model, model_name, in_channels, load_fc=False):
    """ 改写自 efficientnet_pytorch.utils.load_pretrained_weights
        以配合单通道的输入图片
        原input为rgb
     """
    state_dict = model_zoo.load_url(url_map[model_name])
    if in_channels != 3:
        state_dict.pop('_conv_stem.weight')
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        # assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


# 将EfficientNet作为特征提取器
# features = model.extract_features(img)
# print(features.shape)  # torch.Size([1, 1280, 7, 7])


# Print predictions
# print('-----')
# for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
#     prob = torch.softmax(outputs, dim=1)[0, idx].item()
#     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
