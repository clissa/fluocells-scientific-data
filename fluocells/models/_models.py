"""
This submodule contains implementations of suggested architectures to deal with fluocells data.

Currently available:
 - cell ResUnet (c-resunet): Morelli, R., Clissa, L. et al., SciRep (2021) https://www.nature.com/articles/s41598-021-01929-5

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-27
License: Apache License 2.0
"""


__all__ = ["cResUnet", "c_resunet"]

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
from fluocells.config import MODELS_PATH


class cResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=2):
        super(cResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.encoder = nn.ModuleDict(
            {
                "colorspace": nn.Conv2d(3, 1, kernel_size=1, padding=0),
                # block 1
                "conv_block": ConvBlock(1, n_features_start),
                "pool1": nn.MaxPool2d(pool_ks, pool_stride, pool_pad),
                # block 2
                "residual_block1": ResidualBlock(
                    n_features_start, 2 * n_features_start, is_conv=True
                ),
                "pool2": nn.MaxPool2d(pool_ks, pool_stride, pool_pad),
                # block 3
                "residual_block2": ResidualBlock(
                    2 * n_features_start, 4 * n_features_start, is_conv=True
                ),
                "pool3": nn.MaxPool2d(pool_ks, pool_stride, pool_pad),
                # bottleneck
                "bottleneck": Bottleneck(
                    4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2
                ),
            }
        )

        self.decoder = nn.ModuleDict(
            {
                # block 6
                "upconv_block1": UpResidualBlock(
                    n_in=8 * n_features_start, n_out=4 * n_features_start
                ),
                # block 7
                "upconv_block2": UpResidualBlock(
                    4 * n_features_start, 2 * n_features_start
                ),
                # block 8
                "upconv_block3": UpResidualBlock(
                    2 * n_features_start, n_features_start
                ),
            }
        )

        # output
        self.head = Heatmap2d(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if "block" in lbl:
                downblocks.append(x)
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode="fan_in"):
        print("Initializing conv2d weights with Kaiming He normal")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet(
    arch: str,
    n_features_start: int,
    n_out: int,
    #     block: Type[Union[BasicBlock, Bottleneck]],
    #     layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs,
) -> cResUnet:
    model = cResUnet(n_features_start, n_out)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        weights_path = MODELS_PATH / f"{arch}_state_dict.pkl"
        print("loading pretrained Keras weights from", weights_path)
        keras_weights = load_pkl(weights_path)
        keras_state_dict = pt2k_state_dict(model.state_dict())
        assert len(keras_weights) == len(keras_state_dict)
        transfer_weights(model, keras_weights, keras_state_dict)
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #         model.load_state_dict(state_dict)
    else:
        model.init_kaiming_normal()
    return model


def c_resunet(
    arch="c-ResUnet",
    n_features_start: int = 16,
    n_out: int = 2,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> cResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet(
        arch=arch,
        n_features_start=n_features_start,
        n_out=n_out,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )
