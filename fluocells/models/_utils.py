"""
This submodule contains utils for handling fluocells.models components.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-27
License: Apache License 2.0
"""


__all__ = [
    "rgetattr",
    "rsetattr",
    "save_pkl",
    "load_pkl",
    "pt2k_state_dict",
    "transfer_weights",
    "load_model",
    "get_features",
    "get_layer_name",
]

import functools
import pickle
from collections import OrderedDict

import torch
import fastai.layers
from torchvision.models.feature_extraction import create_feature_extractor


def save_pkl(d, path):
    with open(path, "wb") as f:
        pickle.dump(d, f)


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d


def load_model(arch: str = "c-ResUnet", mode: str = "eval"):
    from fluocells.models import c_resunet

    # arch = 'c-ResUnet_noWM'
    model = c_resunet(arch=arch, n_features_start=16, n_out=2, pretrained=True)
    # for m in model.modules():
    #     for child in m.children():
    #         if type(child) == nn.BatchNorm2d:
    #             child.track_running_stats = False
    #             child.running_mean = None
    #             child.running_var = None
    if mode == "eval":
        model.eval()
    return model


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def k2pt_weights(keras_w):
    k_w = keras_w.copy()
    if k_w.ndim == 4:  # convolution filter
        torch_w = torch.from_numpy(k_w.transpose((3, 2, 0, 1)))
    elif k_w.ndim == 1:  # convolution bias; batchnorm weight/gamma and bias/beta
        torch_w = torch.from_numpy(k_w)
    else:
        raise ValueError(
            f"Unexpected shape {k_w.shape} has dimension {k_w.ndim} instead of 1 or 4."
        )
    return torch_w


def transfer_weights(model, k_dict, pt_dict, freeze=True):
    for pt_key, k_weight in zip(pt_dict.keys(), k_dict.values()):
        if freeze:
            with torch.no_grad():
                rsetattr(model, f"{pt_key}.data", k2pt_weights(k_weight))
        else:
            rsetattr(model, f"{pt_key}.data", k2pt_weights(k_weight))


def get_features(img: torch.Tensor, model, layer: list) -> dict:
    """
    Return dictionary with layer as key and features tensor as value.
    :param img: input image in torch format and [0, 1.] range
    :param model: torch model
    :param layer: list containing the name of the layer in string format (as returned by torchvision.models.feature_extraction.get_graph_node_names)
    :return: features dict
    """
    with torch.no_grad():
        feature_extractor = create_feature_extractor(model, return_nodes=layer)
        features = feature_extractor(img)
    return features


def pt2k_state_dict(d):
    """
    Return state_dict without PyTorch-specific layers. This makes it comparable with Keras weight format
    :param d: pytorch model state_dict
    :return: state_dict in Keras-like format
    """
    fixed = OrderedDict({k: v for k, v in d.items() if not "num_batches_tracked" in k})
    return fixed


def get_layer_name(layer, idx):
    # TODO: minimal implementation based on class name + idx
    # type_str = str(type(layer))
    # type_str = type_str.split('.')[1][:-2]
    # return f"{type_str}_{idx}"

    if isinstance(layer, torch.nn.Conv2d):
        layer_name = "Conv2d_{}_{}x{}".format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        layer_name = "ConvT2d_{}_{}x{}".format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.BatchNorm2d):
        layer_name = "BatchNorm2D_{}_{}".format(idx, layer.num_features)
    elif isinstance(layer, torch.nn.Linear):
        layer_name = "Linear_{}_{}x{}".format(
            idx, layer.in_features, layer.out_features
        )
    elif isinstance(layer, fastai.layers.Identity):
        layer_name = "Identity"
    else:
        layer_name = "Activation_{}".format(idx)
    # idx += 1
    # return layer_name, idx
    return "_".join(layer_name.split("_")[:2]).lower()
