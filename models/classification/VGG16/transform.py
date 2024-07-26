from typing import List

from torch.nn import MaxPool2d, Parameter, Linear

from abc import ABC, abstractmethod
from torchvision import models
from torch.nn import Module
from typing import Final

from torchvision.transforms import Resize
import torch.nn.functional as F

import torch
import os


class ClassificationNeuralNetworkModule(ABC, Module):
    """Pytorch Module for working with the ResNeXt networks and adapting them for a given number of classes.

    Parameters
    ----------
    vgg_name: str
            Name of the specific VGG network to be used.
    trainable_backbone_size: int
            Choose how many layers of the network will be trainable.
    """

    def __init__(
        self,
        model_name: str,
        classes_number: int,
        trainable_stages: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        torch.hub.set_dir(
            os.path.join(os.getcwd(), "data/PretrainedModels/Classification")
        )

        self.backbone: Final = models.get_model(model_name, weights="DEFAULT")
        self.n_classes: Final = classes_number

        self.trainable_stages: Final = trainable_stages
        if self.trainable_stages is not None:
            self._freeze_stages()

        self.adapter_layers: Final = self.adapt_output()
        self.resizer = Resize(size=[224, 224], antialias=1)

    @abstractmethod
    def get_backbone_blocks(self):
        raise NotImplementedError

    def _freeze_stages(self):
        blocks = self.get_backbone_blocks()
        n_stages = len(blocks)
        if self.trainable_stages <= n_stages:
            if self.trainable_stages < 1:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                freeze = n_stages - self.trainable_stages
                for i in range(freeze):
                    block = blocks[i]
                    for layer in block:
                        if isinstance(layer, Parameter):
                            layer.requires_grad = False
                        else:
                            for param in layer.parameters():
                                param.requires_grad = False

    def adapt_output(self):
        """Creates layers to adapt the loaded VGG model to a new number of classes.

        Returns
        -------
        ClassifierLayer
                PyTorch linear layer of shape(classifier_in_features, classifier_out_features)
        """

        last_module = list(self.backbone.modules())[-1]
        classifier_in_features = last_module.out_features

        # classifier_out_features = self.n_classes
        classifier_out_features = 1 if self.n_classes == 2 else self.n_classes

        return Linear(classifier_in_features, classifier_out_features)

    def forward(self, input):
        """Foward of the network module.

        Parameters
        ----------
        input: ImagelogWindowsRGBTensor
                ImagelogWindowsRGBTensor of the inputs.

        Returns
        -------
        Tensor
                ClassifierOutputTensor of the adapter_layers.
        """
        x = self.resizer(input)
        x = self.backbone(x)
        x = self.adapter_layers(x)
        return F.softmax(x)


class VGG(ClassificationNeuralNetworkModule):
    """Pytorch Module for working with the VGG networks and adapting them for a given number of classes.

    Parameters
    ----------
    vgg_name: str
            Name of the specific VGG network to be used.
    trainable_backbone_size: int
            Choose how many layers of the network will be trainable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_backbone_blocks(self) -> List:
        """Function that freezes the VGG stages using 'trainable_stages'"""
        blocks, block = [], []

        for layer in self.backbone.features.children():
            block.append(layer)
            if isinstance(layer, MaxPool2d):
                blocks.append(block)
                block = []

        blocks.append([self.backbone.avgpool, self.backbone.classifier])

        return blocks


class VGG16(VGG):
    """Inherits from the VGG class, defining the specific network version to be "VGG16" """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(model_name="VGG16", **kwargs)


def build_classifier(package_path, *args):
    weights_path = f"{package_path}/models/classification/VGG16/model.pt"

    model = VGG16(
        classes_number=3,
        trainable_stages=0,
    )
    model.load_state_dict(torch.load(weights_path))

    return model
