"""
from https://github.com/Vanint/SADE-AgnosticLT/blob/main/model/model.py
"""
import torch
#from base import BaseModel
from .ldam_draw_resnets import resnet_cifar
from .ldam_draw_resnets import expert_resnet_cifar

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError


class Model(BaseModel):
  requires_target = False

  def __init__(self, num_classes, backbone_class=None):
    super().__init__()
    if backbone_class is not None:  # Do not init backbone here if None
      self.backbone = backbone_class(num_classes)

  def _hook_before_iter(self):
    self.backbone._hook_before_iter()

  def forward(self, x, mode=None):
    x = self.backbone(x)

    assert mode is None
    return x


class EAModel(BaseModel):
  requires_target = True
  confidence_model = True

  def __init__(self, num_classes, backbone_class=None):
    super().__init__()
    if backbone_class is not None:  # Do not init backbone here if None
      self.backbone = backbone_class(num_classes)

  def _hook_before_iter(self):
    self.backbone._hook_before_iter()

  def forward(self, x, mode=None, target=None):
    x = self.backbone(x, target=target)

    assert isinstance(x, tuple)  # logits, extra_info
    assert mode is None

    return x


class ResNet32Model(Model):  # From LDAM_DRW
  # make default match
  def __init__(self, num_classes, reduce_dimension=True, layer2_output_dim=None, layer3_output_dim=None,
               use_norm=True, num_experts=1, **kwargs):
    super().__init__(num_classes, None)
    if num_experts == 1:
      self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes,
                                            reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim,
                                            layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
    else:
      self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes,
                                                   reduce_dimension=reduce_dimension,
                                                   layer2_output_dim=layer2_output_dim,
                                                   layer3_output_dim=layer3_output_dim, use_norm=use_norm,
                                                   num_experts=num_experts, **kwargs)

def init_weights(model, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=False, classifier=False):
  """Initialize weights"""
  print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model', weights_path))
  weights = torch.load(weights_path)
  weights1 = {}
  if not classifier:
    if caffe:
      # lower layers are the shared backbones
      for k in model.state_dict():
        if 'layer3s' not in k and 'layer4s' not in k:
          weights1[k] = weights[k] if k in weights else model.state_dict()[k]
        elif 'num_batches_tracked' in k:
          weights1[k] = weights[k] if k in weights else model.state_dict()[k]

        elif 'layer3s.0.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer3s.0.', 'layer3.')]
        elif 'layer3s.1.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer3s.1.', 'layer3.')]
        elif 'layer3s.2.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer3s.2.', 'layer3.')]
        elif 'layer4s.0.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer4s.0.', 'layer4.')]
        elif 'layer4s.1.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer4s.1.', 'layer4.')]
        elif 'layer4s.2.' in k and 'num_batches_tracked' not in k:
          weights1[k] = weights[k.replace('layer4s.2.', 'layer4.')]

    else:
      weights = weights['state_dict_best']['feat_model']
      weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                 for k in model.state_dict()}
  else:
    weights = weights['state_dict_best']['classifier']
    weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
               for k in model.state_dict()}
  model.load_state_dict(weights1)
  return model
