from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
from .cifar10_models.wideresnet_28 import wideresnet28_10
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .cifar10_models.lenet import lenet5
from .cifar10_models.shake_shake import shake_resnet26_2x96d, shake_resnet26_2x32d
from .cifar10_models.resnet_cifar import resnet32_cf, resnet32_cfa
from .cifar10_models.resnet_orig import resnet_orig_110
from .timm_models import timm_eva02_inat, timm_vitl_inat_dcomp, timm_vitl_inat_laion
from .cifar10_models.resnet_sade import ResNet32Model
all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "shake_26_32": shake_resnet26_2x32d,
    "shake_26_96": shake_resnet26_2x96d,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
    "lenet5": lenet5,
    "resnet32_cf": resnet32_cf,
    "resnet32_cfa": resnet32_cfa,
    "resnet32_sade": ResNet32Model,
    "resnet_orig_110": resnet_orig_110,
    "timm_eva02_inat": timm_eva02_inat,
    "timm_vitl_inat_dcomp": timm_vitl_inat_dcomp,
    "timm_vitl_inat_laion": timm_vitl_inat_laion,
}