from pytorch_data.cifar.data import CIFAR10Data, CIFAR10_1Data, CINIC10_Data, CIFAR10_CData, CIFAR100Data, CIFAR100CoarseData
from pytorch_data.cifar.data_imbalanced import IMBALANCECIFAR10Data, IMBALANCECIFAR100Data, IMBALANCECIFAR10DataAug, IMBALANCECIFAR100DataAug, IMBALANCECIFAR10DataAug_v2
from pytorch_data.mnist.data import MNISTModule
from pytorch_data.inaturalist.data import iNaturalistData
all_datasets = {
    "CIFAR10": CIFAR10Data,
    "mnist": MNISTModule,
    "CIFAR100": CIFAR100Data,
    "CIFAR100Coarse": CIFAR100CoarseData,
    "IMBALANCECIFAR10": IMBALANCECIFAR10Data,
    "IMBALANCECIFAR100": IMBALANCECIFAR100Data,
    "IMBALANCECIFAR10Aug": IMBALANCECIFAR10DataAug,
    "IMBALANCECIFAR10Aug_v2": IMBALANCECIFAR10DataAug_v2,
    "IMBALANCECIFAR100Aug": IMBALANCECIFAR100DataAug,
    "cinic10": CINIC10_Data,
    "cifar10_1": CIFAR10_1Data,
    "cifar10_c": CIFAR10_CData,
    "iNaturalistData": iNaturalistData,
}
