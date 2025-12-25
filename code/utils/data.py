import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from scipy.io import loadmat
from . import autoaugment
from . import ops
import torch
from .hrrp_transforms import HRRPTransforms

def mapping(labels):
    labels_set = np.unique(labels)
    map = np.full((1, max(labels_set)+1), -1)
    lab = 0
    for i in range(map.shape[1]):
        if i in labels_set:
            map[0][i] = lab
            lab += 1

    labels = np.array([map[0][labels[i]].item() for i in range(labels.shape[0])])
    return labels

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iHRRP9(iData):
    use_path = False
    class_order = np.arange(9).tolist()

    def download_data(self):
        train_data_mat = loadmat("data/hrrp/原理样机_2024.3.28/TRAIN_RESAMPLE.mat")
        data_tr = train_data_mat['data_resample']
        self.train_data = np.array(data_tr[:, 1:], dtype=np.float32)
        self.train_targets = mapping(np.array(data_tr[:, 0], dtype=np.int64))
        
        test_data_mat = loadmat("data/hrrp/原理样机_2024.3.28/TEST_RESAMPLE.mat")
        data_ev = test_data_mat['TEST']
        self.test_data = np.array(data_ev[:, 1:], dtype=np.float32)
        self.test_targets = mapping(np.array(data_ev[:, 0], dtype=np.int64))

    def get_transforms(self, args):
        # 返回数据增强转换
        return HRRPTransforms(args)

class iHRRP12(iData):
    use_path = False
    class_order = np.arange(12).tolist()

    def download_data(self):
        train_data_mat = loadmat("data/hrrp/hrrp12/TRAIN.mat")
        data_tr = train_data_mat['data']
        self.train_data, self.train_targets = np.array(data_tr[:, 0:256], dtype=np.float32), np.array(data_tr[:, -1], dtype=np.int64)
        test_data_mat = loadmat("data/hrrp/hrrp12/TEST.mat")
        data_ev = test_data_mat['data']
        self.test_data, self.test_targets = np.array(data_ev[:, 0:256], dtype=np.float32), np.array(data_ev[:, -1], dtype=np.int64)
    
    def get_transforms(self, args):
        # 返回数据增强转换
        return HRRPTransforms(args)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
