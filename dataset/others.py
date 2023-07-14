import random

import torchvision.transforms as transforms
from PIL import ImageFilter
from torchvision.transforms import InterpolationMode

from dataset.randaugment import rand_augment_transform
from dataset.autoaugm import Cutout, CIFAR10Policy

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode='train', randaug_n=2, randaug_m=10, plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/ Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]
    )

    transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    if aug_type == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_type == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_type == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randcls_sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]

    if mode == 'train':
        return transform_train[0]
    else:
        return val_transform


def cifar_aug_plus(aug_comb='cifar100', mode='train', plus_plus='False', autoaugm='auto', cutout=False):
    # PaCo's aug: https://github.com/jiequancui/Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if autoaugm == 'auto':
        augmentation_regular.append(CIFAR10Policy())  # add AutoAugm
    elif autoaugm == 'trivial':
        augmentation_regular.append(transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))  # add TrivialAugm

    augmentation_regular.append(transforms.ToTensor())
    if cutout:
        augmentation_regular.append(Cutout(n_holes=1, length=16))  # add AutoAug
    augmentation_regular.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]
    if aug_comb == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_comb == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_comb == 'cifar100':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

    if mode == 'train':
        return transform_train[0]
    else:
        return val_transform