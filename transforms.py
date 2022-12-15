import random
from importlib import import_module

import numpy as np
import torch
from PIL import ImageFilter
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ColorJitter, RandomApply, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, Normalize


class TransformFromPath(object):
    def __init__(self, args, use_augmentation=False):

        channels = 3 if args.GENERAL.rgb else 1
        transforms = [ToTensor()]
        if not args.GENERAL.rgb:
            transforms.append(Grayscale(num_output_channels=1))

        transforms.append(Resize(args.GENERAL.resolution))
        transforms.append(Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)], False))
        if use_augmentation:
            try:
                for transform in args.AUGMENTATION.keys():
                    transforms.append(
                        getattr(import_module(args.AUGMENTATION[transform].origin), transform)(
                            **args.AUGMENTATION[transform].params)
                    )
            except:
                pass

        self.transform = Compose(transforms)

    def __call__(self, data):
        return self.transform(default_loader(data))


class MultiCropTransform(object):
    def __init__(self, all_args):

        args = all_args.CLASSIFIER.pretrain.config

        channels = 3 if all_args.GENERAL.rgb else 1

        ds = args.distortion_strength
        basic_transform = [RandomApply([ColorJitter(0.8 * ds, 0.8 * ds, 0.8 * ds, 0.2 * ds)]),
                           PILRandomGaussianBlur(radius_max=args.sigma_range[0], radius_min=args.sigma_range[1]),
                           ToTensor()]
        if not all_args.GENERAL.rgb:
            basic_transform.append(Grayscale(num_output_channels=1))

        transforms = []
        for i in range(len(args.num_crops)):
            trans = Compose([
                Compose(basic_transform),
                RandomResizedCrop(args.crop_sizes[i], scale=(args.min_scale[i], args.max_scale[i])),
                RandomHorizontalFlip(), RandomVerticalFlip(),
                Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)], False),
                GaussianNoise(std=args.gaussian_std[i]),
            ])
            transforms.extend([trans] * args.num_crops[i])
        self.transforms = transforms

    def __call__(self, path: str):
        return list(map(lambda trans: trans(default_loader(path)), self.transforms))


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor):
        noise = torch.from_numpy(np.random.normal(self.mean, self.std, size=list(img.shape))).double().to(img.device)
        return img + noise
