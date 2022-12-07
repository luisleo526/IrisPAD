from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, Normalize
from torchvision.datasets.folder import default_loader
from importlib import import_module


class TransformFromPath(object):
    def __init__(self, args, use_augmentation=False):

        channels = args.GENERAL.channels

        transforms = [ToTensor()]
        if channels == 1:
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
