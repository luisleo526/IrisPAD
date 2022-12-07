from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, Normalize
from torchvision.datasets.folder import default_loader
from importlib import import_module


class TransformFromPath(object):
    def __init__(self, args, use_augmentation=False):

        transforms = [ToTensor()]
        if args.GENERAL.rgb:
            transforms.append(Grayscale(num_output_channels=3))

        transforms.append(Resize(args.GENERAL.resolution))
        transforms.append(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False))
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
