import imghdr

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset

from .transforms import MultiCropTransform


class MultiCropDataset(Dataset):
    def __init__(self, args):
        self.samples = []
        for path in args.GENERAL.data.pretrain:
            self.samples.extend([path for path, label in
                                 make_dataset(path, find_classes(path)[1], None, lambda x: imghdr.what(x) is not None)])
        self.transform = MultiCropTransform(args)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        elif type(index) is int:
            index = [index]

        _outputs = [self.transform(self.samples[x]) for x in index]
        outputs = [[] for _ in range(len(_outputs[0]))]
        for i in range(len(outputs)):
            outputs[i].extend([x[i] for x in _outputs])
        if len(index) > 1:
            return [torch.stack(x) for x in outputs]
        else:
            return [x[0] for x in outputs]


def get_pseudo_label(args):
    _label = [x for x in range(args.CLASSIFIER.pretrain.batch_size)]
    label = []
    for i in args.CLASSIFIER.pretrain.config.num_crops:
        label.append(torch.tensor(_label * i, dtype=torch.int))
    return label
