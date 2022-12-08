import imghdr
import math

import torch
from accelerate import Accelerator
from monai.data import ThreadDataLoader
from monai.data import ZipDataset
from monai.data.utils import partition_dataset, resample_datalist
from munch import Munch
from torchvision.datasets.folder import find_classes, make_dataset

from transforms import TransformFromPath
from vocab import Vocab


def make_data_loader(args, accelerator: Accelerator):
    paths, vocab = prepare_image_path(args)
    dataloaders = Munch(train=Munch(), test=Munch())

    for mode in ['train', 'test']:
        transform = TransformFromPath(args, use_augmentation=mode != 'test')

        def collator(samples):
            batch = {'image': [], 'label': [], 'path': []}
            for path, label in samples:
                batch['image'].append(transform(path))
                batch['path'].append(vocab.word2index(path))
                batch['label'].append(label)
            batch['image'] = torch.stack(batch['image'])
            batch['label'] = torch.tensor(batch['label'])
            batch['path'] = torch.tensor(batch['path'], dtype=torch.int32)
            return batch

        for name in args.GENERAL.data[mode]:
            dataloaders[mode].update({name: Munch()})
            dataloaders[mode][name].update(dl=accelerator.prepare_data_loader(ThreadDataLoader(
                [[x, 0] for x in paths[mode][name].label_0] + [[x, 1] for x in paths[mode][name].label_1],
                batch_size=args.CLASSIFIER.batch_size,
                collate_fn=collator,
                shuffle=True
            )))
            dataloaders[mode][name].update(config=args.GENERAL.data[mode][name].config)

    return dataloaders, paths, vocab


def make_gan_loader(args, a_path, b_path, accelerator: Accelerator):
    batch_size = args.CUT.batch_size
    a_path = [[x, 0] for x in a_path]
    b_path = [[x, 1] for x in b_path]
    a_bs = round(batch_size * len(a_path) / (len(a_path) + len(b_path)))
    b_bs = round(batch_size * len(b_path) / (len(a_path) + len(b_path)))
    a_num = math.ceil(len(a_path) / a_bs)
    b_num = math.ceil(len(b_path) / b_bs)
    a_path = partition_dataset(a_path, num_partitions=a_bs, shuffle=True, even_divisible=True)
    b_path = partition_dataset(b_path, num_partitions=b_bs, shuffle=True, even_divisible=True)
    if a_num > b_num:
        b_path = resample_datalist(b_path, a_num / b_num, random_pick=True)
    else:
        a_path = resample_datalist(a_path, b_num / a_num, random_pick=True)

    transform = TransformFromPath(args, True)

    def collator(samples):
        samples = samples[0]
        batch_size = int(len(samples) / 2)
        batch = {'a': [], 'b': []}
        for i in range(batch_size):
            if samples[2 * i + 1] == 0:
                batch['a'].append(transform(samples[2 * i]))
            else:
                batch['b'].append(transform(samples[2 * i]))
        batch['a'] = torch.stack(batch['a'])
        batch['b'] = torch.stack(batch['b'])
        return batch

    return accelerator.prepare_data_loader(
        ThreadDataLoader(ZipDataset(a_path + b_path), batch_size=1, collate_fn=collator,
                         shuffle=True))


def prepare_image_path(args):
    paths = Munch(train=Munch(), test=Munch())
    vocab = Vocab()
    for mode in ['train', 'test']:
        for name in args.GENERAL.data[mode]:
            paths[mode].update({name: Munch(label_0=[], label_1=[])})
            for path in args.GENERAL.data[mode][name].paths:
                path_0, path_1 = path_by_label(path)
                vocab.word2index(path_0 + path_1, True)
                paths[mode][name].label_0 += path_0
                paths[mode][name].label_1 += path_1
            paths[mode][name].update(config=args.GENERAL.data[mode][name].config)
    return paths, vocab


def path_by_label(image_folder):
    raw_data = make_dataset(image_folder, find_classes(image_folder)[1], None, lambda x: imghdr.what(x) is not None)
    label_0 = []
    label_1 = []
    for path, label in raw_data:
        if label == 0:
            label_0.append(path)
        elif label == 1:
            label_1.append(path)
        else:
            raise Exception("The number of labels should be 2 only.")
    return label_0, label_1
