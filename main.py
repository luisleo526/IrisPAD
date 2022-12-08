import logging
from argparse import ArgumentParser

import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

from create_networks import get_all_networks
from datasets import make_data_loader
from trainer import run
from utils import get_hypers_config
from tqdm.auto import tqdm

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="config.yaml")
    args = parser.parse_args()
    return args


def main(args):
    use_gan = args.CUT.apply
    iterative = args.CUT.iterative

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        writer = SummaryWriter("./log", filename_suffix=args.GENERAL.name)
    else:
        writer = None

    nets = get_all_networks(args, accelerator)
    loaders, paths, vocab = make_data_loader(args, accelerator)
    paths_from_train = Munch(label_0=[], label_1=[])
    for data in paths.train.values():
        if not data.config.skip:
            paths_from_train.label_0.extend(data.label_0)
            paths_from_train.label_1.extend(data.label_1)

    accelerator.wait_for_everyone()

    # warmup classifier
    logger.info(" *** Start warmup classifier *** ")
    step = run(args, paths_from_train, args.CLASSIFIER.warmup, -1, accelerator, writer, nets, loaders, vocab, use_gan,
               iterative, warmup=True, train_gan=False)
    if use_gan:
        logger.info(" *** Start warmup CUT *** ")
        step = run(args, paths_from_train, args.CUT.warmup, step, accelerator, writer, nets, loaders, vocab, use_gan,
                   iterative, warmup=True, train_gan=True)

    logger.info(" *** Start training *** ")
    for epoch in tqdm(range(args.GENERAL.max_epochs), disable=not accelerator.is_local_main_process):
        step = run(args, paths_from_train, 1, step, accelerator, writer, nets, loaders, vocab, use_gan,
                   iterative, warmup=False, train_gan=(epoch + 1) % args.CUT.update_freq == 0)

    if accelerator.is_main_process:
        writer.close()


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))
    main(args)
