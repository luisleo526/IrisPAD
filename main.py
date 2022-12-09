import logging
from argparse import ArgumentParser

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from create_networks import get_all_networks
from datasets import make_data_loader
from trainer import run

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="config.yaml")
    args = parser.parse_args()
    return args


def main(args):
    torch.set_float32_matmul_precision('high')
    set_seed(args.GENERAL.seed)

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

    loaders, paths, vocab = make_data_loader(args, accelerator)
    paths_from_train = Munch(label_0=[], label_1=[])
    for data in paths.train.values():
        if not data.config.skip:
            paths_from_train.label_0.extend(data.label_0)
            paths_from_train.label_1.extend(data.label_1)

    # Estimate the number of steps
    num_steps = 0
    for loader in loaders.train.values():
        if loader.config.skip:
            num_steps += len(loader.dl) * args.CLASSIFIER.warmup
        else:
            num_steps += len(loader.dl) * (args.CLASSIFIER.warmup * 2 + args.GENERAL.max_epochs)
    if "num_training_steps" in args.CLASSIFIER.scheduler.params:
        args.CLASSIFIER.scheduler.params.num_training_steps = num_steps

    num_steps = int(len(vocab) // args.CUT.batch_size * (args.GENERAL.max_epochs + args.CUT.warmup) * 1.5)
    for scheduler in [args.CUT.netD.scheduler, args.CUT.netF.scheduler, args.CUT.netG.scheduler]:
        if "num_training_steps" in scheduler.params:
            scheduler.params.num_training_steps = num_steps

    nets = get_all_networks(args, accelerator)
    accelerator.wait_for_everyone()

    logger.info(" *** Start warmup classifier *** ")
    step = run(args, paths_from_train, int(args.CLASSIFIER.warmup / 2), -1, accelerator, writer, nets, loaders, vocab,
               use_gan, iterative, warmup=True, train_gan=False, tqdm_no_progress=not accelerator.is_local_main_process,
               self_training=False)
    logger.info(" *** Start warmup classifier + self-training *** ")
    step = run(args, paths_from_train, int(args.CLASSIFIER.warmup / 2), step, accelerator, writer, nets, loaders, vocab,
               use_gan, iterative, warmup=True, train_gan=False, tqdm_no_progress=not accelerator.is_local_main_process,
               self_training=True)
    if use_gan:
        logger.info(" *** Start warmup CUT *** ")
        step = run(args, paths_from_train, args.CUT.warmup, step, accelerator, writer, nets, loaders, vocab, use_gan,
                   iterative, warmup=True, train_gan=True, tqdm_no_progress=not accelerator.is_local_main_process,
                   self_training=True)

    logger.info(" *** Start training *** ")
    for epoch in tqdm(range(args.GENERAL.max_epochs), disable=not accelerator.is_local_main_process):
        step = run(args, paths_from_train, 1, step, accelerator, writer, nets, loaders, vocab, use_gan,
                   iterative, warmup=False, train_gan=(epoch + 1) % args.CUT.update_freq == 0,
                   tqdm_no_progress=True, self_training=True)

    if accelerator.is_main_process:
        writer.close()


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))
    main(args)
