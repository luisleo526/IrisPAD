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
    self_training = args.CLASSIFIER.self_training

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
        
    # accelerator.logger = logger

    loaders, paths, vocab = make_data_loader(args, accelerator)
    paths_from_train = Munch(label_0=[], label_1=[])
    for data in paths.train.values():
        if not data.config.skip and not data.config.selftraing:
            paths_from_train.label_0.extend(data.label_0)
            paths_from_train.label_1.extend(data.label_1)

    # Estimate the number of steps
    warmup_steps = 0
    train_steps = 0
    for loader in loaders.train.values():
        index = 2 if self_training else 1
        warmup_steps += len(loader.dl) * args.CLASSIFIER.warmup * index
        if not loader.config.skip:
            train_steps += len(loader.dl) * (args.CLASSIFIER.warmup * index + args.GENERAL.max_epochs)
            
    if "num_training_steps" in args.CLASSIFIER.scheduler.params:
        args.CLASSIFIER.scheduler.params.num_training_steps = warmup_steps + train_steps
    if "num_warmup_steps" in args.CLASSIFIER.scheduler.params:
        args.CLASSIFIER.scheduler.params.num_warmup_steps = warmup_steps

    if iterative:
        num_steps = max(len(paths_from_train.label_0),len(paths_from_train.label_1)) // args.CUT.batch_size
    else:
        num_steps = len(paths_from_train.label_0 + paths_from_train.label_1) // args.CUT.batch_size
        
    for scheduler in [args.CUT.netD.scheduler, args.CUT.netF.scheduler, args.CUT.netG.scheduler]:
        if "num_training_steps" in scheduler.params:
            scheduler.params.num_training_steps = num_steps * (args.CUT.warmup + args.GENERAL.max_epochs / args.CUT.update_freq )
        if "num_warmup_steps" in scheduler.params:
            scheduler.params.num_warmup_steps = num_steps * args.CUT.warmup
            
    nets = get_all_networks(args, accelerator)
    accelerator.wait_for_everyone()

    logger.info(" *** Start warmup classifier *** ")
    step, paths_for_selftrain = run(args, paths_from_train, None, args.CLASSIFIER.warmup, -1, accelerator, writer, nets,
                                    loaders, vocab,
                                    use_gan, iterative, warmup=True, train_gan=False,
                                    tqdm_no_progress=not accelerator.is_local_main_process,
                                    self_training=False, self_training_refresh=False)
    if self_training:
        logger.info(" *** Start warmup classifier + self-training *** ")
        step, paths_for_selftrain = run(args, paths_from_train, None, args.CLASSIFIER.warmup, step, accelerator, writer,
                                        nets, loaders, vocab,
                                        use_gan, iterative, warmup=True, train_gan=False,
                                        tqdm_no_progress=not accelerator.is_local_main_process,
                                        self_training=True, self_training_refresh=False)
    if use_gan:
        logger.info(" *** Start warmup CUT *** ")
        step, paths_for_selftrain = run(args, paths_from_train, paths_for_selftrain, args.CUT.warmup, step, accelerator,
                                        writer, nets,
                                        loaders, vocab, use_gan,
                                        iterative, warmup=True, train_gan=True,
                                        tqdm_no_progress=not accelerator.is_local_main_process,
                                        self_training=self_training, self_training_refresh=True)

    logger.info(" *** Start training *** ")
    for epoch in tqdm(range(args.GENERAL.max_epochs), disable=not accelerator.is_local_main_process):
        step, paths_for_selftrain = run(args, paths_from_train, paths_for_selftrain, 1, step, accelerator, writer, nets,
                                        loaders, vocab, use_gan,
                                        iterative, warmup=False, train_gan=(epoch + 1) % args.CUT.update_freq == 0,
                                        tqdm_no_progress=True, self_training=self_training,
                                        self_training_refresh=epoch % args.CLASSIFIER.refresh_selftraining == 0)

    if accelerator.is_main_process:
        writer.close()


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))
    main(args)
