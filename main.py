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

from model.create_networks import get_all_networks
from dataset.datasets import make_data_loader
from trainer import run, run_pretrain
import warnings

from utils.tracker import Tracker

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="config.yaml")
    parser.add_argument("--train_all", action="store_true")
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
        writer = SummaryWriter(f"./log/{args.GENERAL.name}", filename_suffix=args.GENERAL.name)
    else:
        writer = None

    loaders, paths, vocab = make_data_loader(args, accelerator)
    paths_from_train = Munch(label_0=[], label_1=[])
    for data in paths.train.values():
        if not data.config.skip and not data.config.selftraining:
            paths_from_train.label_0.extend(data.label_0)
            paths_from_train.label_1.extend(data.label_1)

    nets = get_all_networks(args, accelerator)
    nets.update(tracker=Tracker(milestones=args.GENERAL.milestones,
                                value_names=['acc', 'acer', 'apcer', 'bpcer'],
                                ds_names=list(paths.test.keys())))
    accelerator.wait_for_everyone()

    if args.CLASSIFIER.pretrain.apply:
        logger.info(" *** Start pretrain classifier *** ")
        run_pretrain(args, loaders, nets, accelerator, writer,
                     num_epoch=args.CLASSIFIER.pretrain.epochs, tqdm_no_progress=not accelerator.is_local_main_process)

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

    accelerator.wait_for_everyone()


if __name__ == '__main__':

    opts = parse_args()

    print(f"Loading {opts.yaml} ...")

    with open(opts.yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))

    if opts.train_all:
        for key, value in args.GENERAL.data.train.items():
            if key in ["NotreDame", "IIIT_WVU", "Clarkson"]:
                config = value
                config.paths = [config.paths[0].replace(key, "_TMP_")]
                del args.GENERAL.data.train[key]
                break
        for turn in ["NotreDame", "IIIT_WVU", "Clarkson"]:
            args.GENERAL.name = turn
            config.paths = [x.replace("_TMP_", turn) for x in config.paths]
            args.GENERAL.data.train.update({turn: config})
            main(args)
            config.paths = [config.paths[0].replace(turn, "_TMP_")]
            del args.GENERAL.data.train[turn]
    else:
        main(args)
