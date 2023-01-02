import logging
import socket
import warnings
from argparse import ArgumentParser
from datetime import datetime

import torch
import yaml
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as ddp_kwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from munch import Munch
from tqdm.auto import tqdm

import wandb
from dataset.datasets import make_data_loader
from model.create_networks import get_all_networks
from trainer import run, run_pretrain

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="config.yaml")
    parser.add_argument("--train",type=str, nargs='+', default=[], choices=['all', 'IIIT_WVU', 'NotreDame', 'Clarkson'])
    args = parser.parse_args()
    return args


def main(args):
    torch.set_float32_matmul_precision('high')
    set_seed(args.GENERAL.seed)

    use_gan = args.CUT.apply
    iterative = args.CUT.iterative
    self_training = args.CLASSIFIER.self_training

    accelerator = Accelerator(step_scheduler_with_optimizer=False,
                              kwargs_handlers=[ddp_kwargs(find_unused_parameters=args.CLASSIFIER.pretrain.apply)])

    args.CLASSIFIER.equiv_batch_size = args.CLASSIFIER.batch_size * accelerator.num_processes
    args.CUT.equiv_batch_size = args.CUT.batch_size * accelerator.num_processes

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        args["host"] = socket.gethostname()
        wandb.init(project=args.GENERAL.name, entity="luisleo", config=args)
        wandb.define_metric("acer/*", summary="min")
        wandb.define_metric("acc/*", summary="max")
        wandb.define_metric("loss/*", summary="min")

    loaders, paths, vocab = make_data_loader(args, accelerator)
    paths_from_train = Munch(label_0=[], label_1=[])
    for data in paths.train.values():
        if not data.config.skip and not data.config.selftraining:
            paths_from_train.label_0.extend(data.label_0)
            paths_from_train.label_1.extend(data.label_1)

    nets = get_all_networks(args, accelerator)
    if accelerator.is_main_process:
        wandb.watch(models=nets.classifier.model, idx=0, log='all', log_freq=100)
        if use_gan:
            wandb.watch(models=nets.cut.model, idx=1, log='all', log_freq=100)
            if iterative:
                wandb.watch(models=nets.cut2.model, idx=2, log='all', log_freq=100)
    accelerator.wait_for_everyone()

    if args.CLASSIFIER.pretrain.apply:
        logger.info(" *** Start pretrain classifier *** ")
        run_pretrain(args, loaders, nets, accelerator, num_epoch=args.CLASSIFIER.pretrain.epochs,
                     tqdm_no_progress=not accelerator.is_local_main_process)

    logger.info(" *** Start warmup *** ")
    step, paths_for_selftrain = run(args, paths_from_train, None, args.GENERAL.warmup, -1, accelerator, nets, loaders,
                                    vocab, use_gan, iterative, warmup=True, train_gan=True,
                                    tqdm_no_progress=not accelerator.is_local_main_process, self_training=self_training,
                                    self_training_refresh=False)

    logger.info(" *** Start training *** ")
    for epoch in tqdm(range(args.GENERAL.max_epochs), disable=not accelerator.is_local_main_process):
        step, paths_for_selftrain = run(args, paths_from_train, paths_for_selftrain, 1, step, accelerator, nets,
                                        loaders, vocab, use_gan, iterative, warmup=False,
                                        train_gan=(epoch + 1) % args.CUT.update_freq == 0, tqdm_no_progress=True,
                                        self_training=self_training,
                                        self_training_refresh=epoch % args.CLASSIFIER.refresh_selftraining == 0)
    if accelerator.is_main_process:
        wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == '__main__':

    opts = parse_args()

    if opts.train == ['all']:
        opts.train = ["NotreDame", "IIIT_WVU", "Clarkson"]

    print(f"Loading {opts.yaml} ...")

    with open(opts.yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))

    if len(opts.train) > 0:
        for key, value in args.GENERAL.data.train.items():
            if key in ["NotreDame", "IIIT_WVU", "Clarkson"]:
                config = value
                config.paths = [config.paths[0].replace(key, "_TMP_")]
                del args.GENERAL.data.train[key]
                break
        for turn in opts.train:
            args.GENERAL.name = turn
            config.paths = [x.replace("_TMP_", turn) for x in config.paths]
            args.GENERAL.data.train.update({turn: config})
            main(args)
            config.paths = [config.paths[0].replace(turn, "_TMP_")]
            del args.GENERAL.data.train[turn]
    else:
        main(args)
