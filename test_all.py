import warnings
from argparse import ArgumentParser

import yaml
from accelerate.logging import get_logger
from munch import Munch

from train_funcs.train import main

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train", type=str, nargs='+', default=[],
                        choices=['all', 'IIIT_WVU', 'NotreDame', 'Clarkson'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opts = parse_args()

    if opts.train == ['all']:
        opts.train = ["NotreDame", "IIIT_WVU", "Clarkson"]

    print(f"Loading {opts.yaml} ...")
    with open(opts.yaml, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))

    config = None
    for key, value in args.GENERAL.data.train.items():
        if key in ["NotreDame", "IIIT_WVU", "Clarkson"]:
            config = value
            config.paths = [config.paths[0].replace(key, "_TMP_")]
            del args.GENERAL.data.train[key]
            break

    if config is None:
        raise Exception("NotreDame, IIIT_WVU, Clarkson not found in config.yaml")

    for turn in opts.train:
        args.GENERAL.name = turn
        config.paths = [x.replace("_TMP_", turn) for x in config.paths]
        args.GENERAL.data.train.update({turn: config})
        main(args)
        config.paths = [config.paths[0].replace(turn, "_TMP_")]
        del args.GENERAL.data.train[turn]
