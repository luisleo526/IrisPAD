import warnings
from argparse import ArgumentParser

import yaml
from munch import Munch

from train_funcs.train import main

warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    opts = parse_args()

    print(f"Loading {opts.config} ...")

    with open(opts.config, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))

    main(args)
