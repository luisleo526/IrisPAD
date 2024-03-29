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
    parser.add_argument("--gpu", type=int, default=1, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--train", type=str, nargs='+', default=['all'],
                        choices=['all', 'IIIT_WVU', 'NotreDame', 'Clarkson'])
    parser.add_argument("--test", type=str, nargs='+', default=['all'],
                        choices=['all', 'IIIT_WVU', 'NotreDame', 'Clarkson'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opts = parse_args()

    ds = ['IIIT_WVU', 'NotreDame', 'Clarkson']
    params = {d: {dd: {'bs': 0, 'lambda': 0.0} for dd in ds} for d in ds}

    params['IIIT_WVU']['IIIT_WVU'] = {'bs': 64, 'lambda': 0.25}
    params['IIIT_WVU']['NotreDame'] = {'bs': 128, 'lambda': 0.75}
    params['IIIT_WVU']['Clarkson'] = {'bs': 512, 'lambda': 1.0}

    params['NotreDame']['IIIT_WVU'] = {'bs': 512, 'lambda': 0.50}
    params['NotreDame']['NotreDame'] = {'bs': 128, 'lambda': 0.25}
    params['NotreDame']['Clarkson'] = {'bs': 512, 'lambda': 0.25}

    params['Clarkson']['IIIT_WVU'] = {'bs': 256, 'lambda': 2.0}
    params['Clarkson']['NotreDame'] = {'bs': 256, 'lambda': 0.25}
    params['Clarkson']['Clarkson'] = {'bs': 128, 'lambda': 2.0}

    print(f"Loading {opts.config} ...")
    with open(opts.config, "r") as stream:
        args = Munch.fromDict(yaml.load(stream, Loader=yaml.FullLoader))

    train_ds = opts.train
    if train_ds == ['all']:
        train_ds = ds

    test_ds = opts.test
    if test_ds == ['all']:
        test_ds = ds

    args.GENERAL.accumulation_steps = opts.accumulate

    for train in train_ds:
        args.GENERAL.name = train
        args.GENERAL.data.train = Munch({train: Munch(config=Munch(), paths=[f"LivDet2017/{train}/train"])})
        args.GENERAL.data.train[train].config.skip = False
        args.GENERAL.data.train[train].config.selftraining = False
        for test in test_ds:
            args.GENERAL.data.test = Munch({test: Munch(config=Munch(), paths=[f"LivDet2017/{test}/test"])})
            args.GENERAL.data.test[test].config.gan = True
            args.CLASSIFIER.batch_size = int(params[train][test]['bs'] / opts.gpu / opts.accumulate)
            args.CUT.lambda_NCE = params[train][test]['lambda']
            main(args)
