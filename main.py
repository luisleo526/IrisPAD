import logging
from argparse import ArgumentParser

import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter
from munch import Munch

from utils import get_hypers_config
from cut import get_networks
from datasets import make_gan_loader, prepare_image_path
from tqdm.auto import tqdm

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="config.yaml")
    args = parser.parse_args()
    return args


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs()
    ddp_kwargs.find_unused_parameters = True
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        writer = SummaryWriter("./log", filename_suffix=args.GENERAL.name)
        writer.add_hparams(get_hypers_config(args), {})

    paths = prepare_image_path(args, accelerator)
    loader = make_gan_loader(args, paths.train.default.label_0, paths.train.default.label_1, accelerator)
    model, optimizers = get_networks(args, accelerator)

    progress_bar = tqdm(range(args.GENERAL.max_epochs * len(loader)), disable=not accelerator.is_local_main_process)

    for epoch in range(args.GENERAL.max_epochs):
        total_loss = Munch(netD=0, netGF=0)
        for batch in loader:
            outputs = model(batch, False)
            accelerator.backward(outputs.loss)

            optimizers.netD.optim.step()
            optimizers.netD.scheduler.step()
            optimizers.netD.optim.zero_grad()

            total_loss.netD += outputs.loss.item()

            outputs = model(batch, True)
            accelerator.backward(outputs.loss)

            optimizers.netG.optim.step()
            optimizers.netG.scheduler.step()
            optimizers.netG.optim.zero_grad()

            optimizers.netF.optim.step()
            optimizers.netF.scheduler.step()
            optimizers.netF.optim.zero_grad()

            total_loss.netGF += outputs.loss.item()

            if accelerator.sync_gradients:
                progress_bar.update(1)

        if accelerator.is_main_process:
            writer.add_images("Real Images", outputs.real, global_step=epoch)
            writer.add_images("Fake Images", outputs.fake, global_step=epoch)
            writer.add_scalars("CUT Loss", dict(total_loss), global_step=epoch)
            writer.flush()

    writer.close()


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    args = Munch.fromDict(data)
    main(args)
