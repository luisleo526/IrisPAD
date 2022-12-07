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

    paths = prepare_image_path(args)
    loader = make_gan_loader(args, paths.train.default.label_0, paths.train.default.label_1, accelerator)
    model, optimizers = get_networks(args, accelerator)

    accelerator.wait_for_everyone()

    progress_bar = tqdm(range(args.GENERAL.max_epochs * len(loader)), disable=not accelerator.is_local_main_process)

    for epoch in range(args.GENERAL.max_epochs):
        total_loss = Munch(netD=0, netG=0, netF=0)
        for batch in loader:
            outputs = model(batch, False)

            accelerator.backward(outputs.lossD + outputs.lossG + outputs.lossF)
            for net in ['netD', 'netG', 'netF']:
                getattr(optimizers, net).optim.step()
                getattr(optimizers, net).scheduler.step()
                getattr(optimizers, net).optim.zero_grad()
            total_loss.netG += outputs.lossG.item()
            total_loss.netF += outputs.lossF.item()
            total_loss.netD += outputs.lossD.item()

            if accelerator.sync_gradients:
                progress_bar.update(1)

        if accelerator.is_main_process:
            writer.add_images("Real Images", outputs.real, global_step=epoch)
            writer.add_images("Fake Images", outputs.fake, global_step=epoch)
            writer.add_scalars("CUT Loss", dict(total_loss), global_step=epoch)
            writer.flush()

    if accelerator.is_main_process:
        writer.close()


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    args = Munch.fromDict(data)
    main(args)
