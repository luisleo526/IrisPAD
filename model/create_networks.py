from accelerate import Accelerator
from munch import Munch

from .classifier import get_classifier_networks
from .cut.cut import get_gan_networks


def get_all_networks(args, accelerator: Accelerator):
    classifier, classifier_optim = get_classifier_networks(args, accelerator)
    net = Munch(classifier=Munch(model=classifier, optimizers=classifier_optim))

    if args.CUT.apply:
        cut, cut_optim = get_gan_networks(args, accelerator)
        net.update(cut=Munch(model=cut, optimizers=cut_optim))
        if args.CUT.iterative:
            cut2, cut_optim2 = get_gan_networks(args, accelerator)
            net.update(cut2=Munch(model=cut2, optimizers=cut_optim2))

    return net
