from cut import get_gan_networks
from classifier import get_classifier_networks
from accelerate import Accelerator
from munch import Munch


def get_all_networks(args, accelerator: Accelerator):
    classifier, classifier_optim = get_classifier_networks(args, accelerator)
    cut, cut_optim = get_gan_networks(args, accelerator)
    net = Munch(cut=Munch(model=classifier, optimizers=classifier_optim),
                classifier=Munch(model=cut, optimizers=cut_optim))
    return net
