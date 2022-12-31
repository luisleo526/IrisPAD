import functools
from importlib import import_module

from torch.nn import init


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def get_hypers_config(args):
    hyparam = dict(CUT=args.CUT.apply, iterative=args.CUT.iterative,
                   CUT_NCE_weight=args.CUT.lambda_NCE / args.CUT.lambda_GAN,
                   CUT_update_freq=args.CUT.update_freq,
                   CUT_confidence=args.CLASSIFIER.confidence_CUT,
                   batch_size=args.CLASSIFIER.batch_size,
                   self_training=args.CLASSIFIER.self_training,
                   self_training_confidence=args.CLASSIFIER.confidence_selfTraining,
                   self_training_fresh_freq=args.CLASSIFIER.refresh_selftraining)
    return hyparam


def get_class(x):
    module = x[:x.rfind(".")]
    obj = x[x.rfind(".") + 1:]
    return getattr(import_module(module), obj)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, init_gain=init_gain)
    return net
