from importlib import import_module

from torch.nn import init


def get_hypers_config(args):
    hypers = {}
    hypers['batch_size'] = args.GENERAL.batch_size
    hypers['channels'] = args.GENERAL.channels
    hypers['lambda_GAN'] = args.CUT.lambda_GAN
    hypers['lambda_NCE'] = args.CUT.lambda_NCE
    hypers['nce_idt'] = args.CUT.nce_idt
    hypers['gan_mode'] = args.CUT.mode
    hypers['netG_nc'] = args.CUT.netG.params.ngf
    hypers['netG_blocks'] = args.CUT.netG.params.n_blocks
    hypers['netD_nc'] = args.CUT.netD.params.ndf
    hypers['netD_layers'] = args.CUT.netD.params.n_layers
    hypers['netF_nc'] = args.CUT.netF.params.nc
    hypers['netF.patches'] = args.CUT.netF.params.num_patches
    return hypers

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
