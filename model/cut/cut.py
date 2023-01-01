import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from munch import Munch
from torch.nn import SyncBatchNorm
from .gan_utils import GANLoss, PatchNCELoss
from .netD import NLayerDiscriminator
from .netF import PatchSampleF
from .netG import ResnetGenerator
from utils.utils import get_class, init_net
from torchvision.transforms import ToPILImage


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class CUT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.nce_layers = args.CUT.nce_layers
        self.mlp_sample = args.CUT.netF.params.use_mlp
        self.lambda_GAN = args.CUT.lambda_GAN
        self.lambda_NCE = args.CUT.lambda_NCE
        self.nce_idt = args.CUT.nce_idt
        self.num_patches = args.CUT.netF.params.num_patches
        self.flip_equivariance = args.CUT.flip_equivariance

        channels = 3 if args.GENERAL.rgb else 1
        self.netG = ResnetGenerator(**args.CUT.netG.params, input_nc=channels, output_nc=channels)
        self.netD = NLayerDiscriminator(**args.CUT.netD.params, input_nc=channels)
        self.netF = PatchSampleF(**args.CUT.netF.params, **args.CUT.netF.net_init)

        self.criterionGAN = GANLoss(gan_mode=args.CUT.mode)
        self.criterionNCE = [PatchNCELoss(args.CUT.batch_size, args.CUT.nce_T) for _ in self.nce_layers]

    def calculate_NCE_loss(self, src, tgt):

        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, None)
        feat_q_pool, _ = self.netF(feat_q, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def forward(self, batch):

        self.real_A = batch["a"]
        self.real_B = batch["b"]

        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A

        if self.flip_equivariance:
            self.flipped_for_equivariance = np.random.random() < 0.5
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        real_img = (0.5 * batch['a'] + 0.5).detach().cpu().float()
        fake_img = (0.5 * self.fake_B + 0.5).detach().cpu().float()

        loss_D = self.netD_loss()
        loss_G, loss_F = self.netGF_loss()
        output = Munch(lossG=loss_G, lossF=loss_F, lossD=loss_D,
                       real=[ToPILImage()(x) for x in real_img], 
                       fake=[ToPILImage()(x) for x in fake_img])

        return output

    def a2b(self, images):
        return self.netG(images)

    def netD_loss(self):

        fake = self.fake_B.detach()

        pred_fake = self.netD(fake)
        pred_real = self.netD(self.real_B)

        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real = loss_D_real.mean()

        loss = (loss_D_fake + loss_D_real) * 0.5

        return loss

    def netGF_loss(self):

        fake = self.fake_B

        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
        else:
            loss_G_GAN = 0.0

        if self.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            loss_NCE = 0.0

        if self.nce_idt and self.lambda_NCE > 0.0:
            loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        return loss_G_GAN, loss_NCE_both


def get_gan_networks(args, accelerator: Accelerator):
    model = CUT(args).to(accelerator.device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.compile(model)

    B = args.CUT.batch_size
    H, W = args.GENERAL.resolution
    C = 3 if args.GENERAL.rgb else 1
    sample_batch = torch.rand(B, C, H, W, device=accelerator.device)
    with torch.no_grad():
        model.calculate_NCE_loss(sample_batch, sample_batch)
    del sample_batch

    optimizers = Munch()
    for net in ['netD', 'netG', 'netF']:
        net_args = getattr(args.CUT, net)
        optim = get_class(net_args.optimizer.type)(getattr(model, net).parameters(), **net_args.optimizer.params)
        scheduler = get_class(net_args.scheduler.type)(optim, **net_args.scheduler.params)
        optimizers.update({net: Munch(optim=optim, scheduler=scheduler)})

    model.netG = init_net(model.netG, **args.CUT.netG.net_init)
    model.netD = init_net(model.netD, **args.CUT.netD.net_init)
    model.netF = init_net(model.netF, **args.CUT.netF.net_init)

    model = accelerator.prepare_model(model)
    for net in ['netD', 'netG', 'netF']:
        optim, scheduler = getattr(optimizers, net).optim, getattr(optimizers, net).scheduler
        optim, scheduler = accelerator.prepare(optim, scheduler)
        optim.zero_grad()
        getattr(optimizers, net).update(optim=optim, scheduler=scheduler)

    return model, optimizers
