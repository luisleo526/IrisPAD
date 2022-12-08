import torch
from munch import Munch
from tqdm.auto import tqdm

from datasets import make_gan_loader
from metrics import ISOMetrics
from vocab import Vocab
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


def run(args, paths_from_train, num_epoch: int, step: int,
        accelerator: Accelerator, writer: Optional[SummaryWriter], nets, loaders, vocab: Vocab,
        use_gan: bool, iterative: bool, warmup: bool, train_gan: bool):
    pad_token_id = args.GENERAL.pad_token_id
    for _ in tqdm(range(num_epoch), disable=not accelerator.is_local_main_process):
        step += 1
        metrics = ISOMetrics()
        results = Munch()
        for key in metrics.aggregate().keys():
            results.update({key: Munch()})

        # train classifier
        nets.classifier.model.train()
        for name, loader in loaders.train.items():
            if (not loader.config.skip and not warmup) or warmup:
                for batch in loader.dl:
                    outputs = nets.classifier.model(batch)
                    accelerator.backward(outputs.loss)
                    nets.classifier.optimizers.optim.step()
                    nets.classifier.optimizers.optim.zero_grad()
                    nets.classifier.optimizers.scheduler.step()

                    pred, tgt, loss = accelerator.gather_for_metrics(
                        (outputs.pred, batch['label'], outputs.loss.detach()))
                    metrics(pred, tgt, loss)

                    if not warmup and use_gan:
                        new_images = []
                        with torch.no_grad():
                            if iterative:
                                new_images.append(
                                    accelerator.unwrap_model(nets.cut.model).a2b(batch['image'][batch['label'] == 0]))
                                new_images.append(
                                    accelerator.unwrap_model(nets.cut2.model).a2b(batch['image'][batch['label'] == 1]))
                            else:
                                new_images.append(
                                    accelerator.unwrap_model(nets.cut.model).a2b(batch['image']))
                        batch['image'] = torch.cat(new_images)
                        outputs = nets.classifier.model(batch)
                        accelerator.backward(outputs.loss)
                        nets.classifier.optimizers.optim.step()
                        nets.classifier.optimizers.optim.zero_grad()
                        nets.classifier.optimizers.scheduler.step()

                        pred, tgt, loss = accelerator.gather_for_metrics(
                            (outputs.pred, batch['label'], outputs.loss.detach()))
                        metrics(pred, tgt, loss)

                for key, value in metrics.aggregate().items():
                    results[key].update({name + '_TRAIN': value})

        # test classifier
        nets.classifier.model.eval()
        paths_from_test = Munch(label_0=[], label_1=[])
        for name, loader in loaders.test.items():
            for batch in loader.dl:
                with torch.no_grad():
                    outputs = nets.classifier.model(batch)
                    pred, tgt, loss = accelerator.gather_for_metrics(
                        (outputs.pred, batch['label'], outputs.loss.detach()))
                    metrics(pred, tgt, loss)
                    if loader.config.gan:
                        label_0 = accelerator.pad_across_processes(outputs.label_0,
                                                                   dim=0, pad_index=pad_token_id,
                                                                   pad_first=False)
                        label_1 = accelerator.pad_across_processes(outputs.label_1,
                                                                   dim=0, pad_index=pad_token_id,
                                                                   pad_first=False)
                        label_0 = accelerator.gather(label_0)
                        label_1 = accelerator.gather(label_1)
                        paths_from_test.label_0.extend(label_0[label_0 != pad_token_id].tolist())
                        paths_from_test.label_1.extend(label_1[label_1 != pad_token_id].tolist())

            for key, value in metrics.aggregate().items():
                results[key].update({name + '_TEST': value})

        if accelerator.is_main_process:
            for key, value in results.items():
                writer.add_scalars(key, dict(value), global_step=step)
            writer.flush()

        if use_gan and train_gan:
            # prepare data for gan
            assert len(paths_from_test.label_0) > 0 and len(paths_from_test.label_1) > 0, \
                "Imbalance dataset across labels"
            paths_from_test.label_0 = vocab.index2word(paths_from_test.label_0)
            paths_from_test.label_1 = vocab.index2word(paths_from_test.label_1)
            if iterative:
                gan_ld1 = make_gan_loader(args, paths_from_train.label_0, paths_from_test.label_0, accelerator)
                gan_ld2 = make_gan_loader(args, paths_from_train.label_1, paths_from_test.label_1, accelerator)
                gan_lds = Munch(cut=gan_ld1, cut2=gan_ld2)
            else:
                gan_lds = Munch(cut=make_gan_loader(args,
                                                    paths_from_train.label_0 + paths_from_train.label_1,
                                                    paths_from_test.label_0 + paths_from_test.label_1,
                                                    accelerator
                                                    )
                                )

            # train gan
            results = Munch()
            for name, loader in gan_lds.items():
                results.update({name: Munch(lossG=0, lossD=0, lossF=0)})
                nets[name].model.train()
                for batch in loader:
                    outputs = nets[name].model(batch)
                    accelerator.backward(outputs.lossD + outputs.lossG + outputs.lossF)
                    lossG, lossD, lossF = accelerator.gather((outputs.lossG.detach(),
                                                              outputs.lossD.detach(),
                                                              outputs.lossF.detach()))
                    results[name].lossG += lossG.sum().item()
                    results[name].lossD += lossD.sum().item()
                    results[name].lossF += lossF.sum().item()
                    for net in ['netD', 'netG', 'netF']:
                        nets[name].optimizers[net].optim.step()
                        nets[name].optimizers[net].optim.zero_grad()
                        nets[name].optimizers[net].scheduler.step()
                if accelerator.is_main_process:
                    writer.add_images("Real Images: " + name, outputs.real, global_step=step)
                    writer.add_images("Fake Images: " + name, outputs.fake, global_step=step)
                    writer.flush()

            if accelerator.is_main_process:
                for key, value in results.items():
                    writer.add_scalars(key + " losses", dict(value), global_step=step)
                writer.flush()

    return step
