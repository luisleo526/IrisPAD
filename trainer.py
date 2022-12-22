import inspect
from typing import Optional, List

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from munch import Munch
from sklearn.metrics import RocCurveDisplay
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset.datasets import make_gan_loader, _make_data_loader
from utils.metrics import ISOMetrics
from utils.vocab import Vocab


def run(args, paths_from_train, paths_for_selftraining, num_epoch: int, step: int,
        accelerator: Accelerator, writer: Optional[SummaryWriter], nets, loaders, vocab: Vocab,
        use_gan: bool, iterative: bool, warmup: bool, train_gan: bool, tqdm_no_progress: bool, self_training: bool,
        self_training_refresh: bool):
    pad_token_id = args.CLASSIFIER.pad_token_id

    if self_training_refresh:
        paths_for_selftraining = None

    for _ in tqdm(range(num_epoch), disable=tqdm_no_progress):

        step += 1

        if accelerator.is_main_process:
            for name, weight in accelerator.unwrap_model(nets.classifier.model).model.named_parameters():
                writer.add_histogram(f"TRAIN_classifier/{name}", weight, step)
            writer.flush()

        metrics = ISOMetrics()

        results = Munch()
        for key in metrics.aggregate().keys():
            results.update({key: Munch()})

        # train classifier
        pr_data = Munch()
        for name, loader in loaders.train.items():
            if (not loader.config.skip and not warmup) or warmup:
                if not loader.config.selftraining:
                    pr_data.update({name: Munch(confid=[], truth=[])})
                    nets.classifier.model.train()
                    for batch in loader.dl:
                        if not warmup and use_gan:
                            with torch.no_grad():
                                if iterative:
                                    nets.cut.model.eval()
                                    nets.cut2.model.eval()
                                    label_0_images = batch['image'][batch['label'] == 0]
                                    label_0_paths = batch['path'][batch['label'] == 0]
                                    label_0_images = accelerator.unwrap_model(nets.cut.model).a2b(label_0_images)
                                    label_1_images = batch['image'][batch['label'] == 1]
                                    label_1_paths = batch['path'][batch['label'] == 1]
                                    label_1_images = accelerator.unwrap_model(nets.cut2.model).a2b(label_1_images)
                                    new_images = torch.cat([label_0_images, label_1_images], dim=0)
                                    new_paths = torch.cat([label_0_paths, label_1_paths], dim=0)
                                    new_labels = torch.cat(
                                        [batch['label'][batch['label'] == 0], batch['label'][batch['label'] == 1]],
                                        dim=0)
                                else:
                                    nets.cut.model.eval()
                                    new_images = accelerator.unwrap_model(nets.cut.model).a2b(batch['image'])
                                    new_labels = batch['labels']
                                    new_paths = batch['path']
                            batch["image"] = torch.cat([batch["image"], new_images])
                            batch["label"] = torch.cat([batch["label"], new_labels])
                            batch["path"] = torch.cat([batch["path"], new_paths])

                        outputs = nets.classifier.model(batch)
                        accelerator.backward(outputs.loss)
                        nets.classifier.optimizers.optim.step()
                        nets.classifier.optimizers.optim.zero_grad()

                        pred, tgt, loss = accelerator.gather_for_metrics(
                            (outputs.pred, batch['label'], outputs.loss.detach()))
                        metrics(pred, tgt, loss)

                        # PR curve data
                        pred_confidence = accelerator.gather_for_metrics(outputs.pred_confidence.contiguous())
                        pr_data[name].confid.append(pred_confidence)
                        pr_data[name].truth.append(tgt)

                    nets.classifier.optimizers.optim.step()

                    for key, value in metrics.aggregate().items():
                        results[key].update({name: value})
                else:
                    if self_training:
                        pr_data.update({name: Munch(confid=[], truth=[])})
                        if paths_for_selftraining is None:
                            paths_for_selftraining = Munch(label_0=[], label_1=[])

                        paths_for_selftraining_new = Munch(label_0=[], label_1=[])
                        nets.classifier.model.eval()
                        for batch in loader.dl:
                            with torch.no_grad():
                                outputs = nets.classifier.model(batch)
                                label_0 = accelerator.pad_across_processes(outputs.label_0_sftr,
                                                                           dim=0, pad_index=pad_token_id,
                                                                           pad_first=False)
                                label_1 = accelerator.pad_across_processes(outputs.label_1_sftr,
                                                                           dim=0, pad_index=pad_token_id,
                                                                           pad_first=False)
                                label_0 = accelerator.gather(label_0)
                                label_1 = accelerator.gather(label_1)
                                paths_for_selftraining_new.label_0.extend(label_0[label_0 != pad_token_id].tolist())
                                paths_for_selftraining_new.label_1.extend(label_1[label_1 != pad_token_id].tolist())

                        paths_for_selftraining_new.label_0 = [x for x in paths_for_selftraining_new.label_0
                                                              if x not in paths_for_selftraining.label_1]
                        paths_for_selftraining_new.label_1 = [x for x in paths_for_selftraining_new.label_1
                                                              if x not in paths_for_selftraining.label_0]
                        paths_for_selftraining.label_0.extend(paths_for_selftraining_new.label_0)
                        paths_for_selftraining.label_1.extend(paths_for_selftraining_new.label_1)
                        label_0: List[str] = vocab.index2word(list(set(paths_for_selftraining.label_0)))
                        label_1: List[str] = vocab.index2word(list(set(paths_for_selftraining.label_1)))
                        nets.classifier.model.train()
                        for batch in _make_data_loader(args,
                                                       accelerator, vocab, label_0=label_0, label_1=label_1,
                                                       use_augmentation=True):
                            outputs = nets.classifier.model(batch)
                            accelerator.backward(outputs.loss)
                            nets.classifier.optimizers.optim.step()
                            nets.classifier.optimizers.optim.zero_grad()

                            pred, tgt, loss = accelerator.gather_for_metrics(
                                (outputs.pred, batch['label'], outputs.loss.detach()))
                            metrics(pred, tgt, loss)

                            # PR curve data
                            pred_confidence = accelerator.gather_for_metrics(outputs.pred_confidence.contiguous())
                            pr_data[name].confid.append(pred_confidence)
                            pr_data[name].truth.append(tgt)

                        for key, value in metrics.aggregate().items():
                            results[key].update({name: value})

        if accelerator.is_main_process:
            for key, value in results.items():
                writer.add_scalars(f"TRAIN/{key}", dict(value), global_step=step)
            fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
            for name, data in pr_data.items():
                writer.add_pr_curve(f"TRAIN/pr_curve/{name}", labels=torch.cat(data.truth),
                                    predictions=torch.cat(data.confid), global_step=step)
                RocCurveDisplay.from_predictions(y_true=torch.cat(data.truth).cpu().numpy(),
                                                 y_pred=torch.cat(data.confid).cpu().numpy(),
                                                 ax=ax, name=name)
            writer.add_figure("ROC/TRAIN", figure=fig, global_step=step)
            writer.flush()

        results = Munch()
        for key in metrics.aggregate().keys():
            results.update({key: Munch()})

        # test classifier
        max_acer = 0
        paths_from_test = Munch(label_0=[], label_1=[])
        pr_data = Munch()
        for name, loader in loaders.test.items():
            pr_data.update({name: Munch(confid=[], truth=[])})
            nets.classifier.model.eval()
            for batch in loader.dl:
                with torch.no_grad():
                    outputs = nets.classifier.model(batch)
                    pred, tgt, loss = accelerator.gather_for_metrics(
                        (outputs.pred, batch['label'], outputs.loss.detach()))
                    metrics(pred, tgt, loss)

                    # PR curve data
                    pred_confidence = accelerator.gather_for_metrics(outputs.pred_confidence.contiguous())
                    pr_data[name].confid.append(pred_confidence)
                    pr_data[name].truth.append(tgt)

                    if loader.config.gan:
                        label_0 = accelerator.pad_across_processes(outputs.label_0_cut,
                                                                   dim=0, pad_index=pad_token_id,
                                                                   pad_first=False)
                        label_1 = accelerator.pad_across_processes(outputs.label_1_cut,
                                                                   dim=0, pad_index=pad_token_id,
                                                                   pad_first=False)

                        label_0, label_1 = accelerator.gather((label_0, label_1))

                        paths_from_test.label_0.extend(label_0[label_0 != pad_token_id].tolist())
                        paths_from_test.label_1.extend(label_1[label_1 != pad_token_id].tolist())

            for key, value in metrics.aggregate().items():
                nets.tracker(name, key, value)
                results[key].update({name: value})
                if key == 'acer':
                    max_acer = max(value, max_acer)

        if not warmup:
            scheduler = nets.classifier.optimizers.scheduler
            if "metrics" in list(inspect.signature(scheduler.step).parameters):
                scheduler.step(metrics=max_acer)
            else:
                scheduler.step()

        if accelerator.is_main_process:
            for key, value in results.items():
                writer.add_scalars(f"TEST/{key}", dict(value), global_step=step)
            fig, ax = plt.subplots(dpi=100, figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
            for name, data in pr_data.items():
                writer.add_pr_curve(f"TEST/pr_curve/{name}", labels=torch.cat(data.truth),
                                    predictions=torch.cat(data.confid), global_step=step)
                RocCurveDisplay.from_predictions(y_true=torch.cat(data.truth).cpu().numpy(),
                                                 y_pred=torch.cat(data.confid).cpu().numpy(),
                                                 ax=ax, name=name)
            writer.add_figure("ROC/TEST", figure=fig, global_step=step)
            writer.flush()

        if use_gan and train_gan:

            # prepare data for gan
            paths_from_test.label_0 = vocab.index2word(paths_from_test.label_0)
            paths_from_test.label_1 = vocab.index2word(paths_from_test.label_1)

            if iterative and len(paths_from_test.label_0) > accelerator.num_processes and len(
                    paths_from_test.label_1) > accelerator.num_processes:

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
                cut_labels = dict(cut="Label 0" if iterative else "cut", cut2="Label 1")
                loss_dict = dict(netD="lossD", netG="lossG", netF="lossF")
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

                    if not warmup:
                        for net in ['netD', 'netG', 'netF']:
                            scheduler = nets[name].optimizers[net].scheduler
                            if "metrics" in list(inspect.signature(scheduler.step).parameters):
                                scheduler.step(metrics=results[name][loss_dict[net]])
                            else:
                                scheduler.step()

                    if accelerator.is_main_process:
                        writer.add_images(f"{cut_labels[name]}/RealImages", outputs.real, global_step=step)
                        writer.add_images(f"{cut_labels[name]}/FakeImages", outputs.fake, global_step=step)
                        writer.flush()

                if accelerator.is_main_process:
                    for key, value in results.items():
                        writer.add_scalars(f"CUT/{cut_labels[key]}_losses", dict(value), global_step=step)
                    writer.flush()
                    if iterative:
                        net_list = ['cut', 'cut2']
                    else:
                        net_list = ['cut']
                    for main_net in net_list:
                        for net in ['netD', 'netG', 'netF']:
                            for name, weight in getattr(accelerator.unwrap_model(nets[main_net].model),
                                                        net).named_parameters():
                                writer.add_histogram(f"{main_net}_{net}/{name}", weight, step)
                    writer.flush()

            else:

                accelerator.print("Imblance data configuration, skip CUT training...")

    return step, paths_for_selftraining


def run_pretrain(args, loaders, nets, accelerator: Accelerator, writer: SummaryWriter, num_epoch: int,
                 tqdm_no_progress: bool):
    step = 0
    progress_bar = tqdm(range(num_epoch * len(loaders.pretrain.dl)), disable=tqdm_no_progress)
    for _ in range(num_epoch):

        nets.classifier.model.train()
        for batch in loaders.pretrain.dl:

            results = Munch()
            for layer_name in nets.classifier.model.return_nodes:
                results.update({f"{layer_name}": Munch()})
                for j in range(len(args.CLASSIFIER.pretrain.config.num_crops)):
                    results[f"{layer_name}"].update({f"{j}": 0})

            losses = nets.classifier.model(batch, pretrain=True)
            loss_sum = 0
            for name, loss_list in losses.items():
                for i, loss in enumerate(loss_list):
                    loss_sum += loss
                    loss = accelerator.gather_for_metrics(loss.detach())
                    results[name][f"{i}"] += loss.sum().item()

            accelerator.backward(loss_sum)
            nets.classifier.optimizers.optim_pretrain.step()
            nets.classifier.optimizers.optim_pretrain.zero_grad()
            nets.classifier.optimizers.scheduler_pretrain.step()

            if accelerator.is_main_process:
                for key, value in results.items():
                    writer.add_scalars(f"PRETRAIN/{key}", dict(value), global_step=step)
                writer.flush()
                if step % 10 == 0:
                    for name, weight in nets.classifier.model.model.named_parameters():
                        writer.add_histogram(f"PRETRAIN_classifier/{name}", weight, step)
                    writer.flush()
                step += 1

            accelerator.wait_for_everyone()

            if accelerator.sync_gradients:
                progress_bar.update(1)
