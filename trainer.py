import inspect
from typing import List

import torch
from accelerate import Accelerator
from munch import Munch
from tqdm.auto import tqdm

import wandb
from dataset.datasets import make_gan_loader, _make_data_loader
from utils.metrics import ISOMetrics
from utils.vocab import Vocab


def run(args, paths_from_train, paths_for_selftraining, num_epoch: int, step: int, accelerator: Accelerator, nets,
        loaders, vocab: Vocab, use_gan: bool, iterative: bool, warmup: bool, train_gan: bool, tqdm_no_progress: bool,
        self_training: bool, self_training_refresh: bool):
    pad_token_id = args.CLASSIFIER.pad_token_id

    if self_training_refresh:
        paths_for_selftraining = None

    for _ in tqdm(range(num_epoch), disable=tqdm_no_progress):

        step += 1
        metrics = ISOMetrics()
        results = {}

        # train classifier
        pr_data = Munch()
        for name, loader in loaders.train.items():
            if not loader.config.skip or warmup:
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
                            new_batch = dict(image=new_images, label=new_labels, path=new_paths)
                        else:
                            new_batch = None

                        for data in [batch, new_batch]:
                            if data is not None:
                                outputs = nets.classifier.model(data)
                                accelerator.backward(outputs.loss)
                                nets.classifier.optimizers.optim.step()
                                nets.classifier.optimizers.optim.zero_grad()

                                pred, tgt, loss = accelerator.gather_for_metrics(
                                    (outputs.pred, data['label'], outputs.loss.detach()))
                                metrics(pred, tgt, loss)

                                # PR curve data
                                pred_confidence = accelerator.gather_for_metrics(outputs.pred_confidence.contiguous())
                                pr_data[name].confid.append(pred_confidence)
                                pr_data[name].truth.append(tgt)

                    for key, value in metrics.aggregate().items():
                        results[f"{key}/{name}/TRAIN"] = value
                else:
                    if self_training:
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
                        if len(label_0) + len(label_1) > accelerator.num_processes:
                            pr_data.update({name: Munch(confid=[], truth=[])})
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
                                results[f"{key}/{name}/TRAIN"] = value
                        else:
                            accelerator.print("Model not confident enough for self training...")

        # test classifier
        acer_list = {}
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
                results[f"{key}/{name}/TEST"] = value
                if key == 'acer':
                    acer_list[name] = value

        results["Overall-Score"] = sum([2 ** i * x for i, x in enumerate(sorted(list(acer_list.values())))])

        if not warmup:
            scheduler = nets.classifier.optimizers.scheduler
            if "metrics" in list(inspect.signature(scheduler.step).parameters):
                scheduler.step(metrics=results["Overall-Score"])
            else:
                scheduler.step()

        if use_gan and train_gan and step > 10:

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
                cut_labels = dict(cut="Attack" if iterative else "cut", cut2="Bona fide")
                loss_dict = dict(netD="lossD", netG="lossG", netF="lossF")
                for name, loader in gan_lds.items():
                    results[f"CUT/{cut_labels[name]}/lossG"] = 0
                    results[f"CUT/{cut_labels[name]}/lossD"] = 0
                    results[f"CUT/{cut_labels[name]}/lossF"] = 0
                    nets[name].model.train()
                    for batch in loader:
                        outputs = nets[name].model(batch)
                        accelerator.backward(outputs.lossD + outputs.lossG + outputs.lossF)
                        lossG, lossD, lossF = accelerator.gather((outputs.lossG.detach(),
                                                                  outputs.lossD.detach(),
                                                                  outputs.lossF.detach()))
                        results[f"CUT/{cut_labels[name]}/lossG"] += lossG.sum().item()
                        results[f"CUT/{cut_labels[name]}/lossD"] += lossD.sum().item()
                        results[f"CUT/{cut_labels[name]}/lossF"] += lossF.sum().item()
                        for net in ['netD', 'netG', 'netF']:
                            nets[name].optimizers[net].optim.step()
                            nets[name].optimizers[net].optim.zero_grad()

                    if not warmup:
                        for net in ['netD', 'netG', 'netF']:
                            scheduler = nets[name].optimizers[net].scheduler
                            if "metrics" in list(inspect.signature(scheduler.step).parameters):
                                scheduler.step(metrics=results[f"CUT/{cut_labels[name]}/{loss_dict[net]}"])
                            else:
                                scheduler.step()

                    if accelerator.is_main_process:
                        table = wandb.Table(columns=["Real-Image", "Fake-Image"], allow_mixed_types=True)
                        table.add_data([wandb.Image(x) for x in outputs.real],
                                       [wandb.Image(x) for x in outputs.fake])
                        results[f"CUT/Samples/{cut_labels[name]}"] = table
            else:
                accelerator.print("Imblance data configuration, skip CUT training...")

        if accelerator.is_main_process:
            wandb.log(results)

    return step, paths_for_selftraining


def run_pretrain(args, loaders, nets, accelerator: Accelerator, num_epoch: int, tqdm_no_progress: bool):

    if accelerator.is_main_process:
        wandb.define_metric("pretrain_step")
        wandb.define_metric("SimCLR-loss", step_metric="pretrain_step")

    step = 0
    progress_bar = tqdm(range(num_epoch * len(loaders.pretrain.dl)), disable=tqdm_no_progress)
    return_nodes = accelerator.unwrap_model(nets.classifier.model).return_nodes
    for _ in range(num_epoch):

        nets.classifier.model.train()
        for batch in loaders.pretrain.dl:

            results = {'SimCLR-loss': 0}
            for layer_name in return_nodes:
                results.update({f"{layer_name}": Munch()})
                for j in range(len(args.CLASSIFIER.pretrain.config.num_crops)):
                    results[f"{layer_name}"].update({f"{j}": 0})

            losses = nets.classifier.model(batch, pretrain=True)
            loss_sum = 0
            for name, loss_list in losses.items():
                for i, loss in enumerate(loss_list):
                    loss_sum += loss
                    loss = accelerator.gather_for_metrics(loss.detach())
                    results['SimCLR-loss'] += loss.sum().item()

            accelerator.backward(loss_sum)
            nets.classifier.optimizers.optim_pretrain.step()
            nets.classifier.optimizers.optim_pretrain.zero_grad()
            nets.classifier.optimizers.scheduler_pretrain.step()

            if accelerator.is_main_process:
                step = step + 1
                results['pretrain_step'] = step
                wandb.log(results)

            accelerator.wait_for_everyone()

            if accelerator.sync_gradients:
                progress_bar.update(1)
