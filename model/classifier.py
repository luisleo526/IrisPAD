import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from munch import Munch
from torch.nn import SyncBatchNorm
from torchvision.models.feature_extraction import create_feature_extractor

from dataset.multicropdataset import get_pseudo_label
from model.supconloss import SupConLoss
from utils.utils import get_class, rsetattr, init_net, rgetattr


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.CLASSIFIER.model.type)(**args.CLASSIFIER.model.params)
        if 'replacements' in args.CLASSIFIER.model and len(args.CLASSIFIER.model.replacements) > 0:
            for replacement in args.CLASSIFIER.model.replacements:
                net = get_class(replacement.type)(**replacement.params)
                net = init_net(net, **args.CLASSIFIER.net_init)
                rsetattr(self.model, replacement.name, net)
        if args.CLASSIFIER.model.params.weights is None:
            self.model = init_net(self.model, **args.CLASSIFIER.net_init)

        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.pad_token_id = args.CLASSIFIER.pad_token_id
        self.confidence_selfTraining = args.CLASSIFIER.confidence_selfTraining
        self.confidence_CUT = args.CLASSIFIER.confidence_CUT

        if args.CLASSIFIER.pretrain.apply:
            return_nodes = {}
            for layer_name in args.CLASSIFIER.model.extractor:
                for i, layer in enumerate(args.CLASSIFIER.model.extractor[layer_name]):
                    return_nodes[layer] = f"{layer_name}-{i}"
            self.extractor = create_feature_extractor(self.model, return_nodes=return_nodes)
            self.ConLoss = SupConLoss(temperature=args.CLASSIFIER.pretrain.temperature)
            self.ConLabel = get_pseudo_label(args)
            self.ConNCrops = np.cumsum([0] + args.CLASSIFIER.pretrain.config.num_crops).tolist()
            self.return_nodes = list(return_nodes.values())

    def forward(self, batch, pretrain=False):

        if not pretrain:
            output = self.model(batch['image'])
            loss = self.loss_fn(output, batch['label'])

            padding = [torch.tensor(self.pad_token_id, device=batch['image'].device, dtype=torch.int32)]
            pred_label = output.argmax(dim=-1).detach()
            pred_confidence = torch.nn.Softmax(dim=-1)(output).detach()

            mask = pred_confidence.max(dim=-1)[0] > self.confidence_CUT
            paths = batch['path'][mask]
            preds = pred_label[mask]
            label_0_sftr = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 0])
            label_1_sftr = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 1])

            mask = pred_confidence.max(dim=-1)[0] > self.confidence_selfTraining
            paths = batch['path'][mask]
            preds = pred_label[mask]
            label_0_cut = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 0])
            label_1_cut = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 1])

            pred_confidence = pred_confidence[:, 1]

            return Munch(loss=loss, pred=pred_label, pred_confidence=pred_confidence,
                         label_0_sftr=label_0_sftr, label_1_sftr=label_1_sftr,
                         label_0_cut=label_0_cut, label_1_cut=label_1_cut
                         )
        else:
            _output = [self.extractor(x) for x in batch]
            output = {}
            for layer_name in self.return_nodes:
                output[layer_name] = []
                features = [x[layer_name] for x in _output]
                for i in range(len(self.ConNCrops) - 1):
                    start = self.ConNCrops[i]
                    end = self.ConNCrops[i + 1]
                    loss = self.ConLoss(torch.cat(features[start:end]), self.ConLabel[i])
                    output[layer_name].append(loss)
            return output


def get_classifier_networks(args, accelerator: Accelerator):

    with accelerator.local_main_process_first():
        model = Classifier(args).to(accelerator.device)

    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.compile(model)

    if type(args.CLASSIFIER.optimizer.group) == list and len(args.CLASSIFIER.optimizer.group) > 0:
        params_groups = [dict(x) for x in args.CLASSIFIER.optimizer.group]
        for p in params_groups:
            p['params'] = rgetattr(model.model, p['params']).parameters()
    else:
        params_groups = model.parameters()

    optimizer = get_class(args.CLASSIFIER.optimizer.type)(params_groups, **args.CLASSIFIER.optimizer.params)
    scheduler = get_class(args.CLASSIFIER.scheduler.type)(optimizer, **args.CLASSIFIER.scheduler.params,
                                                          verbose=accelerator.is_local_main_process)
    if args.CLASSIFIER.pretrain.apply:
        optimizer_pretrain = get_class(args.CLASSIFIER.pretrain.optimizer.type)(
            model.parameters(), **args.CLASSIFIER.pretrain.optimizer.params)
        scheduler_pretrain = get_class(args.CLASSIFIER.pretrain.scheduler.type)(
            optimizer_pretrain, **args.CLASSIFIER.pretrain.scheduler.params)
        optimizer_pretrain, scheduler_pretrain = accelerator.prepare(optimizer_pretrain, scheduler_pretrain)
    else:
        scheduler_pretrain = None
        optimizer_pretrain = None

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    return model, Munch(optim=optimizer, optim_pretrain=optimizer_pretrain,
                        scheduler=scheduler, scheduler_pretrain=scheduler_pretrain)
