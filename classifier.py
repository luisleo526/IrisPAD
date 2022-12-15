import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from munch import Munch
from torchvision.models.feature_extraction import create_feature_extractor

from multicropdataset import get_pseudo_label
from supconloss import SupConLoss
from utils import get_class, rsetattr, init_net


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.CLASSIFIER.model.type)(**args.CLASSIFIER.model.params)
        for replacement in args.CLASSIFIER.model.replacements:
            net = get_class(replacement.type)(**replacement.params)
            net = init_net(net, **args.CLASSIFIER.net_init)
            rsetattr(self.model, replacement.name, net)
        if args.CLASSIFIER.model.params.weights is None:
            self.model = init_net(self.model, **args.CLASSIFIER.net_init)

        return_nodes = {args.CLASSIFIER.model.extractor.output: 'output'}
        for i, layer in enumerate(args.CLASSIFIER.model.extractor.features):
            return_nodes[layer] = f"feature_{i}"
        self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
        self.model.double()

        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.pad_token_id = args.CLASSIFIER.pad_token_id
        self.confidence_selfTraining = args.CLASSIFIER.confidence_selfTraining
        self.confidence_CUT = args.CLASSIFIER.confidence_CUT

        self.ConLoss = SupConLoss(temperature=args.CLASSIFIER.pretrain.temperature)
        self.ConLabel = get_pseudo_label(args)
        self.ConNCrops = np.cumsum([0] + args.CLASSIFIER.pretrain.config.num_crops).tolist()
        self.return_nodes = list(return_nodes.values())
        self.return_nodes.remove("output")

    def forward(self, batch, pretrain=False):

        if not pretrain:
            output = self.model(batch['image'])['output']
            loss = self.loss_fn(output, batch['label'])

            padding = [torch.tensor(self.pad_token_id, device=batch['image'].device, dtype=torch.int32)]
            pred_all = output.argmax(dim=-1)

            mask = (torch.nn.Softmax(dim=-1)(output)).max(dim=-1)[0] > self.confidence_CUT
            paths = batch['path'][mask]
            preds = pred_all[mask]
            label_0 = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 0])
            label_1 = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 1])

            mask = (torch.nn.Softmax(dim=-1)(output)).max(dim=-1)[0] > self.confidence_selfTraining
            paths = batch['path'][mask]
            preds = pred_all[mask]
            label_0_mask = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 0])
            label_1_mask = torch.stack(padding + [paths[x] for x in range(len(preds)) if preds[x].item() == 1])

            return Munch(loss=loss, pred=pred_all, label_0=label_0, label_1=label_1,
                         label_0_mask=label_0_mask, label_1_mask=label_1_mask)
        else:
            _output = [self.model(x) for x in batch]
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
    model = Classifier(args).to(accelerator.device)
    # model = torch.compile(model)
    optimizer = get_class(args.CLASSIFIER.optimizer.type)(model.parameters(), **args.CLASSIFIER.optimizer.params)
    scheduler = get_class(args.CLASSIFIER.scheduler.type)(optimizer, **args.CLASSIFIER.scheduler.params)
    if args.CLASSIFIER.pretrain.apply:
        scheduler_pretrain = get_class(args.CLASSIFIER.pretrain.scheduler.type)(
            optimizer, **args.CLASSIFIER.pretrain.scheduler.params)
    else:
        scheduler_pretrain = None
    model, optimizer, scheduler = accelerator.prepare([model, optimizer, scheduler])
    return model, Munch(optim=optimizer, scheduler=scheduler, scheduler_pretrain=scheduler_pretrain)
