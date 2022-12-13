import torch
import torch.nn as nn
from accelerate import Accelerator
from munch import Munch

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
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.pad_token_id = args.GENERAL.pad_token_id
        self.confidence_selfTraining = args.CLASSIFIER.confidence_selfTraining
        self.confidence_CUT = args.CLASSIFIER.confidence_CUT

    def forward(self, batch):
        output = self.model(batch['image'])
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


def get_classifier_networks(args, accelerator: Accelerator):
    model = Classifier(args).to(accelerator.device)
    # model = torch.compile(model)
    optimizer = get_class(args.CLASSIFIER.optimizer.type)(model.parameters(), **args.CLASSIFIER.optimizer.params)
    scheduler = get_class(args.CLASSIFIER.scheduler.type)(optimizer, **args.CLASSIFIER.scheduler.params)
    model, optimizer, scheduler = accelerator.prepare([model, optimizer, scheduler])
    return model, Munch(optim=optimizer, scheduler=scheduler)
