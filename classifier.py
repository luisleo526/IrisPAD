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
            net = init_net(net, **args.GENERAL.net_init)
            rsetattr(self.model, replacement.name, net)
        if args.CLASSIFIER.model.params.weights is None:
            self.model = init_net(self.model, **args.GENERAL.net_init)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.pad_token_id = args.GENERAL.pad_token_id
        self.confidence_threshold = args.CLASSIFIER.confidence_threshold

    def forward(self, batch):
        pred = self.model(batch['image'])
        loss = self.loss_fn(pred, batch['label'])
        mask = (torch.nn.Softmax(dim=-1)(pred)).max(dim=-1)[0] > self.confidence_threshold
        pred = pred.argmax(dim=-1)
        padding = [torch.tensor(self.pad_token_id, device=batch['image'].device, dtype=torch.int32)]
        label_0 = torch.stack(padding + [batch['path'][x] for x in range(len(pred)) if pred[x].item() == 0])
        label_1 = torch.stack(padding + [batch['path'][x] for x in range(len(pred)) if pred[x].item() == 1])
        paths = batch['path'][mask]
        pred_mask = pred[mask]
        label_0_mask = torch.stack(padding + [paths[x] for x in range(len(pred_mask)) if pred_mask[x].item() == 0])
        label_1_mask = torch.stack(padding + [paths[x] for x in range(len(pred_mask)) if pred_mask[x].item() == 1])

        return Munch(loss=loss, pred=pred, label_0=label_0, label_1=label_1,
                     label_0_mask=label_0_mask, label_1_mask=label_1_mask)


def get_classifier_networks(args, accelerator: Accelerator):
    model = Classifier(args).to(accelerator.device)
    # model = torch.compile(model, mode="reduce-overhead")
    optimizer = get_class(args.CLASSIFIER.optimizer.type)(model.parameters(), **args.CLASSIFIER.optimizer.params)
    scheduler = get_class(args.CLASSIFIER.scheduler.type)(optimizer, **args.CLASSIFIER.scheduler.params)
    model, optimizer, scheduler = accelerator.prepare([model, optimizer, scheduler])
    return model, Munch(optim=optimizer, scheduler=scheduler)
