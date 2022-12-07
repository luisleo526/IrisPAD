from utils import get_class
from accelerate import Accelerator
import torch.nn as nn
from munch import Munch


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.CLASSIFIER.model.type)(**args.CLASSIFIER.model.params)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        pred = self.model(batch['image'])
        loss = self.loss_fn(pred, batch['label'])
        pred = pred.argmax(dim=-1)
        label_0 = [batch['path'][x] for x in range(len(pred)) if pred[x].item() == 0]
        label_1 = [batch['path'][x] for x in range(len(pred)) if pred[x].item() == 1]
        return Munch(loss=loss, pred=pred, label_0=label_0, label_1=label_1)


def get_classifier_networks(args, accelerator: Accelerator):
    model = Classifier(args).to(accelerator.device)
    optimizer = get_class(args.CLASSIFIER.optimizer.type)(model.parameters(), **args.CLASSIFIER.optimizer.params)
    scheduler = get_class(args.CLASSIFIER.scheduler.type)(optimizer, **args.CLASSIFIER.scheduler.params)
    model, optimizer, scheduler = accelerator.prepare([model, optimizer, scheduler])
    return model, Munch(optim=optimizer, scheduler=scheduler)
