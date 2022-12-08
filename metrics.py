from munch import Munch
import torch


class ISOMetrics:
    def __init__(self):
        self.n = None
        self.scores = None
        self.loss = None
        self.reset()

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor, loss: torch.Tensor):
        assert len(pred) == len(tgt)
        self.n.all += len(pred)
        self.n.pos += torch.sum(tgt == 1).item()
        self.n.neg += torch.sum(tgt == 0).item()
        self.scores.acc += torch.sum(pred == tgt).item()
        self.scores.apcer += torch.sum((pred == 1) & (tgt == 0)).item()
        self.scores.bpcer += torch.sum((pred == 0) & (tgt == 1)).item()
        self.loss += loss.sum().item()

    def reset(self):
        self.n = Munch(all=1.0e-7, pos=1.0e-7, neg=1.0e-7)
        self.scores = Munch(acc=0, apcer=0, bpcer=0)
        self.loss = 0

    def aggregate(self):
        result = Munch(acc=self.scores.acc / self.n.all,
                       apcer=self.scores.apcer / self.n.neg,
                       bpcer=self.scores.bpcer / self.n.pos,
                       loss=self.loss / self.n.all)
        result.update(acer=(result.apcer + result.bpcer) / 2)
        self.reset()
        return result
