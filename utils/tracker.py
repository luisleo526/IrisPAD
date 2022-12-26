from typing import List
from munch import Munch
from prettytable import PrettyTable
import numpy as np


class Tracker(object):
    def __init__(self, truncate: int, value_names: List[str], ds_names: List[str]):
        self.truncate = truncate
        self.targets = Munch({key: Munch({_key: [] for _key in value_names}) for key in ds_names})
        self.value_names = value_names

    def __call__(self, name: str, key: str, value: float):
        if name in self.targets.keys() and key in self.targets[name].keys():
            self.targets[name][key].append(value)

    def get_table(self, step: int):

        tb = PrettyTable()
        tb.title = f"Milestone at {step}"
        tb.field_names = ["Dataset", "", ] + self.value_names
        for j, (key, value) in enumerate(self.targets.items()):
            if j != 0:
                tb.add_row(["-" * x for x in [12, 5] + [15 for _ in range(len(self.value_names))]])
            for i, (reduction, reduction_fn, comment_fn) in enumerate(
                    [("Mean", lambda x: f"{np.mean(x[-self.truncate:]) * 100:2.2f}",
                      lambda x: f"{np.std(x[-self.truncate:]) * 100:2.2f}"),
                     ("Max", lambda x: f"{np.max(x[-self.truncate:]) * 100:2.2f}",
                      lambda x: f"{int(np.argmax(x[-self.truncate:]))+len(x)-self.truncate:5}"),
                     ("Min", lambda x: f"{np.min(x[-self.truncate:]) * 100:2.2f}",
                      lambda x: f"{int(np.argmin(x[-self.truncate:]))+len(x)-self.truncate:5}")]):
                if i == 1:
                    title = key
                else:
                    title = ""
                tb.add_row(
                    [title, reduction] + [f"{reduction_fn(value[_key])} ({comment_fn(value[_key])})"
                                          for _key in self.value_names])

        return tb
