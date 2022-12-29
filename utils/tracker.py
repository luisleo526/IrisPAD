from typing import List, Dict

import numpy as np
from munch import Munch
from prettytable import PrettyTable

reduction_fns = {'max': lambda x, y: (np.max(x[-y:]), np.argmax(x[-y:]) + len(x) - y),
                 'min': lambda x, y: (np.min(x[-y:]), np.argmin(x[-y:]) + len(x) - y),
                 'mean': lambda x, y: (np.mean(x[-y:]), np.std(x[-y:]))}


class Tracker(object):
    def __init__(self, truncate: int, values: Dict[str, str], ds_names: List[str]):
        self.truncate = truncate
        self.data = Munch({key: Munch({_key: [] for _key in values.keys()}) for key in ds_names})
        self.metrics_name = list(values.keys())
        self.metric_reduction = values

        self.fmt2os = {'max': lambda x, y: "%.4f (%04d)" % reduction_fns['max'](x, y),
                       'min': lambda x, y: "%.4f (%04d)" % reduction_fns['min'](x, y),
                       'mean': lambda x, y: "%.4f (%.4f)" % reduction_fns['mean'](x, y)
                       }

        self.fmt1of = {'max': lambda x, y: float("%.4f" % reduction_fns['max'](x, y)[0]),
                       'min': lambda x, y: float("%.4f" % reduction_fns['min'](x, y)[0]),
                       'mean': lambda x, y: float("%.4f" % reduction_fns['mean'](x, y)[0])
                       }

    def __call__(self, ds: str, metric: str, value: float):
        if ds in self.data.keys() and metric in self.data[ds].keys():
            self.data[ds][metric].append(value)

    def get_table(self, step: int, truncate: int = None):

        if truncate is None:
            truncate = self.truncate

        tb = PrettyTable()
        tb.title = f"Milestone at {step}"
        tb.field_names = ["Dataset"] + [f"{x} ({y})" for x, y in self.metric_reduction.items()]
        for j, (ds, data_dict) in enumerate(self.data.items()):
            if j != 0:
                tb.add_row(["-" * x for x in [12] + [15 for _ in range(len(self.metrics_name))]])
            tb.add_row([ds] + [self.fmt2os[red_name](data_dict[metric], truncate) for metric, red_name in
                               self.metric_reduction.items()])
        return tb

    def overall_metrics(self, items: List[str] = None, truncate: int = None):

        if items is None:
            items = ['acer', 'bpcer', 'apcer']

        if truncate is None:
            truncate = self.truncate

        metrics = {}

        for ds, values in self.data.items():
            for item in items:
                metrics[f"hyparam/{ds}/{item}"] = self.fmt1of[self.metric_reduction[item]](values[item], truncate)

        return metrics
