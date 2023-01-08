import wandb
from prettytable import PrettyTable

ds = ['IIIT_WVU', 'NotreDame', 'Clarkson']
run_result = {key1: {key: dict(acer=100, apcer=1, bpcer=1, run=None) for key in ds} for key1 in ds}
api = wandb.Api(timeout=90)

for train in ds:
    for run in api.runs(f"{train}"):
        summary = run.summary
        for test in ds:
            if summary[f"acer/{test}/TEST"]["min"] * 100 < run_result[train][test]['acer']:
                run_result[train][test]['acer'] = summary[f"acer/{test}/TEST"]["min"] * 100
                run_result[train][test]['run'] = run

for train in ds:
    for test in ds:
        history = run_result[train][test]['run'].scan_history(
            keys=[f"apcer/{test}/TEST", f"bpcer/{test}/TEST", f"acer/{test}/TEST"])
        acer = [x[f"acer/{test}/TEST"] for x in history]
        apcer = [x[f"apcer/{test}/TEST"] for x in history]
        bpcer = [x[f"bpcer/{test}/TEST"] for x in history]
        index = [i for i in range(len(acer)) if abs(acer[i] * 100 - run_result[train][test]['acer']) < 1.0e-8][0]
        run_result[train][test]['apcer'] = apcer[index] * 100
        run_result[train][test]['bpcer'] = bpcer[index] * 100

table = PrettyTable()
table.field_names = ["Test / Train", "Metric"] + ds
for i, test in enumerate(ds):
    for j, metric in enumerate(['apcer', 'bpcer', 'acer']):
        if j == 1:
            table.add_row([test, metric.upper()] + ['%.2f' % run_result[train][test][metric] for train in ds])
        else:
            table.add_row([" ", metric.upper()] + ['%.2f' % run_result[train][test][metric] for train in ds])
    if i + 1 < len(ds):
        table.add_row(["-" * 5 for i in range(5)])

print(table)
