import wandb
from prettytable import PrettyTable
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--net", "-net", type=str, default="efficientnet_b0")
    parser.add_argument("--check", "-check", action='store_true')
    args = parser.parse_args()
    return args


args = parse_args()

ds = ['IIIT_WVU', 'NotreDame', 'Clarkson']
run_result = {key1: {key: dict(acer=100, apcer=1, bpcer=1, run=None, batch_size=0, lambda_NCE=0)
                     for key in ds}
              for key1 in ds}
api = wandb.Api(timeout=90)

bs_list = [64, 128, 256, 512]
lam_list = [0.25, 0.5, 0.75, 1.0, 2.0]

check_board = {d: {key1: {key2: False for key2 in lam_list} for key1 in bs_list} for d in ds}

for train in ds:
    for run in api.runs(f"{train}"):
        batch_size = run.config['CLASSIFIER']['equiv_batch_size']
        lambda_NCE = run.config['CUT']['lambda_NCE']
        if run.config['CLASSIFIER']['model']['type'].replace('torchvision.models.', "") == args.net:
            summary = run.summary
            for test in ds:
                if summary[f"acer/{test}/TEST"]["min"] * 100 < run_result[train][test]['acer']:
                    run_result[train][test]['acer'] = summary[f"acer/{test}/TEST"]["min"] * 100
                    run_result[train][test]['run'] = run
                    run_result[train][test]['batch_size'] = batch_size
                    run_result[train][test]['lambda_NCE'] = lambda_NCE
            if args.check:
                if int(batch_size) in bs_list and lambda_NCE in lam_list:
                    check_board[train][batch_size][lambda_NCE] = True

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

table = PrettyTable()
table.field_names = ["Test / Train", "Hyparam"] + ds
for i, test in enumerate(ds):
    for j, metric in enumerate(['batch_size', "", 'lambda_NCE']):
        if j == 1:
            table.add_row([test, metric.upper()] + [" " for train in ds])
        else:
            table.add_row([" ", metric.upper()] + ['%.2f' % run_result[train][test][metric] for train in ds])
    if i + 1 < len(ds):
        table.add_row(["-" * 5 for i in range(5)])

print(table)

if args.check:
    for train in ds:
        cnt = 0
        print(train)
        print("-" * 30)
        for lambda_NCE in lam_list:
            for batch_size in bs_list:
                if not check_board[train][batch_size][lambda_NCE]:
                    print(lambda_NCE, batch_size)
                    cnt = cnt + 1
        print("Total: ", cnt)
        print("=*" * 15 + "\n")
