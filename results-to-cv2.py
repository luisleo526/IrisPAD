from collections import defaultdict

import wandb

api = wandb.Api(timeout=90)
ds = ['IIIT_WVU', 'NotreDame', 'Clarkson']

params = {d: {dd: {'bs': 0, 'lambda': 0.0} for dd in ds} for d in ds}

params['IIIT_WVU']['IIIT_WVU']  = {'bs': 64 , 'lambda': 0.25}
params['IIIT_WVU']['NotreDame'] = {'bs': 128, 'lambda': 0.75}
params['IIIT_WVU']['Clarkson']  = {'bs': 512, 'lambda': 1.0}

params['NotreDame']['IIIT_WVU']  = {'bs': 512, 'lambda': 0.50}
params['NotreDame']['NotreDame'] = {'bs': 128, 'lambda': 0.25}
params['NotreDame']['Clarkson']  = {'bs': 512, 'lambda': 0.25}

params['Clarkson']['IIIT_WVU']  = {'bs': 256, 'lambda': 2.0}
params['Clarkson']['NotreDame'] = {'bs': 256, 'lambda': 0.25}
params['Clarkson']['Clarkson']  = {'bs': 128, 'lambda': 2.0}

runs = {d: defaultdict(list) for d in ds}

for train in ds:
    for run in api.runs(f"{train}"):
        runs[train][run.config['CLASSIFIER']['model']['type'].replace('torchvision.models.', "")].append(run)

models = sorted(list(set(runs['NotreDame'].keys()).union(set(runs['IIIT_WVU'].keys()), set(runs['Clarkson'].keys()))))

results = {model: {train: {test: None for test in ds} for train in ds} for model in models}

for train in ds:
    for model in runs[train]:
        for run in runs[train][model]:
            batch_size = run.config['CLASSIFIER']['equiv_batch_size']
            lambda_nce = run.config['CUT']['lambda_NCE']
            for test, args in params[train].items():
                if args['bs'] == batch_size and args['lambda'] == lambda_nce:
                    if results[model][train][test] is None:
                        results[model][train][test] = run.summary[f"acer/{test}/TEST"]['min'] * 100
                    else:
                        results[model][train][test] = min(results[model][train][test],
                                                          run.summary[f"acer/{test}/TEST"]['min'] * 100)

for model in models:
    for train in ds:
        for test in ds:
            if results[model][train][test] is not None:
                results[model][train][test] = "{:.02f}".format(results[model][train][test])
            else:
                results[model][train][test] = "-"

model_name = 'PBS'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['IIIT_WVU']['NotreDame'] = 16.86
results[model_name]['IIIT_WVU']['Clarkson'] = 47.17
results[model_name]['NotreDame']['IIIT_WVU'] = 17.49
results[model_name]['NotreDame']['Clarkson'] = 45.31
results[model_name]['Clarkson']['IIIT_WVU'] = 42.28
results[model_name]['Clarkson']['NotreDame'] = 32.42
models.append(model_name)

model_name = 'A-PBS'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['IIIT_WVU']['NotreDame'] = 27.61
results[model_name]['IIIT_WVU']['Clarkson'] = 21.99
results[model_name]['NotreDame']['IIIT_WVU'] = 9.49
results[model_name]['NotreDame']['Clarkson'] = 22.46
results[model_name]['Clarkson']['IIIT_WVU'] = 34.17
results[model_name]['Clarkson']['NotreDame'] = 23.08
models.append(model_name)

model_name = 'FAM+FMM'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['IIIT_WVU']['NotreDame'] = 5.81
results[model_name]['IIIT_WVU']['Clarkson'] = 26.03
results[model_name]['NotreDame']['IIIT_WVU'] = 15.07
results[model_name]['NotreDame']['Clarkson'] = 10.51
results[model_name]['Clarkson']['IIIT_WVU'] = 22.06
results[model_name]['Clarkson']['NotreDame'] = 20.92
models.append(model_name)

model_name = 'CASIA'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 7.10
results[model_name]['NotreDame']['NotreDame'] = 4.03
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 16.70
models.append(model_name)

model_name = 'SpoofNet'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 16.50
results[model_name]['NotreDame']['NotreDame'] = 9.50
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 18.62
models.append(model_name)

model_name = 'D-NetPAD'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 3.36
results[model_name]['NotreDame']['NotreDame'] = 6.81
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 23.27
models.append(model_name)

model_name = 'MSA'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['NotreDame']['NotreDame'] = 6.23
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 11.13
models.append(model_name)

model_name = 'PBS'
# results[model_name] = {train:{test:"-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 4.48
results[model_name]['NotreDame']['NotreDame'] = 4.97
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 7.01
# models.append(model_name)

model_name = 'A-PBS'
# results[model_name] = {train:{test:"-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 3.48
results[model_name]['NotreDame']['NotreDame'] = 3.94
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 6.50
# models.append(model_name)

model_name = 'FAM'
results[model_name] = {train: {test: "-" for test in ds} for train in ds}
results[model_name]['Clarkson']['Clarkson'] = 3.45
results[model_name]['NotreDame']['NotreDame'] = 4.03
results[model_name]['IIIT_WVU']['IIIT_WVU'] = 6.84
models.append(model_name)

with open("LivDet2017-results.csv", "w") as f:
    f.write("Train,IIIT_WVU,,,NotreDame,,,Clarkson,,,\n")
    f.write("Test,IIIT_WVU,NotreDame,Clarkson,IIIT_WVU,NotreDame,Clarkson,IIIT_WVU,NotreDame,Clarkson\n")
    for model in models:
        msg = f"{model},"
        for train in ds:
            for test in ds:
                msg += f"{results[model][train][test]},"
        f.write(msg + "\n")
