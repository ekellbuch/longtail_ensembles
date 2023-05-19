"""
Evaluate the performance of each mode

CIFAR10: worse performance in terms of brier and ece?
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar10" --config-name="base"

CIFAR10-LT:
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="base"
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="base_bloss"
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="weighted_base"

CIFAR100lt:
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar100lt" --config-name="base"
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar100lt" --config-name="base_bloss"
python scripts/compare_model_performance.py --config-path="../configs/comparison_baseline_cifar100lt" --config-name="weighted_base"
"""
from longtail_ensembles.predictions import Model
from itertools import combinations
from omegaconf.errors import ConfigAttributeError
import pandas as pd
import hydra
import os
import numpy as np
from longtail_ensembles.utils_ens import get_model_metrics


def get_player_score(perf):
  player_perf_bins = np.linspace(0, 1, 10)
  return np.digitize(perf, player_perf_bins)


#%%
@hydra.main(config_path="../configs/comparison_baseline_cifar10", config_name="base",version_base=None)
def main(args):
  results = []

  for data_type in args.data_types:
    for sdata_type in args.data_types[data_type]:
      file_ext = args.data_types[data_type][sdata_type]['logit']
      label_ext = args.data_types[data_type][sdata_type]['label']

      for model_architecture in args.models:

        model = args.models[model_architecture]

        kwargs = {}
        try:
          kwargs["npz_flag"] = model.npz_flag
        except ConfigAttributeError:
          pass

        for i, (m, l) in enumerate(zip(model.filepaths, model.labelpaths)):
          model_name = "{}_{}".format(model_architecture, i)
          model = Model(model_name, "data")
          m = os.path.join(m, file_ext) if file_ext is not None else m
          l = os.path.join(l, label_ext) if label_ext is not None else l
          model.register(filename=m,
                         labelpath=l,
                         inputtype=None,
                         **kwargs)

          metrics = get_model_metrics(model, modelname=model_name)
          metrics['data_type'] = data_type
          metrics['sdata_type'] = sdata_type
          # metrics['architecture'] = model_architecture
          results.append(metrics)

  results = pd.concat(results)
  results["models"] = pd.Categorical(results["models"])
  results["data_type"] = pd.Categorical(results["data_type"])
  results["sdata_type"] = pd.Categorical(results["sdata_type"])
  results["train_loss"] = args.train_loss
  results["train_loss"] = pd.Categorical(results["train_loss"])

  results.set_index(["data_type", "train_loss", "sdata_type", "models"], inplace=True)
  # run some checks:
  # model before and after temperature scaling should have the same performance:
  assert (results.groupby(['data_type', 'train_loss', 'models'])['acc'].nunique() == 1).all()
  # model after temperature scaling should have better ece, brier, nll
  for metric_ in ['ece', 'nll', 'brier']:
    ece = results.groupby(['data_type', 'train_loss', 'models']).apply(lambda x: x.groupby(['sdata_type'])[metric_].mean())
    if 'base' in ece.columns and 'temperature' in ece.columns:
      assert (ece['temperature'] < ece['base']).all()


  print(results, '\n\n')
  print(results.to_latex(index=True, float_format="{:.3f}".format))  # multirow=True,multicolumn=True)

if __name__ == "__main__":
    main()
