"""

python scripts/vis_scripts/plot_results_dkl_diff.py --config-path="../../results/configs/comparison_baseline_cifar10lt" --config-name="default"

"""
import hydra
import os
from omegaconf import ListConfig, OmegaConf, open_dict
from pathlib import Path
import yaml
import sys
import copy
from tqdm import tqdm
import fire
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import numpy as np
from longtail_ensembles.utils_ens import build_ensemble, get_model_metrics,get_all_ensemble_pairs

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
output_dir = BASE_DIR / "results"
plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))


@hydra.main(config_path="../../results/configs/comparison_baseline_cifar10lt", config_name="default", version_base=None)
def main(args):
  plot_p_class_box(args)

#%%
def plot_p_class_box(args):
  all_results = []
  for loss_type in args.train_loss:
    print(f"Loss type: {loss_type}", flush=True)
    config_file = args.config_path + "/" + loss_type + ".yaml"
    loss_args = yaml.load(open(str(config_file)), Loader=yaml.FullLoader)
    loss_args = OmegaConf.create(loss_args)

    # process experiment logits:
    results = []
    for ensemble_type_ in loss_args.ensemble_types:
      for data_type in loss_args.data_types:
        for sdata_type in loss_args.data_types[data_type]:
          file_ext = loss_args.data_types[data_type][sdata_type]['logit']
          label_ext = loss_args.data_types[data_type][sdata_type]['label']
          for model_architecture in loss_args.models:
            model = loss_args.models[model_architecture]

            all_new_model_pairs = get_all_ensemble_pairs(model)
            for ensemble_pair_idx, ensemble_pair in enumerate(all_new_model_pairs):
              ens = build_ensemble(model=ensemble_pair,
                                   ensemble_method=ensemble_type_,
                                   file_ext=file_ext,
                                   label_ext=label_ext,
                                   )
              dkl_qbar_qi = ens.get_dkl_qs(per_class=True, metric='dkl_qbar_qi')
              dkl_uni_eq = ens.get_dkl_qs(per_class=True, metric='dkl_uni_eq')
              dkl_qdag_qi = ens.get_dkl_qs(per_class=True, metric='dkl_qdag_qi')
              acc = ens.get_accuracy_per_class()
              metrics0 = pd.DataFrame.from_dict(acc, orient='index', columns=['acc'])
              metrics1 = pd.DataFrame.from_dict(dkl_qbar_qi, orient='index', columns=['dkl_qbar_qi'])
              metrics2 = pd.DataFrame.from_dict(dkl_uni_eq, orient='index', columns=['dkl_uni_eq'])
              metrics3 = pd.DataFrame.from_dict(dkl_qdag_qi, orient='index', columns=['dkl_qdag_qi'])
              metrics = pd.concat([metrics0,metrics1, metrics2, metrics3], axis=1)
              metrics['Class ID'] = metrics.index
              metrics.reset_index(drop=True, inplace=True)
              metrics['models'] = ensemble_type_
              metrics['data_type'] = data_type
              metrics['sdata_type'] = sdata_type
              metrics['architecture'] = model_architecture
              metrics['seed'] = ensemble_pair_idx
              results.append(metrics)

    results = pd.concat(results)
    results["models"] = pd.Categorical(results["models"])
    results["data_type"] = pd.Categorical(results["data_type"])
    results["sdata_type"] = pd.Categorical(results["sdata_type"])
    results["train_loss"] = loss_type
    results["train_loss"] = pd.Categorical(results["train_loss"])
    results["seed"] = pd.Categorical(results["seed"])
    results["architecture"] = pd.Categorical(results["architecture"])
    results.set_index(["data_type", "train_loss", "sdata_type", "models", "architecture"], inplace=True)

    all_results.append(results)

  all_results= pd.concat(all_results)
  all_results = all_results.reset_index()

  all_results.rename(columns={'acc': 'Acc',
                              'f1':'F1',
                              "var":'Var',
                              "cv":"CV",
                              "class_var":"Class Var",
                              "avg_disagreement": "Avg. Disagreement"}, inplace=True)
  all_results.sort_values('Acc', ascending=False, inplace=True)

  data_types = args.data_types
  sdata_types = args.sdata_types
  architectures = args.models

  for data_type, sdata_type, architecture in itertools.product(data_types, sdata_types, architectures):
    print(f'Plotting {data_type}, {sdata_type}, {architecture}', flush=True)

    results = all_results[all_results['data_type'] == data_type]
    results = results[results['sdata_type'] == sdata_type]
    results = results[results['architecture'] == architecture]

    #%%
    results.rename(columns={'models': 'Ensemble Type',
                                'train_loss': 'Loss',
                                }, inplace=True)

    results['Ensemble Type'] = results['Ensemble Type'].replace({'avg_logits': 'avg. logits',
                                                                         'avg_probs': 'avg. probs',
                                                                         'no_avg': 'single model'})

    results['Loss'] = results['Loss'].replace({'base': 'ERM',
                                                       'base_bloss': 'Balanced Softmax CE',
                                                       'weighted_ce': 'Weighted Softmax CE',
                                                       'weighted_softmax': 'd-Weighted Softmax CE',
                                                       }
                                                      )

    # TODO: add error bar:
    if 'seed' in results.columns:
      results.set_index(['data_type', 'Loss','sdata_type', 'Ensemble Type', 'architecture','Class ID'], inplace=True)
      results = results.groupby(results.index.names).agg([('mean', 'mean'), ('std', 'std')])
      w_ci = True
    else:
      w_ci = False

    if 'CIFAR10lt' in args.title:
      extra_kwargs ={
        'col': 'Loss',
      }
      extra_kwargs['col_order'] = ['ERM', 'Balanced Softmax CE', 'Weighted Softmax CE', 'd-Weighted Softmax CE']
    else:
      extra_kwargs = {
        'row': 'Loss',
        'aspect': 5,
      }
      extra_kwargs['row_order'] = ['ERM', 'Balanced Softmax CE', 'Weighted Softmax CE', 'd-Weighted Softmax CE']

    colors = ['yellowgreen', 'tomato', 'dodgerblue']#, 'violet']
    extra_kwargs['palette'] = colors

    # plot two metrics
    results.columns = results.columns.swaplevel(0, 1)
    # plot only mean values
    data_ = results['mean']
    data_std = results['std']

    data_.columns.name = 'Metric'
    data_ = data_.stack().reset_index()
    data_.columns = ['data_type', 'Loss', 'sdata_type', 'Ensemble Type', 'architecture', 'Class ID', 'Metric', 'Value']

    # this should be the same regardless of the ensemble type
    # pick one ensemble type
    data_ = data_[data_['Ensemble Type'] == 'single model']
    data_ = data_[~(data_['Metric'] == 'Acc')]
    data_['Metric'] = data_['Metric'].replace({
      'dkl_qbar_qi': 'diversity', #'diversity_log',
      'dkl_uni_eq': 'dependency',
      'dkl_qdag_qi': 'diversity_prob'
    })
    extra_kwargs['palette'] = ["#4CC9F0","#F72585", "#7209B7"]
    extra_kwargs['palette'] = ["#F72585", "#7209B7"]

    extra_kwargs['hue_order'] = ['diversity', 'dependency']
    #
    g = sns.catplot(data=data_,
                    x='Class ID',
                    y='Value',
                    hue='Metric',
                    s=10,
                    alpha=0.7,
                    **extra_kwargs,
                    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Improve the legend
    g.legend.handletextpad = 0
    g.legend.borderaxespad = 0
    g.legend.borderpadfloat = 0

    if not 'CIFAR10lt' in args.title:
      # if cifar100
      for ax in g.axes.flat:
        labels = ax.get_xticklabels()  # get x labels
        for index, label in enumerate(labels):
          if index % 10 == 0:
            label.set_visible(True)
          else:
            label.set_visible(False)


    g.tight_layout()

    #%
    output_fname = os.path.join(output_dir / "figures" / "per_class_metrics" / f'{args.title}_{data_type}_{sdata_type}_{architecture}_dkl_diff.pdf')
    print(output_fname, flush=True)
    if not os.path.exists(os.path.dirname(output_fname)):
      os.makedirs(os.path.dirname(output_fname))
    g.savefig(output_fname)
    plt.close()


if __name__ == "__main__":
    main()
#%%