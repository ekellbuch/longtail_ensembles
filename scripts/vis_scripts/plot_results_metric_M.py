"""
Make plot ensemble_size vs metric:
python scripts/vis_scripts/plot_results_metric_M.py --config-path="../../configs/comparison_baseline_cifar10lt" --config-name="compare_M"

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
np.random.seed = 42
from longtail_ensembles.utils_ens import build_ensemble, get_model_metrics,get_all_ensemble_pairs

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
output_dir = BASE_DIR / "results"
sys.path.append(str(BASE_DIR / "scripts"))

#%%
plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))
output_dir = BASE_DIR / "results"


@hydra.main(config_path="../../configs/comparison_baseline_cifar10lt", config_name="compare_M", version_base=None)
def main(args):
  plot_ensemble_size_vs_metric(args)

#%%
def plot_ensemble_size_vs_metric(args):
  all_results = []

  ensemble_sizes = range(2, 15)
  max_num_ensembles_per_M = 10

  for ensemble_size in ensemble_sizes:
    for loss_type in args.train_loss:
      print(f"Loss type: {loss_type}", flush=True)
      config_file = args.config_path + "/" + loss_type + ".yaml"
      loss_args = yaml.load(open(str(config_file)), Loader=yaml.FullLoader)
      loss_args = OmegaConf.create(loss_args)

      # process experiment logits:
      results = []
      for data_type in loss_args.data_types:
        for model_architecture in loss_args.models:
          model = loss_args.models[model_architecture]
          all_new_model_pairs = get_all_ensemble_pairs(model, ensemble_size=ensemble_size)
          if len(all_new_model_pairs) > max_num_ensembles_per_M:
            all_new_model_pairs = np.random.choice(all_new_model_pairs, size=max_num_ensembles_per_M, replace=False)
          for ensemble_pair_idx, ensemble_pair in enumerate(all_new_model_pairs):
            for sdata_type in loss_args.data_types[data_type]:
              file_ext = loss_args.data_types[data_type][sdata_type]['logit']
              label_ext = loss_args.data_types[data_type][sdata_type]['label']
              for ensemble_type_ in loss_args.ensemble_types:
                ens = build_ensemble(model=ensemble_pair,
                                     ensemble_size=ensemble_size,
                                     ensemble_method=ensemble_type_,
                                     file_ext=file_ext,
                                     label_ext=label_ext,
                                     )
                metrics = get_model_metrics(model=ens, modelname=ensemble_type_)
                metrics['data_type'] = data_type
                metrics['sdata_type'] = sdata_type
                metrics['architecture'] = model_architecture
                metrics['seed'] = ensemble_pair_idx
                metrics['ensemble_size'] = ensemble_size
                metrics['train_loss'] = loss_type
                results.append(metrics)
      results = pd.concat(results)
      all_results.append(results)


  all_results= pd.concat(all_results)
  all_results = all_results.reset_index()

  all_results.rename(columns={'acc': 'Acc',
                              'f1':'F1',
                              "var":'Var',
                              "cv":"CV",
                              "class_var":"Class Var",
                              "avg_disagreement": "Avg. Disagreement",
                              "brier": "Brier"}, inplace=True)
  all_results.sort_values('Acc', ascending=False, inplace=True)

  data_types = args.data_types
  sdata_types = args.sdata_types
  architectures = args.models

  for data_type, sdata_type, architecture in itertools.product(data_types, sdata_types, architectures):
    print(f'Plotting {data_type}, {sdata_type}, {architecture}', flush=True)

    results = all_results[all_results['data_type'] == data_type]
    results = results[results['sdata_type'] == sdata_type]
    results = results[results['architecture'] == architecture]


    ens_size_pretty= 'Ensemble Size'
    #%%
    results.rename(columns={'models': 'Ensemble Type',
                            'train_loss': 'Loss',
                            'ensemble_size': ens_size_pretty,
                                }, inplace=True)

    results['Ensemble Type'] = results['Ensemble Type'].replace({'avg_logits': 'avg. logits',
                                                                         'avg_probs': 'avg. probs',
                                                                         'no_avg': 'single model'})

    results['Loss'] = results['Loss'].replace({'base_M': 'ERM',
                                                       'base_bloss_M': 'Balanced Softmax CE',
                                                       #'weighted_ce': 'Weighted Softmax CE',
                                                       #'weighted_softmax': 'd-Weighted Softmax CE',
                                                       }
                                                      )

    # drop columns not interesting
    results2 = results[['Ensemble Type', 'Loss', 'seed','Acc','F1',"Brier"] + [ens_size_pretty]].copy()
    extra_kwargs = {}

    colors = ['yellowgreen', 'tomato', 'dodgerblue']
    extra_kwargs['palette'] = colors
    extra_kwargs['hue_order'] = ['single model', 'avg. logits', 'avg. probs']

    results2 = pd.melt(results2, id_vars=['Ensemble Type','seed','Loss', 'Ensemble Size'], value_vars=['Acc', 'F1','Brier'], var_name='Metric', value_name='Value')

    facet_kws = {'sharey': False, 'sharex': True}

    g = sns.relplot(data=results2,
                    x=ens_size_pretty,
                    col="Metric",
                    y="Value",
                    hue='Ensemble Type',
                    style="Loss",
                    #size=10,
                    #alpha=1,
                    kind='line',
                    facet_kws=facet_kws,
                    **extra_kwargs,
                    )

    g.set_ylabels('')

    # Improve the legend
    if g.legend is not None:
      g.legend.handletextpad = 0
      g.legend.borderaxespad = 0
      g.legend.borderpadfloat = 0

    for ax in g.axes.flat:
      ax.set_xticks(ensemble_sizes[::3])


    legend_kwargs = {
      'borderaxespad': 0,
      'handletextpad': 0,
      'loc': 'center left',
      'bbox_to_anchor': (1.0, 0.5),
      'markerscale': 0.5,  # relative size of legend marker compared to originally drawn ones.
      'frameon': True,
    }
    g.add_legend(**legend_kwargs)

    g.tight_layout()
    #%
    output_fname = os.path.join(output_dir / "figures" / "compare_M" / f'{args.title}_{data_type}_{architecture}.pdf')
    print(output_fname, flush=True)
    if not os.path.exists(os.path.dirname(output_fname)):
      os.makedirs(os.path.dirname(output_fname))
    g.savefig(output_fname)
    plt.close()



if __name__ == "__main__":
    main()
#%%