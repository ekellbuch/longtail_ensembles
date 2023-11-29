"""
Plot results for each class.
python scripts/vis_scripts/plot_results_pclass.py --config-path="../../results/configs/comparison_baseline_cifar10lt" --config-name="default"
python scripts/vis_scripts/plot_results_pclass.py --config-path="../../results/configs/comparison_baseline_cifar100lt" --config-name="default"
Plots results comparing performance for each class.
"""
import hydra
import os
from omegaconf import ListConfig, OmegaConf, open_dict
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
from longtail_ensembles.utils_print import process_experiment_logits_pclass
plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))
output_dir = BASE_DIR / "results"


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
    results = process_experiment_logits_pclass(loss_args)
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
    extra_kwargs['hue_order'] = ['single model', 'avg. logits', 'avg. probs']

    #for metric in ["Acc", "F1", "Class Var", "Avg. Disagreement", "Var"]:
    for metric in ["Avg. Disagreement"]:
      if w_ci:
        data_ = results[metric]
        data_ = data_.rename(columns={'mean': metric})
        y = metric
        #extra_kwargs['yerr'] = data_['std']
        #extra_kwargs['errorbar'] = 'black'
      else:
        data_ = results
        y = metric
      data_ = data_.reset_index()

      if metric == "Var":
        # do not want to plot logit and prob variance, just focus on prob variance
        extra_kwargs['hue_order'] = ['avg. probs']
        extra_kwargs['palette'] = ['dodgerblue']
      elif metric == "Avg. Disagreement":
        extra_kwargs['hue_order'] = ['single model']
        extra_kwargs['palette'] = ['dodgerblue']
        extra_kwargs['legend'] = False

      g = sns.catplot(data=data_,
                      x='Class ID',
                      y=y,
                      hue='Ensemble Type',
                      s=10,
                      alpha=0.7,
                      **extra_kwargs,
                      #kind='bar'
                      )

      def errplot(x, y, yerr, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        data.plot(x=x, y=y, yerr=yerr, kind="scatter", ax=ax, **kwargs)

      g.set_titles(col_template="{col_name}", row_template="{row_name}")

      # Improve the legend
      if g.legend is not None:
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
      output_fname = os.path.join(output_dir / "figures" / "per_class_metrics" / f'{args.title}_{data_type}_{sdata_type}_{architecture}_{metric}.pdf')
      print(output_fname, flush=True)
      if not os.path.exists(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))
      g.savefig(output_fname)
      plt.close()


if __name__ == "__main__":
    main()
#%%