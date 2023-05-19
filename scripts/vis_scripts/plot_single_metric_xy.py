"""
Plot the metrics for different ensembles, where the metrics are sorted according to a metric value.
Run:
  python scripts/vis_scripts/plot_single_metric_xy.py --ind_dataset=cifar10 --ood_dataset=cinic10
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import fire
import seaborn as sns
from sklearn.metrics import r2_score

from longtail_ensembles.utils import read_dataset, pretty_data_names,linear_fit
from matplotlib.ticker import MaxNLocator

import itertools
BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%%
metrics_pretty_names = {
  'error': "0-1 Error",
  'nll': 'NLL',
  'ece': 'ECE',
  'brier': 'Brier',
  'f1': "F1",
  "CalPRauc": "Cal. PR auc",
  "CalROCauc": "Cal. ROC auc",
  "EnsVar": "Var",
  "ClsVar": "Class Var",
}
def main(out_name="model_performance/het_run_v6",
         metric="0-1 Error",
         datasets="base",
         ensemble_type=None):

  metric = metrics_pretty_names[metric]
  if datasets == "base":
    all_datasets = ["cifar10", "cinic10"]
    all_datasets += ["imagenet", "imagenetv2mf"]
  elif datasets == "ood_cifar":
    all_datasets = ['cifar10_1']
  elif datasets =="ood_imagenet":
    all_datasets = ["imagenet_c_gaussian_noise_{}".format(i) for i in [1,3,5]]
    all_datasets += ["imagenet_c_fog_{}".format(i) for i in [1,3,5]]

  all_datas = []
  #%% read all datasets
  for dataset in all_datasets:
    data_filename = output_dir / out_name / "{}.csv".format(dataset)
    datas = read_dataset(data_filename)
    all_datas.append(datas)
  all_datas = pd.concat(all_datas, axis=0)

  output_filename = output_dir / "figures" / out_name / ("ens_th_xy_{}_{}.pdf".format(datasets,metric))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Storing output in:\n{}'.format(output_filename), flush=True)

  all_subdatas=all_datas.copy()
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1

  # preprocess model
  all_subdatas['0-1 Error'] = 1 - all_subdatas['acc']
  all_subdatas.rename(
    columns =metrics_pretty_names, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type',
                               'data_type': 'Data Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'Logit ensembles',
     'avg_probs': 'Prob. ensembles',
     'no_avg': 'single model'})

  all_subdatas = all_subdatas[~all_subdatas['models'].str.contains('deepens')]
  plot_results = all_subdatas.copy()
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['dataset'] = pd.Categorical(plot_results['dataset'])

  #%%
  plot_results = plot_results.drop(columns=['model_scores',
                                            'ensemble_size',
                                            'ensemble_type',
                                            'architecture',
                                            #'dataset',
                                            "Data Type",
                                            'seed',
                                            'num_params',
                                            'binning','acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  #%

  plot_results = plot_results.sort_values(by="0-1 Error")

  plot_results.set_index(['models', 'dataset', 'Ensemble Type'], inplace=True)
  plot_results.columns.name = "Metric"
  plot_results = plot_results.unstack(['Ensemble Type']).stack('Metric')

  new_plot_results = plot_results.reset_index().set_index(["models"])

  stacked_df = new_plot_results.dropna()
  #%
  stacked_df["dataset"] = stacked_df["dataset"].replace(pretty_data_names)

  # Plot only the dataset provided:
  row_order =[pretty_data_names[dataset] for dataset in all_datasets]
  stacked_df = stacked_df[stacked_df['Metric'] == metric]
  col_wrap= 3 if datasets == "ood_imagenet" else None
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']
  model_types = ['Prob. ensembles', 'Logit ensembles'] #new_plot_results['Ensemble Type'].unique()

  aspect = 1
  new_plot_results = stacked_df
  #%
  g = sns.FacetGrid(
    data=new_plot_results,
    col="dataset",
    col_wrap=col_wrap,
    hue_order=model_types,
    margin_titles=True,
    despine=False,
    legend_out=False,
    aspect=aspect,
    height=4,
    sharey=False,
    sharex=False,
    col_order=row_order,
    palette=colors,
  )

  def plot_data(x, y, **kwargs):
    ax = sns.scatterplot(x=x, y=y, **kwargs)
    # get the current axis
    ax = plt.gca()

  g.map_dataframe(plot_data, x=model_types[0], y=model_types[1])
  g.set_titles(col_template="{col_name}", row_template="{row_name}")
  g.tight_layout()
  textbox_props = dict(boxstyle='round', facecolor='white', alpha=0.5)

  all_axes = g.axes.flat
  for ii, g_ax in enumerate(all_axes):
    all_total = new_plot_results.loc[(new_plot_results['dataset'] == row_order[ii])]
    x = all_total[model_types[0]].values
    y = all_total[model_types[1]].values
    y_pred_total, params_total, sd_b_total, ts_b_total, p_values_total = linear_fit(x, y)
    r2_val = r2_score(y, x)
    mse_val = np.mean((y - x)**2)
    g_ax.plot(x, x, linewidth=1, c='k', linestyle='--', alpha=0.5)
    g_ax.plot(x, y_pred_total, linewidth=1, c='r', linestyle='-', alpha=0.5)

    # Get the x and y limits of the plot
    x_min, x_max = g_ax.get_xlim()
    y_min, y_max = g_ax.get_ylim()

    # Calculate the coordinates for the top-left corner (adjust as needed)
    x_coord = x_min + 0.05 * (x_max - x_min)
    y_coord = y_max - 0.09 * (y_max - y_min)

    txt_str = "MSE = {:.3e}".format(mse_val)
    g_ax.text(x_coord, y_coord, txt_str, bbox=textbox_props, verticalalignment='top')

    # Todo update for multiple datasets
    if (ii == 0):
      g_ax.set_ylabel("{} \n Logit ensembles".format(metric))
    if not(col_wrap is None) and ii % col_wrap == 0:
      g_ax.set_ylabel("{} \n Logit ensembles".format(metric))

  g.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')

  #%%
  plt.close()

  print('Done', flush=True)
  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
