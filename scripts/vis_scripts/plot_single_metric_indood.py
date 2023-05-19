"""
Plot the metrics for different ensembles, where the metrics are sorted according to a metric value.
Run:
  python scripts/vis_scripts/plot_ens_avg_comparison_th.py --ind_dataset=cifar10 --ood_dataset=cinic10
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

#%

#%%

def main(out_name="model_performance/het_run_v6",
         metric="0-1 Error",
         datasets="base",
         ensemble_type=None):

  if datasets == "base":
    all_datasets  = ["cifar10","cinic10"]
    all_datasets += ["imagenet", "imagenetv2mf"]
  elif datasets == "ood_cifar":
    all_datasets = ['cifar10_1']
  elif datasets =="ood_imagenet":
    all_datasets  = ["imagenet_c_gaussian_noise_{}".format(i) for i in [1,3,5]]
    all_datasets  += ["imagenet_c_fog_{}".format(i) for i in [1,3,5]]

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
    columns={'nll': 'NLL',
             'ece': 'ECE',
             'brier': 'Brier',
             'f1': 'F1',
             #'CalROCauc': 'Calib. ROC AUC'
             }, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type',
                               'data_type': 'Data Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     'no_avg': 'single model'})


  #all_subdatas = all_subdatas[all_subdatas['acc'] >= 0.25]
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

  #%%
  # Separate logit variance vs ensemble variance
  metrics_columns=["0-1 Error",  "F1", "NLL", "Brier", "CalROCauc"]  #,"EnsVar","ClsVar"]
  metrics_columns=[metric]
  #%%
  """
  # stacked_df = stacked_df[:21]
  #TODO: merge with sortby
  model_idx = stacked_df[np.logical_and(stacked_df['Metric'] == '0-1 Error', stacked_df['Data Type'] == 'InD')]
  model_idx = model_idx[model_idx['Ensemble Type'] == 'single model']
  assert len(model_idx['models'].values) == len(np.unique(model_idx['models'].values))
  metric_vals = np.argsort(model_idx['Value'].values)
  model_vals = model_idx['models'].values
  model_vals_dict = dict(zip(model_vals, metric_vals))
  stacked_df['models'] = stacked_df['models'].replace(model_vals_dict)
  """
  #%%
  stacked_df["dataset"] = stacked_df["dataset"].replace(pretty_data_names)

  # Plot only the dataset provided:
  row_order =[pretty_data_names[dataset] for dataset in all_datasets]
  #%%
  #num_models = len(model_vals)
  # stacked_df = stacked_df[stacked_df['Ensemble Type'] == 'single model']
  #stacked_df = stacked_df.sort_values(by="models").reset_index()
  #%%
  subplot_kws ={
    #'xlim': (0, num_models),
  }
  hue_kws = {
      #'marker': ['o', '+', 'x'],

  }
  stacked_df = stacked_df[stacked_df['Metric'] == "0-1 Error"]
  col_wrap= 3 if datasets == "ood_imagenet" else None
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']
  model_types = ['avg. probs', 'avg. logits']#new_plot_results['Ensemble Type'].unique()

  #%%
  new_plot_results = stacked_df
  #%%
  g = sns.FacetGrid(
    data=new_plot_results,
    col="dataset",
    col_wrap=col_wrap,
    #row="data_type",
    # hue='Ensemble Type',
    hue_order=model_types,
    margin_titles=True,
    despine=False,
    legend_out=True,
    aspect=1, height=4.,
    sharey=False,
    sharex=False,
    col_order=row_order,
    palette=colors,
  )

  def plot_data(x, y, **kwargs):
    ax = sns.scatterplot(x=x, y=y, **kwargs)


    # get the current axis
    ax = plt.gca()

    # fit a line?

    #xmin, xmax = ax.get_xlim()
    #ymin, ymax = ax.get_ylim()
    #ax.axline([xmin, xmax], [ymin, ymax], color='k', linestyle='--', linewidth=1)  # , alpha=0.75, linewidth=0.75)


  g.map_dataframe(plot_data, x=model_types[0], y=model_types[1])
  g.set_titles(col_template="{col_name}", row_template="{row_name}")
  g.tight_layout()

  for ii, g_ax in enumerate(g.axes.flat):
    all_total = new_plot_results.loc[(new_plot_results['dataset'] == row_order[ii])]
    x = all_total[model_types[0]].values
    y = all_total[model_types[1]].values
    y_pred_total, params_total, sd_b_total, ts_b_total, p_values_total = linear_fit(x, y)
    r2_val = r2_score(y, y_pred_total)

    #if r2_val > 0.1:
    #  pass
    #else:
    g_ax.plot(x, y_pred_total, linewidth=1, c='k', linestyle='--', alpha=0.5)

    # Get the x and y limits of the plot
    x_min, x_max = g_ax.get_xlim()
    y_min, y_max = g_ax.get_ylim()

    # Calculate the coordinates for the top-left corner (adjust as needed)
    x_coord = x_min + 0.05 * (x_max - x_min)
    y_coord = y_max - 0.09 * (y_max - y_min)

    g_ax.text(x_coord, y_coord, 'r2 = {:.3f}'.format(r2_val))
  g.add_legend(handletextpad=0)
  g.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')
  # plt.show()
  g.savefig(output_filename, bbox_inches='tight')

  #%%
  plt.close()

  print('Done', flush=True)
  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
