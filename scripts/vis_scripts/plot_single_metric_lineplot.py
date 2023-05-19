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
from longtail_ensembles.utils import read_dataset, pretty_data_names
from matplotlib.ticker import MaxNLocator

import itertools
BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"


#%%

def main(out_name="model_performance/het_run_v6",
         metric="0-1 Error",
         datasets="base",
         ensemble_type=None):

  if metric == "error":
    metric = "0-1 Error"

  if datasets == "base":
    all_datasets = ["cifar10", "cifar10_1"]
    all_datasets += ["imagenet", "imagenetv2mf"]
  elif datasets == "ood_cifar":
    all_datasets = ['cinic10']
  elif datasets == "ood_imagenet":
    all_datasets = ["imagenet_c_gaussian_noise_{}".format(i) for i in [1,3,5]]
    all_datasets += ["imagenet_c_fog_{}".format(i) for i in [1,3,5]]

  all_datas = []
  #%% read all datasets
  for dataset in all_datasets:
    data_filename = output_dir / out_name / "{}.csv".format(dataset)
    datas = read_dataset(data_filename)
    all_datas.append(datas)
  all_datas = pd.concat(all_datas, axis=0)

  #output_filename = output_dir / "figures" / out_name / ("ens_th_{}.pdf".format(ind_dataset, ood_dataset))
  output_filename = output_dir / "figures" / out_name / ("ens_th_{}_{}.pdf".format(datasets,metric))

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
  plot_results["Data Type"] = plot_results["Data Type"].replace({"ind": "InD", "ood": "OOD"})
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

  stacked_df = plot_results.set_index(['models','dataset','Ensemble Type']).stack().reset_index()
  stacked_df.columns = ['models','dataset','Ensemble Type','Metric','Value']

  model_types =['single model', 'avg. logits', 'avg. probs']
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']

  stacked_df = stacked_df.dropna()

  #%%
  # Separate logit variance vs ensemble variance
  metrics_columns=["0-1 Error",  "F1", "NLL", "Brier", "CalROCauc"]  #,"EnsVar","ClsVar"]
  metrics_columns=[metric]
  #%%
  """
  # sort according to dataset:
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
  # import pdb; pdb.set_trace()
  col_order =[pretty_data_names[dataset] for dataset in all_datasets]
  #%%
  #num_models = len(model_vals)
  # stacked_df = stacked_df[stacked_df['Ensemble Type'] == 'single model']
  #stacked_df = stacked_df.sort_values(by="models").reset_index()
  #%%
  linestyle_mapping = {'single model': '--',
                       'avg. logits': '-',
                       'avg. probs': '-',
                       }

  subplot_kws ={
    #'xlim': (0, num_models),

  }
  hue_kws = {
      #'marker': ['o', '+', 'x'],

  }
  stacked_df = stacked_df[stacked_df['Metric'] == metric]
  col_wrap = 3 if datasets =="ood_imagenet" else None
  #%%
  g = sns.FacetGrid(
    stacked_df,
    col_wrap=col_wrap,
    #row="Metric",
    col="dataset",
    hue='Ensemble Type',
    sharey='col',
    sharex=False,
    #aspect=2.5,
    height=3,
    palette=colors,
    row_order=metrics_columns,
    col_order=col_order,
    hue_order=model_types,
    legend_out=True,
    despine=False,
    margin_titles=True,
    subplot_kws=subplot_kws,
    hue_kws=hue_kws,

  )


  other_kwargs= {
    #'s': 5,
    'alpha': 0.7,
    #'edgewidth': [5, 5, 5]
    #'facecolor': [None, None, None]
    #'linestyle': ["--","-","-"]
    #'linestyle': lambda label: linestyle_mapping[label]

  }

  def plot_data(x, y, **kwargs):
    linestyle = linestyle_mapping[kwargs['label']]
    #import pdb; pdb.set_trace()
    sns.lineplot(x=x, y=y, linestyle=linestyle, **kwargs)

    # get the current axis
    ax = plt.gca()
    #xmin, xmax = ax.get_xlim()
    #ymin, ymax = ax.get_ylim()
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    # Set MaxNLocator for y-axis
    #y_locator = MaxNLocator(nbins=3)  # Customize the number of bins as needed
    #ax.yaxis.set_major_locator(y_locator)
    #x_locator = MaxNLocator(integer=True, nbins=2)  # Customize the number of bins as needed
    #ax.xaxis.set_major_locator(x_locator)
    #ax.set_xlim(0, num_models)
    # statistically the same?

  g.map_dataframe(plot_data, y="Value", x="models", **other_kwargs)
  g.set_xticklabels([])
  g.set_xlabels('')
  g.set_ylabels('')
  #g.set_xlim(0, num_models)
  #g.set_axis_labels(x_var='models')
  g.set_titles(col_template="{col_name}", row_template="{row_name}")
  #plt.subplots_adjust(right=0.85)  # Adjust the value as needed

  # Add a shared xlabel to the entire figure
  if datasets == "base":
    g.fig.text(0.25, -0.01, 'ensembles', ha='center')
    g.fig.text(0.75, -0.01, 'ensembles', ha='center')
  else:
    g.fig.text(0.5, -0.01, 'ensembles', ha='center')
  #elif datasets == "ood_cifar":
  #  g.fig.text(0.5, -0.01, 'CIFAR 10 ensembles', ha='center')
  #elif datasets == "ood_cifar":
  #  g.fig.text(0.5, -0.01, 'ImageNet ensembles', ha='center')

  legend_kwargs= {
    'borderaxespad': 0,
    'handletextpad': 0.1,
    'handlelength': 0.5,
    'loc': 'center left',
    'bbox_to_anchor': (1.0, 0.5),
    'markerscale': 0.5, # relative size of legend marker compared to originally drawn ones.
    'frameon': True,
    #'markeredgewidth': 1.5,
    #'borderpadfloat':0,
  }
  g.add_legend(label_order=model_types,
               **legend_kwargs)

  g.tight_layout()
  plt.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')

  #%%
  plt.close()

  print('Done', flush=True)
  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
