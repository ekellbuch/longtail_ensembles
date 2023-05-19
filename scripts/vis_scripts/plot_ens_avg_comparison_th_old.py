"""
Plot the metrics for different ensembles, where the metrics are sorted according to the metric value.
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
from longtail_ensembles.utils import read_dataset

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
#BASE_DIR = Path("/data/Projects/linear_ensembles/longtail_ensembles")

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%

#%%

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v2"):

  #%%
  #ind_dataset = 'cifar10'
  #ood_dataset = 'cinic10'
  ##ind_dataset = 'imagenet'
  #ood_dataset = 'imagenetv2mf'
  #out_name = "model_performance/het_run_v6"

  #%%
  data_filename = output_dir / out_name / "{}.csv".format(ind_dataset)
  all_datas = read_dataset(data_filename)
  data_filename = output_dir / out_name / "{}.csv".format(ood_dataset)
  ood_datas = read_dataset(data_filename)
  all_datas = pd.concat([all_datas, ood_datas], axis=0)

  output_filename = output_dir / "figures" / out_name / ("ens_th_{}_{}.pdf".format(ind_dataset, ood_dataset))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Storing output in:\n{}'.format(output_filename), flush=True)

  all_subdatas=all_datas.copy()
  all_subdatas = all_subdatas[np.logical_or(all_subdatas['dataset']==ind_dataset,all_subdatas['dataset']==ood_dataset)]
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1

  all_subdatas['0-1 Error'] = 1 - all_subdatas['acc']
  all_subdatas.rename(columns={'nll': 'NLL',
                               'ece':'ECE',
                               'brier':'Brier',
                               'f1':'F1'}, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     'no_avg': 'single model'})

  #all_subdatas = all_subdatas[all_subdatas['acc'] >= 0.25]
  all_subdatas = all_subdatas[~all_subdatas['models'].str.contains('deepens')]
  plot_results = all_subdatas.copy()
  plot_results["data_type"] = plot_results["data_type"].replace({"ind": "InD", "ood": "OOD"})
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['dataset'] = pd.Categorical(plot_results['dataset'])
  plot_results['data_type'] = pd.Categorical(plot_results['data_type'])
  #%
  plot_results = plot_results.drop(columns=['model_scores',
                                            'ensemble_size',
                                            'ensemble_type',
                                            'architecture',
                                            'dataset',
                                            'seed',
                                            'num_params',
                                            'binning',
                                            'acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  plot_results.rename(columns={'data_type': 'Data Type'},inplace=True)
  metrics_columns=["0-1 Error", "NLL", "Brier","F1", "ECE", "CalROCauc", "CalPRauc","EnsVar","ClsVar"]
  # does this sort ind and ood separately?
  #%%
  #import pdb; pdb.set_trace()
  plot_results = plot_results.sort_values(by="0-1 Error")
  plot_results = plot_results.reset_index(drop=True)

  new_plot_results = pd.melt(plot_results, id_vars=['models', 'Data Type', 'Ensemble Type'], value_vars=metrics_columns, var_name='Metric', value_name='Value')
  new_plot_results = new_plot_results.dropna()

  model_types =['single model', 'avg. logits', 'avg. probs']
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']

  #%%
  # Separate logit variance vs ensemble variance
  #%%
  metrics_columns=["0-1 Error", "NLL", "Brier","F1", "CalPRauc","EnsVar","ClsVar"]
  metrics_columns=["0-1 Error", "NLL", "Brier", "F1", "CalROCauc"]#,"EnsVar","ClsVar"]

  #debug
  #import pdb; pdb.set_trace()
  #new_plot_results= new_plot_results[:21]
  # replace model name by model type

  #all_model_names = (new_plot_results['models'].unique())
  # sort model name by performance
  
  model_idx = new_plot_results[np.logical_and(new_plot_results['Metric'] == '0-1 Error', new_plot_results['Data Type'] == 'InD')]
  model_idx = model_idx[model_idx['Ensemble Type'] == 'single model']
  assert len(model_idx['models'].values) == len(np.unique(model_idx['models'].values))
  metric_vals = np.argsort(model_idx['Value'].values)
  model_vals = model_idx['models'].values
  model_vals_dict = dict(zip(model_vals, metric_vals))
  new_plot_results['models'] = new_plot_results['models'].replace(model_vals_dict)
  #%%
  import pdb; pdb.set_trace()
  #%%
  g = sns.FacetGrid(new_plot_results, col="Metric", row="Data Type", hue='Ensemble Type')
  g.map(sns.scatterplot, y="Value", x="models" )
  #%%
  """
  fig, ax = plt.subplots(1, 1, figsize=(12, 4))
  g = sns.catplot(
    data=new_plot_results,
    row="Data Type",
    x="models",
    y="Value",
    col="Metric",
    hue='Ensemble Type',
    hue_order=model_types,
    ax=ax,
    col_order=metrics_columns,
    legend_out=True,
    #aspect=1, height=4.,
    sharey=False,
    sharex=True,
    #col_order=metrics,
    palette=colors,
    margin_titles=True,
    #kind='box',
  )

  #%%
  #g.set_titles("{col_name} {col_var}")
  g.set_xticklabels([])
  g.set_xlabels('')
  g.set_ylabels('')
  g.legend.handletextpad = 0
  g.legend.borderaxespad = 0
  g.legend.borderpadfloat = 0

  # add metric name to columns names:
  #import pdb; pdb.set_trace()
  #for g_ax in g.axes[0]:
  #  g_ax.set_title("{col_name} {col_var}")
  #g.axes[0][0].set_ylabel('{row_name} {row_var}')
  #g.axes[1][0].set_ylabel('{row_name} {row_var}')
  #g.axes[0][0].set_ylabel('InD')
  #g.axes[1][0].set_ylabel('OOD')
  #g.axes[0][0].set_ylim([-5, 5])
  #g.axes[1][0].set_ylim([-5, 5])
  
  """

  g.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')

  #plt.show()
  #%%
  plt.close()

  print('Done', flush=True)
  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
