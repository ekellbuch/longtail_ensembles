"""
Plot the error for ensemble sorted by classes
Run:
  python scripts/vis_scripts/plot_ens_acc_class.py --ind_dataset=cifar10 --ood_dataset=cinic10

Figure: barplot
x-axis: class ID,
y-axis: error value
col: Data Type (ind or OOD)
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
from matplotlib.ticker import MaxNLocator

import itertools
BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%

#%%

cifar10_cls_idx = {
              0: 'airplane',
              1: 'automobile',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck'}

imagenet_cls_idx = {'0': 'tench'}

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v2"):

  #%%
  if 'imagenet' in ind_dataset:
    cls_naming = None #imagenet_cls_idx
  else:
    cls_naming = cifar10_cls_idx

  #%%
  data_filename = output_dir / out_name / "pclass_{}.csv".format(ind_dataset)
  all_datas = read_dataset(data_filename)
  data_filename = output_dir / out_name / "pclass_{}.csv".format(ood_dataset)
  ood_datas = read_dataset(data_filename)
  all_datas = pd.concat([all_datas, ood_datas], axis=0)

  output_filename = output_dir / "figures" / out_name / ("ens_pclass_acc_{}_{}.pdf".format(ind_dataset, ood_dataset))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Storing output in:\n{}'.format(output_filename), flush=True)

  all_subdatas=all_datas.copy()
  all_subdatas = all_subdatas[np.logical_or(all_subdatas['dataset']==ind_dataset,all_subdatas['dataset']==ood_dataset)]
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1

  # preprocess model
  all_subdatas['0-1 Error'] = 1 - all_subdatas['acc']
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type',
                               'data_type': 'Data Type',
                               'f1':'F1',
                               }, inplace=True)
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
                                            'dataset',
                                            'seed',
                                            'num_params',
                                            'binning',
                                            'acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  #%
  # drop single model performance
  metrics_columns=["0-1 Error", "F1"] #,"EnsVar","ClsVar"]

  plot_results = plot_results.sort_values(by="0-1 Error")
  #stacked_df = plot_results.set_index(['models','Data Type','Ensemble Type','Class ID']).stack().reset_index()
  #stacked_df.columns = ['models','Data Type','Ensemble Type','Class ID','Metric','Value']
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['Data Type'] = pd.Categorical(plot_results['Data Type'])
  plot_results['Class ID'] = pd.Categorical(plot_results['Class ID'])

  stacked_df = pd.melt(plot_results, id_vars=['models', 'Data Type', 'Ensemble Type','Class ID'], value_vars=metrics_columns, var_name='Metric', value_name='Value')

  #%%

  # Separate logit variance vs ensemble variance
  #%%

  model_vals = stacked_df['models'].values

  #%%
  num_models = len(model_vals)
  # only use 1 metric
  #stacked_df = stacked_df.sort_values(by="models").reset_index()
  # cbar has to be
  #%%
  subplot_kws ={
    'xlim': (0, num_models),
  }

  hue_kws = {
      #'marker': ['o', '+', 'x'],

  }
  #stacked_df = stacked_df[:10]
  #%%
  model_idx = stacked_df[np.logical_and(stacked_df['Metric'] == '0-1 Error', stacked_df['Data Type'] == 'InD')]
  model_idx = model_idx[model_idx['Ensemble Type'] == 'single model']
  model_idx = model_idx.groupby(['models']).agg({'Value': sum}).reset_index()
  assert len(model_idx['models'].values) == len(np.unique(model_idx['models'].values))
  metric_vals = np.argsort(model_idx['Value'].values)
  model_vals = model_idx['models'].values
  model_vals_dict = dict(zip(model_vals, metric_vals))
  stacked_df['models'] = stacked_df['models'].replace(model_vals_dict)

  #%%
  # use only 0-1 error
  stacked_df = stacked_df[stacked_df['Metric'] == '0-1 Error']
  #stacked_df = stacked_df[stacked_df['Ensemble Type'] == 'single model']
  #%% sort classes

  #%%
  model_idx = stacked_df[np.logical_and(stacked_df['Metric'] == '0-1 Error', stacked_df['Data Type'] == 'InD')]
  model_idx = model_idx[model_idx['Ensemble Type'] == 'single model']
  model_idx = model_idx.groupby(['Class ID']).agg({'Value': sum}).reset_index()
  metric_vals = np.argsort(model_idx['Value'].values)
  model_vals = model_idx['Class ID'].values
  model_vals_dict = dict(zip(model_vals, metric_vals))
  # this overwrites the class id
  #stacked_df['Class ID'] = stacked_df['Class ID'].replace(model_vals_dict)
  order = metric_vals

  # if we have the class name use that instead
  if not(cls_naming is None):
    stacked_df['Class ID'] = stacked_df['Class ID'].cat.rename_categories(cifar10_cls_idx)
    order = [cifar10_cls_idx[i] for i in order]
  #%%
  data_order =['InD', 'OOD']
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']
  model_types =['single model', 'avg. logits', 'avg. probs']

  #%%
  g = sns.catplot(
    data=stacked_df,
    order =order,
    x="Class ID",
    y="Value",
    col="Data Type",
    #row="Data Type",
    sharex=True,
    sharey=True,
    aspect=2.5,
    height=3,
    #row_order=['InD','OOD'],
    col_order=data_order,
    legend_out=True,
    #despine=False,
    margin_titles=True,
    hue='Ensemble Type',
    hue_order=model_types,
    palette=colors,
    kind='bar',
  )

  g.set_xlabels('')
  g.set_ylabels('')
  #g.set_xlim(0, num_models)
  #g.set_axis_labels(x_var='models')
  g.set_titles(col_template="{col_name}", row_template="{row_name}")

  #plt.subplots_adjust(right=0.85)  # Adjust the value as needed

  # Add a shared xlabel to the entire figure
  # g.fig.text(0.5, -0.01, 'models', ha='center')
  g.set_xticklabels(rotation=30, ha='right')
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
