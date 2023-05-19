"""
Having used ../metrics__

Plot boxplots for different ensembles
Run:
python scripts/vis_scripts/plot_ens_means.py --ind_dataset=cifar10 --ood_dataset=cinic10
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import fire
import ast
import seaborn as sns
from longtail_ensembles.utils import read_dataset

import matplotlib

#BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
BASE_DIR = Path("/data/Projects/linear_ensembles/longtail_ensembles")

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"



#%%

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v2"):

  #%%
  #ind_dataset = 'cifar10'
  #ood_dataset = 'cinic10'
  ##ind_dataset = 'imagenet'
  #ood_dataset = 'imagenetv2mf'
  #out_name = "model_performance/het_run_v20"
  #%%
  data_filename = output_dir / out_name / "{}--{}.csv".format(ind_dataset, ood_dataset)
  all_datas = read_dataset(data_filename)
  output_filename = output_dir / "figures" / out_name / ("ens_box_{}_{}_box.pdf".format(ind_dataset, ood_dataset))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Output stored at {}'.format(output_filename))

  all_subdatas=all_datas.copy()
  all_subdatas = all_subdatas[np.logical_or(all_subdatas['ind_dataset']==ind_dataset,all_subdatas['ood_dataset']==ood_dataset)]
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  #assert len(ens_size) == 1
  ens_size = ens_size[0]

  # color given score
  colors = ['blue', 'purple','green','red']
  colors = ['red', 'green','purple', 'blue']
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']

  cmap = matplotlib.colors.ListedColormap(colors)

  #%% color by std in predictive accuracy?
  apply_f = lambda x: [ast.literal_eval(x_) for x_ in x ]

  color_method = 'ensemble_avg_type'
  if color_method =='std':
    color_method_name = 'Ensemble Acc std.'
    ensemble_hyper = apply_f(all_subdatas['ensemble_hyperparameters'])
    label_flag = [np.asarray(ensemble_hyp_['model_acc']).std() for ensemble_hyp_ in ensemble_hyper]
  elif color_method == 'avgnll':
      color_method_name = 'Avg. model CE'
      label_flag = all_subdatas['Avg. NLL']
      #label_flag = np.digitize(ensemble_hyper, bins=np.linspace(0, 10, 5))
  elif color_method == 'ensemble_avg_type':
    color_method_name = 'Ensemble avg. type'
    label_flag = range(all_subdatas['ensemble_avg_type'].unique().shape[0])
    label_flag = None
  else:
    label_flag = None

  #%%
  kwargs={
        's': 15,
        'alpha': 0.3,
        'c': label_flag,
        'cmap': cmap,
  }
  all_subdatas['0-1 Error'] = 1 - all_subdatas['acc']
  all_subdatas.rename(columns={'nll': 'NLL', 'ece':'ECE', 'brier':'Brier', 'f1':'F1'}, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     'no_avg': 'single model.'})
  # filter out model's which didn't achieve min performance
  all_subdatas = all_subdatas[all_subdatas['acc'] >= 0.25]
  all_subdatas = all_subdatas[~all_subdatas['models'].str.contains('deepens')]
  # filter models with deepens?
  plot_results = all_subdatas.copy()
  plot_results["data_type"] = plot_results["data_type"].replace({"ind": "InD", "ood": "OOD"})
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['ind_dataset'] = pd.Categorical(plot_results['ind_dataset'])
  plot_results['ood_dataset'] = pd.Categorical(plot_results['ood_dataset'])
  plot_results['data_type'] = pd.Categorical(plot_results['data_type'])
  #%
  plot_results = plot_results.drop(columns=['model_scores','ensemble_size','ensemble_type',
                                            'architecture','ind_dataset','ood_dataset','seed',
                                            'num_params','model_class_id','binning','acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  plot_results.rename(columns={'data_type': 'Data Type'},inplace=True)
  metrics_columns=["0-1 Error", "NLL", "Brier","F1", "ECE", "CalROCauc", "CalPRauc"]

  new_plot_results = pd.melt(plot_results, id_vars=['models', 'Data Type', 'Ensemble Type'], value_vars=metrics_columns,
                      var_name='Metric', value_name='Value')
  new_plot_results = new_plot_results.dropna()

  #model_types = new_plot_results['Ensemble Type'].unique()
  model_types =['single model', 'avg. logits', 'avg. probs']
  metrics = metrics_columns
  #%%
  fig, ax = plt.subplots(1, 1, figsize=(8, 4))
  g = sns.catplot(
    data=new_plot_results,
    x="Data Type",
    y="Value",
    col="Metric",
    hue='Ensemble Type',
    hue_order=model_types,
    ax=ax,
    col_order=metrics,
    legend_out=True,
    #aspect=1, height=4.,
    sharey=False,
    #sharex=False,
    #col_order=metrics,
    palette=colors,
    kind='box',
  )

  g.set_ylabels('')
  g.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')
  plt.close()
  #plt.show()

  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
#%%

