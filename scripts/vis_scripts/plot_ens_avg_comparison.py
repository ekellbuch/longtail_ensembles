"""
Having used ../metrics__
to store ensemble model prediction metrics

Plot the ensemble metrics x-y avg probs vs avg logits.
python scripts/vis_scripts/plot_ens_avg_comparison.py --ind_dataset=cifar10 --ood_dataset=cinic10
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import fire
import ast
import seaborn as sns
from sklearn.metrics import r2_score
from longtail_ensembles.utils import read_dataset
import matplotlib
from scipy import stats

#BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
BASE_DIR = Path("/data/Projects/linear_ensembles/longtail_ensembles")

plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%%

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v2"):

  #%%
  data_filename = output_dir / out_name / "{}--{}.csv".format(ind_dataset, ood_dataset)
  all_datas = read_dataset(data_filename)
  output_filename = output_dir / "figures" / out_name / ("one_vs_{}_{}.pdf".format(ind_dataset, ood_dataset))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Output stored at {}'.format(output_filename))

  all_subdatas=all_datas.copy()
  all_subdatas = all_subdatas[np.logical_or(all_subdatas['ind_dataset']==ind_dataset,all_subdatas['ood_dataset']==ood_dataset)]
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1
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
  all_subdatas.rename(columns={'nll': 'NLL','ece':'ECE','brier':'Brier', 'f1':'F1'}, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     'no_avg': 'single model'})

  all_subdatas = all_subdatas[all_subdatas['acc'] >= 0.25]
  all_subdatas = all_subdatas[~all_subdatas['models'].str.contains('deepens')]
  plot_results = all_subdatas.copy()
  plot_results["data_type"] = plot_results["data_type"].replace({"ind": "InD", "ood": "OOD"})
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['ind_dataset'] = pd.Categorical(plot_results['ind_dataset'])
  plot_results['ood_dataset'] = pd.Categorical(plot_results['ood_dataset'])
  plot_results['data_type'] = pd.Categorical(plot_results['data_type'])
  #%%
  plot_results = plot_results.drop(columns=['model_scores','ensemble_size','ensemble_type', 'architecture','ind_dataset','ood_dataset','seed',
                                            'num_params','model_class_id','binning','acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  #%
  plot_results.set_index(['models','Ensemble Type','data_type'], inplace=True)
  plot_results.columns.name = "Metric"
  plot_results = plot_results.unstack(['Ensemble Type']).stack('Metric')
  #%%
  new_plot_results = plot_results.reset_index().set_index("models")

  model_types = ['avg. probs', 'avg. logits']#new_plot_results['Ensemble Type'].unique()
  metrics=["0-1 Error", "NLL", "Brier","F1", "ECE", "CalROCauc", "CalPRauc"]
  #%%
  g = sns.FacetGrid(
    data=new_plot_results,
    col="Metric",
    row="data_type",
    #hue='Ensemble Type',
    hue_order=model_types,
    margin_titles=True,
    despine=False,
    legend_out=True,
    aspect=1, height=4.,
    sharey=False,
    sharex=False,
    col_order=metrics,
    palette=colors,
  )

  g.map_dataframe(sns.scatterplot, x=model_types[0], y=model_types[1])
  g.tight_layout()
  #%
  for m_idx, metric_type in enumerate(metrics):
    single_models = new_plot_results.loc[(new_plot_results['Metric'] == metric_type)]

    for row_idx, row_val in enumerate(['InD', 'OOD']):
      all_total = single_models[single_models['data_type'] == row_val]

      x_total = all_total[model_types[0]].values
      y_total = all_total[model_types[1]].values

      lims_x = np.round(min(x_total.min(), y_total.min()),3)
      lims_y = np.round(min(x_total.max(), y_total.max()),3)
      g.axes[row_idx, m_idx].axline([lims_x, lims_x], [lims_y, lims_y], color='k', linestyle='--',
                              linewidth=1)  # , alpha=0.75, linewidth=0.75)
      #"""
      #
  g.add_legend(handletextpad=0)
  g.tight_layout()
  g.savefig(output_filename, bbox_inches='tight')
  #plt.show()
  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
#%%

