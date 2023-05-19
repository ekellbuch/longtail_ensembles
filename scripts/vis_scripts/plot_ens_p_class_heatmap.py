"""
Plot the metrics for different ensembles, where the metrics are sorted according to the metric value.
Run:
  python scripts/vis_scripts/plot_ens_per_class.py --ind_dataset=cifar10 --ood_dataset=cinic10

Figure: per class metric comparison where the metric is the sum across classes.
as heatmaps
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

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v2"):

  #%%
  data_filename = output_dir / out_name / "pclass_{}.csv".format(ind_dataset)
  all_datas = read_dataset(data_filename)
  data_filename = output_dir / out_name / "pclass_{}.csv".format(ood_dataset)
  ood_datas = read_dataset(data_filename)
  all_datas = pd.concat([all_datas, ood_datas], axis=0)

  output_filename = output_dir / "figures" / out_name / ("ens_pclass_{}_{}.pdf".format(ind_dataset, ood_dataset))

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
  plot_results = plot_results.sort_values(by="0-1 Error")
  stacked_df = plot_results.set_index(['models','Data Type','Ensemble Type','Class ID']).stack().reset_index()
  stacked_df.columns = ['models','Data Type','Ensemble Type','Class ID','Metric','Value']

  # just do accuracy
  #%%

  # Separate logit variance vs ensemble variance
  #%%
  metrics_columns=["0-1 Error", "F1"] #,"EnsVar","ClsVar"]

  model_vals = stacked_df['models'].values

  #%%
  num_models = len(model_vals)
  # stacked_df = stacked_df[stacked_df['Ensemble Type'] == 'single model']
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
  stacked_df= stacked_df[stacked_df['Metric'] == '0-1 Error']

  col_order = ['single model', 'avg. logits', 'avg. probs']
  #%%
  g = sns.FacetGrid(
    stacked_df,
    row="Data Type",
    col="Ensemble Type",
    #hue='Ensemble Type',
    #sharey='col',
    #sharex=True,
    #aspect=2.5,
    height=3,
    #palette=colors,
    row_order=['InD','OOD'],
    col_order=col_order,
    #hue_order=model_types,
    legend_out=True,
    despine=False,
    margin_titles=True,
    #subplot_kws=subplot_kws,
    #hue_kws=hue_kws,
  )

  other_kwargs= {
    #'s': 5,
    #'alpha': 0.7,
    #'edgewidth': [5, 5, 5]
    #'facecolor': [None, None, None]

  }

  def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)

  g.map_dataframe(draw_heatmap, 'Class ID', 'models', 'Value', cbar=True, square=False, vmin=0, vmax=1, cmap='RdBu_r')

  #g.set_xticklabels([])
  g.set_xlabels('')
  g.set_ylabels('')
  #g.set_xlim(0, num_models)
  #g.set_axis_labels(x_var='models')
  g.set_titles(col_template="{col_name}", row_template="{row_name}")

  #plt.subplots_adjust(right=0.85)  # Adjust the value as needed

  # Add a shared xlabel to the entire figure
  # g.fig.text(0.5, -0.01, 'models', ha='center')

  legend_kwargs= {
    'borderaxespad': 0,
    'handletextpad': 0,
    'loc': 'center left',
    'bbox_to_anchor': (1.0, 0.5),
    'markerscale': 0.5, # relative size of legend marker compared to originally drawn ones.
    'frameon': True,
    #'markeredgewidth': 1.5,
    #'borderpadfloat':0,
  }
  #g.add_legend(label_order=model_types,
  #             **legend_kwargs)

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
